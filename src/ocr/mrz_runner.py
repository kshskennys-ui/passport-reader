"""Targeted second-pass OCR for MRZ regions found in the baseline output."""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Callable, Protocol

import cv2
import numpy as np

from config import MRZConfig, OCRConfig
from image_utils import read_image, write_png
from ocr.models import OCRLine
from ocr.mrz_locator import MRZRegion, locate_mrz_region
from ocr.path_utils import resolve_project_path
from ocr.paddle_engine import PaddleOCREngine
from ocr.mrz_rows import build_row_crops, merge_row_lines
from ocr.mrz_parser import parse_mrz_row_results


class MRZOCRBackend(Protocol):
    def recognize(self, image_path: Path) -> list[OCRLine]: ...


MRZ_ALLOWED = re.compile(r"[A-Z0-9<]")
ProgressCallback = Callable[[int, int, dict], None]


class MRZSecondPassRunner:
    def __init__(self, ocr_config: OCRConfig, mrz_config: MRZConfig, engine: MRZOCRBackend | None = None):
        self.mrz_config = mrz_config
        self.engine = engine or PaddleOCREngine(ocr_config)

    def run(
        self,
        baseline_root: str | Path,
        output_root: str | Path,
        *,
        all_pages: bool = False,
        resume: bool = True,
        callback: ProgressCallback | None = None,
    ) -> tuple[list[dict], Path]:
        baseline = Path(baseline_root).resolve()
        output = Path(output_root).resolve()
        inputs = sorted((baseline / "results").rglob("ocr.json"))
        if not all_pages:
            inputs = [path for path in inputs if "mrz_candidate_incomplete" in _read_json(path).get("warnings", [])]
        results: list[dict] = []
        batch_started = time.perf_counter()
        for index, json_path in enumerate(inputs, start=1):
            result = self._process(json_path, output, resume)
            results.append(result)
            if callback:
                callback(index, len(inputs), result)
        report = write_mrz_report(
            output,
            results,
            all_pages=all_pages,
            wall_elapsed_ms=(time.perf_counter() - batch_started) * 1000,
        )
        return results, report

    def _process(self, baseline_json: Path, output_root: Path, resume: bool) -> dict:
        payload = _read_json(baseline_json)
        source_path = resolve_project_path(payload["source_image"])
        document = str(payload.get("document", baseline_json.parent.parent.name))
        page_number = int(payload.get("page_number", 0))
        relative_dir = Path(document) / baseline_json.parent.name
        result_dir = output_root / "results" / relative_dir
        result_json = result_dir / "mrz.json"
        if resume and result_json.exists():
            saved = _read_json(result_json)
            if saved.get("source_fingerprint") == _fingerprint(source_path):
                return saved | {"reused": True}

        started = time.perf_counter()
        result: dict = {
            "source_image": str(source_path),
            "source_fingerprint": _fingerprint(source_path),
            "document": document,
            "page_number": page_number,
            "status": "error",
            "warnings": [],
        }
        try:
            image = read_image(source_path)
            lines = [_line_from_dict(item) for item in payload.get("lines", [])]
            region = locate_mrz_region(lines, image.shape[:2], self.mrz_config)
            if region is None:
                result["status"] = "warning"
                result["warnings"] = ["mrz_region_not_located"]
                result["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 2)
                return _write_result(result_json, result)
            crop = _crop(image, region)
            crop_path = result_dir / "mrz_crop.png"
            write_png(crop_path, crop)
            source_overlay = _draw_region_overlay(image, region)
            source_overlay_path = result_dir / "mrz_region_overlay.png"
            write_png(source_overlay_path, source_overlay)
            second_lines = self.engine.recognize(crop_path)
            ocr_overlay_path = result_dir / "mrz_ocr_overlay.png"
            write_png(ocr_overlay_path, _draw_ocr_overlay(crop, second_lines))
            row_results = build_direct_row_results(second_lines, self.mrz_config)
            parsed = parse_mrz_row_results(row_results)
            row_files: list[dict] = []
            if parsed.get("status") != "valid":
                row_results, row_files = self._run_row_fallback(image, region, result_dir, output_root)
                parsed = parse_mrz_row_results(row_results)
            result.update(
                {
                    "status": "ok",
                    "region": region.as_dict(),
                    "source_candidate_line_count": len(lines),
                    "second_pass_lines": [line.as_dict() for line in second_lines],
                    "row_results": row_results,
                    "mrz_parse": parsed,
                    "metrics": _second_pass_metrics(second_lines, row_results),
                    "files": {
                        "mrz_crop": str(crop_path.relative_to(output_root)),
                        "region_overlay": str(source_overlay_path.relative_to(output_root)),
                        "ocr_overlay": str(ocr_overlay_path.relative_to(output_root)),
                        "row_passes": row_files,
                    },
                }
            )
            parse_status = result["mrz_parse"].get("status")
            if parse_status == "invalid":
                result["warnings"].append("mrz_parse_invalid")
            elif parse_status == "partial":
                result["warnings"].append("mrz_parse_partial")
            elif parse_status == "incomplete":
                result["warnings"].append("mrz_parse_incomplete")
            elif parse_status == "unsupported_format":
                result["warnings"].append("mrz_parse_unsupported_format")
        except Exception as exc:
            result["error"] = f"{type(exc).__name__}: {exc}"
        result["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 2)
        return _write_result(result_json, result)

    def _run_row_fallback(
        self, image: np.ndarray, region: MRZRegion, result_dir: Path, output_root: Path
    ) -> tuple[list[dict], list[dict]]:
        return run_row_ocr_fallback(
            self.engine, image, region, result_dir, output_root, self.mrz_config
        )


def _second_pass_metrics(lines: list[OCRLine], row_results: list[dict] | None = None) -> dict:
    mrz_like = []
    for line in lines:
        compact = "".join(line.text.upper().split())
        if len(compact) < 28 or len(compact) > 46:
            continue
        allowed = len(MRZ_ALLOWED.findall(compact)) / len(compact)
        if allowed >= 0.85:
            mrz_like.append(line)
    confidences = [line.confidence for line in lines]
    row_results = row_results or []
    row_lengths = [len(item.get("normalized_text", "")) for item in row_results]
    row_texts = [length for length in row_lengths if length > 0]
    row_confidences = [
        float(item["confidence"])
        for item in row_results
        if item.get("confidence") is not None
    ]
    return {
        "line_count": len(lines),
        "mrz_like_line_count": len(mrz_like),
        "mean_confidence": round(sum(confidences) / len(confidences), 6) if confidences else None,
        "mrz_like_confidence": round(
            sum(line.confidence for line in mrz_like) / len(mrz_like), 6
        )
        if mrz_like
        else None,
        "row_count": len(row_results),
        "rows_with_text": len(row_texts),
        "row_lengths": row_lengths,
        "row_lengths_equal": len(row_texts) >= 2 and len(set(row_texts)) == 1,
        "row_mean_confidence": round(sum(row_confidences) / len(row_confidences), 6)
        if row_confidences
        else None,
    }


def build_direct_row_results(lines: list[OCRLine], config: MRZConfig) -> list[dict]:
    """Turn complete OCR lines into parser rows before using expensive row crops."""
    candidates: list[dict] = []
    for index, line in enumerate(lines, start=1):
        compact = "".join(line.text.upper().split())
        allowed_ratio = len(MRZ_ALLOWED.findall(compact)) / max(1, len(compact))
        if (
            28 <= len(compact) <= 46
            and ("<" in compact or _looks_like_td3_data_row(compact))
            and allowed_ratio >= config.row_minimum_allowed_ratio
        ):
            candidates.append(
                {
                    "row_index": index,
                    "raw_text": line.text,
                    "normalized_text": compact,
                    "length": len(compact),
                    "fragment_count": 1,
                    "confidence": line.confidence,
                    "ocr_lines": [line.as_dict()],
                }
            )
    return candidates


def _looks_like_td3_data_row(value: str) -> bool:
    """Accept a TD3 data row whose optional-data field has no filler characters."""
    return (
        len(value) == 44
        and value[0].isalnum()
        and sum(character.isdigit() for character in value) >= 8
    )


def run_row_ocr_fallback(
    engine: MRZOCRBackend,
    image: np.ndarray,
    region: MRZRegion,
    result_dir: Path,
    output_root: Path,
    config: MRZConfig,
) -> tuple[list[dict], list[dict]]:
    row_results: list[dict] = []
    row_files: list[dict] = []
    for row_crop in build_row_crops(image, region, config):
        row_path = result_dir / f"mrz_row_{row_crop.row_index:02d}.png"
        row_overlay_path = result_dir / f"mrz_row_{row_crop.row_index:02d}_ocr_overlay.png"
        write_png(row_path, row_crop.image)
        row_lines = engine.recognize(row_path)
        row_text = merge_row_lines(row_lines, row_crop, config)
        write_png(row_overlay_path, _draw_ocr_overlay(row_crop.image, row_lines))
        row_result = _row_result_payload(row_text, row_crop, row_lines)
        variants: list[dict] = []
        if len(row_text.normalized_text) != 44 or (row_text.confidence or 0.0) < 0.75:
            gray = cv2.cvtColor(row_crop.image, cv2.COLOR_BGR2GRAY)
            otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            otsu_image = cv2.cvtColor(otsu, cv2.COLOR_GRAY2BGR)
            otsu_path = result_dir / f"mrz_row_{row_crop.row_index:02d}_otsu.png"
            otsu_overlay_path = result_dir / f"mrz_row_{row_crop.row_index:02d}_otsu_ocr_overlay.png"
            write_png(otsu_path, otsu_image)
            otsu_lines = engine.recognize(otsu_path)
            otsu_text = merge_row_lines(otsu_lines, row_crop, config)
            write_png(otsu_overlay_path, _draw_ocr_overlay(otsu_image, otsu_lines))
            if _row_candidate_rank(otsu_text, row_crop.row_index) > _row_candidate_rank(
                row_text, row_crop.row_index
            ):
                row_text = otsu_text
                row_lines = otsu_lines
                row_result = _row_result_payload(row_text, row_crop, row_lines)
            variants.append(
                {
                    "name": "otsu",
                    "image": str(otsu_path.relative_to(output_root)),
                    "overlay": str(otsu_overlay_path.relative_to(output_root)),
                    "length": len(otsu_text.normalized_text),
                    "confidence": otsu_text.confidence,
                }
            )
        row_results.append(row_result)
        row_files.append({
            "row_index": row_crop.row_index,
            "crop": str(row_path.relative_to(output_root)),
            "overlay": str(row_overlay_path.relative_to(output_root)),
            "variants": variants,
        })
    return row_results, row_files


def _row_result_payload(row_text, row_crop, row_lines) -> dict:
    payload = row_text.as_dict()
    payload.update(
        {
            "source_rect": row_crop.source_rect.as_dict(),
            "crop_rect": row_crop.crop_rect.as_dict(),
            "scale": row_crop.scale,
            "ocr_lines": [line.as_dict() for line in row_lines],
        }
    )
    return payload


def _row_candidate_rank(row_text, row_index: int) -> tuple[int, float, float]:
    text = row_text.normalized_text
    exact_length = int(len(text) == 44)
    first_row_marker = int(row_index == 1 and text.startswith("P<"))
    confidence = float(row_text.confidence or 0.0)
    allowed_ratio = len(MRZ_ALLOWED.findall(text)) / max(1, len(text))
    return exact_length, first_row_marker + confidence, allowed_ratio


def _crop(image: np.ndarray, region: MRZRegion) -> np.ndarray:
    rect = region.rect
    return image[rect.y : rect.y + rect.h, rect.x : rect.x + rect.w].copy()


def _draw_region_overlay(image: np.ndarray, region: MRZRegion) -> np.ndarray:
    overlay = image.copy()
    rect = region.rect
    cv2.rectangle(overlay, (rect.x, rect.y), (rect.x + rect.w, rect.y + rect.h), (255, 0, 180), 4)
    for row in region.rows:
        r = row.rect
        cv2.rectangle(overlay, (r.x, r.y), (r.x + r.w, r.y + r.h), (255, 180, 0), 2)
    return overlay


def _draw_ocr_overlay(image: np.ndarray, lines: list[OCRLine]) -> np.ndarray:
    overlay = image.copy()
    for line in lines:
        if len(line.polygon) < 3:
            continue
        points = np.asarray(line.polygon, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(overlay, [points], True, (40, 180, 60), 2, cv2.LINE_AA)
    return overlay


def write_mrz_report(
    output_root: Path,
    results: list[dict],
    *,
    all_pages: bool,
    wall_elapsed_ms: float | None = None,
) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    summary = {
        "pages": len(results),
        "located": len(results)
        - sum(_region_not_located(result) for result in results),
        "errors": sum(result.get("status") == "error" for result in results),
        "region_not_located": sum(_region_not_located(result) for result in results),
        "second_pass_two_lines": sum(
            result.get("metrics", {}).get("mrz_like_line_count", 0) >= 2 for result in results
        ),
        "fast_band_pages": sum(result.get("mode") == "fast_band" for result in results),
        "fallback_pages": sum(result.get("mode") == "full_page_fallback" for result in results),
        "parse_valid": sum(result.get("mrz_parse", {}).get("status") == "valid" for result in results),
        "parse_partial": sum(result.get("mrz_parse", {}).get("status") == "partial" for result in results),
        "parse_invalid": sum(result.get("mrz_parse", {}).get("status") == "invalid" for result in results),
        "parse_incomplete": sum(
            result.get("mrz_parse", {}).get("status") == "incomplete" for result in results
        ),
        "parse_unsupported_format": sum(
            result.get("mrz_parse", {}).get("status") == "unsupported_format" for result in results
        ),
        "all_pages": all_pages,
        "total_elapsed_ms": round(sum(float(result.get("elapsed_ms", 0.0)) for result in results), 2),
        "wall_elapsed_ms": round(wall_elapsed_ms, 2) if wall_elapsed_ms is not None else None,
    }
    (output_root / "summary.json").write_text(
        json.dumps({"summary": summary, "pages": results}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    rows = "\n".join(_report_row(output_root, result) for result in results)
    html = f"""<!doctype html><html lang=\"zh-CN\"><meta charset=\"utf-8\"><title>MRZ Second Pass</title>
<style>body{{font:14px Segoe UI,Microsoft YaHei,sans-serif;margin:24px;color:#17202a}}table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #d8dde3;padding:8px;text-align:left;vertical-align:top}}th{{background:#eef1f4}}img{{width:260px;max-height:180px;object-fit:contain}}.ok{{color:#147d46}}.warning{{color:#a45b00}}.error{{color:#b42318}}</style>
<h1>Phase 3 MRZ解析与校验报告</h1><p>处理页数：{summary['pages']}；MRZ定位成功：{summary['located']}；快速路径：{summary['fast_band_pages']}；全页回退：{summary['fallback_pages']}；完全通过：{summary['parse_valid']}；部分通过：{summary['parse_partial']}；解析异常：{summary['parse_invalid'] + summary['parse_incomplete'] + summary['parse_unsupported_format']}；运行错误：{summary['errors']}；批次墙钟时间：{_format_elapsed(summary['wall_elapsed_ms'])}</p>
<table><thead><tr><th>文件/页</th><th>状态</th><th>定位区域</th><th>二次OCR</th><th>图像</th></tr></thead><tbody>{rows}</tbody></table></html>"""
    report = output_root / "mrz_report.html"
    report.write_text(html, encoding="utf-8")
    return report


def _report_row(output_root: Path, result: dict) -> str:
    status = result.get("status", "error")
    metrics = result.get("metrics", {})
    parsed = result.get("mrz_parse", {})
    files = result.get("files", {})
    image = "-"
    if files.get("ocr_overlay"):
        image = f'<a href="{files["ocr_overlay"]}"><img src="{files["ocr_overlay"]}"></a>'
    warning = ", ".join(result.get("warnings", [])) or "-"
    region = result.get("region", {}).get("rect", {})
    region_text = f"x={region.get('x')} y={region.get('y')} w={region.get('w')} h={region.get('h')}"
    return (
        f"<tr><td>{result.get('document')} / {int(result.get('page_number', 0)):03d}</td>"
        f"<td class=\"{status}\">{status}</td><td>{region_text}<br>{warning}</td>"
        f"<td>{metrics.get('line_count', 0)}行 / MRZ样式 {metrics.get('mrz_like_line_count', 0)}行<br>"
        f"按行：{metrics.get('rows_with_text', 0)}/{metrics.get('row_count', 0)}；长度 {metrics.get('row_lengths', [])}；"
        f"等长：{metrics.get('row_lengths_equal', False)}<br>"
        f"解析：{parsed.get('status', 'N/A')}；格式：{parsed.get('format', 'N/A')}</td>"
        f"<td>{image}</td></tr>"
    )


def _format_elapsed(milliseconds: float | None) -> str:
    if milliseconds is None:
        return "N/A"
    return f"{milliseconds / 1000:.1f}s"


def _region_not_located(result: dict) -> int:
    warnings = result.get("warnings", [])
    return int(
        "mrz_region_not_located" in warnings
        or "mrz_region_not_located_after_fast_retry" in warnings
    )


def _line_from_dict(payload: dict) -> OCRLine:
    return OCRLine(
        polygon=[[int(value) for value in point] for point in payload.get("polygon", [])],
        text=str(payload.get("text", "")),
        confidence=float(payload.get("confidence", 0.0)),
    )


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_result(path: Path, result: dict) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def _fingerprint(path: Path) -> dict[str, int]:
    stat = path.stat()
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
