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
        for index, json_path in enumerate(inputs, start=1):
            result = self._process(json_path, output, resume)
            results.append(result)
            if callback:
                callback(index, len(inputs), result)
        report = write_mrz_report(output, results, all_pages=all_pages)
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
            row_results: list[dict] = []
            row_files: list[dict] = []
            for row_crop in build_row_crops(image, region, self.mrz_config):
                row_path = result_dir / f"mrz_row_{row_crop.row_index:02d}.png"
                row_overlay_path = result_dir / f"mrz_row_{row_crop.row_index:02d}_ocr_overlay.png"
                write_png(row_path, row_crop.image)
                row_lines = self.engine.recognize(row_path)
                row_text = merge_row_lines(row_lines, row_crop, self.mrz_config)
                write_png(row_overlay_path, _draw_ocr_overlay(row_crop.image, row_lines))
                row_result = row_text.as_dict()
                row_result.update(
                    {
                        "source_rect": row_crop.source_rect.as_dict(),
                        "crop_rect": row_crop.crop_rect.as_dict(),
                        "scale": row_crop.scale,
                        "ocr_lines": [line.as_dict() for line in row_lines],
                    }
                )
                row_results.append(row_result)
                row_files.append(
                    {
                        "row_index": row_crop.row_index,
                        "crop": str(row_path.relative_to(output_root)),
                        "overlay": str(row_overlay_path.relative_to(output_root)),
                    }
                )
            result.update(
                {
                    "status": "ok",
                    "region": region.as_dict(),
                    "source_candidate_line_count": len(lines),
                    "second_pass_lines": [line.as_dict() for line in second_lines],
                    "row_results": row_results,
                    "metrics": _second_pass_metrics(second_lines, row_results),
                    "files": {
                        "mrz_crop": str(crop_path.relative_to(output_root)),
                        "region_overlay": str(source_overlay_path.relative_to(output_root)),
                        "ocr_overlay": str(ocr_overlay_path.relative_to(output_root)),
                        "row_passes": row_files,
                    },
                }
            )
        except Exception as exc:
            result["error"] = f"{type(exc).__name__}: {exc}"
        result["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 2)
        return _write_result(result_json, result)


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


def write_mrz_report(output_root: Path, results: list[dict], *, all_pages: bool) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    summary = {
        "pages": len(results),
        "located": sum(result.get("status") == "ok" for result in results),
        "errors": sum(result.get("status") == "error" for result in results),
        "region_not_located": sum("mrz_region_not_located" in result.get("warnings", []) for result in results),
        "second_pass_two_lines": sum(
            result.get("metrics", {}).get("mrz_like_line_count", 0) >= 2 for result in results
        ),
        "all_pages": all_pages,
        "total_elapsed_ms": round(sum(float(result.get("elapsed_ms", 0.0)) for result in results), 2),
    }
    (output_root / "summary.json").write_text(
        json.dumps({"summary": summary, "pages": results}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    rows = "\n".join(_report_row(output_root, result) for result in results)
    html = f"""<!doctype html><html lang=\"zh-CN\"><meta charset=\"utf-8\"><title>MRZ Second Pass</title>
<style>body{{font:14px Segoe UI,Microsoft YaHei,sans-serif;margin:24px;color:#17202a}}table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #d8dde3;padding:8px;text-align:left;vertical-align:top}}th{{background:#eef1f4}}img{{width:260px;max-height:180px;object-fit:contain}}.ok{{color:#147d46}}.warning{{color:#a45b00}}.error{{color:#b42318}}</style>
<h1>Phase 2C MRZ按行识别报告</h1><p>处理页数：{summary['pages']}；定位成功：{summary['located']}；二次识别检出至少两条长行：{summary['second_pass_two_lines']}；错误：{summary['errors']}</p>
<table><thead><tr><th>文件/页</th><th>状态</th><th>定位区域</th><th>二次OCR</th><th>图像</th></tr></thead><tbody>{rows}</tbody></table></html>"""
    report = output_root / "mrz_report.html"
    report.write_text(html, encoding="utf-8")
    return report


def _report_row(output_root: Path, result: dict) -> str:
    status = result.get("status", "error")
    metrics = result.get("metrics", {})
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
        f"等长：{metrics.get('row_lengths_equal', False)}</td>"
        f"<td>{image}</td></tr>"
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
