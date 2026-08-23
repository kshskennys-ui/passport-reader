"""Fast MRZ-first OCR with a full-page fallback for uncertain pages."""

from __future__ import annotations

import json
import re
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Protocol

from config import MRZConfig, OCRConfig
from image_utils import read_image, write_png
from ocr.models import OCRLine
from ocr.mrz_locator import locate_mrz_region
from ocr.mrz_parser import parse_mrz_row_results
from ocr.mrz_runner import (
    _draw_ocr_overlay,
    _fingerprint,
    _second_pass_metrics,
    build_direct_row_results,
    run_row_ocr_fallback,
    write_mrz_report,
)
from ocr.paddle_engine import PaddleOCREngine

FastProgressCallback = Callable[[int, int, dict], None]
_WORKER_RUNNER: "FastMRZRunner | None" = None
_WORKER_INPUT_ROOT: Path | None = None
_WORKER_OUTPUT_ROOT: Path | None = None


class FastMRZOCRBackend(Protocol):
    def recognize(self, image_path: Path) -> list[OCRLine]: ...


class FastMRZRunner:
    """Process only the lower page band first, then retry uncertain pages fully."""

    pipeline_mode = "fast_mrz_band_v1"

    def __init__(
        self,
        ocr_config: OCRConfig,
        mrz_config: MRZConfig,
        engine: FastMRZOCRBackend | None = None,
    ):
        self.ocr_config = ocr_config
        self.mrz_config = mrz_config
        self._engine = engine

    @property
    def engine(self) -> FastMRZOCRBackend:
        """Load PaddleOCR lazily so the parent does not create an unused model."""
        if self._engine is None:
            self._engine = PaddleOCREngine(self.ocr_config)
        return self._engine

    def run(
        self,
        input_root: str | Path,
        output_root: str | Path,
        *,
        resume: bool = True,
        workers: int = 1,
        pages: set[int] | None = None,
        callback: FastProgressCallback | None = None,
    ) -> tuple[list[dict], Path]:
        source_root = Path(input_root).resolve()
        target_root = Path(output_root).resolve()
        images = self.discover_images(source_root)
        if pages is not None:
            images = [image for image in images if _page_number(image) in pages]
        batch_started = time.perf_counter()
        if workers < 1:
            raise ValueError("workers must be at least 1")
        if workers > 1:
            return self._run_parallel(
                images,
                source_root,
                target_root,
                workers=workers,
                resume=resume,
                callback=callback,
                batch_started=batch_started,
            )
        results: list[dict] = []
        for index, image_path in enumerate(images, start=1):
            result = self._process(image_path, source_root, target_root, resume)
            results.append(result)
            if callback:
                callback(index, len(images), result)
        report = write_mrz_report(
            target_root,
            results,
            all_pages=True,
            wall_elapsed_ms=(time.perf_counter() - batch_started) * 1000,
        )
        return results, report

    def _run_parallel(
        self,
        images: list[Path],
        input_root: Path,
        output_root: Path,
        *,
        workers: int,
        resume: bool,
        callback: FastProgressCallback | None,
        batch_started: float,
    ) -> tuple[list[dict], Path]:
        """Run independent pages in separate processes, each with its own OCR model."""
        results: list[dict] = []
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_fast_worker,
            initargs=(self.ocr_config, self.mrz_config, input_root, output_root),
        ) as pool:
            futures = {
                pool.submit(_process_fast_worker, image_path, resume): image_path
                for image_path in images
            }
            for completed, future in enumerate(as_completed(futures), start=1):
                result = future.result()
                results.append(result)
                if callback:
                    callback(completed, len(images), result)
        results.sort(key=lambda result: (str(result.get("document", "")), int(result.get("page_number", 0))))
        report = write_mrz_report(
            output_root,
            results,
            all_pages=True,
            wall_elapsed_ms=(time.perf_counter() - batch_started) * 1000,
        )
        return results, report

    @staticmethod
    def discover_images(input_root: Path) -> list[Path]:
        if not input_root.exists():
            raise FileNotFoundError(f"Fast MRZ input directory does not exist: {input_root}")
        return sorted(
            path
            for path in input_root.rglob("*")
            if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
        )

    def _process(
        self, image_path: Path, input_root: Path, output_root: Path, resume: bool
    ) -> dict:
        relative = image_path.relative_to(input_root)
        document = relative.parent.as_posix() if relative.parent != Path(".") else input_root.name
        page_number = _page_number(image_path)
        result_dir = output_root / "results" / relative.parent / image_path.stem
        result_json = result_dir / "mrz.json"
        fingerprint = _fingerprint(image_path)
        if resume and result_json.exists():
            try:
                saved = json.loads(result_json.read_text(encoding="utf-8"))
                if (
                    saved.get("source_fingerprint") == fingerprint
                    and saved.get("pipeline_mode") == self.pipeline_mode
                ):
                    return saved | {"reused": True}
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                pass

        started = time.perf_counter()
        result: dict = {
            "source_image": str(image_path),
            "source_fingerprint": fingerprint,
            "document": document,
            "page_number": page_number,
            "pipeline_mode": self.pipeline_mode,
            "status": "warning",
            "warnings": [],
            "reused": False,
        }
        try:
            image = read_image(image_path)
            height, width = image.shape[:2]
            y1 = max(0, min(height - 1, round(height * self.mrz_config.fast_band_start_ratio)))
            y2 = max(y1 + 1, min(height, round(height * self.mrz_config.fast_band_end_ratio)))
            fast_crop = image[y1:y2, :].copy()
            fast_crop_path = result_dir / "fast_mrz_band.png"
            write_png(fast_crop_path, fast_crop)
            fast_lines = self.engine.recognize(fast_crop_path)
            fast_rows = build_direct_row_results(fast_lines, self.mrz_config)
            parsed = parse_mrz_row_results(fast_rows)
            selected_lines = fast_lines
            selected_rows = fast_rows
            selected_region = _band_region(width, height, y1, y2)
            mode = "fast_band"
            files = {
                "mrz_crop": str(fast_crop_path.relative_to(output_root)),
                "fast_band": str(fast_crop_path.relative_to(output_root)),
                "row_passes": [],
            }
            overlay_path = result_dir / "fast_mrz_ocr_overlay.png"
            write_png(overlay_path, _draw_ocr_overlay(image, _offset_lines(fast_lines, y1)))
            files["ocr_overlay"] = str(overlay_path.relative_to(output_root))

            if parsed.get("status") not in {"valid", "partial"} and self.mrz_config.fast_fallback_enabled:
                mode = "full_page_fallback"
                full_lines = self.engine.recognize(image_path)
                region = locate_mrz_region(full_lines, image.shape[:2], self.mrz_config)
                if region is not None:
                    crop = image[
                        region.rect.y : region.rect.y + region.rect.h,
                        region.rect.x : region.rect.x + region.rect.w,
                    ].copy()
                    fallback_crop_path = result_dir / "fallback_mrz_crop.png"
                    write_png(fallback_crop_path, crop)
                    crop_lines = self.engine.recognize(fallback_crop_path)
                    crop_rows = build_direct_row_results(crop_lines, self.mrz_config)
                    crop_parsed = parse_mrz_row_results(crop_rows)
                    selected_lines = crop_lines
                    selected_rows = crop_rows
                    parsed = crop_parsed
                    selected_region = region.as_dict()
                    files["mrz_crop"] = str(fallback_crop_path.relative_to(output_root))
                    fallback_overlay_path = result_dir / "fallback_mrz_ocr_overlay.png"
                    write_png(fallback_overlay_path, _draw_ocr_overlay(crop, crop_lines))
                    files["ocr_overlay"] = str(fallback_overlay_path.relative_to(output_root))
                    if parsed.get("status") != "valid":
                        selected_rows, row_files = run_row_ocr_fallback(
                            self.engine, image, region, result_dir, output_root, self.mrz_config
                        )
                        parsed = parse_mrz_row_results(selected_rows)
                        files["row_passes"] = row_files
                else:
                    result["warnings"].append("mrz_region_not_located_after_fast_retry")

            if parsed.get("status") not in {"valid"}:
                result["warnings"].append(f"mrz_parse_{parsed.get('status', 'invalid')}")
            result.update(
                {
                    "status": "ok" if parsed.get("status") == "valid" else "warning",
                    "mode": mode,
                    "region": selected_region,
                    "second_pass_lines": [line.as_dict() for line in selected_lines],
                    "row_results": selected_rows,
                    "mrz_parse": parsed,
                    "metrics": _second_pass_metrics(selected_lines, selected_rows),
                    "files": files,
                }
            )
        except Exception as exc:
            result["status"] = "error"
            result["error"] = f"{type(exc).__name__}: {exc}"
        result["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 2)
        result_dir.mkdir(parents=True, exist_ok=True)
        result_json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return result


def _band_region(width: int, height: int, y1: int, y2: int) -> dict:
    return {
        "rect": {"x": 0, "y": y1, "w": width, "h": y2 - y1},
        "rows": [],
        "confidence": 0.0,
        "method": "fast_bottom_band",
    }


def _init_fast_worker(
    ocr_config: OCRConfig,
    mrz_config: MRZConfig,
    input_root: Path,
    output_root: Path,
) -> None:
    global _WORKER_RUNNER, _WORKER_INPUT_ROOT, _WORKER_OUTPUT_ROOT
    _WORKER_RUNNER = FastMRZRunner(ocr_config, mrz_config)
    _WORKER_INPUT_ROOT = input_root
    _WORKER_OUTPUT_ROOT = output_root


def _process_fast_worker(image_path: Path, resume: bool) -> dict:
    if _WORKER_RUNNER is None or _WORKER_INPUT_ROOT is None or _WORKER_OUTPUT_ROOT is None:
        raise RuntimeError("fast OCR worker was not initialized")
    return _WORKER_RUNNER._process(image_path, _WORKER_INPUT_ROOT, _WORKER_OUTPUT_ROOT, resume)


def _offset_lines(lines: list[OCRLine], y_offset: int) -> list[OCRLine]:
    return [
        OCRLine(
            polygon=[[point[0], point[1] + y_offset] for point in line.polygon],
            text=line.text,
            confidence=line.confidence,
        )
        for line in lines
    ]


def _page_number(path: Path) -> int:
    matches = re.findall(r"\d+", path.stem)
    return int(matches[-1]) if matches else 0
