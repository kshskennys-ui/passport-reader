"""Resumable OCR baseline runner over immutable Phase 1 data-page images."""

from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path
from typing import Callable, Protocol

from config import OCRConfig
from image_utils import read_image, write_png
from ocr.analyzer import analyze_ocr_lines
from ocr.models import OCRLine, OCRPageResult, source_fingerprint
from ocr.overlay import draw_ocr_overlay
from ocr.paddle_engine import PaddleOCREngine
from ocr.report import write_ocr_report

ProgressCallback = Callable[[int, int, OCRPageResult], None]


class OCREngine(Protocol):
    def recognize(self, image_path: Path) -> list[OCRLine]: ...


class OCRBaselineRunner:
    def __init__(self, config: OCRConfig, engine: OCREngine | None = None) -> None:
        self.config = config
        self.engine = engine or PaddleOCREngine(config)

    def run(
        self,
        input_root: str | Path,
        output_root: str | Path,
        *,
        resume: bool = True,
        limit: int | None = None,
        callback: ProgressCallback | None = None,
    ) -> tuple[list[OCRPageResult], Path]:
        source_root = Path(input_root).resolve()
        target_root = Path(output_root).resolve()
        images = self.discover_images(source_root)
        if limit is not None:
            images = images[: max(0, limit)]
        results: list[OCRPageResult] = []
        for index, image_path in enumerate(images, start=1):
            result = self._process_image(image_path, source_root, target_root, resume)
            results.append(result)
            if callback:
                callback(index, len(images), result)
            if index % 10 == 0:
                write_ocr_report(target_root, results, self.config)
        report = write_ocr_report(target_root, results, self.config)
        return results, report

    @staticmethod
    def discover_images(input_root: Path) -> list[Path]:
        if not input_root.exists():
            raise FileNotFoundError(f"OCR input directory does not exist: {input_root}")
        extensions = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
        return sorted(
            path for path in input_root.rglob("*") if path.is_file() and path.suffix.lower() in extensions
        )

    def _process_image(
        self, image_path: Path, input_root: Path, output_root: Path, resume: bool
    ) -> OCRPageResult:
        relative = image_path.relative_to(input_root)
        document = relative.parent.as_posix() if relative.parent != Path(".") else input_root.name
        page_number = _page_number(image_path)
        result_dir = output_root / self.config.results_subdirectory / relative.parent / image_path.stem
        json_path = result_dir / "ocr.json"
        overlay_path = result_dir / "overlay.png"
        current_inference_signature = inference_signature(self.config)
        current_analysis_signature = analysis_signature(self.config)
        if resume:
            reused = self._read_reusable(
                json_path,
                image_path,
                output_root,
                current_inference_signature,
                current_analysis_signature,
            )
            if reused:
                return reused

        start = time.perf_counter()
        try:
            image = read_image(image_path)
            lines = self.engine.recognize(image_path)
            metrics, warnings = analyze_ocr_lines(lines, image.shape[:2], self.config)
            overlay = draw_ocr_overlay(
                image,
                lines,
                set(metrics.get("mrz_line_indices", [])),
                self.config,
            )
            write_png(overlay_path, overlay)
            result = OCRPageResult(
                source_image=image_path,
                document=document,
                page_number=page_number,
                status="warning" if warnings else "ok",
                elapsed_ms=(time.perf_counter() - start) * 1000,
                lines=lines,
                metrics=metrics,
                warnings=warnings,
                overlay_path=overlay_path,
                inference_signature=current_inference_signature,
                analysis_signature=current_analysis_signature,
            )
        except Exception as exc:
            result = OCRPageResult(
                source_image=image_path,
                document=document,
                page_number=page_number,
                status="error",
                elapsed_ms=(time.perf_counter() - start) * 1000,
                error=f"{type(exc).__name__}: {exc}",
                inference_signature=current_inference_signature,
                analysis_signature=current_analysis_signature,
            )
        result_dir.mkdir(parents=True, exist_ok=True)
        json_path.write_text(
            json.dumps(result.as_dict(output_root), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return result

    def _read_reusable(
        self,
        json_path: Path,
        image_path: Path,
        output_root: Path,
        current_inference_signature: str,
        current_analysis_signature: str,
    ) -> OCRPageResult | None:
        if not json_path.exists():
            return None
        try:
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            if payload.get("source_fingerprint") != source_fingerprint(image_path):
                return None
            result = OCRPageResult.from_dict(payload, output_root)
            if result.status == "error":
                return None
            saved_inference_signature = payload.get("inference_signature")
            if saved_inference_signature and saved_inference_signature != current_inference_signature:
                return None
            if (
                result.analysis_signature != current_analysis_signature
                or not result.overlay_path
                or not result.overlay_path.exists()
            ):
                image = read_image(image_path)
                metrics, warnings = analyze_ocr_lines(result.lines, image.shape[:2], self.config)
                result.metrics = metrics
                result.warnings = warnings
                result.status = "warning" if warnings else "ok"
                result.overlay_path = json_path.parent / "overlay.png"
                result.inference_signature = current_inference_signature
                result.analysis_signature = current_analysis_signature
                overlay = draw_ocr_overlay(
                    image,
                    result.lines,
                    set(metrics.get("mrz_line_indices", [])),
                    self.config,
                )
                write_png(result.overlay_path, overlay)
                json_path.write_text(
                    json.dumps(result.as_dict(output_root), ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
            return result
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
            return None


def _page_number(path: Path) -> int:
    matches = re.findall(r"\d+", path.stem)
    return int(matches[-1]) if matches else 0


def inference_signature(config: OCRConfig) -> str:
    return _signature(
        {
            "language": config.language,
            "ocr_version": config.ocr_version,
            "text_detection_model_name": config.text_detection_model_name,
            "text_recognition_model_name": config.text_recognition_model_name,
            "use_doc_orientation_classify": config.use_doc_orientation_classify,
            "use_doc_unwarping": config.use_doc_unwarping,
            "use_textline_orientation": config.use_textline_orientation,
        }
    )


def analysis_signature(config: OCRConfig) -> str:
    return _signature(
        {
            "minimum_text_confidence": config.minimum_text_confidence,
            "low_confidence_threshold": config.low_confidence_threshold,
            "minimum_mean_confidence": config.minimum_mean_confidence,
            "mrz_minimum_length": config.mrz_minimum_length,
            "mrz_maximum_length": config.mrz_maximum_length,
            "mrz_minimum_allowed_ratio": config.mrz_minimum_allowed_ratio,
            "mrz_minimum_content_width_ratio": config.mrz_minimum_content_width_ratio,
            "overlay_line_thickness": config.overlay_line_thickness,
        }
    )


def _signature(payload: dict[str, object]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]
