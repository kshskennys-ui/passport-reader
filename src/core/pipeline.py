"""End-to-end, inspectable data-page extraction pipeline."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Callable

from config import PipelineConfig, load_config
from debug.logger import DebugRun
from detector.data_page_classifier import DataPageClassifier
from detector.document_detector import DocumentAnalyzer
from detector.page_segmenter import PageSegmenter
from image_utils import write_png
from loader.factory import load_pages
from models import CandidateResult, PageResult, ProcessResult, SourcePage
from normalize.normalizer import Normalizer
from preprocess.border_trim import WhiteBorderRemover
from preprocess.orientation import OrientationCorrector

LogCallback = Callable[[str], None]


class ExtractionPipeline:
    """Coordinates modules while keeping their inputs and outputs independently testable."""

    def __init__(self, config: PipelineConfig | None = None, log_callback: LogCallback | None = None) -> None:
        self.config = config or load_config()
        self.log_callback = log_callback
        self.orientation = OrientationCorrector(self.config.orientation)
        self.border_remover = WhiteBorderRemover(self.config.border_trim)
        self.document_analyzer = DocumentAnalyzer(self.config.document)
        self.segmenter = PageSegmenter(self.config.segmenter)
        self.classifier = DataPageClassifier(self.config.classifier)
        self.normalizer = Normalizer(self.config.normalizer, self.config.border_trim)

    def process(self, input_path: str | Path, output_dir: str | Path) -> ProcessResult:
        source = Path(input_path).expanduser().resolve()
        destination = Path(output_dir).expanduser().resolve()
        destination.mkdir(parents=True, exist_ok=True)
        self._log(f"Loading {source.name}")
        pages = load_pages(source, self.config.loader)
        self._log(f"Loaded {len(pages)} source page(s)")
        debug_run = DebugRun(destination, source, self.config.output, self.log_callback)
        results = [self._process_page(page, debug_run, destination, source.stem) for page in pages]
        return ProcessResult(source, results, destination)

    def _process_page(self, page: SourcePage, debug_run: DebugRun, output_root: Path, input_stem: str) -> PageResult:
        debug = debug_run.start_page(page.source_index)
        output_path = self._output_path(output_root, input_stem, page.source_index)
        output_path.unlink(missing_ok=True)
        debug.log("Started")
        timings: dict[str, float] = {}
        try:
            debug.save_stage(1, "original", page.image, 0.0, {"shape": list(page.image.shape)})

            started = perf_counter()
            oriented = self.orientation.correct(page.image)
            timings["orientation_ms"] = self._elapsed_ms(started)
            debug.save_stage(
                2,
                "rotated",
                oriented.image,
                timings["orientation_ms"],
                {
                    "angle": oriented.angle,
                    "confidence": round(oriented.confidence, 4),
                    "candidate_scores": {str(key): round(value, 4) for key, value in oriented.candidate_scores.items()},
                    "method": oriented.method,
                },
            )

            started = perf_counter()
            trimmed = self.border_remover.trim(oriented.image)
            timings["border_trim_ms"] = self._elapsed_ms(started)
            debug.save_stage(3, "trimmed", trimmed.image, timings["border_trim_ms"], {"roi": trimmed.rect.as_dict()})

            started = perf_counter()
            document = self.document_analyzer.detect(trimmed.image)
            timings["document_detection_ms"] = self._elapsed_ms(started)
            debug.save_stage(
                4,
                "document",
                document.image,
                timings["document_detection_ms"],
                {"roi": document.rect.as_dict(), "confidence": round(document.confidence, 4), "method": document.method},
            )

            started = perf_counter()
            segments = self.segmenter.segment(document.image)
            timings["segmentation_ms"] = self._elapsed_ms(started)
            for segment in segments:
                debug.save_stage(
                    5,
                    f"segment_{segment.index}",
                    segment.image,
                    timings["segmentation_ms"],
                    {"segment": segment.index, "roi": segment.rect.as_dict(), "count": len(segments)},
                )

            candidates: list[CandidateResult] = []
            for segment in segments:
                started = perf_counter()
                scores, overlay = self.classifier.classify(segment.image)
                scores = self.classifier.apply_rotation_score(scores, oriented.confidence * 100.0)
                overlay = self.classifier.annotate(segment.image, scores)
                elapsed_ms = self._elapsed_ms(started)
                candidates.append(CandidateResult(segment, scores, overlay, elapsed_ms))
                debug.save_stage(
                    6,
                    f"scored_{segment.index}",
                    overlay,
                    elapsed_ms,
                    {"segment": segment.index, "scores": scores.as_dict()},
                )
            candidate = max(
                candidates,
                key=lambda item: self.classifier.selection_score(item.scores),
            )
            accepted, confidence_path = self.classifier.confidence_decision(candidate.scores)
            debug.save_stage(
                7,
                "selected",
                candidate.segment.image,
                0.0,
                {"segment": candidate.segment.index, "scores": candidate.scores.as_dict()},
            )

            safe_input = candidate.segment.safe_image if candidate.segment.safe_image is not None else candidate.segment.image
            safe_rect = candidate.segment.safe_rect if candidate.segment.safe_rect is not None else candidate.segment.rect
            debug.save_stage(
                8,
                "safe_input",
                safe_input,
                0.0,
                {"segment": candidate.segment.index, "roi": safe_rect.as_dict()},
            )

            metrics = {
                "orientation": {
                    "angle": oriented.angle,
                    "confidence": round(oriented.confidence, 4),
                    "candidate_scores": {str(key): round(value, 4) for key, value in oriented.candidate_scores.items()},
                    "method": oriented.method,
                },
                "trim_roi": trimmed.rect.as_dict(),
                "document_roi": document.rect.as_dict(),
                "document_method": document.method,
                "segments": len(segments),
                "selected_segment": candidate.segment.index,
                "scores": candidate.scores.as_dict(),
                "selection_score": round(self.classifier.selection_score(candidate.scores), 4),
                "confidence_path": confidence_path,
                "timings_ms": {key: round(value, 2) for key, value in timings.items()},
            }
            if not accepted:
                message = (
                    f"Low confidence {candidate.scores.final:.1f} < "
                    f"{self.config.classifier.confidence_threshold:.1f}; preserved as failed case"
                )
                debug.update_metrics(**metrics)
                debug.log(message)
                debug.finalize("low_confidence", message)
                failed_path = debug.archive_failure()
                return PageResult(
                    page.source_index,
                    candidate.segment.index,
                    candidate.scores.final,
                    "low_confidence",
                    None,
                    debug.directory,
                    f"{message}; {failed_path}",
                    metrics,
                )

            started = perf_counter()
            protected_content_risk = self._protected_content_near_safe_edge(candidate)
            normalized = self.normalizer.normalize(safe_input)
            if protected_content_risk:
                normalized.ocr_safe = False
                normalized.warnings.append("protected_content_near_safe_roi_edge")
            timings["normalization_ms"] = self._elapsed_ms(started)
            debug.save_stage(
                9,
                "normalized",
                normalized.image,
                timings["normalization_ms"],
                {
                    "estimated_skew_degrees": round(normalized.estimated_skew_degrees, 3),
                    "deskew_degrees": round(normalized.deskew_degrees, 3),
                    "deskew_confidence": round(normalized.deskew_confidence, 4),
                    "residual_skew_degrees": round(normalized.residual_skew_degrees, 3),
                    "padding_px": normalized.padding_px,
                    "fallback_level": normalized.fallback_level,
                    "ocr_safe": normalized.ocr_safe,
                    "warnings": normalized.warnings,
                    "output_roi": normalized.trim_rect.as_dict(),
                },
            )
            write_png(output_path, normalized.image, self.config.output.png_compression)
            metrics["timings_ms"] = {key: round(value, 2) for key, value in timings.items()}
            metrics["normalization"] = {
                "estimated_skew_degrees": round(normalized.estimated_skew_degrees, 3),
                "deskew_degrees": round(normalized.deskew_degrees, 3),
                "deskew_confidence": round(normalized.deskew_confidence, 4),
                "residual_skew_degrees": round(normalized.residual_skew_degrees, 3),
                "padding_px": normalized.padding_px,
                "fallback_level": normalized.fallback_level,
                "protected_content_risk": protected_content_risk,
                "ocr_safe": normalized.ocr_safe,
                "warnings": normalized.warnings,
                "output_roi": normalized.trim_rect.as_dict(),
            }
            debug.update_metrics(**metrics)
            debug.log(
                f"Selected segment {candidate.segment.index}, score {candidate.scores.final:.1f}, "
                f"confidence path {confidence_path}"
            )
            debug.finalize("selected", str(output_path))
            return PageResult(
                page.source_index,
                candidate.segment.index,
                candidate.scores.final,
                "selected",
                output_path,
                debug.directory,
                "Data page written",
                metrics,
            )
        except Exception as error:  # Keep one bad scan from stopping the remaining pages.
            message = f"Processing error: {type(error).__name__}: {error}"
            debug.log(message)
            debug.finalize("error", message)
            failed_path = debug.archive_failure()
            return PageResult(page.source_index, None, 0.0, "error", None, debug.directory, f"{message}; {failed_path}")

    def _output_path(self, root: Path, input_stem: str, source_page: int) -> Path:
        safe_stem = DebugRun._safe_name(input_stem)
        return root / self.config.output.output_subdirectory / safe_stem / f"DataPage_{source_page:03d}.png"

    @staticmethod
    def _protected_content_near_safe_edge(candidate: CandidateResult) -> bool:
        segment = candidate.segment
        safe_rect = segment.safe_rect or segment.rect
        protected = [
            rect
            for rect in [candidate.scores.portrait_rect, *candidate.scores.mrz_bands]
            if rect is not None
        ]
        if not protected:
            return False
        guard = max(15, round(min(safe_rect.w, safe_rect.h) * 0.015))
        for rect in protected:
            x0 = segment.rect.x + rect.x
            y0 = segment.rect.y + rect.y
            x1 = x0 + rect.w
            y1 = y0 + rect.h
            distances = (
                x0 - safe_rect.x,
                y0 - safe_rect.y,
                safe_rect.x + safe_rect.w - x1,
                safe_rect.y + safe_rect.h - y1,
            )
            if min(distances) < guard:
                return True
        return False

    def _log(self, message: str) -> None:
        if self.log_callback:
            self.log_callback(message)

    @staticmethod
    def _elapsed_ms(started: float) -> float:
        return (perf_counter() - started) * 1000
