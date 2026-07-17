"""Compare saved pipeline runs with Ground Truth and build a portable HTML report."""

from __future__ import annotations

import html
import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean
from typing import Any, Iterable
from urllib.parse import quote

from config import ValidationConfig
from debug.logger import DebugRun
from validation.ground_truth import GroundTruth, GroundTruthStore

FAILURE_REASONS = (
    "rotation",
    "crop",
    "split",
    "classifier",
    "multiple_candidates",
    "unknown",
)


@dataclass
class Prediction:
    source_file: str
    source_page: int
    status: str
    selected_segment: int | None
    score: float
    rotation: int | None
    elapsed_ms: float
    debug_dir: Path
    output_path: Path | None
    candidate_scores: list[float] = field(default_factory=list)
    normalization: dict[str, Any] = field(default_factory=dict)

    @property
    def selected(self) -> bool:
        return self.status == "selected"


@dataclass
class PageEvaluation:
    truth: GroundTruth
    prediction: Prediction | None
    result: str
    failure_reason: str | None
    note: str
    review: dict | None
    flags: list[str] = field(default_factory=list)

    @property
    def detection_correct(self) -> bool:
        return self.prediction is not None and self.prediction.selected == self.truth.contains_data_page


@dataclass
class ValidationSummary:
    pages: int
    detection_correct: int
    candidates_pending_review: int
    automatic_passes: int
    failures: int
    reviews: int
    verified_correct: int
    quality_reviews: int
    crop_complete: int
    mrz_complete: int
    portrait_complete: int
    ocr_ready: int

    @property
    def detection_accuracy(self) -> float:
        return self.detection_correct / self.pages if self.pages else 0.0

    @property
    def verified_accuracy(self) -> float | None:
        return self.verified_correct / self.reviews if self.reviews else None

    @property
    def crop_completeness(self) -> float | None:
        return self.crop_complete / self.quality_reviews if self.quality_reviews else None

    @property
    def mrz_completeness(self) -> float | None:
        return self.mrz_complete / self.quality_reviews if self.quality_reviews else None

    @property
    def portrait_completeness(self) -> float | None:
        return self.portrait_complete / self.quality_reviews if self.quality_reviews else None

    @property
    def ocr_ready_rate(self) -> float | None:
        return self.ocr_ready / self.quality_reviews if self.quality_reviews else None


class ValidationService:
    def __init__(self, store: GroundTruthStore, config: ValidationConfig) -> None:
        self.store = store
        self.config = config

    def evaluate_output(
        self, output_root: str | Path, source_files: Iterable[str] | None = None
    ) -> list[PageEvaluation]:
        root = Path(output_root)
        evaluations: list[PageEvaluation] = []
        sources = sorted(source_files or {label.source_file for label in self.store.labels})
        for source_file in sources:
            for truth in self.store.labels_for_source(source_file):
                prediction = self._read_prediction(root, truth)
                evaluations.append(self._evaluate_page(truth, prediction))
        return evaluations

    def write_report(self, output_root: str | Path, evaluations: list[PageEvaluation]) -> Path:
        root = Path(output_root)
        root.mkdir(parents=True, exist_ok=True)
        self.archive_failures(root, evaluations)
        report_path = root / self.config.report_filename
        report_path.write_text(self._render_html(root, evaluations), encoding="utf-8")
        report_path.with_suffix(".json").write_text(
            json.dumps(
                {
                    "summary": self._summary_json(self.summary(evaluations)),
                    "pages": [self._json_page(item) for item in evaluations],
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return report_path

    def summary(self, evaluations: list[PageEvaluation]) -> ValidationSummary:
        reviews = [item for item in evaluations if item.review]
        quality_reviews = [
            item
            for item in evaluations
            if item.truth.contains_data_page and item.review and isinstance(item.review.get("quality"), dict)
        ]
        qualities = [item.review["quality"] for item in quality_reviews if item.review]
        return ValidationSummary(
            pages=len(evaluations),
            detection_correct=sum(item.detection_correct for item in evaluations),
            candidates_pending_review=sum(item.result == "candidate" for item in evaluations),
            automatic_passes=sum(item.result == "automatic_pass" for item in evaluations),
            failures=sum(item.result.endswith("fail") for item in evaluations),
            reviews=len(reviews),
            verified_correct=sum(item.result == "verified_pass" for item in evaluations),
            quality_reviews=len(quality_reviews),
            crop_complete=sum(
                bool(quality.get("page_complete"))
                and bool(quality.get("portrait_complete"))
                and bool(quality.get("mrz_complete"))
                for quality in qualities
            ),
            mrz_complete=sum(bool(quality.get("mrz_complete")) for quality in qualities),
            portrait_complete=sum(bool(quality.get("portrait_complete")) for quality in qualities),
            ocr_ready=sum(
                all(
                    bool(quality.get(name))
                    for name in ("page_complete", "portrait_complete", "mrz_complete", "ocr_ready")
                )
                for quality in qualities
            ),
        )

    def archive_failures(self, output_root: Path, evaluations: list[PageEvaluation]) -> None:
        failure_root = output_root / self.config.failure_subdirectory
        for reason in FAILURE_REASONS:
            (failure_root / reason).mkdir(parents=True, exist_ok=True)
        for item in evaluations:
            if not item.result.endswith("fail") or not item.prediction:
                continue
            reason = item.failure_reason or "unknown"
            target = (
                failure_root
                / reason
                / item.truth.dataset_name
                / f"page{item.truth.source_page:03d}"
            )
            if item.prediction.debug_dir.exists():
                shutil.copytree(item.prediction.debug_dir, target, dirs_exist_ok=True)
            target.mkdir(parents=True, exist_ok=True)
            (target / "failure.json").write_text(
                json.dumps(self._json_page(item), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )

    def _read_prediction(self, output_root: Path, truth: GroundTruth) -> Prediction | None:
        run_name = DebugRun._safe_name(Path(truth.source_file).stem)
        debug_dir = output_root / "debug" / run_name / f"page{truth.source_page:03d}"
        log_path = debug_dir / "log.json"
        if not log_path.exists():
            return None
        payload = json.loads(log_path.read_text(encoding="utf-8"))
        metrics = payload.get("metrics", {})
        score_data = metrics.get("scores", {})
        orientation = metrics.get("orientation", {})
        scores = [
            float(stage.get("metadata", {}).get("scores", {}).get("final", 0.0))
            for stage in payload.get("stages", [])
            if str(stage.get("name", "")).startswith("scored_")
        ]
        output_path = output_root / "data_pages" / run_name / f"DataPage_{truth.source_page:03d}.png"
        return Prediction(
            source_file=truth.source_file,
            source_page=truth.source_page,
            status=str(payload.get("status", "error")),
            selected_segment=metrics.get("selected_segment"),
            score=float(score_data.get("final", 0.0)),
            rotation=orientation.get("angle"),
            elapsed_ms=sum(float(stage.get("elapsed_ms", 0.0)) for stage in payload.get("stages", [])),
            debug_dir=debug_dir,
            output_path=output_path if output_path.exists() else None,
            candidate_scores=sorted(scores, reverse=True),
            normalization=metrics.get("normalization", {}),
        )

    def _evaluate_page(self, truth: GroundTruth, prediction: Prediction | None) -> PageEvaluation:
        review = self.store.latest_review(truth)
        flags = self._flags(prediction)
        if prediction is None:
            return PageEvaluation(truth, None, "automatic_fail", "unknown", "Missing pipeline output", review, flags)
        if review:
            outcome = review.get("review")
            if outcome == "correct":
                quality = review.get("quality")
                if truth.contains_data_page and isinstance(quality, dict) and not all(
                    bool(quality.get(name))
                    for name in ("page_complete", "portrait_complete", "mrz_complete", "ocr_ready")
                ):
                    return PageEvaluation(
                        truth,
                        prediction,
                        "verified_fail",
                        "crop",
                        "Selection confirmed but output is not OCR ready",
                        review,
                        flags,
                    )
                return PageEvaluation(truth, prediction, "verified_pass", None, "Confirmed by reviewer", review, flags)
            reason = str(review.get("failure_reason") or "unknown")
            return PageEvaluation(truth, prediction, "verified_fail", reason, "Rejected by reviewer", review, flags)
        if not truth.contains_data_page:
            if prediction.selected:
                return PageEvaluation(
                    truth, prediction, "automatic_fail", "classifier", "False positive on a non-data page", None, flags
                )
            return PageEvaluation(truth, prediction, "automatic_pass", None, "Correctly rejected", None, flags)
        if not prediction.selected:
            reason = "rotation" if prediction.rotation != truth.rotation else "classifier"
            return PageEvaluation(truth, prediction, "automatic_fail", reason, "Data page was not selected", None, flags)
        if prediction.rotation != truth.rotation:
            return PageEvaluation(truth, prediction, "automatic_fail", "rotation", "Incorrect cardinal orientation", None, flags)
        return PageEvaluation(
            truth,
            prediction,
            "candidate",
            None,
            "Detection and rotation pass; image crop requires Review confirmation",
            None,
            flags,
        )

    def _flags(self, prediction: Prediction | None) -> list[str]:
        if prediction is None:
            return []
        flags: list[str] = []
        if len(prediction.candidate_scores) >= 2:
            delta = prediction.candidate_scores[0] - prediction.candidate_scores[1]
            if delta <= self.config.multiple_candidate_margin:
                flags.append("multiple_candidates")
        if prediction.normalization and not prediction.normalization.get("ocr_safe", True):
            flags.append("normalization_warning")
        return flags

    def _render_html(self, output_root: Path, evaluations: list[PageEvaluation]) -> str:
        summary = self.summary(evaluations)
        verification = "N/A" if summary.verified_accuracy is None else f"{summary.verified_accuracy * 100:.1f}%"
        crop_completeness = self._format_rate(summary.crop_completeness)
        mrz_completeness = self._format_rate(summary.mrz_completeness)
        portrait_completeness = self._format_rate(summary.portrait_completeness)
        ocr_ready = self._format_rate(summary.ocr_ready_rate)
        rows = "".join(self._render_row(output_root, item) for item in evaluations)
        failures = [item for item in evaluations if item.result.endswith("fail")]
        failure_counts = {reason: sum(item.failure_reason == reason for item in failures) for reason in FAILURE_REASONS}
        failure_list = "".join(
            f"<li><span>{html.escape(reason)}</span><strong>{count}</strong></li>"
            for reason, count in failure_counts.items()
            if count
        ) or "<li><span>No failures</span><strong>0</strong></li>"
        elapsed_values = [item.prediction.elapsed_ms for item in evaluations if item.prediction is not None]
        average_time = mean(elapsed_values) if elapsed_values else 0.0
        return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Data Page Validation Report</title>
<style>
body{{margin:0;background:#f4f6f8;color:#17202a;font:14px/1.45 Arial,sans-serif}}main{{max-width:1440px;margin:auto;padding:28px}}h1{{font-size:26px;margin:0 0 4px}}p{{margin:0;color:#56616e}}.cards{{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:10px;margin:22px 0}}.card{{background:#fff;border:1px solid #d9dee5;border-radius:6px;padding:14px}}.card span{{display:block;color:#667384;font-size:12px}}.card strong{{display:block;font-size:24px;margin-top:5px}}section{{background:#fff;border:1px solid #d9dee5;border-radius:6px;padding:18px;margin-top:14px}}.failures{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:8px;padding:0;list-style:none}}.failures li{{border:1px solid #e4e8ed;padding:9px;display:flex;justify-content:space-between}}table{{border-collapse:collapse;width:100%;margin-top:12px}}th,td{{border-bottom:1px solid #e4e8ed;padding:10px;text-align:left;vertical-align:middle}}th{{font-size:12px;color:#586575;background:#f8fafb;position:sticky;top:0}}img{{width:118px;height:82px;object-fit:contain;background:#f3f5f6;border:1px solid #d9dee5}}.badge{{display:inline-block;border-radius:3px;padding:3px 7px;font-size:12px;font-weight:bold}}.candidate{{background:#fff4cc;color:#805a00}}.automatic_pass,.verified_pass{{background:#dff5e8;color:#12633c}}.automatic_fail,.verified_fail{{background:#fde4e4;color:#9e2020}}.timeline{{height:6px;background:#e6ebef;width:120px;display:inline-block;vertical-align:middle}}.timeline i{{height:100%;background:#1677b9;display:block}}.muted{{color:#6c7784;font-size:12px}}@media(max-width:900px){{.cards{{grid-template-columns:repeat(2,1fr)}}main{{padding:12px}}table{{font-size:12px}}}}
</style></head><body><main>
<h1>Data Page Validation Report</h1><p>Ground Truth comparison and reviewer audit trail. Generated from saved debug artifacts.</p>
<div class="cards"><div class="card"><span>Pages</span><strong>{summary.pages}</strong></div><div class="card"><span>Selection Accuracy</span><strong>{summary.detection_accuracy * 100:.1f}%</strong></div><div class="card"><span>Crop Completeness</span><strong>{crop_completeness}</strong></div><div class="card"><span>MRZ Completeness</span><strong>{mrz_completeness}</strong></div><div class="card"><span>Portrait Completeness</span><strong>{portrait_completeness}</strong></div><div class="card"><span>OCR Ready</span><strong>{ocr_ready}</strong></div><div class="card"><span>Candidate Review Queue</span><strong>{summary.candidates_pending_review}</strong></div><div class="card"><span>Failures</span><strong>{summary.failures}</strong></div><div class="card"><span>Verified Accuracy</span><strong>{verification}</strong></div><div class="card"><span>Average Pipeline Time</span><strong>{average_time:.0f} ms</strong></div></div>
<section><h2>Failure Taxonomy</h2><ul class="failures">{failure_list}</ul></section>
<section><h2>Pages</h2><table><thead><tr><th>Dataset / Page</th><th>Result</th><th>Score</th><th>Rotation</th><th>OCR Safety</th><th>Timeline</th><th>Failure / Review</th><th>Image</th></tr></thead><tbody>{rows}</tbody></table></section>
</main></body></html>"""

    def _render_row(self, output_root: Path, item: PageEvaluation) -> str:
        prediction = item.prediction
        score = "-" if prediction is None else f"{prediction.score:.1f}"
        rotation = "-" if prediction is None else f"{prediction.rotation} / expected {item.truth.rotation}"
        elapsed = 0.0 if prediction is None else prediction.elapsed_ms
        width = min(100.0, elapsed / 10.0)
        image_path = None
        if prediction:
            image_path = prediction.output_path or prediction.debug_dir / "07_selected.png"
        image = "-" if image_path is None or not image_path.exists() else f'<img src="{self._relative_url(output_root, image_path)}" alt="page thumbnail">'
        flags = ", ".join(item.flags)
        note = item.note if not flags else f"{item.note}; {flags}"
        normalization = {} if prediction is None else prediction.normalization
        automatic_safety = "-"
        if normalization:
            state = "safe" if normalization.get("ocr_safe", False) else "warning"
            automatic_safety = f"auto {state}; {normalization.get('fallback_level', '-')}"
        quality = (item.review or {}).get("quality") or {}
        if quality:
            automatic_safety += (
                f"<br><span class=\"muted\">review: page={bool(quality.get('page_complete'))}, "
                f"portrait={bool(quality.get('portrait_complete'))}, mrz={bool(quality.get('mrz_complete'))}, "
                f"ocr={bool(quality.get('ocr_ready'))}</span>"
            )
        return (
            "<tr>"
            f"<td><strong>{html.escape(item.truth.dataset_name)}</strong><br>page {item.truth.source_page:03d}</td>"
            f"<td><span class=\"badge {item.result}\">{html.escape(item.result)}</span></td>"
            f"<td>{score}</td><td>{html.escape(rotation)}</td><td>{automatic_safety}</td>"
            f"<td>{elapsed:.0f} ms <span class=\"timeline\"><i style=\"width:{width:.0f}%\"></i></span></td>"
            f"<td>{html.escape(note)}</td><td>{image}</td></tr>"
        )

    @staticmethod
    def _relative_url(root: Path, path: Path) -> str:
        return quote(path.relative_to(root).as_posix())

    @staticmethod
    def _format_rate(value: float | None) -> str:
        return "N/A" if value is None else f"{value * 100:.1f}%"

    @staticmethod
    def _summary_json(summary: ValidationSummary) -> dict:
        return {
            "pages": summary.pages,
            "detection_correct": summary.detection_correct,
            "selection_accuracy": summary.detection_accuracy,
            "candidates_pending_review": summary.candidates_pending_review,
            "automatic_passes": summary.automatic_passes,
            "failures": summary.failures,
            "reviews": summary.reviews,
            "verified_correct": summary.verified_correct,
            "verified_accuracy": summary.verified_accuracy,
            "quality_reviews": summary.quality_reviews,
            "crop_complete": summary.crop_complete,
            "crop_completeness": summary.crop_completeness,
            "mrz_complete": summary.mrz_complete,
            "mrz_completeness": summary.mrz_completeness,
            "portrait_complete": summary.portrait_complete,
            "portrait_completeness": summary.portrait_completeness,
            "ocr_ready": summary.ocr_ready,
            "ocr_ready_rate": summary.ocr_ready_rate,
        }

    @staticmethod
    def _json_page(item: PageEvaluation) -> dict:
        prediction = None if item.prediction is None else {
            "status": item.prediction.status,
            "selected_segment": item.prediction.selected_segment,
            "score": item.prediction.score,
            "rotation": item.prediction.rotation,
            "elapsed_ms": item.prediction.elapsed_ms,
            "debug_dir": str(item.prediction.debug_dir),
            "output_path": str(item.prediction.output_path) if item.prediction.output_path else None,
            "normalization": item.prediction.normalization,
        }
        return {
            "dataset": item.truth.dataset_name,
            "source_file": item.truth.source_file,
            "source_page": item.truth.source_page,
            "expected": {
                "contains_data_page": item.truth.contains_data_page,
                "expected_pages": item.truth.expected_pages,
                "rotation": item.truth.rotation,
            },
            "prediction": prediction,
            "result": item.result,
            "failure_reason": item.failure_reason,
            "note": item.note,
            "flags": item.flags,
            "review": item.review,
        }
