"""Run one or more labeled PDF/image inputs and append a comparable validation result."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from config import load_config
from core.pipeline import ExtractionPipeline
from validation.evaluator import ValidationService
from validation.ground_truth import GroundTruthStore


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--label", required=True, help="Stable version label, for example phase1b-v0.1")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--history", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--reuse-output", action="store_true", help="Only evaluate an already-complete output directory")
    arguments = parser.parse_args()

    config = load_config(arguments.config)
    output_root = arguments.output_root.resolve()
    if not arguments.reuse_output:
        pipeline = ExtractionPipeline(config=config)
        for source in arguments.inputs:
            pipeline.process(source, output_root)

    store = GroundTruthStore(config.validation.dataset_root)
    service = ValidationService(store, config.validation)
    source_files = [path.name for path in arguments.inputs]
    evaluations = service.evaluate_output(output_root, source_files)
    report = service.write_report(output_root, evaluations)
    summary = service.summary(evaluations)
    processed_pages = len(list((output_root / config.output.debug_subdirectory).rglob("log.json")))
    history_path = arguments.history or output_root.parent / "regression_history.json"
    _append_history(history_path, arguments.label, output_root, report, summary, processed_pages)
    print(f"report={report}")
    print(
        f"label={arguments.label} processed_pages={processed_pages} "
        f"evaluated_pages={summary.pages} detection_accuracy={summary.detection_accuracy * 100:.1f}% "
        f"ocr_ready={_format_rate(summary.ocr_ready_rate)} "
        f"candidates={summary.candidates_pending_review} failures={summary.failures}"
    )
    return 0


def _append_history(
    history_path: Path,
    label: str,
    output_root: Path,
    report: Path,
    summary,  # type: ignore[no-untyped-def]
    processed_pages: int,
) -> None:
    payload = {"runs": []}
    if history_path.exists():
        payload = json.loads(history_path.read_text(encoding="utf-8"))
    record = {
        "label": label,
        "recorded_at": datetime.now().isoformat(timespec="seconds"),
        "output_root": str(output_root),
        "report": str(report),
        "processed_pages": processed_pages,
        "pages": summary.pages,
        "detection_accuracy": round(summary.detection_accuracy, 6),
        "candidate_review_queue": summary.candidates_pending_review,
        "failures": summary.failures,
        "reviewed_pages": summary.reviews,
        "verified_accuracy": summary.verified_accuracy,
        "crop_completeness": summary.crop_completeness,
        "mrz_completeness": summary.mrz_completeness,
        "portrait_completeness": summary.portrait_completeness,
        "ocr_ready_rate": summary.ocr_ready_rate,
    }
    runs = payload.setdefault("runs", [])
    payload["runs"] = [existing for existing in runs if existing.get("label") != label]
    payload["runs"].append(record)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _format_rate(value: float | None) -> str:
    return "N/A" if value is None else f"{value * 100:.1f}%"


if __name__ == "__main__":
    raise SystemExit(main())
