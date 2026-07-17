"""Create a Phase 1B validation report from a saved pipeline output directory."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from config import load_config
from validation.evaluator import ValidationService
from validation.ground_truth import GroundTruthStore


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--config", type=Path, default=None)
    arguments = parser.parse_args()
    config = load_config(arguments.config)
    store = GroundTruthStore(config.validation.dataset_root)
    service = ValidationService(store, config.validation)
    evaluations = service.evaluate_output(arguments.output_root)
    report = service.write_report(arguments.output_root, evaluations)
    summary = service.summary(evaluations)
    print(f"report={report}")
    print(
        f"pages={summary.pages} detection_accuracy={summary.detection_accuracy * 100:.1f}% "
        f"ocr_ready={'N/A' if summary.ocr_ready_rate is None else f'{summary.ocr_ready_rate * 100:.1f}%'} "
        f"candidates={summary.candidates_pending_review} failures={summary.failures}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
