"""Run Phase 2A OCR against Phase 1 safe outputs without modifying them."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from config import load_config
from ocr.baseline import OCRBaselineRunner
from ocr.report import summarize


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-resume", action="store_true")
    arguments = parser.parse_args()

    config = load_config(arguments.config)
    runner = OCRBaselineRunner(config.ocr)

    def progress(index, total, result):  # type: ignore[no-untyped-def]
        confidence = result.metrics.get("mean_confidence")
        confidence_text = "N/A" if confidence is None else f"{float(confidence) * 100:.1f}%"
        reused = " reused" if result.reused else ""
        print(
            f"[{index:03d}/{total:03d}] {result.document}/page{result.page_number:03d} "
            f"{result.status} lines={result.metrics.get('accepted_line_count', 0)} "
            f"confidence={confidence_text} time={result.elapsed_ms / 1000:.1f}s{reused}",
            flush=True,
        )

    results, report = runner.run(
        arguments.input_root,
        arguments.output_root,
        resume=not arguments.no_resume,
        limit=arguments.limit,
        callback=progress,
    )
    summary = summarize(results)
    print(f"report={report}")
    print(
        f"pages={summary['pages']} completed={summary['completed']} errors={summary['errors']} "
        f"complete_mrz_candidate_groups={summary['pages_with_complete_mrz_candidates']} "
        f"mean_confidence={summary['mean_page_confidence']}",
    )
    return 0 if not summary["errors"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
