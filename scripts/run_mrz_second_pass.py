"""Run targeted MRZ localization and second-pass OCR."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from config import load_config
from ocr.mrz_runner import MRZSecondPassRunner


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--all", action="store_true", help="Process all baseline pages")
    parser.add_argument("--no-resume", action="store_true")
    arguments = parser.parse_args()
    config = load_config(arguments.config)
    runner = MRZSecondPassRunner(config.ocr, config.mrz)

    def progress(index, total, result):  # type: ignore[no-untyped-def]
        metrics = result.get("metrics", {})
        print(
            f"[{index:03d}/{total:03d}] {result.get('document')}/page{int(result.get('page_number', 0)):03d} "
            f"{result.get('status')} mrz_like={metrics.get('mrz_like_line_count', 0)} "
            f"time={float(result.get('elapsed_ms', 0)) / 1000:.1f}s",
            flush=True,
        )

    results, report = runner.run(
        arguments.baseline_root,
        arguments.output_root,
        all_pages=arguments.all,
        resume=not arguments.no_resume,
        callback=progress,
    )
    print(f"report={report}")
    print(f"pages={len(results)}")
    return 0 if not any(result.get("status") == "error" for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
