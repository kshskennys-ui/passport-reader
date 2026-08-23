"""Run fast MRZ-first OCR directly against Phase 1 data-page images."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from config import load_config
from ocr.fast_mrz_runner import FastMRZRunner


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--no-fallback", action="store_true")
    parser.add_argument(
        "--pages",
        nargs="+",
        type=int,
        default=None,
        help="only process selected page numbers, for example --pages 7 10 12",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="parallel page workers; each worker owns one PaddleOCR model (default: 2)",
    )
    arguments = parser.parse_args()

    config = load_config(arguments.config)
    mrz_config = config.mrz
    if arguments.no_fallback:
        from dataclasses import replace

        mrz_config = replace(mrz_config, fast_fallback_enabled=False)
    runner = FastMRZRunner(config.ocr, mrz_config)

    def progress(index, total, result):  # type: ignore[no-untyped-def]
        metrics = result.get("metrics", {})
        print(
            f"[{index:03d}/{total:03d}] {result.get('document')}/page{int(result.get('page_number', 0)):03d} "
            f"{result.get('status')} mode={result.get('mode', 'error')} "
            f"parse={result.get('mrz_parse', {}).get('status', 'N/A')} "
            f"time={float(result.get('elapsed_ms', 0)) / 1000:.1f}s",
            flush=True,
        )

    results, report = runner.run(
        arguments.input_root,
        arguments.output_root,
        resume=not arguments.no_resume,
        workers=arguments.workers,
        pages=set(arguments.pages) if arguments.pages else None,
        callback=progress,
    )
    print(f"report={report}")
    print(
        f"pages={len(results)} valid={sum(r.get('mrz_parse', {}).get('status') == 'valid' for r in results)} "
        f"fallback={sum(r.get('mode') == 'full_page_fallback' for r in results)} "
        f"errors={sum(r.get('status') == 'error' for r in results)}"
    )
    return 0 if not any(result.get("status") == "error" for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
