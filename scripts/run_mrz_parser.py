"""Parse and validate existing MRZ second-pass OCR results without rerunning OCR."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ocr.mrz_parser import parse_mrz_row_results
from ocr.mrz_runner import write_mrz_report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--all-pages", action="store_true")
    arguments = parser.parse_args()
    input_root = arguments.input_root.resolve()
    output_root = (arguments.output_root or input_root).resolve()
    results: list[dict] = []
    for path in sorted((input_root / "results").rglob("mrz.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        result["mrz_parse"] = parse_mrz_row_results(result.get("row_results", []))
        result["warnings"] = [
            warning for warning in result.get("warnings", []) if not warning.startswith("mrz_parse_")
        ]
        status = result["mrz_parse"].get("status")
        if status in {"invalid", "incomplete", "unsupported_format"}:
            result["warnings"].append(f"mrz_parse_{status}")
        destination = output_root / path.relative_to(input_root)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        results.append(result)
    report = write_mrz_report(output_root, results, all_pages=arguments.all_pages)
    summary = json.loads((output_root / "summary.json").read_text(encoding="utf-8"))["summary"]
    print(f"report={report}")
    print(f"pages={summary['pages']} parse_valid={summary['parse_valid']} parse_invalid={summary['parse_invalid']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
