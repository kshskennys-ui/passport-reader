"""Create the audited seed labels for the two Phase 1B validation PDFs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


DATASETS: dict[str, dict[str, object]] = {
    "中联海防0704": {
        "source_file": "中联海防0704证件.pdf",
        "page_count": 20,
        "without_data_page": {7},
        "rotations": {9: 90},
        "remarks": {
            7: "Visa-only spread. No personal information page is present.",
            9: "Source scan requires a clockwise 90 degree rotation.",
        },
    },
    "环球01": {
        "source_file": "环球01证件.pdf",
        "page_count": 19,
        "without_data_page": set(),
        "rotations": {},
        "remarks": {},
    },
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("dataset"))
    parser.add_argument("--overwrite", action="store_true")
    arguments = parser.parse_args()

    for dataset_name, spec in DATASETS.items():
        page_count = int(spec["page_count"])
        excluded = set(spec["without_data_page"])
        rotations = dict(spec["rotations"])
        remarks = dict(spec["remarks"])
        for page_number in range(1, page_count + 1):
            path = arguments.root / dataset_name / f"page{page_number:03d}" / "expected.json"
            if path.exists() and not arguments.overwrite:
                continue
            has_data_page = page_number not in excluded
            payload = {
                "schema_version": 1,
                "source_file": spec["source_file"],
                "source_page": page_number,
                "contains_data_page": has_data_page,
                "expected_pages": 1 if has_data_page else 0,
                "rotation": int(rotations.get(page_number, 0)),
                "remarks": remarks.get(page_number, ""),
                "annotator": "bootstrap_visual_annotation",
            }
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
