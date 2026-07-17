"""Create visually audited labels for the expanded Phase 1B sample set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


DATASETS: dict[str, dict[str, object]] = {
    "3_Seaman_Books": {
        "source_file": "3.Seaman Books.pdf",
        "page_count": 21,
    },
    "All_crew_s_passport_copy": {
        "source_file": "All crew's passport copy.pdf",
        "page_count": 21,
    },
    "CREW_PASSPORT_W026": {
        "source_file": "CREW PASSPORT W026.pdf",
        "page_count": 24,
        "without_data_page": {4, 10},
        "remarks": {
            4: "Passport address and observation pages only; no personal data page is present.",
            10: "Passport title and request pages only; no personal data page is present.",
        },
    },
    "宝舟丰泽_证件": {
        "source_file": "宝舟丰泽 证件.pdf",
        "page_count": 20,
    },
    "海安铃_证件": {
        "source_file": "海安铃 证件.pdf",
        "page_count": 21,
        "rotations": {9: 270},
    },
    "证件": {
        "source_file": "证件.pdf",
        "page_count": 22,
        "rotations": {1: 270},
    },
    "金星卡兰德拉护照": {
        "source_file": "金星卡兰德拉护照.pdf",
        "page_count": 22,
    },
    "高丽紫丁香护照": {
        "source_file": "高丽紫丁香护照.pdf",
        "page_count": 18,
    },
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("dataset"))
    parser.add_argument("--overwrite", action="store_true")
    arguments = parser.parse_args()

    for dataset_name, spec in DATASETS.items():
        page_count = int(spec["page_count"])
        excluded = set(spec.get("without_data_page", set()))
        rotations = dict(spec.get("rotations", {}))
        remarks = dict(spec.get("remarks", {}))
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
                "annotator": "codex_visual_audit_2026-07-16",
            }
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
