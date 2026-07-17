"""Versioned ground-truth labels and reviewer decisions."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class GroundTruth:
    dataset_name: str
    directory: Path
    source_file: str
    source_page: int
    contains_data_page: bool
    expected_pages: int
    rotation: int
    remarks: str

    @property
    def review_path(self) -> Path:
        return self.directory / "review.json"

    @property
    def expected_image_path(self) -> Path:
        return self.directory / "expected.png"


class GroundTruthStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self._labels = self._load_labels()

    @property
    def labels(self) -> tuple[GroundTruth, ...]:
        return tuple(self._labels)

    def labels_for_source(self, source_file: str) -> list[GroundTruth]:
        return sorted(
            (label for label in self._labels if label.source_file == source_file),
            key=lambda label: label.source_page,
        )

    def get(self, source_file: str, source_page: int) -> GroundTruth | None:
        return next(
            (
                label
                for label in self._labels
                if label.source_file == source_file and label.source_page == source_page
            ),
            None,
        )

    def latest_review(self, truth: GroundTruth) -> dict[str, Any] | None:
        if not truth.review_path.exists():
            return None
        payload = json.loads(truth.review_path.read_text(encoding="utf-8"))
        return payload.get("latest")

    def record_review(
        self,
        truth: GroundTruth,
        prediction: dict[str, Any],
        outcome: str,
        failure_reason: str | None = None,
        quality: dict[str, bool] | None = None,
    ) -> Path:
        if outcome not in {"correct", "incorrect"}:
            raise ValueError(f"Unsupported review outcome: {outcome}")
        event = {
            "reviewed_at": datetime.now().isoformat(timespec="seconds"),
            "prediction": prediction,
            "ground_truth": {
                "contains_data_page": truth.contains_data_page,
                "expected_pages": truth.expected_pages,
                "rotation": truth.rotation,
            },
            "review": outcome,
            "failure_reason": failure_reason if outcome == "incorrect" else None,
            "quality": quality,
        }
        existing: dict[str, Any] = {"schema_version": 2, "history": []}
        if truth.review_path.exists():
            existing = json.loads(truth.review_path.read_text(encoding="utf-8"))
        history = list(existing.get("history", []))
        history.append(event)
        payload = {"schema_version": 2, "latest": event, "history": history}
        truth.review_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

        output_path = prediction.get("output_path")
        quality_accepted = quality is None or all(
            quality.get(name, False)
            for name in ("page_complete", "portrait_complete", "mrz_complete", "ocr_ready")
        )
        if outcome == "correct" and quality_accepted and truth.contains_data_page and output_path:
            source = Path(output_path)
            if source.exists():
                shutil.copy2(source, truth.expected_image_path)
        return truth.review_path

    def _load_labels(self) -> list[GroundTruth]:
        if not self.root.exists():
            return []
        labels: list[GroundTruth] = []
        for path in sorted(self.root.rglob("expected.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            required = {
                "source_file",
                "source_page",
                "contains_data_page",
                "expected_pages",
                "rotation",
                "remarks",
            }
            missing = required - set(payload)
            if missing:
                names = ", ".join(sorted(missing))
                raise ValueError(f"Missing Ground Truth field(s) in {path}: {names}")
            labels.append(
                GroundTruth(
                    dataset_name=path.parents[1].name,
                    directory=path.parent,
                    source_file=str(payload["source_file"]),
                    source_page=int(payload["source_page"]),
                    contains_data_page=bool(payload["contains_data_page"]),
                    expected_pages=int(payload["expected_pages"]),
                    rotation=int(payload["rotation"]) % 360,
                    remarks=str(payload["remarks"]),
                )
            )
        return labels
