from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from config import DEFAULT_CONFIG, ValidationConfig
from validation.evaluator import ValidationService
from validation.ground_truth import GroundTruthStore
from validation.saved_run import load_saved_process


def test_validation_report_classifies_missing_selection_as_classifier_failure(tmp_path: Path) -> None:
    store = _store_with_label(tmp_path, contains_data_page=True, rotation=0)
    output = tmp_path / "output"
    _write_debug_log(output, status="low_confidence", rotation=0, score=62.0)

    service = ValidationService(store, ValidationConfig())
    evaluations = service.evaluate_output(output)
    report = service.write_report(output, evaluations)

    assert len(evaluations) == 1
    assert evaluations[0].result == "automatic_fail"
    assert evaluations[0].failure_reason == "classifier"
    assert report.exists()
    assert (output / "failed" / "classifier" / "demo" / "page001" / "failure.json").exists()


def test_reviewer_confirmation_is_persisted_and_creates_canonical_image(tmp_path: Path) -> None:
    store = _store_with_label(tmp_path, contains_data_page=True, rotation=0)
    truth = store.get("demo.pdf", 1)
    assert truth is not None
    output = tmp_path / "output"
    output_image = output / "data_pages" / "demo" / "DataPage_001.png"
    output_image.parent.mkdir(parents=True)
    cv2.imencode(".png", np.full((30, 40, 3), 180, dtype=np.uint8))[1].tofile(str(output_image))
    _write_debug_log(output, status="selected", rotation=0, score=88.0)

    store.record_review(
        truth,
        {"status": "selected", "segment": "segment_1", "score": 88.0, "output_path": str(output_image)},
        "correct",
    )
    service = ValidationService(store, ValidationConfig())
    evaluation = service.evaluate_output(output)[0]

    assert evaluation.result == "verified_pass"
    assert truth.review_path.exists()
    assert truth.expected_image_path.exists()


def test_quality_review_drives_crop_and_ocr_ready_metrics(tmp_path: Path) -> None:
    store = _store_with_label(tmp_path, contains_data_page=True, rotation=0)
    truth = store.get("demo.pdf", 1)
    assert truth is not None
    output = tmp_path / "output"
    output_image = output / "data_pages" / "demo" / "DataPage_001.png"
    output_image.parent.mkdir(parents=True)
    cv2.imencode(".png", np.full((30, 40, 3), 180, dtype=np.uint8))[1].tofile(str(output_image))
    _write_debug_log(output, status="selected", rotation=0, score=88.0)
    store.record_review(
        truth,
        {"status": "selected", "output_path": str(output_image)},
        "correct",
        quality={
            "page_complete": True,
            "portrait_complete": True,
            "mrz_complete": False,
            "ocr_ready": False,
        },
    )

    service = ValidationService(store, ValidationConfig())
    evaluations = service.evaluate_output(output)
    summary = service.summary(evaluations)

    assert evaluations[0].result == "verified_fail"
    assert evaluations[0].failure_reason == "crop"
    assert summary.crop_completeness == 0.0
    assert summary.mrz_completeness == 0.0
    assert summary.portrait_completeness == 1.0
    assert summary.ocr_ready_rate == 0.0
    assert not truth.expected_image_path.exists()


def test_saved_run_loader_reconstructs_page_results(tmp_path: Path) -> None:
    source = tmp_path / "demo.pdf"
    source.write_bytes(b"saved-run-placeholder")
    output = tmp_path / "output"
    _write_debug_log(output, status="selected", rotation=0, score=88.0)
    output_image = output / "data_pages" / "demo" / "DataPage_001.png"
    output_image.parent.mkdir(parents=True)
    cv2.imencode(".png", np.full((20, 30, 3), 180, dtype=np.uint8))[1].tofile(
        str(output_image)
    )

    result = load_saved_process(source, output, DEFAULT_CONFIG)

    assert result.input_path == source.resolve()
    assert result.successful_pages == 1
    assert result.page_results[0].source_page == 1
    assert result.page_results[0].score == 88.0
    assert result.page_results[0].output_path == output_image.resolve()


def _store_with_label(tmp_path: Path, contains_data_page: bool, rotation: int) -> GroundTruthStore:
    directory = tmp_path / "dataset" / "demo" / "page001"
    directory.mkdir(parents=True)
    (directory / "expected.json").write_text(
        json.dumps(
            {
                "source_file": "demo.pdf",
                "source_page": 1,
                "contains_data_page": contains_data_page,
                "expected_pages": 1 if contains_data_page else 0,
                "rotation": rotation,
                "remarks": "",
            }
        ),
        encoding="utf-8",
    )
    return GroundTruthStore(tmp_path / "dataset")


def _write_debug_log(output: Path, status: str, rotation: int, score: float) -> None:
    directory = output / "debug" / "demo" / "page001"
    directory.mkdir(parents=True)
    payload = {
        "status": status,
        "metrics": {
            "selected_segment": 1,
            "scores": {"final": score},
            "orientation": {"angle": rotation},
        },
        "stages": [
            {"name": "scored_1", "elapsed_ms": 12.0, "metadata": {"scores": {"final": score}}}
        ],
    }
    (directory / "log.json").write_text(json.dumps(payload), encoding="utf-8")
