"""Measurements for deciding whether OCR failures originate upstream."""

from __future__ import annotations

import re
from statistics import mean
from typing import Any

from config import OCRConfig
from ocr.models import OCRLine

MRZ_ALLOWED = re.compile(r"[A-Z0-9<]")


def analyze_ocr_lines(
    lines: list[OCRLine], image_shape: tuple[int, int], config: OCRConfig
) -> tuple[dict[str, Any], list[str]]:
    height, width = image_shape
    accepted = [line for line in lines if line.confidence >= config.minimum_text_confidence]
    confidences = [line.confidence for line in accepted]
    bbox = _content_bbox(accepted, width, height)
    mrz_indices = [
        index
        for index, line in enumerate(lines)
        if is_mrz_candidate(line, image_shape, config, bbox)
    ]
    metrics: dict[str, Any] = {
        "line_count": len(lines),
        "accepted_line_count": len(accepted),
        "mean_confidence": round(mean(confidences), 6) if confidences else None,
        "minimum_confidence": round(min(confidences), 6) if confidences else None,
        "low_confidence_line_count": sum(
            line.confidence < config.low_confidence_threshold for line in accepted
        ),
        "mrz_candidate_count": len(mrz_indices),
        "mrz_line_indices": mrz_indices,
        "content_bbox": bbox,
        "content_bbox_area_ratio": round((bbox["w"] * bbox["h"]) / (width * height), 6)
        if bbox
        else 0.0,
    }
    warnings: list[str] = []
    if not accepted:
        warnings.append("ocr_detection_no_text")
    elif metrics["mean_confidence"] < config.minimum_mean_confidence:
        warnings.append("ocr_recognition_low_confidence")
    if not mrz_indices:
        warnings.append("mrz_candidate_not_detected")
    elif len(mrz_indices) == 1:
        warnings.append("mrz_candidate_incomplete")
    return metrics, warnings


def is_mrz_candidate(
    line: OCRLine,
    image_shape: tuple[int, int],
    config: OCRConfig,
    content_bbox: dict[str, int] | None = None,
) -> bool:
    compact = "".join(line.text.upper().split())
    if (
        len(compact) < config.mrz_minimum_length
        or len(compact) > config.mrz_maximum_length
        or not line.polygon
    ):
        return False
    allowed_ratio = len(MRZ_ALLOWED.findall(compact)) / len(compact)
    marker_present = "<" in compact or any(character.isdigit() for character in compact)
    content = content_bbox or {"x": 0, "y": 0, "w": image_shape[1], "h": image_shape[0]}
    line_width = max(point[0] for point in line.polygon) - min(point[0] for point in line.polygon)
    width_ratio = line_width / max(1, content["w"])
    bottom = max(point[1] for point in line.polygon)
    lower_boundary = content["y"] + content["h"] * 0.55
    return (
        allowed_ratio >= config.mrz_minimum_allowed_ratio
        and marker_present
        and width_ratio >= config.mrz_minimum_content_width_ratio
        and bottom >= lower_boundary
    )


def _content_bbox(lines: list[OCRLine], width: int, height: int) -> dict[str, int] | None:
    points = [point for line in lines for point in line.polygon]
    if not points:
        return None
    x1 = max(0, min(point[0] for point in points))
    y1 = max(0, min(point[1] for point in points))
    x2 = min(width, max(point[0] for point in points))
    y2 = min(height, max(point[1] for point in points))
    return {"x": x1, "y": y1, "w": max(0, x2 - x1), "h": max(0, y2 - y1)}
