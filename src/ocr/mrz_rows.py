"""Prepare and merge OCR results for individual MRZ rows."""

from __future__ import annotations

import re
from dataclasses import dataclass

import cv2
import numpy as np

from config import MRZConfig
from models import Rect
from ocr.models import OCRLine
from ocr.mrz_locator import MRZRegion, MRZRow

MRZ_ALLOWED = re.compile(r"[A-Z0-9<]")


@dataclass(frozen=True)
class MRZRowCrop:
    row_index: int
    source_rect: Rect
    crop_rect: Rect
    scale: float
    image: np.ndarray


@dataclass(frozen=True)
class MRZRowText:
    row_index: int
    raw_text: str
    normalized_text: str
    fragment_count: int
    confidence: float | None
    source_lines: list[OCRLine]

    def as_dict(self) -> dict:
        return {
            "row_index": self.row_index,
            "raw_text": self.raw_text,
            "normalized_text": self.normalized_text,
            "length": len(self.normalized_text),
            "fragment_count": self.fragment_count,
            "confidence": round(self.confidence, 6) if self.confidence is not None else None,
            "source_lines": [line.as_dict() for line in self.source_lines],
        }


def build_row_crops(image: np.ndarray, region: MRZRegion, config: MRZConfig) -> list[MRZRowCrop]:
    """Crop each located row with independent vertical padding and upscale."""
    height, width = image.shape[:2]
    scale = max(1.0, config.row_upscale_factor)
    crops: list[MRZRowCrop] = []
    for row_index, row in enumerate(region.rows, start=1):
        vertical_padding = max(8, round(row.rect.h * config.row_vertical_padding_ratio))
        tall_row_threshold = max(
            config.minimum_row_height_px * 5,
            round(height * 0.04),
        )
        if row.rect.h >= tall_row_threshold:
            vertical_padding = max(8, round(row.rect.h * config.row_tall_vertical_padding_ratio))
        horizontal_padding = max(8, round(width * config.row_horizontal_padding_ratio))
        x1 = max(0, region.rect.x - horizontal_padding)
        x2 = min(width, region.rect.x + region.rect.w + horizontal_padding)
        y1 = max(0, row.rect.y - vertical_padding)
        y2 = min(height, row.rect.y + row.rect.h + vertical_padding)
        crop_rect = Rect(x1, y1, max(1, x2 - x1), max(1, y2 - y1))
        crop = image[y1:y2, x1:x2].copy()
        if scale != 1.0:
            crop = cv2.resize(
                crop,
                (round(crop.shape[1] * scale), round(crop.shape[0] * scale)),
                interpolation=cv2.INTER_CUBIC,
            )
        crops.append(MRZRowCrop(row_index, row.rect, crop_rect, scale, crop))
    return crops


def merge_row_lines(
    lines: list[OCRLine], crop: MRZRowCrop, config: MRZConfig
) -> MRZRowText:
    """Merge split OCR boxes from one row in left-to-right order."""
    crop_height = crop.image.shape[0]
    expected_center = crop.image.shape[0] / 2
    accepted: list[OCRLine] = []
    for line in lines:
        if not line.polygon or len(line.text.strip()) < config.row_minimum_fragment_length:
            continue
        compact = "".join(line.text.upper().split())
        allowed_ratio = len(MRZ_ALLOWED.findall(compact)) / max(1, len(compact))
        center_y = sum(point[1] for point in line.polygon) / len(line.polygon)
        if allowed_ratio < config.row_minimum_allowed_ratio:
            continue
        if abs(center_y - expected_center) > crop_height * 0.36:
            continue
        accepted.append(line)
    accepted.sort(key=lambda line: min(point[0] for point in line.polygon))
    raw_text = "".join("".join(line.text.split()) for line in accepted)
    normalized = "".join(character for character in raw_text.upper() if MRZ_ALLOWED.fullmatch(character))
    confidences = [line.confidence for line in accepted]
    return MRZRowText(
        row_index=crop.row_index,
        raw_text=raw_text,
        normalized_text=normalized,
        fragment_count=len(accepted),
        confidence=sum(confidences) / len(confidences) if confidences else None,
        source_lines=accepted,
    )
