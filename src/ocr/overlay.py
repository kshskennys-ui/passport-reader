"""Draw OCR geometry without rendering personal text into the image."""

from __future__ import annotations

import cv2
import numpy as np

from config import OCRConfig
from ocr.models import OCRLine


def draw_ocr_overlay(
    image: np.ndarray, lines: list[OCRLine], mrz_indices: set[int], config: OCRConfig
) -> np.ndarray:
    overlay = image.copy()
    for index, line in enumerate(lines):
        if len(line.polygon) < 3:
            continue
        points = np.asarray(line.polygon, dtype=np.int32).reshape((-1, 1, 2))
        if index in mrz_indices:
            color = (255, 180, 0)
        elif line.confidence >= config.low_confidence_threshold:
            color = (40, 180, 60)
        else:
            color = (0, 120, 255)
        cv2.polylines(overlay, [points], True, color, config.overlay_line_thickness, cv2.LINE_AA)
        anchor = tuple(int(value) for value in points[0, 0])
        cv2.putText(
            overlay,
            f"{index + 1}:{line.confidence:.2f}",
            (anchor[0], max(14, anchor[1] - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            color,
            1,
            cv2.LINE_AA,
        )
    return overlay
