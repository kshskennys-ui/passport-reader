"""Shared OpenCV face-orientation geometry checks."""

from __future__ import annotations

from itertools import combinations
from typing import Protocol

import numpy as np


class EyePairConfig(Protocol):
    eye_pair_min_x_ratio: float
    eye_pair_max_x_ratio: float
    eye_pair_min_y_ratio: float
    eye_pair_max_y_ratio: float
    eye_pair_min_separation_ratio: float
    eye_pair_max_separation_ratio: float
    eye_pair_max_vertical_delta_ratio: float


def eye_pair_quality(
    eyes: object,
    face_w: int,
    face_h: int,
    config: EyePairConfig,
) -> float:
    if face_w <= 0 or face_h <= 0:
        return 0.0
    candidates = []
    for eye in eyes:  # type: ignore[union-attr]
        eye_x, eye_y, eye_w, eye_h = (int(value) for value in eye)
        candidates.append(
            ((eye_x + eye_w / 2.0) / face_w, (eye_y + eye_h / 2.0) / face_h)
        )
    best = 0.0
    for first, second in combinations(candidates, 2):
        left, right = sorted((first, second), key=lambda point: point[0])
        if not (
            config.eye_pair_min_x_ratio <= left[0] <= config.eye_pair_max_x_ratio
            and config.eye_pair_min_x_ratio <= right[0] <= config.eye_pair_max_x_ratio
            and config.eye_pair_min_y_ratio <= left[1] <= config.eye_pair_max_y_ratio
            and config.eye_pair_min_y_ratio <= right[1] <= config.eye_pair_max_y_ratio
        ):
            continue
        separation = right[0] - left[0]
        vertical_delta = abs(right[1] - left[1])
        if not (
            config.eye_pair_min_separation_ratio
            <= separation
            <= config.eye_pair_max_separation_ratio
            and vertical_delta <= config.eye_pair_max_vertical_delta_ratio
        ):
            continue
        vertical_quality = 1.0 - vertical_delta / config.eye_pair_max_vertical_delta_ratio
        center_y = (left[1] + right[1]) / 2.0
        position_quality = 1.0 - min(1.0, abs(center_y - 0.38) / 0.20)
        best = max(best, 0.65 * vertical_quality + 0.35 * position_quality)
    return float(np.clip(best, 0.0, 1.0))
