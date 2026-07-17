"""Conservative removal of uniform scanner borders."""

from __future__ import annotations

import cv2
import numpy as np

from config import BorderTrimConfig
from image_utils import gray
from models import CropResult, Rect


class WhiteBorderRemover:
    def __init__(self, config: BorderTrimConfig) -> None:
        self.config = config

    def trim(self, image: np.ndarray) -> CropResult:
        channel = gray(image)
        foreground_mask = channel < self.config.white_threshold
        foreground = foreground_mask.astype(np.uint8) * 255
        points = cv2.findNonZero(foreground)
        height, width = channel.shape
        full_rect = Rect(0, 0, width, height)
        if points is None:
            return CropResult(image.copy(), full_rect, 0.0)

        x, y, content_w, content_h = cv2.boundingRect(points)
        content_ratio = (content_w * content_h) / (width * height)
        if content_ratio < self.config.minimum_content_ratio:
            return CropResult(image.copy(), full_rect, 0.0)

        margin = max(self.config.minimum_margin_px, round(min(width, height) * self.config.safety_margin_ratio))
        x0 = max(0, x - margin)
        y0 = max(0, y - margin)
        x1 = min(width, x + content_w + margin)
        y1 = min(height, y + content_h + margin)
        rect = Rect(x0, y0, x1 - x0, y1 - y0)
        if (
            self.config.projection_enabled
            and max(rect.w / width, rect.h / height)
            >= self.config.projection_fallback_trigger_extent_ratio
        ):
            projection_rect = self._projection_rect(foreground_mask)
            if projection_rect is not None:
                rect = projection_rect
        if rect.area / (width * height) < 1.0 - self.config.maximum_crop_ratio:
            return CropResult(image.copy(), full_rect, 0.0)
        confidence = float(1.0 - rect.area / (width * height))
        return CropResult(
            image[rect.y : rect.y + rect.h, rect.x : rect.x + rect.w].copy(),
            rect,
            confidence,
        )

    def _projection_rect(self, foreground: np.ndarray) -> Rect | None:
        height, width = foreground.shape
        edge_x = max(1, round(width * self.config.projection_edge_ignore_ratio))
        edge_y = max(1, round(height * self.config.projection_edge_ignore_ratio))
        if width <= edge_x * 2 or height <= edge_y * 2:
            return None

        row_profile = np.mean(foreground[:, edge_x : width - edge_x], axis=1)
        column_profile = np.mean(foreground[edge_y : height - edge_y, :], axis=0)
        row_bounds = self._active_projection_bounds(row_profile, height)
        column_bounds = self._active_projection_bounds(column_profile, width)
        if row_bounds is None or column_bounds is None:
            return None

        margin = max(
            self.config.minimum_margin_px,
            round(min(width, height) * self.config.safety_margin_ratio),
        )
        y0, y1 = row_bounds
        x0, x1 = column_bounds
        x0, y0 = max(0, x0 - margin), max(0, y0 - margin)
        x1, y1 = min(width, x1 + margin), min(height, y1 + margin)
        rect = Rect(x0, y0, x1 - x0, y1 - y0)

        area_reduction = 1.0 - rect.area / (width * height)
        if area_reduction < self.config.projection_minimum_area_reduction:
            return None
        if self._retained_interior_ink_ratio(foreground, rect, edge_x, edge_y) < self.config.projection_minimum_retained_ink_ratio:
            return None
        return rect

    def _active_projection_bounds(self, profile: np.ndarray, size: int) -> tuple[int, int] | None:
        kernel_size = max(3, round(size * self.config.projection_smoothing_ratio))
        kernel = np.ones(kernel_size, dtype=np.float32) / kernel_size
        smoothed = np.convolve(profile, kernel, mode="same")
        active = np.flatnonzero(smoothed >= self.config.projection_minimum_ink_density)
        if active.size == 0:
            return None
        return int(active[0]), int(active[-1] + 1)

    @staticmethod
    def _retained_interior_ink_ratio(
        foreground: np.ndarray,
        rect: Rect,
        edge_x: int,
        edge_y: int,
    ) -> float:
        interior = foreground.copy()
        interior[:edge_y, :] = False
        interior[-edge_y:, :] = False
        interior[:, :edge_x] = False
        interior[:, -edge_x:] = False
        total = int(np.count_nonzero(interior))
        if total == 0:
            return 1.0
        retained = int(
            np.count_nonzero(
                interior[rect.y : rect.y + rect.h, rect.x : rect.x + rect.w]
            )
        )
        return retained / total
