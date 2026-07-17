"""Find the primary document region using contours and conservative fallbacks."""

from __future__ import annotations

import cv2
import numpy as np

from config import DocumentDetectorConfig
from image_utils import gray, resize_for_analysis
from models import DocumentROI, Rect


class DocumentAnalyzer:
    def __init__(self, config: DocumentDetectorConfig) -> None:
        self.config = config

    def detect(self, image: np.ndarray) -> DocumentROI:
        sample, scale = resize_for_analysis(image, 1800)
        channel = gray(sample)
        blurred = cv2.GaussianBlur(channel, (5, 5), 0)
        edges = cv2.Canny(blurred, 45, 135)
        kernel_size = max(3, round(min(sample.shape[:2]) * self.config.closing_kernel_ratio))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        sample_h, sample_w = sample.shape[:2]
        image_area = sample_h * sample_w
        best: tuple[float, tuple[int, int, int, int]] | None = None
        for contour in contours:
            area = cv2.contourArea(contour)
            if area <= 0:
                continue
            x, y, width, height = cv2.boundingRect(contour)
            rect_area = width * height
            area_ratio = rect_area / image_area
            if not self.config.minimum_area_ratio <= area_ratio <= self.config.maximum_area_ratio:
                continue
            rectangularity = area / max(1, rect_area)
            if rectangularity < self.config.rectangularity_threshold:
                continue
            score = area_ratio * 0.7 + rectangularity * 0.3
            if best is None or score > best[0]:
                best = score, (x, y, width, height)

        height, width = image.shape[:2]
        if best is None:
            return DocumentROI(image.copy(), Rect(0, 0, width, height), 0.0, "fallback_full_image")

        _, (x, y, rect_w, rect_h) = best
        x, y, rect_w, rect_h = self._scale_rect(x, y, rect_w, rect_h, scale, width, height)
        margin = max(8, round(min(width, height) * self.config.safety_margin_ratio))
        x0, y0 = max(0, x - margin), max(0, y - margin)
        x1, y1 = min(width, x + rect_w + margin), min(height, y + rect_h + margin)
        x0, y0, x1, y1 = self._expand_while_content_touches_edge(image, x0, y0, x1, y1)
        rect = Rect(x0, y0, x1 - x0, y1 - y0)
        if self._outside_ink_density(image, rect) > self.config.maximum_outside_ink_density:
            return DocumentROI(
                image.copy(),
                Rect(0, 0, width, height),
                min(1.0, best[0]),
                "fallback_content_outside_roi",
            )
        return DocumentROI(image[y0:y1, x0:x1].copy(), rect, min(1.0, best[0]), "largest_rectangular_contour")

    def _outside_ink_density(self, image: np.ndarray, rect: Rect) -> float:
        channel = gray(image)
        outside = np.ones(channel.shape, dtype=bool)
        outside[rect.y : rect.y + rect.h, rect.x : rect.x + rect.w] = False
        area = int(np.count_nonzero(outside))
        if area == 0:
            return 0.0
        dark = channel < self.config.edge_ink_threshold
        return float(np.count_nonzero(dark & outside) / area)

    def _expand_while_content_touches_edge(
        self, image: np.ndarray, x0: int, y0: int, x1: int, y1: int
    ) -> tuple[int, int, int, int]:
        height, width = image.shape[:2]
        base_step = max(16, round(min(width, height) * self.config.expansion_step_ratio))
        step = base_step * self.config.expansion_context_multiplier
        for _ in range(self.config.max_expansion_steps):
            roi = gray(image[y0:y1, x0:x1])
            guard = max(6, round(min(roi.shape) * self.config.edge_guard_ratio))
            dark = roi < self.config.edge_ink_threshold
            densities = {
                "top": float(np.mean(dark[:guard, :])),
                "bottom": float(np.mean(dark[-guard:, :])),
                "left": float(np.mean(dark[:, :guard])),
                "right": float(np.mean(dark[:, -guard:])),
            }
            updated = (x0, y0, x1, y1)
            if densities["top"] >= self.config.edge_ink_density_threshold and y0 > 0:
                y0 = max(0, y0 - step)
            if densities["bottom"] >= self.config.edge_ink_density_threshold and y1 < height:
                y1 = min(height, y1 + step)
            if densities["left"] >= self.config.edge_ink_density_threshold and x0 > 0:
                x0 = max(0, x0 - step)
            if densities["right"] >= self.config.edge_ink_density_threshold and x1 < width:
                x1 = min(width, x1 + step)
            if updated == (x0, y0, x1, y1):
                break
        return x0, y0, x1, y1

    @staticmethod
    def _scale_rect(
        x: int, y: int, width: int, height: int, scale: float, max_width: int, max_height: int
    ) -> tuple[int, int, int, int]:
        if scale == 1.0:
            return x, y, width, height
        return (
            max(0, round(x / scale)),
            max(0, round(y / scale)),
            min(max_width, round(width / scale)),
            min(max_height, round(height / scale)),
        )
