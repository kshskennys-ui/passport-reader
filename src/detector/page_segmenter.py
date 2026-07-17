"""Detect a central seam and split an expanded scan without page-side assumptions."""

from __future__ import annotations

import cv2
import numpy as np

from config import SegmenterConfig
from image_utils import gray, resize_for_analysis
from models import Rect, Segment


class PageSegmenter:
    def __init__(self, config: SegmenterConfig) -> None:
        self.config = config

    def segment(self, image: np.ndarray) -> list[Segment]:
        height, width = image.shape[:2]
        if not self.config.enabled or width < 160:
            return [Segment(image.copy(), Rect(0, 0, width, height), 1, image.copy(), Rect(0, 0, width, height))]

        vertical = self._find_seam(image, axis="vertical")
        horizontal = self._find_seam(image, axis="horizontal")
        candidates = [item for item in (vertical, horizontal) if item is not None]
        if not candidates:
            return [Segment(image.copy(), Rect(0, 0, width, height), 1)]
        axis, seam, _ = max(candidates, key=lambda item: item[2])

        if axis == "vertical":
            minimum = round(width * self.config.min_segment_width_ratio)
            if seam < minimum or width - seam < minimum:
                return [Segment(image.copy(), Rect(0, 0, width, height), 1)]
            padding = max(8, round(width * self.config.safety_padding_ratio))
            left_safe_end = min(width, seam + padding)
            right_safe_start = max(0, seam - padding)
            left = Segment(
                image[:, :seam].copy(),
                Rect(0, 0, seam, height),
                1,
                image[:, :left_safe_end].copy(),
                Rect(0, 0, left_safe_end, height),
            )
            right = Segment(
                image[:, seam:].copy(),
                Rect(seam, 0, width - seam, height),
                2,
                image[:, right_safe_start:].copy(),
                Rect(right_safe_start, 0, width - right_safe_start, height),
            )
            return [left, right]

        minimum = round(height * self.config.min_segment_width_ratio)
        if seam < minimum or height - seam < minimum:
            return [Segment(image.copy(), Rect(0, 0, width, height), 1)]
        padding = max(8, round(height * self.config.safety_padding_ratio))
        top_safe_end = min(height, seam + padding)
        bottom_safe_start = max(0, seam - padding)
        top = Segment(
            image[:seam, :].copy(),
            Rect(0, 0, width, seam),
            1,
            image[:top_safe_end, :].copy(),
            Rect(0, 0, width, top_safe_end),
        )
        bottom = Segment(
            image[seam:, :].copy(),
            Rect(0, seam, width, height - seam),
            2,
            image[bottom_safe_start:, :].copy(),
            Rect(0, bottom_safe_start, width, height - bottom_safe_start),
        )
        return [top, bottom]

    def _find_seam(self, image: np.ndarray, axis: str) -> tuple[str, int, float] | None:
        sample, scale = resize_for_analysis(image, 1800)
        channel = gray(sample)
        binary = cv2.adaptiveThreshold(
            channel, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 11
        )
        height, width = binary.shape
        if axis == "vertical":
            length, cross_length = width, height
            ink_density = np.mean(binary > 0, axis=0)
            white_coverage = np.mean(channel > 240, axis=0)
        else:
            length, cross_length = height, width
            ink_density = np.mean(binary > 0, axis=1)
            white_coverage = np.mean(channel > 240, axis=1)
        start = int(length * self.config.candidate_start_ratio)
        end = int(length * self.config.candidate_end_ratio)
        if end <= start:
            return None

        radius = max(2, round(length * self.config.seam_neighbourhood_ratio))
        candidates: list[tuple[float, int]] = []
        for x in range(start, end):
            x0, x1 = max(0, x - radius), min(length, x + radius + 1)
            local_ink = float(np.mean(ink_density[x0:x1]))
            local_white = float(np.mean(white_coverage[x0:x1]))
            if local_ink > self.config.seam_ink_density_threshold:
                continue
            if local_white < self.config.seam_min_height_coverage:
                continue
            center_bias = abs(x - length / 2) / (length / 2)
            candidates.append(((1.0 - local_ink) * local_white - center_bias * 0.08, x))
        valley = max(candidates, default=(0.0, -1))
        line = self._line_seam(channel, axis, start, end, length, cross_length)
        gradient = self._gradient_seam(channel, axis, start, end)
        methods = (("valley", valley), ("line", line), ("gradient", gradient))
        method, (best_score, seam_sample) = max(methods, key=lambda item: item[1][0])
        minimum_score = (
            self.config.line_seam_minimum_score if method == "line" else self.config.minimum_seam_score
        )
        if seam_sample < 0 or best_score < minimum_score:
            return None
        return axis, round(seam_sample / scale), best_score

    def _line_seam(
        self, channel: np.ndarray, axis: str, start: int, end: int, length: int, cross_length: int
    ) -> tuple[float, int]:
        edges = cv2.Canny(cv2.GaussianBlur(channel, (3, 3), 0), 45, 135)
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 360,
            threshold=self.config.line_seam_hough_threshold,
            minLineLength=max(30, round(cross_length * self.config.line_seam_min_length_ratio)),
            maxLineGap=max(8, round(cross_length * 0.02)),
        )
        if lines is None:
            return 0.0, -1
        best = (0.0, -1)
        for x1, y1, x2, y2 in lines[:, 0]:
            delta_x, delta_y = x2 - x1, y2 - y1
            if axis == "vertical":
                if abs(delta_x) > abs(delta_y) * np.tan(np.deg2rad(self.config.line_seam_max_angle_degrees)):
                    continue
                coordinate = round((x1 + x2) / 2)
                span = abs(delta_y)
            else:
                if abs(delta_y) > abs(delta_x) * np.tan(np.deg2rad(self.config.line_seam_max_angle_degrees)):
                    continue
                coordinate = round((y1 + y2) / 2)
                span = abs(delta_x)
            if not start <= coordinate < end:
                continue
            coverage = span / max(1, cross_length)
            center_bias = abs(coordinate - length / 2) / (length / 2)
            score = coverage - center_bias * 0.08
            if score > best[0]:
                best = score, coordinate
        return best

    def _gradient_seam(self, channel: np.ndarray, axis: str, start: int, end: int) -> tuple[float, int]:
        kernel_size = max(3, self.config.gradient_blur_kernel | 1)
        blurred = cv2.GaussianBlur(channel, (kernel_size, kernel_size), 0)
        difference_axis = 1 if axis == "vertical" else 0
        differences = np.abs(np.diff(blurred.astype(np.float32), axis=difference_axis))
        profile = np.mean(differences, axis=0 if axis == "vertical" else 1)
        region = profile[start:end]
        if len(region) == 0:
            return 0.0, -1
        offset = int(np.argmax(region))
        coordinate = start + offset
        median = float(np.median(region))
        p99 = float(np.percentile(region, 99))
        peak = float(profile[coordinate])
        if peak <= median or p99 <= median:
            return 0.0, -1
        cross_section = differences[:, coordinate] if axis == "vertical" else differences[coordinate, :]
        coverage = float(np.mean(cross_section >= self.config.gradient_minimum_pixel_delta))
        if coverage < self.config.gradient_minimum_coverage:
            return 0.0, -1
        prominence = min(1.0, (peak - median) / (p99 - median))
        score = prominence * 0.62 + coverage * 0.38
        return score, coordinate
