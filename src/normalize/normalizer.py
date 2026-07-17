"""Produce a padded OCR-safe image without destructive post-normalization cropping."""

from __future__ import annotations

from math import atan2, degrees

import cv2
import numpy as np

from config import BorderTrimConfig, NormalizerConfig
from image_utils import gray
from models import NormalizedResult, Rect
from normalize.content_safety import ContentSafetyChecker, ContentSafetyResult


class Normalizer:
    def __init__(self, config: NormalizerConfig, border_config: BorderTrimConfig) -> None:
        del border_config
        self.config = config
        self.safety_checker = ContentSafetyChecker(config)

    def normalize(
        self,
        image: np.ndarray,
        fallbacks: list[tuple[str, np.ndarray]] | None = None,
    ) -> NormalizedResult:
        levels = [("selected_safe_roi", image), *(fallbacks or [])]
        last: NormalizedResult | None = None
        for level, candidate in levels:
            result, safety = self._normalize_one(candidate, level)
            last = result
            if safety.safe:
                return result
        if last is None:
            raise ValueError("Normalizer requires at least one image")
        return last

    def _normalize_one(self, image: np.ndarray, level: str) -> tuple[NormalizedResult, ContentSafetyResult]:
        estimated, confidence = self._estimate_skew(image) if self.config.enable_deskew else (0.0, 0.0)
        applied = 0.0
        residual = estimated
        working = image.copy()
        warnings: list[str] = []
        should_deskew = (
            self.config.enable_deskew
            and self.config.minimum_abs_deskew_degrees <= abs(estimated) <= self.config.max_abs_deskew_degrees
            and confidence >= self.config.minimum_deskew_confidence
        )
        if should_deskew:
            rotated = self._rotate_with_bounds(image, estimated)
            residual, _ = self._estimate_skew(rotated)
            required_improvement = abs(estimated) * self.config.minimum_deskew_improvement_ratio
            if abs(estimated) - abs(residual) >= required_improvement:
                working = rotated
                applied = estimated
            else:
                residual = estimated
                warnings.append("deskew_reverted_no_measured_improvement")
        elif abs(estimated) >= self.config.minimum_abs_deskew_degrees:
            warnings.append("deskew_skipped_low_confidence")

        safety = self.safety_checker.assess(working)
        warnings.extend(safety.warnings)
        padded, padding = self._add_padding(working)
        height, width = padded.shape[:2]
        return (
            NormalizedResult(
                image=padded,
                deskew_degrees=applied,
                trim_rect=Rect(0, 0, width, height),
                estimated_skew_degrees=estimated,
                deskew_confidence=confidence,
                residual_skew_degrees=residual,
                padding_px=padding,
                fallback_level=level,
                ocr_safe=safety.safe,
                warnings=warnings,
            ),
            safety,
        )

    def _estimate_skew(self, image: np.ndarray) -> tuple[float, float]:
        channel = gray(image)
        edges = cv2.Canny(cv2.GaussianBlur(channel, (3, 3), 0), 55, 150)
        min_length = max(40, round(image.shape[1] * 0.14))
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 360,
            threshold=self.config.hough_threshold,
            minLineLength=min_length,
            maxLineGap=max(8, min_length // 4),
        )
        if lines is None:
            return 0.0, 0.0
        angles: list[tuple[float, float]] = []
        for x1, y1, x2, y2 in lines[:, 0]:
            angle = degrees(atan2(y2 - y1, x2 - x1))
            normalized = ((angle + 90) % 180) - 90
            if abs(normalized) <= self.config.max_abs_deskew_degrees:
                length = float(np.hypot(x2 - x1, y2 - y1))
                angles.append((normalized, length))
        if len(angles) < 3:
            return 0.0, 0.0
        values = np.array([value for value, _ in angles])
        weights = np.array([weight for _, weight in angles])
        estimate = self._weighted_median(values, weights)
        deviations = np.abs(values - estimate)
        dispersion = self._weighted_median(deviations, weights)
        inliers = deviations <= max(0.6, dispersion * 2.5)
        support = min(1.0, float(np.count_nonzero(inliers)) / 8.0)
        agreement = max(0.0, 1.0 - dispersion / 1.5)
        confidence = support * agreement
        if abs(estimate) > self.config.max_abs_deskew_degrees:
            return 0.0, 0.0
        return estimate, confidence

    @staticmethod
    def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
        order = np.argsort(values)
        sorted_values = values[order]
        cumulative = np.cumsum(weights[order])
        midpoint = cumulative[-1] / 2.0
        return float(sorted_values[int(np.searchsorted(cumulative, midpoint, side="left"))])

    def _add_padding(self, image: np.ndarray) -> tuple[np.ndarray, int]:
        height, width = image.shape[:2]
        padding = max(self.config.minimum_padding_px, round(min(width, height) * self.config.safe_padding_ratio))
        padded = cv2.copyMakeBorder(
            image,
            padding,
            padding,
            padding,
            padding,
            cv2.BORDER_CONSTANT,
            value=(255, 255, 255),
        )
        return padded, padding

    @staticmethod
    def _rotate_with_bounds(image: np.ndarray, angle: float) -> np.ndarray:
        height, width = image.shape[:2]
        matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
        cosine, sine = abs(matrix[0, 0]), abs(matrix[0, 1])
        bound_w = int(height * sine + width * cosine)
        bound_h = int(height * cosine + width * sine)
        matrix[0, 2] += bound_w / 2 - width / 2
        matrix[1, 2] += bound_h / 2 - height / 2
        return cv2.warpAffine(
            image,
            matrix,
            (bound_w, bound_h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255),
        )
