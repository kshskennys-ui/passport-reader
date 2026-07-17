"""Cardinal-orientation selection using OpenCV layout heuristics only."""

from __future__ import annotations

import cv2
import numpy as np

from config import OrientationConfig
from face_geometry import eye_pair_quality
from image_utils import gray, resize_for_analysis
from models import OrientationResult


class OrientationCorrector:
    """Select a 0/90/180/270 degree orientation without reading any text."""

    def __init__(self, config: OrientationConfig) -> None:
        self.config = config
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

    def correct(self, image: np.ndarray) -> OrientationResult:
        if not self.config.enabled:
            return OrientationResult(image, 0, 1.0, {0: 1.0})

        candidates = {angle: self._rotate(image, angle) for angle in (0, 90, 180, 270)}
        scores = {angle: self._layout_score(candidate) for angle, candidate in candidates.items()}
        ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        best_angle, best_score = ordered[0]
        method = "layout"
        if (
            best_angle != 0
            and scores[best_angle] - scores[0] < self.config.minimum_rotation_score_gain
        ):
            best_angle = 0
            method = "conservative_original"
        runner_up = max(score for angle, score in scores.items() if angle != best_angle)
        confidence = max(0.0, min(1.0, (scores[best_angle] - runner_up + 0.05) / 0.35))

        return OrientationResult(
            candidates[best_angle],
            best_angle,
            confidence,
            scores,
            method,
        )

    @staticmethod
    def _rotate(image: np.ndarray, angle: int) -> np.ndarray:
        if angle == 0:
            return image.copy()
        if angle == 90:
            return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        if angle == 180:
            return cv2.rotate(image, cv2.ROTATE_180)
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

    def _layout_score(self, image: np.ndarray) -> float:
        sample, _ = resize_for_analysis(image, 1500)
        channel = gray(sample)
        block_size = max(3, self.config.text_threshold_block_size | 1)
        binary = cv2.adaptiveThreshold(
            channel,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            block_size,
            self.config.text_threshold_constant,
        )
        height, width = binary.shape
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(3, width // 60), 1))
        connected = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        component_count, _, stats, _ = cv2.connectedComponentsWithStats(connected)
        useful = stats[1:] if component_count > 1 else np.empty((0, 5), dtype=np.int32)
        if len(useful) == 0:
            return 0.0

        widths = useful[:, cv2.CC_STAT_WIDTH]
        heights = useful[:, cv2.CC_STAT_HEIGHT]
        areas = useful[:, cv2.CC_STAT_AREA]
        horizontal = areas[(widths >= heights * 1.6) & (widths >= width * 0.025)].sum()
        horizontal_score = float(horizontal / max(1, areas.sum()))

        top = np.mean(binary[: max(1, int(height * 0.35))] > 0)
        bottom = np.mean(binary[int(height * 0.65) :] > 0)
        lower_preference = float(np.clip(0.5 + (bottom - top) * 5.0, 0.0, 1.0))

        points = cv2.findNonZero(binary)
        if points is None:
            compactness = 0.0
        else:
            _, _, content_w, content_h = cv2.boundingRect(points)
            compactness = float(np.clip((content_w * content_h) / (width * height), 0.0, 1.0))

        face_alignment, eye_alignment = self._face_alignment(channel)
        machine_readable_alignment = self._bottom_machine_readable_texture(binary)
        return (
            horizontal_score * self.config.horizontal_structure_weight
            + lower_preference * self.config.lower_band_weight
            + compactness * self.config.compactness_weight
            + face_alignment * self.config.face_weight
            + eye_alignment * self.config.eye_weight
            + machine_readable_alignment * self.config.machine_readable_texture_weight
        )

    def _face_alignment(self, channel: np.ndarray) -> tuple[float, float]:
        if self.face_cascade.empty():
            return 0.0, 0.0
        minimum = max(24, round(min(channel.shape) * 0.045))
        try:
            faces = self.face_cascade.detectMultiScale(
                channel,
                scaleFactor=self.config.face_scale_factor,
                minNeighbors=self.config.face_min_neighbors,
                minSize=(minimum, minimum),
            )
        except cv2.error:
            return 0.0, 0.0
        if len(faces) == 0:
            return 0.0, 0.0
        height, width = channel.shape
        best_face = 0.0
        best_eyes = 0.0
        for x, y, face_w, face_h in faces:
            area_ratio = (face_w * face_h) / (width * height)
            if not 0.001 <= area_ratio <= 0.15:
                continue
            placement = 1.0 if y < height * 0.84 else 0.55
            face_score = min(1.0, 0.65 + area_ratio / 0.06) * placement
            if self.eye_cascade.empty():
                best_face = max(best_face, face_score * self.config.unverified_face_weight)
                continue
            face_roi = channel[y : y + face_h, x : x + face_w]
            min_eye_size = max(8, round(min(face_w, face_h) * 0.10))
            try:
                eyes = self.eye_cascade.detectMultiScale(
                    face_roi,
                    scaleFactor=self.config.eye_scale_factor,
                    minNeighbors=self.config.eye_min_neighbors,
                    minSize=(min_eye_size, min_eye_size),
                )
            except cv2.error:
                eyes = ()
            eye_quality = eye_pair_quality(eyes, face_w, face_h, self.config)
            evidence_weight = (
                self.config.unverified_face_weight
                + (1.0 - self.config.unverified_face_weight) * eye_quality
            )
            best_face = max(best_face, face_score * evidence_weight)
            best_eyes = max(best_eyes, eye_quality)
        return best_face, best_eyes

    def _bottom_machine_readable_texture(self, binary: np.ndarray) -> float:
        height, width = binary.shape
        start = round(height * self.config.machine_readable_start_ratio)
        lower = binary[start:, :]
        active = np.mean(lower > 0, axis=1) > 0.008
        max_height = max(3, round(height * self.config.machine_readable_max_band_height_ratio))
        bands: list[tuple[int, int, float]] = []
        for begin, end in self._runs(active):
            if not 2 <= end - begin <= max_height:
                continue
            xs = np.where(np.any(lower[begin:end] > 0, axis=0))[0]
            if len(xs) == 0:
                continue
            coverage = (xs[-1] - xs[0] + 1) / width
            if coverage >= self.config.machine_readable_min_band_coverage:
                bands.append((begin, end, coverage))
        for first, second in zip(bands, bands[1:]):
            gap = second[0] - first[1]
            if 0 <= gap <= height * 0.11:
                return min(1.0, (first[2] + second[2]) / 1.4)
        return min(0.55, max((band[2] for band in bands), default=0.0) * 0.55)

    @staticmethod
    def _runs(active: np.ndarray) -> list[tuple[int, int]]:
        padded = np.pad(active.astype(np.int8), (1, 1))
        transitions = np.diff(padded)
        return list(zip(np.where(transitions == 1)[0], np.where(transitions == -1)[0]))
