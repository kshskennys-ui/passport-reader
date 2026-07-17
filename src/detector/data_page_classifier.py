"""Score likely data pages from visual features without OCR or MRZ decoding."""

from __future__ import annotations

import cv2
import numpy as np

from config import ClassifierConfig
from face_geometry import eye_pair_quality
from image_utils import ensure_bgr, gray, resize_for_analysis
from models import FeatureScores, Rect


class DataPageClassifier:
    """OpenCV-only classifier for portrait, layout, text and MRZ-like texture cues."""

    def __init__(self, config: ClassifierConfig) -> None:
        self.config = config
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_eye.xml"
        )

    def classify(self, image: np.ndarray) -> tuple[FeatureScores, np.ndarray]:
        sample, scale = resize_for_analysis(ensure_bgr(image), 1800)
        channel = gray(sample)
        binary = self._ink_mask(channel)
        portrait_score, portrait_rect, portrait_source, portrait_eye_quality = self._portrait_feature(
            channel
        )
        mrz_score, mrz_bands = self._mrz_texture_feature(binary)
        text_score, text_details = self._text_density_feature(binary)
        layout_score = self._layout_feature(
            portrait_score, portrait_rect, mrz_score, mrz_bands, text_score, sample.shape[:2]
        )
        final_score = self._final_score(portrait_score, mrz_score, layout_score, text_score)

        scores = FeatureScores(
            portrait=portrait_score,
            mrz_texture=mrz_score,
            layout=layout_score,
            text_density=text_score,
            final=final_score,
            portrait_rect=self._restore_rect(portrait_rect, scale),
            mrz_bands=[self._restore_rect(band, scale) for band in mrz_bands],
            details={
                "portrait_source": portrait_source,
                "portrait_eye_quality": round(portrait_eye_quality, 4),
                "analysis_scale": scale,
                **text_details,
            },
        )
        return scores, self.annotate(image, scores)

    def apply_rotation_score(self, scores: FeatureScores, rotation_score: float) -> FeatureScores:
        scores.rotation = max(0.0, min(100.0, rotation_score))
        scores.final = self._final_score(
            scores.portrait,
            scores.mrz_texture,
            scores.layout,
            scores.text_density,
            scores.rotation,
        )
        return scores

    def confidence_decision(self, scores: FeatureScores) -> tuple[bool, str]:
        eye_quality = float(scores.details.get("portrait_eye_quality", 0.0))
        line_count = float(scores.details.get("line_count", 0.0))
        if (
            self.config.narrative_rejection_enabled
            and line_count >= self.config.narrative_minimum_line_count
            and eye_quality <= self.config.narrative_maximum_eye_quality
            and scores.mrz_texture <= self.config.narrative_maximum_mrz_score
        ):
            return False, "narrative_page_rejection"
        if scores.final >= self.config.confidence_threshold:
            return True, "standard_threshold"
        if (
            self.config.strong_layout_fallback_enabled
            and scores.final >= self.config.fallback_confidence_threshold
            and scores.portrait >= self.config.fallback_portrait_threshold
            and scores.layout >= self.config.fallback_layout_threshold
        ):
            return True, "strong_portrait_layout_fallback"
        if (
            self.config.verified_face_fallback_enabled
            and scores.final >= self.config.verified_face_fallback_confidence_threshold
            and scores.portrait >= self.config.verified_face_fallback_portrait_threshold
            and scores.layout >= self.config.verified_face_fallback_layout_threshold
            and eye_quality >= self.config.verified_face_fallback_eye_quality
        ):
            return True, "verified_face_layout_fallback"
        return False, "low_confidence"

    def selection_score(self, scores: FeatureScores) -> float:
        eye_quality = float(scores.details.get("portrait_eye_quality", 0.0))
        return scores.final + self.config.verified_face_selection_bonus * eye_quality

    def annotate(self, image: np.ndarray, scores: FeatureScores) -> np.ndarray:
        overlay = ensure_bgr(image).copy()
        if scores.portrait_rect:
            self._draw_rect(overlay, scores.portrait_rect, (40, 180, 40), "portrait")
        for index, band in enumerate(scores.mrz_bands, start=1):
            self._draw_rect(overlay, band, (40, 160, 240), f"texture {index}")
        lines = [
            f"Portrait  {scores.portrait:05.1f}",
            f"MRZ texture {scores.mrz_texture:05.1f}",
            f"Layout    {scores.layout:05.1f}",
            f"Text      {scores.text_density:05.1f}",
            f"Rotation  {scores.rotation:05.1f}",
            f"Final     {scores.final:05.1f}",
        ]
        font_scale = max(0.45, min(0.9, overlay.shape[1] / 1900))
        line_height = max(20, round(26 * font_scale))
        box_w = max(210, round(250 * font_scale))
        box_h = line_height * len(lines) + 14
        cv2.rectangle(overlay, (12, 12), (12 + box_w, 12 + box_h), (25, 25, 25), -1)
        for index, line in enumerate(lines):
            cv2.putText(
                overlay,
                line,
                (20, 12 + line_height * (index + 1)),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (245, 245, 245),
                1,
                cv2.LINE_AA,
            )
        return overlay

    def _ink_mask(self, channel: np.ndarray) -> np.ndarray:
        block_size = max(3, self.config.adaptive_block_size | 1)
        return cv2.adaptiveThreshold(
            channel,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            block_size,
            self.config.adaptive_constant,
        )

    def _portrait_feature(
        self, channel: np.ndarray
    ) -> tuple[float, Rect | None, str, float]:
        height, width = channel.shape
        face_score, face_rect, eye_quality = self._face_feature(channel)
        rectangle_score, rectangle_rect = self._portrait_rectangle_feature(channel)
        if face_score >= rectangle_score:
            return face_score, face_rect, "haar_face" if face_rect else "none", eye_quality
        return (
            rectangle_score,
            rectangle_rect,
            "portrait_rectangle" if rectangle_rect else "none",
            0.0,
        )

    def _face_feature(self, channel: np.ndarray) -> tuple[float, Rect | None, float]:
        if self.face_cascade.empty():
            return 0.0, None, 0.0
        height, width = channel.shape
        min_size = max(24, round(min(height, width) * 0.045))
        try:
            faces = self.face_cascade.detectMultiScale(
                channel,
                scaleFactor=self.config.face_scale_factor,
                minNeighbors=self.config.face_min_neighbors,
                minSize=(min_size, min_size),
            )
        except cv2.error:
            return 0.0, None, 0.0
        if len(faces) == 0:
            return 0.0, None, 0.0
        candidates: list[tuple[float, Rect, float]] = []
        for x, y, face_w, face_h in faces:
            area_ratio = (face_w * face_h) / (width * height)
            if not 0.001 <= area_ratio <= 0.12 or y > height * 0.83:
                continue
            aspect = min(face_w, face_h) / max(face_w, face_h)
            score = 72.0 + 18.0 * min(1.0, area_ratio / 0.025) + 10.0 * aspect
            eye_quality = self._face_eye_quality(channel, x, y, face_w, face_h)
            candidates.append(
                (min(100.0, score), Rect(int(x), int(y), int(face_w), int(face_h)), eye_quality)
            )
        return max(
            candidates,
            default=(0.0, None, 0.0),
            key=lambda item: (item[2] > 0.0, item[0]),
        )

    def _face_eye_quality(
        self,
        channel: np.ndarray,
        x: int,
        y: int,
        face_w: int,
        face_h: int,
    ) -> float:
        if self.eye_cascade.empty():
            return 0.0
        face_roi = channel[y : y + face_h, x : x + face_w]
        minimum = max(8, round(min(face_w, face_h) * 0.10))
        try:
            eyes = self.eye_cascade.detectMultiScale(
                face_roi,
                scaleFactor=self.config.eye_scale_factor,
                minNeighbors=self.config.eye_min_neighbors,
                minSize=(minimum, minimum),
            )
        except cv2.error:
            return 0.0
        return eye_pair_quality(eyes, face_w, face_h, self.config)

    def _portrait_rectangle_feature(self, channel: np.ndarray) -> tuple[float, Rect | None]:
        height, width = channel.shape
        edges = cv2.Canny(cv2.GaussianBlur(channel, (5, 5), 0), 55, 160)
        closed = cv2.morphologyEx(
            edges,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
        )
        contours, _ = cv2.findContours(closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        candidates: list[tuple[float, Rect]] = []
        for contour in contours:
            x, y, rect_w, rect_h = cv2.boundingRect(contour)
            area_ratio = (rect_w * rect_h) / (width * height)
            aspect = rect_w / max(1, rect_h)
            if not self.config.portrait_min_area_ratio <= area_ratio <= self.config.portrait_max_area_ratio:
                continue
            if not self.config.portrait_min_aspect_ratio <= aspect <= self.config.portrait_max_aspect_ratio:
                continue
            if y > height * 0.82:
                continue
            crop = channel[y : y + rect_h, x : x + rect_w]
            texture = min(1.0, float(np.std(crop)) / 38.0)
            edge_density = min(1.0, float(np.mean(edges[y : y + rect_h, x : x + rect_w] > 0)) / 0.10)
            shape = 1.0 - min(1.0, abs(aspect - 0.78) / 0.78)
            score = 30.0 + 36.0 * texture + 22.0 * edge_density + 12.0 * shape
            candidates.append((min(self.config.portrait_rectangle_max_score, score), Rect(x, y, rect_w, rect_h)))
        return max(candidates, default=(0.0, None), key=lambda item: item[0])

    def _mrz_texture_feature(self, binary: np.ndarray) -> tuple[float, list[Rect]]:
        height, width = binary.shape
        start_y = round(height * self.config.lower_region_start_ratio)
        lower = binary[start_y:, :]
        row_density = np.mean(lower > 0, axis=1)
        active = row_density > 0.008
        bands: list[Rect] = []
        max_band_height = max(3, round(height * self.config.mrz_max_band_height_ratio))
        for begin, end in self._runs(active):
            band_height = end - begin
            if not 2 <= band_height <= max_band_height:
                continue
            band = lower[begin:end, :]
            if float(np.mean(band > 0)) < self.config.mrz_min_band_ink_density:
                continue
            xs = np.where(np.any(band > 0, axis=0))[0]
            if len(xs) == 0:
                continue
            coverage = (xs[-1] - xs[0] + 1) / width
            if coverage >= self.config.mrz_min_band_coverage:
                bands.append(Rect(int(xs[0]), start_y + begin, int(xs[-1] - xs[0] + 1), band_height))
        bands = sorted(bands, key=lambda item: item.y)
        if not bands:
            return 0.0, []

        best_pair: tuple[Rect, Rect] | None = None
        for first, second in zip(bands, bands[1:]):
            gap = second.y - (first.y + first.h)
            if 0 <= gap <= height * 0.11:
                best_pair = first, second
                break
        selected = list(best_pair) if best_pair else [max(bands, key=lambda item: item.w)]
        mean_coverage = float(np.mean([band.w / width for band in selected]))
        pair_bonus = 30.0 if best_pair else 0.0
        lower_bonus = 10.0 * min(1.0, selected[-1].y / max(1, height * 0.75))
        score = min(100.0, mean_coverage * 60.0 + pair_bonus + lower_bonus)
        return score, selected

    def _text_density_feature(self, binary: np.ndarray) -> tuple[float, dict[str, float]]:
        height, width = binary.shape
        ink_density = float(np.mean(binary > 0))
        density_quality = max(0.0, 1.0 - abs(ink_density - 0.055) / 0.09)
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(3, width // 85), 1))
        grouped = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, horizontal_kernel)
        contours, _ = cv2.findContours(grouped, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        line_widths: list[float] = []
        for contour in contours:
            x, _, line_w, line_h = cv2.boundingRect(contour)
            if line_h < max(2, height * 0.003) or line_w < width * 0.025:
                continue
            line_widths.append(line_w / width)
        if not line_widths:
            return 0.0, {"ink_density": ink_density, "line_count": 0.0, "long_line_ratio": 1.0}
        long_line_ratio = sum(value >= self.config.field_long_line_ratio for value in line_widths) / len(line_widths)
        field_line_ratio = sum(0.06 <= value < self.config.field_long_line_ratio for value in line_widths) / len(line_widths)
        line_count_quality = min(1.0, len(line_widths) / 9.0)
        score = 100.0 * (
            0.36 * density_quality + 0.38 * field_line_ratio + 0.18 * (1.0 - long_line_ratio) + 0.08 * line_count_quality
        )
        return min(100.0, score), {
            "ink_density": round(ink_density, 5),
            "line_count": float(len(line_widths)),
            "long_line_ratio": round(long_line_ratio, 4),
            "field_line_ratio": round(field_line_ratio, 4),
        }

    def _layout_feature(
        self,
        portrait: float,
        portrait_rect: Rect | None,
        mrz: float,
        mrz_bands: list[Rect],
        text: float,
        image_shape: tuple[int, int],
    ) -> float:
        height, _ = image_shape
        base = 0.45 * (portrait / 100.0) + 0.30 * (text / 100.0) + 0.25 * (mrz / 100.0)
        relationship = 0.0
        if portrait_rect and portrait_rect.y < height * 0.82:
            relationship += 0.22
        if portrait_rect and mrz_bands and mrz_bands[0].y > portrait_rect.y + portrait_rect.h:
            relationship += 0.18
        if portrait >= 65.0 and text >= 50.0:
            relationship += 0.12
        return min(100.0, (base + relationship) * 100.0)

    def _final_score(
        self, portrait: float, mrz: float, layout: float, text: float, rotation: float = 0.0
    ) -> float:
        score = (
            portrait * self.config.portrait_weight
            + mrz * self.config.mrz_weight
            + layout * self.config.layout_weight
            + text * self.config.text_weight
            + rotation * self.config.rotation_weight
        )
        return min(100.0, max(0.0, score))

    @staticmethod
    def _runs(active: np.ndarray) -> list[tuple[int, int]]:
        padded = np.pad(active.astype(np.int8), (1, 1))
        change = np.diff(padded)
        return list(zip(np.where(change == 1)[0], np.where(change == -1)[0]))

    @staticmethod
    def _restore_rect(rect: Rect | None, scale: float) -> Rect | None:
        if rect is None or scale == 1.0:
            return rect
        return Rect(
            round(rect.x / scale),
            round(rect.y / scale),
            round(rect.w / scale),
            round(rect.h / scale),
        )

    @staticmethod
    def _draw_rect(image: np.ndarray, rect: Rect, color: tuple[int, int, int], label: str) -> None:
        cv2.rectangle(image, (rect.x, rect.y), (rect.x + rect.w, rect.y + rect.h), color, 2)
        cv2.putText(
            image,
            label,
            (rect.x, max(18, rect.y - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )
