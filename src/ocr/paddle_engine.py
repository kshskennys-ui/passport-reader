"""Lazy PaddleOCR 3.x adapter with a small, testable output surface."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from config import OCRConfig
from ocr.models import OCRLine


class PaddleOCREngine:
    def __init__(self, config: OCRConfig) -> None:
        self.config = config
        self._engine: Any | None = None

    def recognize(self, image_path: Path) -> list[OCRLine]:
        engine = self._get_engine()
        results = list(engine.predict(str(image_path), text_rec_score_thresh=0.0))
        lines: list[OCRLine] = []
        for result in results:
            lines.extend(parse_paddle_result(result))
        return lines

    def _get_engine(self) -> Any:
        if self._engine is None:
            from paddleocr import PaddleOCR

            options: dict[str, Any] = {
                "text_detection_model_name": self.config.text_detection_model_name,
                "text_recognition_model_name": self.config.text_recognition_model_name,
                "use_doc_orientation_classify": self.config.use_doc_orientation_classify,
                "use_doc_unwarping": self.config.use_doc_unwarping,
                "use_textline_orientation": self.config.use_textline_orientation,
                "text_recognition_batch_size": self.config.text_recognition_batch_size,
                "enable_mkldnn": self.config.enable_mkldnn,
                "device": self.config.device,
            }
            if not self.config.text_detection_model_name and not self.config.text_recognition_model_name:
                options.update(lang=self.config.language, ocr_version=self.config.ocr_version)
            self._engine = PaddleOCR(
                **options,
            )
        return self._engine


def parse_paddle_result(result: Any) -> list[OCRLine]:
    payload = _as_payload(result)
    data = payload.get("res", payload)
    texts = list(data.get("rec_texts", []))
    scores = list(data.get("rec_scores", []))
    polygons = data.get("rec_polys") or data.get("dt_polys") or []
    if hasattr(polygons, "tolist"):
        polygons = polygons.tolist()

    lines: list[OCRLine] = []
    for index, text in enumerate(texts):
        polygon = polygons[index] if index < len(polygons) else []
        score = scores[index] if index < len(scores) else 0.0
        if hasattr(polygon, "tolist"):
            polygon = polygon.tolist()
        lines.append(
            OCRLine(
                polygon=[[int(round(float(value))) for value in point] for point in polygon],
                text=str(text),
                confidence=float(score),
            )
        )
    return lines


def _as_payload(result: Any) -> dict[str, Any]:
    value = getattr(result, "json", result)
    if callable(value):
        value = value()
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, dict):
        raise TypeError(f"Unsupported PaddleOCR result type: {type(value).__name__}")
    return value
