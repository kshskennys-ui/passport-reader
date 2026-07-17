"""Image conversion and Unicode-safe image persistence helpers."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def ensure_bgr(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image


def gray(image: np.ndarray) -> np.ndarray:
    image = ensure_bgr(image)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def read_image(path: Path) -> np.ndarray:
    raw = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(raw, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"OpenCV cannot decode image: {path}")
    return ensure_bgr(image)


def write_png(path: Path, image: np.ndarray, compression: int = 3) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    success, buffer = cv2.imencode(".png", ensure_bgr(image), [cv2.IMWRITE_PNG_COMPRESSION, compression])
    if not success:
        raise ValueError(f"Cannot encode PNG: {path}")
    buffer.tofile(str(path))


def resize_for_analysis(image: np.ndarray, max_dimension: int) -> tuple[np.ndarray, float]:
    height, width = image.shape[:2]
    longest = max(height, width)
    if longest <= max_dimension:
        return image, 1.0
    scale = max_dimension / longest
    return cv2.resize(image, (round(width * scale), round(height * scale))), scale


def scale_rect(rect: tuple[int, int, int, int], factor: float) -> tuple[int, int, int, int]:
    if factor == 1.0:
        return rect
    return tuple(round(value / factor) for value in rect)  # type: ignore[return-value]
