"""Raster image adapters, including multipage TIFF support."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps, ImageSequence

from image_utils import ensure_bgr, read_image
from models import SourcePage


class ImageLoader:
    def load(self, path: Path) -> list[SourcePage]:
        if path.suffix.lower() in {".tif", ".tiff"}:
            return self._load_tiff(path)
        return [SourcePage(path, 1, read_image(path))]

    @staticmethod
    def _load_tiff(path: Path) -> list[SourcePage]:
        pages: list[SourcePage] = []
        with Image.open(path) as source:
            for index, frame in enumerate(ImageSequence.Iterator(source), start=1):
                corrected = ImageOps.exif_transpose(frame).convert("RGB")
                rgb = np.asarray(corrected)
                pages.append(SourcePage(path, index, ensure_bgr(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))))
        if not pages:
            raise ValueError(f"TIFF has no readable frames: {path}")
        return pages
