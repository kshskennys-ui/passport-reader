"""PDF rendering adapter based on PyMuPDF."""

from __future__ import annotations

from pathlib import Path

import cv2
import fitz
import numpy as np

from config import LoaderConfig
from models import SourcePage


class PdfLoader:
    def __init__(self, config: LoaderConfig) -> None:
        self.config = config

    def load(self, path: Path) -> list[SourcePage]:
        pages: list[SourcePage] = []
        with fitz.open(path) as document:
            for index, page in enumerate(document):
                scale = self._scale_for_page(page)
                pixmap = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
                rgb = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(
                    pixmap.height, pixmap.width, pixmap.n
                )
                image = cv2.cvtColor(rgb[:, :, :3], cv2.COLOR_RGB2BGR)
                pages.append(SourcePage(path, index + 1, image))
        return pages

    def _scale_for_page(self, page: fitz.Page) -> float:
        requested = self.config.pdf_dpi / 72.0
        maximum_scale = self.config.max_render_dimension / max(page.rect.width, page.rect.height)
        return min(requested, maximum_scale)
