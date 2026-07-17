"""Route a supported input file to the correct loader."""

from __future__ import annotations

from pathlib import Path

from config import LoaderConfig
from loader.image_loader import ImageLoader
from loader.pdf_loader import PdfLoader
from models import SourcePage

PDF_SUFFIXES = {".pdf"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
SUPPORTED_SUFFIXES = PDF_SUFFIXES | IMAGE_SUFFIXES


def load_pages(path: Path, config: LoaderConfig) -> list[SourcePage]:
    normalized = path.expanduser().resolve()
    if not normalized.exists():
        raise FileNotFoundError(normalized)
    suffix = normalized.suffix.lower()
    if suffix in PDF_SUFFIXES:
        return PdfLoader(config).load(normalized)
    if suffix in IMAGE_SUFFIXES:
        return ImageLoader().load(normalized)
    supported = ", ".join(sorted(SUPPORTED_SUFFIXES))
    raise ValueError(f"Unsupported input type '{suffix}'. Supported: {supported}")
