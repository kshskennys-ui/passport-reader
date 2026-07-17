"""Conservative edge-risk checks for OCR input images."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from config import NormalizerConfig
from image_utils import gray


@dataclass(frozen=True)
class ContentSafetyResult:
    safe: bool
    guard_px: int
    edge_ink_density: dict[str, float]
    warnings: list[str] = field(default_factory=list)


class ContentSafetyChecker:
    """Flag dark content touching an image edge before output padding is added."""

    def __init__(self, config: NormalizerConfig) -> None:
        self.config = config

    def assess(self, image: np.ndarray) -> ContentSafetyResult:
        channel = gray(image)
        height, width = channel.shape
        guard = max(6, round(min(width, height) * self.config.edge_guard_ratio))
        dark = channel < self.config.edge_ink_threshold
        densities = {
            "top": float(np.mean(dark[:guard, :])),
            "bottom": float(np.mean(dark[-guard:, :])),
            "left": float(np.mean(dark[:, :guard])),
            "right": float(np.mean(dark[:, -guard:])),
        }
        warnings = [
            f"dark_content_near_{edge}:{density:.4f}"
            for edge, density in densities.items()
            if density >= self.config.edge_ink_density_threshold
        ]
        return ContentSafetyResult(not warnings, guard, densities, warnings)
