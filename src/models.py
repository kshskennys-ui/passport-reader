"""Typed values shared between independent processing modules."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Rect:
    x: int
    y: int
    w: int
    h: int

    @property
    def area(self) -> int:
        return self.w * self.h

    def as_tuple(self) -> tuple[int, int, int, int]:
        return self.x, self.y, self.w, self.h

    def as_dict(self) -> dict[str, int]:
        return {"x": self.x, "y": self.y, "w": self.w, "h": self.h}


@dataclass
class SourcePage:
    source_path: Path
    source_index: int
    image: np.ndarray


@dataclass
class OrientationResult:
    image: np.ndarray
    angle: int
    confidence: float
    candidate_scores: dict[int, float]
    method: str = "layout"


@dataclass
class CropResult:
    image: np.ndarray
    rect: Rect
    confidence: float


@dataclass
class DocumentROI:
    image: np.ndarray
    rect: Rect
    confidence: float
    method: str


@dataclass
class Segment:
    image: np.ndarray
    rect: Rect
    index: int
    safe_image: np.ndarray | None = None
    safe_rect: Rect | None = None


@dataclass
class FeatureScores:
    portrait: float
    mrz_texture: float
    layout: float
    text_density: float
    final: float
    rotation: float = 0.0
    portrait_rect: Rect | None = None
    mrz_bands: list[Rect] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        if self.portrait_rect:
            result["portrait_rect"] = self.portrait_rect.as_dict()
        result["mrz_bands"] = [band.as_dict() for band in self.mrz_bands]
        return result


@dataclass
class CandidateResult:
    segment: Segment
    scores: FeatureScores
    overlay: np.ndarray
    elapsed_ms: float


@dataclass
class NormalizedResult:
    image: np.ndarray
    deskew_degrees: float
    trim_rect: Rect
    estimated_skew_degrees: float = 0.0
    deskew_confidence: float = 0.0
    residual_skew_degrees: float = 0.0
    padding_px: int = 0
    fallback_level: str = "selected_safe_roi"
    ocr_safe: bool = True
    warnings: list[str] = field(default_factory=list)


@dataclass
class PageResult:
    source_page: int
    selected_segment: int | None
    score: float
    status: str
    output_path: Path | None
    debug_dir: Path
    message: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)


@dataclass
class ProcessResult:
    input_path: Path
    page_results: list[PageResult]
    output_dir: Path

    @property
    def successful_pages(self) -> int:
        return sum(result.status == "selected" for result in self.page_results)
