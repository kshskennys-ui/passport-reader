"""Typed OCR results independent of the PaddleOCR runtime."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ocr.path_utils import resolve_project_path

@dataclass(frozen=True)
class OCRLine:
    polygon: list[list[int]]
    text: str
    confidence: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class OCRPageResult:
    source_image: Path
    document: str
    page_number: int
    status: str
    elapsed_ms: float
    lines: list[OCRLine] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    error: str | None = None
    overlay_path: Path | None = None
    inference_signature: str | None = None
    analysis_signature: str | None = None
    reused: bool = False

    def as_dict(self, output_root: Path | None = None) -> dict[str, Any]:
        overlay = self.overlay_path
        if overlay and output_root:
            try:
                overlay_value = str(overlay.relative_to(output_root))
            except ValueError:
                overlay_value = str(overlay)
        else:
            overlay_value = str(overlay) if overlay else None
        return {
            "source_image": str(self.source_image),
            "source_fingerprint": source_fingerprint(self.source_image),
            "document": self.document,
            "page_number": self.page_number,
            "status": self.status,
            "elapsed_ms": round(self.elapsed_ms, 2),
            "lines": [line.as_dict() for line in self.lines],
            "metrics": self.metrics,
            "warnings": self.warnings,
            "error": self.error,
            "overlay_path": overlay_value,
            "inference_signature": self.inference_signature,
            "analysis_signature": self.analysis_signature,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any], output_root: Path) -> "OCRPageResult":
        overlay_value = payload.get("overlay_path")
        overlay = Path(overlay_value) if overlay_value else None
        if overlay and not overlay.is_absolute():
            overlay = output_root / overlay
        return cls(
            source_image=resolve_project_path(payload["source_image"]),
            document=str(payload.get("document", "")),
            page_number=int(payload.get("page_number", 0)),
            status=str(payload.get("status", "error")),
            elapsed_ms=float(payload.get("elapsed_ms", 0.0)),
            lines=[
                OCRLine(
                    polygon=[[int(value) for value in point] for point in item.get("polygon", [])],
                    text=str(item.get("text", "")),
                    confidence=float(item.get("confidence", 0.0)),
                )
                for item in payload.get("lines", [])
            ],
            metrics=dict(payload.get("metrics", {})),
            warnings=list(payload.get("warnings", [])),
            error=payload.get("error"),
            overlay_path=overlay,
            inference_signature=payload.get("inference_signature"),
            analysis_signature=payload.get("analysis_signature"),
            reused=True,
        )


def source_fingerprint(path: Path) -> dict[str, int]:
    stat = path.stat()
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
