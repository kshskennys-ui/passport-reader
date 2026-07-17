"""Locate a safe MRZ band from first-pass OCR geometry."""

from __future__ import annotations

import re
from dataclasses import dataclass

from config import MRZConfig
from models import Rect
from ocr.models import OCRLine

MRZ_CHARACTER = re.compile(r"[A-Z0-9<]")


@dataclass(frozen=True)
class MRZRow:
    line_indices: list[int]
    rect: Rect
    text: str
    confidence: float
    width_ratio: float


@dataclass(frozen=True)
class MRZRegion:
    rect: Rect
    rows: list[MRZRow]
    confidence: float
    method: str = "ocr_geometry"

    def as_dict(self) -> dict:
        return {
            "rect": self.rect.as_dict(),
            "rows": [
                {
                    "line_indices": row.line_indices,
                    "rect": row.rect.as_dict(),
                    "text": row.text,
                    "confidence": round(row.confidence, 6),
                    "width_ratio": round(row.width_ratio, 6),
                }
                for row in self.rows
            ],
            "confidence": round(self.confidence, 6),
            "method": self.method,
        }


def locate_mrz_region(
    lines: list[OCRLine], image_shape: tuple[int, int], config: MRZConfig
) -> MRZRegion | None:
    height, width = image_shape
    if not lines or width <= 0 or height <= 0:
        return None
    content = _content_bbox(lines, width, height)
    if content is None:
        return None
    fragments = [
        (index, line)
        for index, line in enumerate(lines)
        if _is_mrz_fragment(line, content, height, config)
    ]
    if not fragments:
        return None
    rows = _cluster_rows(fragments, width, height, config)
    if not rows:
        return None
    selected = rows[-config.maximum_rows :]
    if len(selected) < 2:
        selected = _expand_single_row(selected, lines, width, height, config)
    if len(selected) < 2:
        if len(selected) != 1 or selected[0].rect.h < config.minimum_row_height_px * 2:
            return None
    x1 = min(row.rect.x for row in selected)
    y1 = min(row.rect.y for row in selected)
    x2 = max(row.rect.x + row.rect.w for row in selected)
    y2 = max(row.rect.y + row.rect.h for row in selected)
    horizontal_padding = max(8, round(width * config.horizontal_padding_ratio))
    vertical_padding = max(8, round(height * config.vertical_padding_ratio))
    crop_x1 = max(0, min(content["x"], x1) - horizontal_padding)
    crop_x2 = min(width, max(content["x"] + content["w"], x2) + horizontal_padding)
    crop_y1 = max(0, y1 - vertical_padding)
    crop_y2 = min(height, y2 + vertical_padding)
    minimum_height = max(round(height * config.minimum_crop_height_ratio), round(height * 0.08))
    if crop_y2 - crop_y1 < minimum_height:
        center = (crop_y1 + crop_y2) // 2
        crop_y1 = max(0, center - minimum_height // 2)
        crop_y2 = min(height, crop_y1 + minimum_height)
        crop_y1 = max(0, crop_y2 - minimum_height)
    rect = Rect(crop_x1, crop_y1, max(1, crop_x2 - crop_x1), max(1, crop_y2 - crop_y1))
    return MRZRegion(
        rect=rect,
        rows=selected,
        confidence=sum(row.confidence for row in selected) / len(selected),
    )


def _expand_single_row(
    selected: list[MRZRow],
    lines: list[OCRLine],
    width: int,
    height: int,
    config: MRZConfig,
) -> list[MRZRow]:
    """Recover the neighboring MRZ row when OCR misreads its markers."""
    if len(selected) != 1:
        return selected
    row = selected[0]
    tall_row_threshold = max(
        config.minimum_row_height_px * 5,
        round(height * 0.04),
    )
    if row.rect.h >= tall_row_threshold:
        return selected
    candidates: list[tuple[float, MRZRow]] = []
    max_gap = max(24, round(row.rect.h * 2.25), round(height * 0.06))
    for index, line in enumerate(lines):
        if index in row.line_indices or not line.polygon:
            continue
        compact = "".join(line.text.upper().split())
        if len(compact) < 15 or line.confidence < config.minimum_fragment_confidence:
            continue
        allowed_ratio = len(MRZ_CHARACTER.findall(compact)) / len(compact)
        if allowed_ratio < config.minimum_allowed_ratio:
            continue
        candidate = _line_rect(index, line, width)
        overlap = _horizontal_overlap_ratio(candidate, row.rect)
        if overlap < 0.55:
            continue
        if candidate.y + candidate.h <= row.rect.y:
            gap = row.rect.y - candidate.y - candidate.h
        elif row.rect.y + row.rect.h <= candidate.y:
            gap = candidate.y - row.rect.y - row.rect.h
        else:
            continue
        if gap < 0 or gap > max_gap:
            continue
        candidates.append((gap + abs(candidate.w - row.rect.w) / max(1, width), _line_to_row(index, line, width)))
    if not candidates:
        return selected
    neighbor = min(candidates, key=lambda item: item[0])[1]
    return sorted([neighbor, row], key=lambda item: item.rect.y)


def _is_mrz_fragment(
    line: OCRLine, content: dict[str, int], height: int, config: MRZConfig
) -> bool:
    if line.confidence < config.minimum_fragment_confidence or not line.polygon:
        return False
    compact = "".join(line.text.upper().split())
    if len(compact) < 3:
        return False
    allowed_ratio = len(MRZ_CHARACTER.findall(compact)) / len(compact)
    marker_present = "<" in compact or (
        len(compact) >= 20 and any(character.isdigit() for character in compact)
    )
    bottom = max(point[1] for point in line.polygon)
    lower_boundary = content["y"] + content["h"] * config.lower_content_start_ratio
    return allowed_ratio >= config.minimum_allowed_ratio and marker_present and bottom >= lower_boundary


def _cluster_rows(
    fragments: list[tuple[int, OCRLine]], width: int, height: int, config: MRZConfig
) -> list[MRZRow]:
    ordered = sorted(fragments, key=lambda item: _center_y(item[1]))
    groups: list[list[tuple[int, OCRLine]]] = []
    current: list[tuple[int, OCRLine]] = []
    previous_center = None
    for item in ordered:
        center = _center_y(item[1])
        line_height = _height(item[1])
        tolerance = max(
            config.minimum_row_height_px,
            round(height * config.row_y_tolerance_ratio),
            round(line_height * 1.1),
        )
        if current and previous_center is not None and center - previous_center > tolerance:
            groups.append(current)
            current = []
        current.append(item)
        previous_center = center
    if current:
        groups.append(current)

    rows: list[MRZRow] = []
    for group in groups:
        points = [point for _, line in group for point in line.polygon]
        x1, y1 = min(point[0] for point in points), min(point[1] for point in points)
        x2, y2 = max(point[0] for point in points), max(point[1] for point in points)
        rect = Rect(x1, y1, max(1, x2 - x1), max(1, y2 - y1))
        width_ratio = rect.w / max(1, width)
        if width_ratio < config.minimum_row_width_ratio:
            continue
        rows.append(
            MRZRow(
                line_indices=[index for index, _ in sorted(group, key=lambda item: _left_x(item[1]))],
                rect=rect,
                text="".join(line.text for _, line in sorted(group, key=lambda item: _left_x(item[1]))),
                confidence=sum(line.confidence for _, line in group) / len(group),
                width_ratio=width_ratio,
            )
        )
    return rows


def _line_to_row(index: int, line: OCRLine, width: int) -> MRZRow:
    rect = _line_rect(index, line, width)
    return MRZRow(
        line_indices=[index],
        rect=rect,
        text=line.text,
        confidence=line.confidence,
        width_ratio=rect.w / max(1, width),
    )


def _line_rect(index: int, line: OCRLine, width: int) -> Rect:
    points = line.polygon
    x1, y1 = min(point[0] for point in points), min(point[1] for point in points)
    x2, y2 = max(point[0] for point in points), max(point[1] for point in points)
    return Rect(x1, y1, max(1, x2 - x1), max(1, y2 - y1))


def _horizontal_overlap_ratio(first: Rect, second: Rect) -> float:
    overlap = max(0, min(first.x + first.w, second.x + second.w) - max(first.x, second.x))
    return overlap / max(1, min(first.w, second.w))


def _content_bbox(lines: list[OCRLine], width: int, height: int) -> dict[str, int] | None:
    points = [point for line in lines if line.polygon for point in line.polygon]
    if not points:
        return None
    x1, y1 = max(0, min(point[0] for point in points)), max(0, min(point[1] for point in points))
    x2, y2 = min(width, max(point[0] for point in points)), min(height, max(point[1] for point in points))
    return {"x": x1, "y": y1, "w": max(1, x2 - x1), "h": max(1, y2 - y1)}


def _center_y(line: OCRLine) -> float:
    return sum(point[1] for point in line.polygon) / len(line.polygon)


def _left_x(line: OCRLine) -> int:
    return min(point[0] for point in line.polygon)


def _height(line: OCRLine) -> int:
    return max(point[1] for point in line.polygon) - min(point[1] for point in line.polygon)
