"""Resolve saved project paths after the repository is moved between drives."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    if path.exists():
        return path
    for marker in ("output", "dataset"):
        if marker not in path.parts:
            continue
        marker_index = path.parts.index(marker)
        candidate = PROJECT_ROOT.joinpath(*path.parts[marker_index:])
        if candidate.exists():
            return candidate
    return path
