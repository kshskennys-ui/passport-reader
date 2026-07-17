"""Reconstruct GUI result models from an existing debug run."""

from __future__ import annotations

import json
from pathlib import Path

from config import PipelineConfig
from debug.logger import DebugRun
from models import PageResult, ProcessResult


def load_saved_process(
    input_path: str | Path,
    output_dir: str | Path,
    config: PipelineConfig,
) -> ProcessResult:
    source = Path(input_path).expanduser().resolve()
    output_root = Path(output_dir).expanduser().resolve()
    run_name = DebugRun._safe_name(source.stem)
    debug_root = output_root / config.output.debug_subdirectory / run_name
    if not debug_root.is_dir():
        raise FileNotFoundError(f"Saved debug run not found: {debug_root}")

    results: list[PageResult] = []
    for log_path in sorted(debug_root.glob("page*/log.json")):
        payload = json.loads(log_path.read_text(encoding="utf-8"))
        metrics = dict(payload.get("metrics", {}))
        source_page = int(payload.get("source_page", log_path.parent.name.removeprefix("page")))
        status = str(payload.get("status", "error"))
        scores = metrics.get("scores", {})
        output_path = (
            output_root
            / config.output.output_subdirectory
            / run_name
            / f"DataPage_{source_page:03d}.png"
        )
        if status != "selected" or not output_path.is_file():
            output_path = None
        results.append(
            PageResult(
                source_page=source_page,
                selected_segment=metrics.get("selected_segment"),
                score=float(scores.get("final", 0.0)),
                status=status,
                output_path=output_path,
                debug_dir=log_path.parent,
                message=str(payload.get("message", "")),
                metrics=metrics,
            )
        )
    if not results:
        raise ValueError(f"Saved run contains no page logs: {debug_root}")
    return ProcessResult(source, results, output_root)
