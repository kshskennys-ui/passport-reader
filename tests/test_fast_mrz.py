from __future__ import annotations

from pathlib import Path

import numpy as np

from config import MRZConfig, OCRConfig
from image_utils import write_png
from ocr.fast_mrz_runner import FastMRZRunner
from ocr.models import OCRLine


class FakeFastEngine:
    def __init__(self) -> None:
        self.calls: list[Path] = []

    def recognize(self, image_path: Path) -> list[OCRLine]:
        self.calls.append(image_path)
        return [
            OCRLine([], "P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<", 0.99),
            OCRLine([], "L898902C36UTO7408122F1204159ZE184226B<<<<<10", 0.99),
        ]


def test_fast_runner_uses_band_without_fallback(tmp_path: Path) -> None:
    input_root = tmp_path / "data_pages" / "sample"
    image_path = input_root / "DataPage_001.png"
    write_png(image_path, np.full((200, 300, 3), 255, dtype=np.uint8))
    engine = FakeFastEngine()

    results, _ = FastMRZRunner(OCRConfig(), MRZConfig(), engine=engine).run(
        input_root.parent, tmp_path / "fast", resume=False
    )

    assert len(results) == 1
    assert results[0]["status"] == "ok"
    assert results[0]["mode"] == "fast_band"
    assert results[0]["mrz_parse"]["status"] == "valid"
    assert len(engine.calls) == 1


def test_fast_runner_rejects_invalid_worker_count(tmp_path: Path) -> None:
    input_root = tmp_path / "data_pages" / "sample"
    write_png(input_root / "DataPage_001.png", np.full((200, 300, 3), 255, dtype=np.uint8))

    try:
        FastMRZRunner(OCRConfig(), MRZConfig(), engine=FakeFastEngine()).run(
            input_root.parent, tmp_path / "fast", workers=0, resume=False
        )
    except ValueError as exc:
        assert "workers" in str(exc)
    else:
        raise AssertionError("workers=0 should be rejected")
