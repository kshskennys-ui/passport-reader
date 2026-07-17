from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from config import OCRConfig
from config import MRZConfig
from image_utils import write_png
from ocr.analyzer import analyze_ocr_lines, is_mrz_candidate
from ocr.baseline import OCRBaselineRunner
from ocr.models import OCRLine
from ocr.paddle_engine import parse_paddle_result
from ocr.mrz_locator import locate_mrz_region


class FakeEngine:
    def __init__(self) -> None:
        self.calls = 0

    def recognize(self, image_path: Path) -> list[OCRLine]:
        self.calls += 1
        return [
            OCRLine([[20, 20], [180, 20], [180, 40], [20, 40]], "PASSPORT", 0.96),
            OCRLine(
                [[10, 105], [190, 105], [190, 120], [10, 120]],
                "P<CHNLI<<MING<<<<<<<<<<<<<<<<<<<<<<<<<<<<",
                0.98,
            ),
            OCRLine(
                [[10, 130], [190, 130], [190, 145], [10, 145]],
                "E123456789CHN9001011M3001012<<<<<<<<<<<<04",
                0.97,
            ),
        ]


def test_parse_paddle_3_result() -> None:
    class Result:
        json = {
            "res": {
                "rec_texts": ["NAME", "P<CHNTEST<<<<<<<<<<<<<<<<<<<<"],
                "rec_scores": [0.9, 0.95],
                "rec_polys": [
                    [[1, 2], [10, 2], [10, 5], [1, 5]],
                    [[1, 20], [30, 20], [30, 25], [1, 25]],
                ],
            }
        }

    lines = parse_paddle_result(Result())

    assert [line.text for line in lines] == ["NAME", "P<CHNTEST<<<<<<<<<<<<<<<<<<<<"]
    assert lines[1].confidence == 0.95
    assert lines[0].polygon[0] == [1, 2]


def test_analyzer_identifies_lower_machine_readable_line() -> None:
    config = OCRConfig()
    upper = OCRLine([[0, 10], [100, 10], [100, 20], [0, 20]], "P<CHN" + "<" * 30, 0.9)
    lower = OCRLine([[0, 70], [100, 70], [100, 80], [0, 80]], "P<CHN" + "<" * 30, 0.9)

    assert not is_mrz_candidate(upper, (100, 120), config)
    assert is_mrz_candidate(lower, (100, 120), config)

    metrics, warnings = analyze_ocr_lines([upper, lower], (100, 120), config)
    assert metrics["mrz_candidate_count"] == 1
    assert warnings == ["mrz_candidate_incomplete"]


def test_runner_writes_report_and_reuses_matching_result(tmp_path: Path) -> None:
    input_root = tmp_path / "phase1" / "data_pages"
    image_path = input_root / "sample" / "DataPage_001.png"
    write_png(image_path, np.full((160, 220, 3), 255, dtype=np.uint8))
    output_root = tmp_path / "ocr"
    engine = FakeEngine()
    runner = OCRBaselineRunner(OCRConfig(), engine=engine)

    first, report = runner.run(input_root, output_root)
    second, _ = runner.run(input_root, output_root)

    assert engine.calls == 1
    assert first[0].status == "ok"
    assert second[0].reused
    assert report.exists()
    assert (output_root / "summary.json").exists()
    result_path = output_root / "results" / "sample" / "DataPage_001" / "ocr.json"
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    assert payload["metrics"]["mrz_candidate_count"] == 2
    assert (result_path.parent / "overlay.png").exists()


def test_mrz_locator_merges_split_rows_with_safe_padding() -> None:
    lines = [
        OCRLine([[40, 120], [120, 120], [120, 140], [40, 140]], "NAME", 0.95),
        OCRLine([[100, 800], [480, 800], [480, 824], [100, 824]], "P<CHNTEST<<<<<<<<<<<<<<<<", 0.90),
        OCRLine([[500, 801], [900, 801], [900, 824], [500, 824]], "<<<<<<<<<<<<<<<<<<<", 0.88),
        OCRLine([[100, 840], [520, 840], [520, 864], [100, 864]], "E123456789CHN900101<<<<<<<<", 0.92),
        OCRLine([[540, 841], [900, 841], [900, 864], [540, 864]], "1M3001012<<<<<<<<<<", 0.91),
    ]

    region = locate_mrz_region(lines, (1000, 1000), MRZConfig())

    assert region is not None
    assert len(region.rows) == 2
    assert region.rect.x < 100
    assert region.rect.y < 800
    assert region.rect.y + region.rect.h > 864
    assert region.rows[0].line_indices == [1, 2]
    assert region.rows[1].line_indices == [3, 4]


def test_mrz_locator_recovers_neighbor_when_first_row_loses_markers() -> None:
    lines = [
        OCRLine([[40, 120], [220, 120], [220, 145], [40, 145]], "NAME", 0.95),
        OCRLine(
            [[68, 675], [1055, 671], [1055, 714], [66, 718]],
            "PAIDNMULYANTOAAASHARIAAAAAAAAAAAAAAAAAAAAAAA",
            0.76,
        ),
        OCRLine(
            [[68, 732], [1056, 731], [1056, 765], [68, 766]],
            "X4439894<8IDN8101316M34121903216093101000398",
            0.95,
        ),
    ]

    region = locate_mrz_region(lines, (1124, 1100), MRZConfig())

    assert region is not None
    assert len(region.rows) == 2
    assert region.rows[0].line_indices == [1]
    assert region.rows[1].line_indices == [2]
    assert region.rect.y < 675
