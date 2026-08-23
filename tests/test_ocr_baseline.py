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
from ocr.mrz_rows import build_row_crops, merge_row_lines
from ocr.mrz_runner import build_direct_row_results
from ocr.mrz_parser import parse_mrz_row_results


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


def test_mrz_locator_does_not_expand_an_already_tall_row() -> None:
    lines = [
        OCRLine([[800, 520], [1050, 520], [1050, 550], [800, 550]], "AUTHORITYTEXT", 0.95),
        OCRLine(
            [[100, 600], [900, 600], [900, 680], [100, 680]],
            "A904788527CHN0309183M3001204NDNKNBPD<<<<A078PMCHNYU<<YANG<<<<<<<<<<<<<<<<<<<",
            0.90,
        ),
    ]

    region = locate_mrz_region(lines, (1000, 1100), MRZConfig())

    assert region is not None
    assert len(region.rows) == 1
    assert region.rows[0].line_indices == [1]


def test_mrz_locator_does_not_expand_a_short_date_neighbor() -> None:
    lines = [
        OCRLine([[100, 600], [700, 600], [700, 640], [100, 640]], "PROFESSIONALTEREDTOSEAMAN", 0.95),
        OCRLine([[100, 690], [300, 690], [300, 720], [100, 720]], "DATE09102023", 0.95),
    ]

    region = locate_mrz_region(lines, (800, 900), MRZConfig())

    assert region is None


def test_mrz_rows_upscale_and_merge_split_ocr_boxes() -> None:
    lines = [
        OCRLine([[100, 800], [480, 800], [480, 824], [100, 824]], "P<CHNTEST<<<<<<<<", 0.90),
        OCRLine([[500, 801], [900, 801], [900, 824], [500, 824]], "<<<<<<<<<<<<<<", 0.88),
        OCRLine([[100, 840], [520, 840], [520, 864], [100, 864]], "E123456789CHN900101<<<<<<<<", 0.92),
        OCRLine([[540, 841], [900, 841], [900, 864], [540, 864]], "1M3001012<<<<<<<<", 0.91),
    ]
    region = locate_mrz_region(lines, (1000, 1000), MRZConfig())

    assert region is not None
    crops = build_row_crops(np.zeros((1000, 1000, 3), dtype=np.uint8), region, MRZConfig())
    assert len(crops) == 2
    assert crops[0].scale == 3.0
    assert crops[0].image.shape[0] > crops[0].crop_rect.h

    first_row_lines = [
        OCRLine([[20, 50], [220, 50], [220, 70], [20, 70]], "P<CHNTEST<<<<<<<<", 0.90),
        OCRLine([[230, 51], [420, 51], [420, 71], [230, 71]], "<<<<<<<<<<<<<<", 0.88),
    ]
    merged = merge_row_lines(first_row_lines, crops[0], MRZConfig())

    assert merged.fragment_count == 2
    assert merged.normalized_text == "P<CHNTEST<<<<<<<<" + "<<<<<<<<<<<<<<"


def test_mrz_parser_reconstructs_concatenated_td3_and_validates_checks() -> None:
    first = "P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<"
    second = "L898902C36UTO7408122F1204159ZE184226B<<<<<10"

    parsed = parse_mrz_row_results([{"row_index": 1, "normalized_text": first + second}])

    assert parsed["status"] == "valid"
    assert parsed["format"] == "TD3"
    assert parsed["reconstruction"]["method"] == "split_concatenated_row"
    assert parsed["fields"]["surname"] == "ERIKSSON"
    assert parsed["fields"]["given_names"] == "ANNA MARIA"
    assert parsed["fields"]["passport_number"] == "L898902C3"
    assert parsed["validation"]["all_check_digits_valid"]


def test_mrz_parser_reorders_reversed_concatenated_td3_rows() -> None:
    first = "P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<"
    second = "L898902C36UTO7408122F1204159ZE184226B<<<<<10"

    parsed = parse_mrz_row_results([{"row_index": 1, "normalized_text": second + first}])

    assert parsed["status"] == "valid"
    assert parsed["reconstruction"]["row_order"] == "swapped"
    assert parsed["fields"]["surname"] == "ERIKSSON"
    assert parsed["fields"]["passport_number"] == "L898902C3"


def test_mrz_parser_does_not_correct_invalid_check_digit() -> None:
    first = "P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<"
    second = "L898902C36UTO7408122F1204159ZE184226B<<<<<11"

    parsed = parse_mrz_row_results(
        [
            {"row_index": 1, "normalized_text": first},
            {"row_index": 2, "normalized_text": second},
        ]
    )

    assert parsed["status"] == "invalid"
    assert "check_digit_failed" in parsed["reasons"]
    assert parsed["validation"]["check_digits"][-1]["value"] == "1"


def test_mrz_parser_accepts_validated_missing_filler_character() -> None:
    first = "P<IDNPRAMONO<<SETYO<<<<<<<<<<<<<<<<<<<<<<<".ljust(43, "<")
    second = "X4973624<2IDN9209264M30021203273132609000644"

    parsed = parse_mrz_row_results(
        [
            {"row_index": 1, "normalized_text": first},
            {"row_index": 2, "normalized_text": second},
        ]
    )

    assert parsed["status"] == "valid"
    assert parsed["reconstruction"]["recovery"] == "validated_filler_padding"
    assert parsed["fields"]["passport_number"] == "X4973624"


def test_mrz_parser_accepts_validated_document_marker_repair() -> None:
    first = "PCIDNRACO<<HERBY<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
    second = "E2994177<9IDN7102039M33031593201010302001218"

    parsed = parse_mrz_row_results(
        [
            {"row_index": 1, "normalized_text": first},
            {"row_index": 2, "normalized_text": second},
        ]
    )

    assert parsed["status"] == "valid"
    assert parsed["reconstruction"]["recovery"] == "validated_structural_marker"
    assert parsed["fields"]["document_code"] == "P<"


def test_mrz_parser_accepts_validated_filler_and_optional_check_repair() -> None:
    first = "P<INDPUSHPALINGAM<<PRABU<<<<<<<<<<<<<<<"
    second = "Z4715450<5IND8207137M2812139<<<<<<<<<<<<<<<8"

    parsed = parse_mrz_row_results(
        [
            {"row_index": 1, "normalized_text": first},
            {"row_index": 2, "normalized_text": second},
        ]
    )

    assert parsed["status"] == "valid"
    assert parsed["reconstruction"]["recovery"] == (
        "validated_filler_padding+validated_optional_data_check"
    )


def test_mrz_parser_accepts_blank_optional_data_check_as_zero() -> None:
    first = "P<INDYADAV<<ASHOK<KUMAR<<<<<<<<<<<<<<<<<<<<<"
    second = "V0052150<4IND8411154M2911102<<<<<<<<<<<<<<<6"

    parsed = parse_mrz_row_results(
        [
            {"row_index": 1, "normalized_text": first},
            {"row_index": 2, "normalized_text": second},
        ]
    )

    assert parsed["status"] == "valid"
    assert parsed["reconstruction"]["recovery"] == "validated_optional_data_check"


def test_mrz_parser_accepts_validated_passport_character_confusion() -> None:
    first = "PINDGIRI<SUNDAR<<"
    second = "H4965412<9IND0111159M32101122066947084822<84"

    parsed = parse_mrz_row_results(
        [
            {"row_index": 1, "normalized_text": first},
            {"row_index": 2, "normalized_text": second},
        ]
    )

    assert parsed["status"] == "valid"
    assert parsed["fields"]["passport_number"] == "W4965412"
    assert parsed["reconstruction"]["recovery"] == (
        "validated_filler_padding+validated_structural_marker+"
        "validated_passport_character_correction"
    )


def test_mrz_parser_accepts_seafarer_pm_document_code() -> None:
    first = "PMCHNYANG<<QINGSHU<<<<<<<<<<<<<<<<<<<<<<<<<<"
    second = "A905326773CHN6610012M3008085NB00MH0FMKPHA934"

    parsed = parse_mrz_row_results(
        [
            {"row_index": 1, "normalized_text": first},
            {"row_index": 2, "normalized_text": second},
        ]
    )

    assert parsed["status"] == "valid"
    assert parsed["fields"]["document_code"] == "PM"
    assert "recovery" not in parsed["reconstruction"]


def test_mrz_parser_marks_seafarer_optional_check_failure_as_partial() -> None:
    first = "PMCHNWU<<YONGBING<<<<<<<<<<<<<<<<<<<<<<<<<<<"
    second = "A904448889CHN8607232M2908133M00CNDMALBPIA980"

    parsed = parse_mrz_row_results(
        [
            {"row_index": 1, "normalized_text": first},
            {"row_index": 2, "normalized_text": second},
        ]
    )

    assert parsed["status"] == "partial"
    assert parsed["validation"]["essential_check_digits_valid"]
    assert parsed["fields"]["passport_number"] == "A90444888"


def test_mrz_parser_pads_omitted_pm_name_fillers() -> None:
    first = "PMCHNLI<CHENGLONG<<<<<<<<<<"
    second = "A906222724CHN8809232M3106246MA00LDMJMBPKA946"

    parsed = parse_mrz_row_results(
        [
            {"row_index": 1, "normalized_text": first},
            {"row_index": 2, "normalized_text": second},
        ]
    )

    assert parsed["status"] == "partial"
    assert parsed["reconstruction"]["recovery"] == "pm_filler_padding"
    assert parsed["fields"]["surname"] == "LI CHENGLONG"
    assert parsed["fields"]["given_names"] == ""
    assert parsed["validation"]["essential_check_digits_valid"]


def test_direct_mrz_rows_keep_data_row_without_filler_characters() -> None:
    lines = [
        OCRLine([], "PMCHNWU<<CHENXU<<<<<<<<<<<<<<<<<<<<<<<<<<<<<", 0.98),
        OCRLine([], "A905501761CHN0409094M3009299M00CLDLPNAPBA018", 0.98),
    ]

    rows = build_direct_row_results(lines, MRZConfig())

    assert len(rows) == 2
    assert rows[1]["normalized_text"] == "A905501761CHN0409094M3009299M00CLDLPNAPBA018"


def test_mrz_parser_removes_noise_after_final_name_filler_run() -> None:
    first = "PMCHNWU<<YONGBING<<<<<<<<<<<<<<<<<<<<<<<N<<<"
    second = "A904448889CHN8607232M2908133M00CNDMALBPIA980"

    parsed = parse_mrz_row_results(
        [
            {"row_index": 1, "normalized_text": first},
            {"row_index": 2, "normalized_text": second},
        ]
    )

    assert parsed["fields"]["given_names"] == "YONGBING"
