from __future__ import annotations

from web.server import _find_artifact_module, _page_number, _stage_name


def test_web_artifact_helpers_extract_page_and_stage() -> None:
    assert _page_number("phase1/debug/document/page018/01_original.png") == 18
    assert _page_number("mrz/results/document/DataPage_007/fast_mrz_band.png") == 7
    assert _stage_name("fallback_mrz_crop") == "MRZ回退裁剪"


def test_web_export_runtime_module_name() -> None:
    assert _find_artifact_module().name == "artifact_tool.mjs"
