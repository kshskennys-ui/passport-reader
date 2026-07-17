"""Local HTML and JSON reports for an OCR baseline run."""

from __future__ import annotations

import html
import json
import platform
from collections import Counter
from dataclasses import asdict
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from statistics import mean
from urllib.parse import quote

from config import OCRConfig
from ocr.models import OCRPageResult


def summarize(results: list[OCRPageResult]) -> dict[str, object]:
    completed = [result for result in results if result.status != "error"]
    confidence_values = [
        float(result.metrics["mean_confidence"])
        for result in completed
        if result.metrics.get("mean_confidence") is not None
    ]
    candidate_counts = [int(result.metrics.get("mrz_candidate_count", 0)) for result in completed]
    return {
        "pages": len(results),
        "completed": len(completed),
        "errors": sum(result.status == "error" for result in results),
        "warnings": sum(bool(result.warnings) for result in completed),
        "pages_with_mrz_candidates": sum(
            int(result.metrics.get("mrz_candidate_count", 0)) > 0 for result in completed
        ),
        "pages_with_complete_mrz_candidates": sum(count >= 2 for count in candidate_counts),
        "mrz_candidate_line_distribution": {
            str(count): pages for count, pages in sorted(Counter(candidate_counts).items())
        },
        "pages_without_text": sum(
            "ocr_detection_no_text" in result.warnings for result in completed
        ),
        "mean_page_confidence": round(mean(confidence_values), 6)
        if confidence_values
        else None,
        "total_elapsed_ms": round(sum(result.elapsed_ms for result in results), 2),
        "mean_elapsed_ms": round(mean(result.elapsed_ms for result in results), 2)
        if results
        else 0.0,
        "reused_pages": sum(result.reused for result in results),
    }


def write_ocr_report(
    output_root: Path, results: list[OCRPageResult], config: OCRConfig
) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    summary = summarize(results)
    summary_path = output_root / config.summary_filename
    summary_path.write_text(
        json.dumps(
            {
                "run": {
                    "generated_at": datetime.now().isoformat(timespec="seconds"),
                    "ocr_config": asdict(config),
                    "runtime": {
                        "python": platform.python_version(),
                        "paddleocr": _package_version("paddleocr"),
                        "paddlepaddle": _package_version("paddlepaddle"),
                    },
                },
                "summary": summary,
                "pages": [result.as_dict(output_root) for result in results],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    report_path = output_root / config.report_filename
    report_path.write_text(_render_html(output_root, results, summary), encoding="utf-8")
    return report_path


def _render_html(
    output_root: Path, results: list[OCRPageResult], summary: dict[str, object]
) -> str:
    rows = "\n".join(_render_row(output_root, result) for result in results)
    confidence = summary["mean_page_confidence"]
    confidence_text = "N/A" if confidence is None else f"{float(confidence) * 100:.1f}%"
    mean_seconds = float(summary["mean_elapsed_ms"]) / 1000
    mrz_rate = (
        int(summary["pages_with_complete_mrz_candidates"]) / int(summary["completed"]) * 100
        if summary["completed"]
        else 0.0
    )
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>OCR Baseline Report</title>
<style>
:root {{ color-scheme: light; --ink:#17202a; --muted:#65707c; --line:#d8dde3; --ok:#147d46; --warn:#a45b00; --bad:#b42318; --paper:#fff; --bg:#f4f6f8; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--bg); color:var(--ink); font:14px/1.45 "Segoe UI", "Microsoft YaHei", sans-serif; letter-spacing:0; }}
header {{ background:#fff; border-bottom:1px solid var(--line); padding:22px 28px 18px; position:sticky; top:0; z-index:2; }}
h1 {{ font-size:22px; margin:0 0 14px; letter-spacing:0; }}
.summary {{ display:grid; grid-template-columns:repeat(6,minmax(110px,1fr)); gap:1px; background:var(--line); border:1px solid var(--line); max-width:1100px; }}
.metric {{ background:#fff; padding:10px 12px; }} .metric b {{ display:block; font-size:19px; }} .metric span {{ color:var(--muted); font-size:12px; }}
main {{ padding:20px 28px 40px; }}
.note {{ color:var(--muted); margin:0 0 14px; }}
table {{ width:100%; border-collapse:collapse; background:var(--paper); border:1px solid var(--line); }}
th,td {{ border-bottom:1px solid var(--line); padding:9px 10px; text-align:left; vertical-align:top; }}
th {{ background:#eef1f4; font-size:12px; color:#46515c; position:sticky; top:143px; z-index:1; }}
tr:last-child td {{ border-bottom:0; }}
.thumb {{ width:220px; max-height:150px; object-fit:contain; background:#f7f7f7; border:1px solid var(--line); }}
.status {{ font-weight:700; }} .ok {{ color:var(--ok); }} .warning {{ color:var(--warn); }} .error {{ color:var(--bad); }}
.warning-list {{ color:var(--warn); max-width:240px; overflow-wrap:anywhere; }}
.text {{ max-width:440px; white-space:pre-wrap; overflow-wrap:anywhere; font-family:Consolas, monospace; font-size:12px; }}
details summary {{ cursor:pointer; color:#245c8a; }}
@media (max-width:900px) {{ header {{ position:static; }} .summary {{ grid-template-columns:repeat(2,1fr); }} main {{ padding:12px; overflow-x:auto; }} th {{ position:static; }} }}
</style>
</head>
<body>
<header>
<h1>Phase 2A OCR 基线报告</h1>
<div class="summary">
<div class="metric"><b>{summary['pages']}</b><span>总页数</span></div>
<div class="metric"><b>{summary['completed']}</b><span>完成</span></div>
<div class="metric"><b>{summary['errors']}</b><span>运行错误</span></div>
<div class="metric"><b>{confidence_text}</b><span>平均置信度</span></div>
<div class="metric"><b>{mrz_rate:.1f}%</b><span>MRZ候选组(≥2行)</span></div>
<div class="metric"><b>{mean_seconds:.1f}s</b><span>平均每页</span></div>
</div>
</header>
<main>
<p class="note">模型：PP-OCRv5 mobile，CPU。蓝框表示MRZ候选，绿框为高置信度文字，橙框为较低置信度文字。候选组要求至少两行，但仍不代表MRZ字符准确或通过ICAO校验。</p>
<table>
<thead><tr><th>文件 / 页</th><th>状态</th><th>文字框</th><th>MRZ候选</th><th>平均置信度</th><th>耗时</th><th>警告</th><th>叠加图</th><th>识别文本</th></tr></thead>
<tbody>{rows}</tbody>
</table>
</main>
</body>
</html>"""


def _render_row(output_root: Path, result: OCRPageResult) -> str:
    confidence = result.metrics.get("mean_confidence")
    confidence_text = "N/A" if confidence is None else f"{float(confidence) * 100:.1f}%"
    warning_text = "<br>".join(html.escape(item) for item in result.warnings) or "-"
    recognized = "\n".join(line.text for line in result.lines)
    text_block = html.escape(recognized) if recognized else html.escape(result.error or "-")
    image_html = "-"
    if result.overlay_path and result.overlay_path.exists():
        relative = result.overlay_path.relative_to(output_root).as_posix()
        image_html = f'<a href="{quote(relative)}"><img class="thumb" src="{quote(relative)}" loading="lazy"></a>'
    return f"""<tr>
<td><b>{html.escape(result.document)}</b><br>Page {result.page_number:03d}</td>
<td class="status {html.escape(result.status)}">{html.escape(result.status.upper())}</td>
<td>{result.metrics.get('accepted_line_count', 0)}</td>
<td>{result.metrics.get('mrz_candidate_count', 0)}行</td>
<td>{confidence_text}</td>
<td>{result.elapsed_ms / 1000:.1f}s</td>
<td class="warning-list">{warning_text}</td>
<td>{image_html}</td>
<td><details><summary>查看文本</summary><div class="text">{text_block}</div></details></td>
</tr>"""


def _package_version(package: str) -> str | None:
    try:
        return version(package)
    except PackageNotFoundError:
        return None
