"""Local FastAPI server for drag-and-drop document processing."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import quote

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from config import load_config
from core.pipeline import ExtractionPipeline
from ocr.fast_mrz_runner import FastMRZRunner

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WEB_OUTPUT_ROOT = PROJECT_ROOT / "output" / "web_jobs"
STATIC_ROOT = Path(__file__).resolve().parent / "static"
SUPPORTED_SUFFIXES = {".pdf", ".jpg", ".jpeg", ".png", ".tif", ".tiff"}


@dataclass
class Job:
    job_id: str
    root: Path
    input_path: Path
    workers: int
    status: str = "queued"
    message: str = "等待开始"
    events: list[dict[str, Any]] = field(default_factory=list)
    condition: threading.Condition = field(default_factory=threading.Condition)
    finished: threading.Event = field(default_factory=threading.Event)

    def emit(self, event_type: str, payload: dict[str, Any]) -> None:
        event = {"type": event_type, "time": time.time(), **payload}
        with self.condition:
            self.events.append(event)
            self.condition.notify_all()

    def set_status(self, status: str, message: str) -> None:
        self.status = status
        self.message = message
        self.emit("status", {"status": status, "message": message})


class JobManager:
    def __init__(self) -> None:
        self.jobs: dict[str, Job] = {}
        self.lock = threading.Lock()

    def create(self, upload: UploadFile, workers: int) -> Job:
        if workers < 1 or workers > 3:
            raise ValueError("并发数只能设置为 1 到 3")
        filename = Path(upload.filename or "document").name
        suffix = Path(filename).suffix.lower()
        if suffix not in SUPPORTED_SUFFIXES:
            raise ValueError("仅支持 PDF、JPG、PNG、TIFF 文件")
        job_id = uuid.uuid4().hex[:12]
        root = WEB_OUTPUT_ROOT / job_id
        input_dir = root / "input"
        input_dir.mkdir(parents=True, exist_ok=True)
        input_path = input_dir / filename
        with input_path.open("wb") as destination:
            shutil.copyfileobj(upload.file, destination)
        job = Job(job_id, root, input_path, workers)
        with self.lock:
            self.jobs[job_id] = job
        thread = threading.Thread(target=self._run, args=(job,), name=f"passport-job-{job_id}", daemon=True)
        thread.start()
        return job

    def get(self, job_id: str) -> Job:
        with self.lock:
            job = self.jobs.get(job_id)
            if job is None:
                job = self._restore_from_disk(job_id)
                if job is not None:
                    self.jobs[job_id] = job
        if job is None:
            raise KeyError(job_id)
        return job

    @staticmethod
    def _restore_from_disk(job_id: str) -> Job | None:
        root = WEB_OUTPUT_ROOT / job_id
        input_files = list((root / "input").glob("*")) if (root / "input").exists() else []
        if not root.is_dir() or not input_files:
            return None
        job = Job(
            job_id=job_id,
            root=root,
            input_path=input_files[0],
            workers=1,
            status="done",
            message="历史任务",
        )
        job.finished.set()
        return job

    def _run(self, job: Job) -> None:
        watcher = threading.Thread(target=self._watch_artifacts, args=(job,), name=f"watch-{job.job_id}", daemon=True)
        watcher.start()
        try:
            job.set_status("running", "正在进行个人信息页提取")
            config = load_config()
            phase1_root = job.root / "phase1"
            pipeline = ExtractionPipeline(config, log_callback=lambda message: job.emit("log", {"message": message}))
            process_result = pipeline.process(job.input_path, phase1_root)
            job.emit(
                "phase1_complete",
                {
                    "pages": len(process_result.page_results),
                    "selected": process_result.successful_pages,
                },
            )
            data_pages_root = phase1_root / config.output.output_subdirectory
            if data_pages_root.exists():
                job.set_status("running", "个人信息页提取完成，正在定位 MRZ")
                runner = FastMRZRunner(config.ocr, config.mrz)
                _, report = runner.run(
                    data_pages_root,
                    job.root / "mrz",
                    resume=False,
                    workers=job.workers,
                    callback=lambda index, total, result: self._emit_mrz_page(job, index, total, result),
                )
                job.emit("report", {"url": self._url(job, report.relative_to(job.root))})
                try:
                    export_path = self._export_excel(job)
                    job.emit("export", {"url": self._url(job, export_path.relative_to(job.root))})
                except Exception as exc:
                    job.emit("export_error", {"message": str(exc)})
            job.set_status("done", "处理完成")
        except Exception as exc:
            job.set_status("error", f"{type(exc).__name__}: {exc}")
            job.emit("error", {"message": job.message})
        finally:
            job.emit("done", {"status": job.status})
            job.finished.set()
            watcher.join(timeout=2)

    @staticmethod
    def _export_excel(job: Job) -> Path:
        node = _find_node()
        artifact_module = _find_artifact_module()
        output_path = job.root / "export" / "船员名单.xlsx"
        command = [
            str(node),
            str(PROJECT_ROOT / "scripts" / "export_mrz_excel.mjs"),
            str(job.root / "mrz"),
            str(output_path),
            str(artifact_module),
        ]
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=180,
            check=False,
        )
        if completed.returncode != 0:
            detail = completed.stderr.strip() or completed.stdout.strip() or "未知错误"
            raise RuntimeError(f"Excel导出失败: {detail[-1000:]}")
        if not output_path.is_file():
            raise RuntimeError("Excel导出未生成文件")
        return output_path

    def _emit_mrz_page(self, job: Job, index: int, total: int, result: dict[str, Any]) -> None:
        files = result.get("files", {})
        urls: dict[str, str] = {}
        for key, value in files.items():
            if isinstance(value, str) and value:
                urls[key] = self._url(job, Path("mrz") / value)
        job.emit(
            "mrz_page",
            {
                "index": index,
                "total": total,
                "page": int(result.get("page_number", 0)),
                "status": result.get("status", "error"),
                "mode": result.get("mode", ""),
                "elapsed_ms": result.get("elapsed_ms", 0),
                "warnings": result.get("warnings", []),
                "parse": result.get("mrz_parse", {}),
                "metrics": result.get("metrics", {}),
                "files": urls,
            },
        )

    def _watch_artifacts(self, job: Job) -> None:
        seen: dict[str, tuple[int, int]] = {}
        while not job.finished.is_set():
            self._scan_artifacts(job, seen)
            job.finished.wait(0.35)
        self._scan_artifacts(job, seen)

    def _scan_artifacts(self, job: Job, seen: dict[str, tuple[int, int]]) -> None:
        if not job.root.exists():
            return
        for path in job.root.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}:
                continue
            if "input" in path.relative_to(job.root).parts:
                continue
            stat = path.stat()
            key = path.as_posix()
            signature = (stat.st_size, stat.st_mtime_ns)
            if seen.get(key) == signature:
                continue
            seen[key] = signature
            relative = path.relative_to(job.root)
            job.emit(
                "artifact",
                {
                    "path": relative.as_posix(),
                    "url": self._url(job, relative),
                    "page": _page_number(relative.as_posix()),
                    "stage": _stage_name(path.stem),
                },
            )

    @staticmethod
    def _url(job: Job, relative: Path) -> str:
        encoded = quote(relative.as_posix(), safe="/")
        return f"/api/jobs/{job.job_id}/artifact/{encoded}"


manager = JobManager()
app = FastAPI(title="Identity Document Data Page Extractor")
app.mount("/static", StaticFiles(directory=STATIC_ROOT), name="static")


@app.get("/", response_class=HTMLResponse)
def index() -> FileResponse:
    return FileResponse(STATIC_ROOT / "index.html")


@app.post("/api/jobs")
async def create_job(file: UploadFile = File(...), workers: int = Form(2)) -> dict[str, Any]:
    try:
        job = manager.create(file, workers)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"job_id": job.job_id, "status": job.status, "filename": job.input_path.name}


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str) -> dict[str, Any]:
    try:
        job = manager.get(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="任务不存在") from exc
    return {"job_id": job.job_id, "status": job.status, "message": job.message, "workers": job.workers}


@app.get("/api/jobs/{job_id}/events")
def job_events(job_id: str) -> StreamingResponse:
    try:
        job = manager.get(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="任务不存在") from exc

    def stream():
        cursor = 0
        while True:
            with job.condition:
                if cursor >= len(job.events) and not job.finished.is_set():
                    job.condition.wait(timeout=1)
                pending = job.events[cursor:]
                cursor = len(job.events)
            for event in pending:
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
            if job.finished.is_set() and cursor >= len(job.events):
                break

    return StreamingResponse(stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})


@app.get("/api/jobs/{job_id}/artifact/{artifact_path:path}")
def artifact(job_id: str, artifact_path: str) -> FileResponse:
    try:
        job = manager.get(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="任务不存在") from exc
    root = job.root.resolve()
    path = (root / artifact_path).resolve()
    if root not in path.parents or not path.is_file():
        raise HTTPException(status_code=404, detail="文件不存在")
    return FileResponse(path)


@app.get("/api/jobs/{job_id}/export.xlsx")
def export_excel(job_id: str) -> FileResponse:
    try:
        job = manager.get(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="任务不存在") from exc
    path = job.root / "export" / "船员名单.xlsx"
    if not path.is_file():
        raise HTTPException(status_code=409, detail="Excel尚未生成，请等待处理完成")
    return FileResponse(
        path,
        filename="船员名单.xlsx",
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


def _page_number(value: str) -> int | None:
    matches = re.findall(r"(?:page|DataPage_)(\d+)", value, flags=re.IGNORECASE)
    return int(matches[-1]) if matches else None


def _stage_name(stem: str) -> str:
    labels = {
        "01_original": "原图",
        "02_rotated": "旋转",
        "03_trimmed": "去白边",
        "04_document": "文档区域",
        "05_segment_1": "页面分割",
        "08_safe_input": "安全输入",
        "09_normalized": "标准化",
        "fast_mrz_band": "MRZ快速区域",
        "fast_mrz_ocr_overlay": "MRZ快速定位",
        "fallback_mrz_crop": "MRZ回退裁剪",
        "fallback_mrz_ocr_overlay": "MRZ回退定位",
    }
    return labels.get(stem, stem)


def _find_node() -> Path:
    configured = os.environ.get("PASSPORT_READER_NODE")
    candidates = [
        Path(configured) if configured else None,
        Path.home() / ".cache" / "codex-runtimes" / "codex-primary-runtime" / "dependencies" / "node" / "bin" / "node.exe",
        Path(shutil.which("node") or ""),
    ]
    for candidate in candidates:
        if candidate and candidate.is_file():
            return candidate
    raise RuntimeError("未找到 Node.js，无法生成 Excel")


def _find_artifact_module() -> Path:
    configured = os.environ.get("PASSPORT_READER_ARTIFACT_MODULE")
    candidates = [
        Path(configured) if configured else None,
        Path.home() / ".cache" / "codex-runtimes" / "codex-primary-runtime" / "dependencies" / "node" / "node_modules" / "@oai" / "artifact-tool" / "dist" / "artifact_tool.mjs",
    ]
    for candidate in candidates:
        if candidate and candidate.is_file():
            return candidate
    raise RuntimeError("未找到 @oai/artifact-tool，无法生成 Excel")
