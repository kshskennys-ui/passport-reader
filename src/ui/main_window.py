"""PySide6 UI for launching the pipeline and inspecting every saved stage."""

from __future__ import annotations

import json
from pathlib import Path

from PySide6.QtCore import QObject, Qt, QThread, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QCheckBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
    QStatusBar,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from config import PipelineConfig, load_config
from core.pipeline import ExtractionPipeline
from debug.logger import DebugRun
from models import PageResult, ProcessResult
from validation.evaluator import FAILURE_REASONS, ValidationService
from validation.ground_truth import GroundTruthStore
from validation.saved_run import load_saved_process


class ImagePreview(QLabel):
    def __init__(self, placeholder: str) -> None:
        super().__init__(placeholder)
        self._path: Path | None = None
        self._pixmap: QPixmap | None = None
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(320, 250)
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setStyleSheet("background: #f6f7f8; color: #5d6875;")

    def set_image(self, path: Path | None) -> None:
        self._path = path
        self._pixmap = QPixmap(str(path)) if path and path.exists() else None
        if self._pixmap is None or self._pixmap.isNull():
            self.setText("无可显示图像")
            self.setPixmap(QPixmap())
            return
        self.setText("")
        self._render()

    def resizeEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        super().resizeEvent(event)
        self._render()

    def _render(self) -> None:
        if self._pixmap is None or self._pixmap.isNull():
            return
        self.setPixmap(
            self._pixmap.scaled(
                self.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )


class PipelineWorker(QObject):
    log = Signal(str)
    completed = Signal(object)
    failed = Signal(str)

    def __init__(self, input_path: Path, output_dir: Path, config: PipelineConfig) -> None:
        super().__init__()
        self.input_path = input_path
        self.output_dir = output_dir
        self.config = config

    def run(self) -> None:
        try:
            pipeline = ExtractionPipeline(config=self.config, log_callback=self.log.emit)
            self.completed.emit(pipeline.process(self.input_path, self.output_dir))
        except Exception as error:
            self.failed.emit(f"{type(error).__name__}: {error}")


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Identity Document Data Page Extractor")
        self.resize(1440, 920)
        self._thread: QThread | None = None
        self._worker: PipelineWorker | None = None
        self._results: list[PageResult] = []
        self._current_result: ProcessResult | None = None
        self._active_source_files: list[str] = []
        self._current_config = load_config()
        self._store = GroundTruthStore(self._project_root / self._current_config.validation.dataset_root)
        self._validation_service = ValidationService(self._store, self._current_config.validation)
        self._build_ui()

    @property
    def _project_root(self) -> Path:
        return Path(__file__).resolve().parents[2]

    def _build_ui(self) -> None:
        root = QWidget()
        layout = QVBoxLayout(root)
        layout.setContentsMargins(14, 14, 14, 10)
        layout.setSpacing(10)
        layout.addWidget(self._source_controls())

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._result_panel())
        splitter.addWidget(self._inspection_panel())
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 7)
        layout.addWidget(splitter, 1)

        log_box = QGroupBox("实时日志")
        log_layout = QVBoxLayout(log_box)
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(1000)
        self.log_view.setMinimumHeight(150)
        log_layout.addWidget(self.log_view)
        layout.addWidget(log_box)
        self.setCentralWidget(root)
        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage("就绪：选择 PDF、JPG、PNG 或 TIFF 开始处理")

    def _source_controls(self) -> QWidget:
        group = QGroupBox("输入与输出")
        form = QFormLayout(group)
        self.input_edit = QLineEdit()
        self.input_edit.setReadOnly(True)
        choose_input = QPushButton("选择文件")
        choose_input.clicked.connect(self._choose_input)
        input_row = QHBoxLayout()
        input_row.addWidget(self.input_edit, 1)
        input_row.addWidget(choose_input)

        self.output_edit = QLineEdit(str((Path.cwd() / "output").resolve()))
        choose_output = QPushButton("选择输出目录")
        choose_output.clicked.connect(self._choose_output)
        output_row = QHBoxLayout()
        output_row.addWidget(self.output_edit, 1)
        output_row.addWidget(choose_output)

        self.run_button = QPushButton("开始处理")
        self.run_button.clicked.connect(self._start_processing)
        self.load_results_button = QPushButton("加载已有结果")
        self.load_results_button.clicked.connect(self._load_existing_results)
        action_row = QHBoxLayout()
        action_row.addWidget(self.run_button)
        action_row.addWidget(self.load_results_button)
        form.addRow("输入文件", self._layout_widget(input_row))
        form.addRow("输出目录", self._layout_widget(output_row))
        form.addRow("", self._layout_widget(action_row))
        return group

    def _result_panel(self) -> QWidget:
        group = QGroupBox("页面结果")
        layout = QVBoxLayout(group)
        self.result_table = QTableWidget(0, 4)
        self.result_table.setHorizontalHeaderLabels(["来源页", "选中分段", "得分", "状态"])
        self.result_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.result_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.result_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.result_table.itemSelectionChanged.connect(self._show_selected_result)
        self.result_table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.result_table)
        layout.addWidget(self._review_controls())
        return group

    def _review_controls(self) -> QWidget:
        group = QGroupBox("人工确认结果 (Review)")
        layout = QVBoxLayout(group)
        self.review_status = QLabel("选择一页后进行确认")
        self.failure_reason = QComboBox()
        self.failure_reason.addItems(FAILURE_REASONS)
        self.failure_reason.setEnabled(False)
        self.page_complete = QCheckBox("页面完整")
        self.portrait_complete = QCheckBox("头像完整")
        self.mrz_complete = QCheckBox("MRZ完整")
        self.ocr_ready = QCheckBox("可以OCR")
        self.quality_checks = (
            self.page_complete,
            self.portrait_complete,
            self.mrz_complete,
            self.ocr_ready,
        )
        quality_layout = QGridLayout()
        for index, checkbox in enumerate(self.quality_checks):
            checkbox.setEnabled(False)
            quality_layout.addWidget(checkbox, index // 2, index % 2)
        self.review_correct_button = QPushButton("正确")
        self.review_incorrect_button = QPushButton("错误")
        self.review_correct_button.clicked.connect(lambda: self._record_review("correct"))
        self.review_incorrect_button.clicked.connect(lambda: self._record_review("incorrect"))
        self.review_correct_button.setEnabled(False)
        self.review_incorrect_button.setEnabled(False)
        buttons = QHBoxLayout()
        buttons.addWidget(self.review_correct_button)
        buttons.addWidget(self.review_incorrect_button)
        layout.addWidget(self.review_status)
        layout.addLayout(quality_layout)
        layout.addWidget(QLabel("错误原因"))
        layout.addWidget(self.failure_reason)
        layout.addLayout(buttons)
        return group

    def _inspection_panel(self) -> QWidget:
        tabs = QTabWidget()
        tabs.addTab(self._preview_tab(), "输入 / 输出")
        tabs.addTab(self._debug_tab(), "Debug 流水线")
        return tabs

    def _preview_tab(self) -> QWidget:
        tab = QWidget()
        layout = QGridLayout(tab)
        layout.addWidget(QLabel("原图"), 0, 0)
        layout.addWidget(QLabel("选中并标准化后的 Data Page"), 0, 1)
        self.original_preview = ImagePreview("原图将在这里显示")
        self.output_preview = ImagePreview("处理结果将在这里显示")
        layout.addWidget(self.original_preview, 1, 0)
        layout.addWidget(self.output_preview, 1, 1)
        return tab

    def _debug_tab(self) -> QWidget:
        tab = QWidget()
        layout = QHBoxLayout(tab)
        self.stage_list = QListWidget()
        self.stage_list.setMinimumWidth(190)
        self.stage_list.currentItemChanged.connect(self._show_selected_stage)
        layout.addWidget(self.stage_list, 2)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        self.debug_preview = ImagePreview("选择左侧阶段查看中间图像")
        right_layout.addWidget(self.debug_preview, 3)
        right_layout.addWidget(QLabel("ROI、耗时与评分"))
        self.metrics_view = QPlainTextEdit()
        self.metrics_view.setReadOnly(True)
        self.metrics_view.setMinimumHeight(190)
        right_layout.addWidget(self.metrics_view, 2)
        layout.addWidget(right, 7)
        return tab

    @staticmethod
    def _layout_widget(layout: QHBoxLayout) -> QWidget:
        widget = QWidget()
        widget.setLayout(layout)
        return widget

    def _choose_input(self) -> None:
        selected, _ = QFileDialog.getOpenFileName(
            self,
            "选择扫描件",
            str(Path.home()),
            "Supported files (*.pdf *.jpg *.jpeg *.png *.tif *.tiff)",
        )
        if selected:
            self.input_edit.setText(selected)

    def _choose_output(self) -> None:
        selected = QFileDialog.getExistingDirectory(self, "选择输出目录", self.output_edit.text())
        if selected:
            self.output_edit.setText(selected)

    def _start_processing(self) -> None:
        if self._thread is not None:
            return
        input_text = self.input_edit.text().strip()
        output_text = self.output_edit.text().strip()
        if not input_text or not Path(input_text).is_file():
            QMessageBox.warning(self, "缺少输入", "请选择存在的 PDF、JPG、PNG 或 TIFF 文件。")
            return
        if not output_text:
            QMessageBox.warning(self, "缺少输出", "请选择输出目录。")
            return

        self.result_table.setRowCount(0)
        self.stage_list.clear()
        self.metrics_view.clear()
        self.original_preview.set_image(None)
        self.output_preview.set_image(None)
        self.debug_preview.set_image(None)
        self.log_view.clear()
        self.run_button.setEnabled(False)
        self.load_results_button.setEnabled(False)
        self.statusBar().showMessage("正在处理，实时日志与中间结果会持续写入输出目录")

        self._current_config = load_config()
        self._store = GroundTruthStore(self._project_root / self._current_config.validation.dataset_root)
        self._validation_service = ValidationService(self._store, self._current_config.validation)
        self._thread = QThread(self)
        self._worker = PipelineWorker(Path(input_text), Path(output_text), self._current_config)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.log.connect(self._append_log)
        self._worker.completed.connect(self._processing_complete)
        self._worker.failed.connect(self._processing_failed)
        self._worker.completed.connect(self._thread.quit)
        self._worker.failed.connect(self._thread.quit)
        self._thread.finished.connect(self._thread_finished)
        self._thread.start()

    def _load_existing_results(self) -> None:
        input_text = self.input_edit.text().strip()
        output_text = self.output_edit.text().strip()
        if not input_text or not Path(input_text).is_file():
            QMessageBox.warning(self, "缺少输入", "请选择已有结果对应的原始 PDF 或图片。")
            return
        if not output_text or not Path(output_text).is_dir():
            QMessageBox.warning(self, "缺少输出", "请选择已有结果的输出目录。")
            return
        try:
            self._current_config = load_config()
            self._store = GroundTruthStore(
                self._project_root / self._current_config.validation.dataset_root
            )
            self._validation_service = ValidationService(
                self._store,
                self._current_config.validation,
            )
            result = load_saved_process(input_text, output_text, self._current_config)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            QMessageBox.critical(self, "加载失败", f"{type(error).__name__}: {error}")
            return
        self.log_view.clear()
        self._active_source_files = self._saved_source_files(result.output_dir)
        self._display_process_result(result)
        self._append_log(f"Loaded saved result: {result.output_dir}")

    def _append_log(self, message: str) -> None:
        self.log_view.appendPlainText(message)
        self.log_view.verticalScrollBar().setValue(self.log_view.verticalScrollBar().maximum())

    def _processing_complete(self, result: ProcessResult) -> None:
        self._active_source_files = [result.input_path.name]
        self._display_process_result(result)
        self._write_validation_report()

    def _display_process_result(self, result: ProcessResult) -> None:
        self._current_result = result
        self._results = result.page_results
        self.result_table.setRowCount(len(self._results))
        for row, item in enumerate(self._results):
            values = [
                str(item.source_page),
                "-" if item.selected_segment is None else str(item.selected_segment),
                f"{item.score:.1f}",
                self._status_text(item.status),
            ]
            for column, value in enumerate(values):
                self.result_table.setItem(row, column, QTableWidgetItem(value))
        self.result_table.resizeColumnsToContents()
        selected = result.successful_pages
        self.statusBar().showMessage(f"完成：{selected}/{len(self._results)} 页输出 Data Page")
        self._append_log(f"Completed: {selected}/{len(self._results)} page(s) selected")
        if self._results:
            self.result_table.selectRow(0)

    def _processing_failed(self, message: str) -> None:
        self.statusBar().showMessage("处理无法启动")
        self._append_log(f"Fatal: {message}")
        QMessageBox.critical(self, "处理失败", message)

    def _thread_finished(self) -> None:
        if self._worker is not None:
            self._worker.deleteLater()
        if self._thread is not None:
            self._thread.deleteLater()
        self._worker = None
        self._thread = None
        self.run_button.setEnabled(True)
        self.load_results_button.setEnabled(True)

    def _show_selected_result(self) -> None:
        selected = self.result_table.selectedItems()
        if not selected:
            return
        row = selected[0].row()
        if not 0 <= row < len(self._results):
            return
        result = self._results[row]
        self.original_preview.set_image(result.debug_dir / "01_original.png")
        self.output_preview.set_image(result.output_path or result.debug_dir / "07_selected.png")
        self._load_debug(result)
        self._update_review_controls(result)

    def _load_debug(self, result: PageResult) -> None:
        self.stage_list.clear()
        log_path = result.debug_dir / "log.json"
        try:
            payload = json.loads(log_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            self.metrics_view.setPlainText(result.message)
            return
        for stage in payload.get("stages", []):
            item = QListWidgetItem(
                f"{stage['name']}  ({stage.get('elapsed_ms', 0):.1f} ms)"
            )
            item.setData(Qt.ItemDataRole.UserRole, str(result.debug_dir / stage["file"]))
            self.stage_list.addItem(item)
        display = {
            "status": payload.get("status"),
            "message": payload.get("message"),
            "metrics": payload.get("metrics", result.metrics),
            "selected_result": {
                "score": result.score,
                "status": result.status,
                "message": result.message,
            },
        }
        self.metrics_view.setPlainText(json.dumps(display, ensure_ascii=False, indent=2))
        if self.stage_list.count():
            self.stage_list.setCurrentRow(self.stage_list.count() - 1)

    def _update_review_controls(self, result: PageResult) -> None:
        if self._current_result is None:
            return
        truth = self._store.get(self._current_result.input_path.name, result.source_page)
        enabled = truth is not None
        self.review_correct_button.setEnabled(enabled)
        self.review_incorrect_button.setEnabled(enabled)
        self.failure_reason.setEnabled(enabled)
        for checkbox in self.quality_checks:
            checkbox.setEnabled(enabled and truth is not None and truth.contains_data_page)
        if truth is None:
            self.review_status.setText("当前输入未配置 Ground Truth，无法写入 Review")
            return
        review = self._store.latest_review(truth)
        if review is None:
            self.review_status.setText("未确认")
            self.failure_reason.setCurrentText("unknown")
            for checkbox in self.quality_checks:
                checkbox.setChecked(False)
            return
        outcome = review.get("review")
        reason = review.get("failure_reason") or "unknown"
        self.failure_reason.setCurrentText(str(reason))
        quality = review.get("quality") or {}
        self.page_complete.setChecked(bool(quality.get("page_complete", False)))
        self.portrait_complete.setChecked(bool(quality.get("portrait_complete", False)))
        self.mrz_complete.setChecked(bool(quality.get("mrz_complete", False)))
        self.ocr_ready.setChecked(bool(quality.get("ocr_ready", False)))
        self.review_status.setText("已确认正确" if outcome == "correct" else f"已确认错误：{reason}")

    def _record_review(self, outcome: str) -> None:
        selected = self.result_table.selectedItems()
        if not selected or self._current_result is None:
            return
        result = self._results[selected[0].row()]
        truth = self._store.get(self._current_result.input_path.name, result.source_page)
        if truth is None:
            QMessageBox.warning(self, "缺少标签", "当前页没有对应的 Ground Truth 标签。")
            return
        orientation = result.metrics.get("orientation", {}).get("angle")
        prediction = {
            "status": result.status,
            "segment": None if result.selected_segment is None else f"segment_{result.selected_segment}",
            "score": round(result.score, 2),
            "rotation": orientation,
            "output_path": str(result.output_path) if result.output_path else None,
            "debug_dir": str(result.debug_dir),
        }
        reason = self.failure_reason.currentText() if outcome == "incorrect" else None
        quality = None
        if truth.contains_data_page:
            quality = {
                "page_complete": self.page_complete.isChecked(),
                "portrait_complete": self.portrait_complete.isChecked(),
                "mrz_complete": self.mrz_complete.isChecked(),
                "ocr_ready": self.ocr_ready.isChecked(),
            }
        path = self._store.record_review(truth, prediction, outcome, reason, quality)
        self._append_log(f"Review saved: {path}")
        self._update_review_controls(result)
        self._write_validation_report()

    def _write_validation_report(self) -> None:
        if self._current_result is None:
            return
        try:
            source_files = self._active_source_files or [self._current_result.input_path.name]
            evaluations = self._validation_service.evaluate_output(
                self._current_result.output_dir, source_files
            )
            report = self._validation_service.write_report(self._current_result.output_dir, evaluations)
            self._append_log(f"Validation report: {report}")
        except Exception as error:
            self._append_log(f"Validation report error: {type(error).__name__}: {error}")

    def _saved_source_files(self, output_dir: Path) -> list[str]:
        debug_root = output_dir / self._current_config.output.debug_subdirectory
        source_files = {
            label.source_file
            for label in self._store.labels
            if (debug_root / DebugRun._safe_name(Path(label.source_file).stem)).is_dir()
        }
        return sorted(source_files)

    def _show_selected_stage(self, current: QListWidgetItem | None, previous: QListWidgetItem | None) -> None:
        del previous
        if current is None:
            return
        self.debug_preview.set_image(Path(current.data(Qt.ItemDataRole.UserRole)))

    @staticmethod
    def _status_text(status: str) -> str:
        return {
            "selected": "已输出",
            "low_confidence": "低置信度，已归档",
            "error": "处理错误，已归档",
        }.get(status, status)
