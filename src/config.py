"""Centralized, immutable pipeline settings."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class LoaderConfig:
    pdf_dpi: int = 220
    max_render_dimension: int = 5000


@dataclass(frozen=True)
class OrientationConfig:
    enabled: bool = True
    minimum_rotation_score_gain: float = 0.07
    text_threshold_block_size: int = 31
    text_threshold_constant: int = 11
    lower_band_weight: float = 0.08
    horizontal_structure_weight: float = 0.22
    compactness_weight: float = 0.05
    face_weight: float = 0.35
    eye_weight: float = 0.20
    machine_readable_texture_weight: float = 0.10
    face_scale_factor: float = 1.08
    face_min_neighbors: int = 4
    eye_scale_factor: float = 1.08
    eye_min_neighbors: int = 3
    eye_pair_min_x_ratio: float = 0.12
    eye_pair_max_x_ratio: float = 0.88
    eye_pair_min_y_ratio: float = 0.18
    eye_pair_max_y_ratio: float = 0.58
    eye_pair_min_separation_ratio: float = 0.18
    eye_pair_max_separation_ratio: float = 0.62
    eye_pair_max_vertical_delta_ratio: float = 0.16
    unverified_face_weight: float = 0.45
    machine_readable_start_ratio: float = 0.56
    machine_readable_min_band_coverage: float = 0.54
    machine_readable_max_band_height_ratio: float = 0.07


@dataclass(frozen=True)
class BorderTrimConfig:
    white_threshold: int = 245
    minimum_content_ratio: float = 0.006
    safety_margin_ratio: float = 0.015
    minimum_margin_px: int = 12
    maximum_crop_ratio: float = 0.96
    projection_enabled: bool = True
    projection_fallback_trigger_extent_ratio: float = 0.98
    projection_edge_ignore_ratio: float = 0.02
    projection_minimum_ink_density: float = 0.008
    projection_smoothing_ratio: float = 0.01
    projection_minimum_area_reduction: float = 0.55
    projection_minimum_retained_ink_ratio: float = 0.995


@dataclass(frozen=True)
class DocumentDetectorConfig:
    minimum_area_ratio: float = 0.20
    maximum_area_ratio: float = 0.995
    rectangularity_threshold: float = 0.58
    closing_kernel_ratio: float = 0.025
    fallback_to_trimmed_image: bool = True
    safety_margin_ratio: float = 0.04
    edge_guard_ratio: float = 0.012
    edge_ink_threshold: int = 205
    edge_ink_density_threshold: float = 0.012
    maximum_outside_ink_density: float = 0.05
    expansion_step_ratio: float = 0.04
    expansion_context_multiplier: int = 3
    max_expansion_steps: int = 8


@dataclass(frozen=True)
class SegmenterConfig:
    enabled: bool = True
    candidate_start_ratio: float = 0.28
    candidate_end_ratio: float = 0.72
    seam_ink_density_threshold: float = 0.012
    min_segment_width_ratio: float = 0.25
    seam_min_height_coverage: float = 0.55
    seam_neighbourhood_ratio: float = 0.015
    line_seam_min_length_ratio: float = 0.68
    line_seam_max_angle_degrees: float = 5.0
    line_seam_hough_threshold: int = 85
    line_seam_minimum_score: float = 0.67
    minimum_seam_score: float = 0.72
    gradient_blur_kernel: int = 31
    gradient_minimum_pixel_delta: float = 1.0
    gradient_minimum_coverage: float = 0.90
    max_segments: int = 2
    safety_padding_ratio: float = 0.04


@dataclass(frozen=True)
class ClassifierConfig:
    adaptive_block_size: int = 31
    adaptive_constant: int = 11
    portrait_min_area_ratio: float = 0.018
    portrait_max_area_ratio: float = 0.34
    portrait_min_aspect_ratio: float = 0.45
    portrait_max_aspect_ratio: float = 1.35
    face_scale_factor: float = 1.08
    face_min_neighbors: int = 4
    eye_scale_factor: float = 1.08
    eye_min_neighbors: int = 3
    eye_pair_min_x_ratio: float = 0.12
    eye_pair_max_x_ratio: float = 0.88
    eye_pair_min_y_ratio: float = 0.18
    eye_pair_max_y_ratio: float = 0.58
    eye_pair_min_separation_ratio: float = 0.18
    eye_pair_max_separation_ratio: float = 0.62
    eye_pair_max_vertical_delta_ratio: float = 0.16
    verified_face_selection_bonus: float = 12.0
    lower_region_start_ratio: float = 0.55
    mrz_min_band_coverage: float = 0.52
    mrz_min_band_ink_density: float = 0.03
    mrz_max_band_height_ratio: float = 0.075
    field_long_line_ratio: float = 0.58
    confidence_threshold: float = 58.0
    strong_layout_fallback_enabled: bool = True
    fallback_confidence_threshold: float = 55.0
    fallback_portrait_threshold: float = 90.0
    fallback_layout_threshold: float = 65.0
    verified_face_fallback_enabled: bool = True
    verified_face_fallback_confidence_threshold: float = 50.0
    verified_face_fallback_portrait_threshold: float = 85.0
    verified_face_fallback_layout_threshold: float = 65.0
    verified_face_fallback_eye_quality: float = 0.80
    narrative_rejection_enabled: bool = True
    narrative_minimum_line_count: int = 50
    narrative_maximum_eye_quality: float = 0.20
    narrative_maximum_mrz_score: float = 10.0
    portrait_rectangle_max_score: float = 55.0
    portrait_weight: float = 0.38
    mrz_weight: float = 0.14
    layout_weight: float = 0.24
    text_weight: float = 0.19
    rotation_weight: float = 0.05


@dataclass(frozen=True)
class NormalizerConfig:
    enable_deskew: bool = True
    max_abs_deskew_degrees: float = 6.0
    hough_threshold: int = 70
    trim_after_deskew: bool = False
    minimum_abs_deskew_degrees: float = 1.0
    minimum_deskew_confidence: float = 0.65
    minimum_deskew_improvement_ratio: float = 0.20
    safe_padding_ratio: float = 0.04
    minimum_padding_px: int = 24
    edge_guard_ratio: float = 0.015
    edge_ink_threshold: int = 205
    edge_ink_density_threshold: float = 0.012


@dataclass(frozen=True)
class OutputConfig:
    output_subdirectory: str = "data_pages"
    debug_subdirectory: str = "debug"
    failed_subdirectory: str = "failed_cases"
    png_compression: int = 3


@dataclass(frozen=True)
class ValidationConfig:
    dataset_root: str = "dataset"
    report_filename: str = "validation_report.html"
    failure_subdirectory: str = "failed"
    multiple_candidate_margin: float = 5.0


@dataclass(frozen=True)
class OCRConfig:
    language: str = "ch"
    ocr_version: str = "PP-OCRv5"
    text_detection_model_name: str = "PP-OCRv5_mobile_det"
    text_recognition_model_name: str = "PP-OCRv5_mobile_rec"
    device: str = "cpu"
    enable_mkldnn: bool = False
    use_doc_orientation_classify: bool = False
    use_doc_unwarping: bool = False
    use_textline_orientation: bool = False
    text_recognition_batch_size: int = 8
    minimum_text_confidence: float = 0.50
    low_confidence_threshold: float = 0.75
    minimum_mean_confidence: float = 0.80
    mrz_minimum_length: int = 28
    mrz_maximum_length: int = 46
    mrz_minimum_allowed_ratio: float = 0.85
    mrz_minimum_content_width_ratio: float = 0.55
    results_subdirectory: str = "results"
    report_filename: str = "ocr_report.html"
    summary_filename: str = "summary.json"
    overlay_line_thickness: int = 2


@dataclass(frozen=True)
class MRZConfig:
    enabled: bool = True
    minimum_fragment_confidence: float = 0.35
    minimum_allowed_ratio: float = 0.70
    lower_content_start_ratio: float = 0.60
    row_y_tolerance_ratio: float = 0.015
    minimum_row_width_ratio: float = 0.18
    minimum_row_height_px: int = 8
    maximum_rows: int = 3
    horizontal_padding_ratio: float = 0.02
    vertical_padding_ratio: float = 0.05
    minimum_crop_height_ratio: float = 0.10
    row_upscale_factor: float = 3.0
    row_vertical_padding_ratio: float = 0.70
    row_tall_vertical_padding_ratio: float = 0.25
    row_horizontal_padding_ratio: float = 0.01
    row_minimum_allowed_ratio: float = 0.65
    row_minimum_fragment_length: int = 2


@dataclass(frozen=True)
class PipelineConfig:
    loader: LoaderConfig = field(default_factory=LoaderConfig)
    orientation: OrientationConfig = field(default_factory=OrientationConfig)
    border_trim: BorderTrimConfig = field(default_factory=BorderTrimConfig)
    document: DocumentDetectorConfig = field(default_factory=DocumentDetectorConfig)
    segmenter: SegmenterConfig = field(default_factory=SegmenterConfig)
    classifier: ClassifierConfig = field(default_factory=ClassifierConfig)
    normalizer: NormalizerConfig = field(default_factory=NormalizerConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    ocr: OCRConfig = field(default_factory=OCRConfig)
    mrz: MRZConfig = field(default_factory=MRZConfig)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"


def load_config(path: str | Path | None = None) -> PipelineConfig:
    """Load operational settings from YAML while preserving typed defaults."""
    values = asdict(PipelineConfig())
    source = Path(path) if path else DEFAULT_CONFIG_PATH
    if source.exists():
        loaded = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"Configuration must be a mapping: {source}")
        _merge_known(values, loaded, source)
    return PipelineConfig(
        loader=LoaderConfig(**values["loader"]),
        orientation=OrientationConfig(**values["orientation"]),
        border_trim=BorderTrimConfig(**values["border_trim"]),
        document=DocumentDetectorConfig(**values["document"]),
        segmenter=SegmenterConfig(**values["segmenter"]),
        classifier=ClassifierConfig(**values["classifier"]),
        normalizer=NormalizerConfig(**values["normalizer"]),
        output=OutputConfig(**values["output"]),
        validation=ValidationConfig(**values["validation"]),
        ocr=OCRConfig(**values["ocr"]),
        mrz=MRZConfig(**values["mrz"]),
    )


def _merge_known(target: dict[str, Any], updates: dict[str, Any], source: Path) -> None:
    for key, value in updates.items():
        if key not in target:
            raise ValueError(f"Unknown configuration section '{key}' in {source}")
        if not isinstance(value, dict):
            raise ValueError(f"Configuration section '{key}' must be a mapping in {source}")
        unknown = set(value) - set(target[key])
        if unknown:
            names = ", ".join(sorted(unknown))
            raise ValueError(f"Unknown configuration value(s) in '{key}': {names}")
        target[key].update(value)


DEFAULT_CONFIG = load_config()
