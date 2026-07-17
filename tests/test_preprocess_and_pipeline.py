from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import cv2
import fitz
import numpy as np

from config import DEFAULT_CONFIG
from core.pipeline import ExtractionPipeline
from detector.data_page_classifier import DataPageClassifier
from detector.document_detector import DocumentAnalyzer
from detector.page_segmenter import PageSegmenter
from face_geometry import eye_pair_quality
from loader.factory import load_pages
from models import FeatureScores, Rect
from normalize.normalizer import Normalizer
from preprocess.border_trim import WhiteBorderRemover
from preprocess.orientation import OrientationCorrector


def test_white_border_remover_keeps_content_with_safety_margin() -> None:
    image = np.full((500, 700, 3), 255, dtype=np.uint8)
    cv2.rectangle(image, (120, 80), (580, 420), (180, 180, 180), -1)
    result = WhiteBorderRemover(DEFAULT_CONFIG.border_trim).trim(image)

    assert result.rect.x < 120
    assert result.rect.y < 80
    assert result.rect.x + result.rect.w > 580
    assert result.rect.y + result.rect.h > 420
    assert result.image.shape[0] < image.shape[0]


def test_white_border_remover_ignores_scanner_edge_and_sparse_noise() -> None:
    image = np.full((1000, 800, 3), 255, dtype=np.uint8)
    cv2.rectangle(image, (260, 0), (795, 520), (225, 225, 225), -1)
    cv2.line(image, (798, 0), (798, 999), (15, 15, 15), 3)
    for y in range(600, 950, 70):
        cv2.circle(image, (400 + y % 90, y), 1, (80, 80, 80), -1)

    result = WhiteBorderRemover(DEFAULT_CONFIG.border_trim).trim(image)

    assert result.rect.x < 260
    assert result.rect.x + result.rect.w > 780
    assert 520 < result.rect.h < 700


def test_orientation_keeps_original_when_rotation_gain_is_too_small() -> None:
    corrector = OrientationCorrector(DEFAULT_CONFIG.orientation)
    scores = iter((0.70, 0.10, 0.73, 0.20))
    corrector._layout_score = lambda _: next(scores)  # type: ignore[method-assign]

    image = np.full((120, 80, 3), 255, dtype=np.uint8)
    result = corrector.correct(image)

    assert result.angle == 0
    assert np.array_equal(result.image, image)


def test_orientation_eye_pair_rejects_inverted_face_geometry() -> None:
    corrector = OrientationCorrector(DEFAULT_CONFIG.orientation)
    upright = np.array([[20, 26, 18, 12], [61, 27, 18, 12]], dtype=np.int32)
    inverted = np.array([[20, 62, 18, 12], [61, 64, 18, 12]], dtype=np.int32)

    assert eye_pair_quality(upright, 100, 100, DEFAULT_CONFIG.orientation) > 0.8
    assert eye_pair_quality(inverted, 100, 100, DEFAULT_CONFIG.orientation) == 0.0


def test_classifier_accepts_strong_portrait_layout_fallback() -> None:
    classifier = DataPageClassifier(DEFAULT_CONFIG.classifier)
    valid = FeatureScores(100.0, 0.0, 67.0, 1.0, 57.5)
    invalid = FeatureScores(55.0, 0.0, 60.0, 1.0, 57.5)

    assert classifier.confidence_decision(valid) == (
        True,
        "strong_portrait_layout_fallback",
    )
    assert classifier.confidence_decision(invalid) == (False, "low_confidence")


def test_classifier_selection_prefers_verified_face_geometry() -> None:
    classifier = DataPageClassifier(DEFAULT_CONFIG.classifier)
    false_positive = FeatureScores(100.0, 80.0, 100.0, 60.0, 88.0)
    false_positive.details["portrait_eye_quality"] = 0.0
    verified = FeatureScores(100.0, 75.0, 100.0, 55.0, 84.0)
    verified.details["portrait_eye_quality"] = 0.9

    assert classifier.selection_score(verified) > classifier.selection_score(false_positive)


def test_classifier_accepts_verified_face_without_mrz_texture() -> None:
    classifier = DataPageClassifier(DEFAULT_CONFIG.classifier)
    valid = FeatureScores(90.0, 0.0, 68.0, 18.0, 54.0)
    valid.details["portrait_eye_quality"] = 0.9
    unverified = FeatureScores(90.0, 0.0, 68.0, 18.0, 54.0)
    unverified.details["portrait_eye_quality"] = 0.0

    assert classifier.confidence_decision(valid) == (
        True,
        "verified_face_layout_fallback",
    )
    assert classifier.confidence_decision(unverified) == (False, "low_confidence")


def test_classifier_rejects_narrative_page_false_positive() -> None:
    classifier = DataPageClassifier(DEFAULT_CONFIG.classifier)
    narrative = FeatureScores(97.0, 0.0, 78.0, 41.0, 66.0)
    narrative.details.update({"portrait_eye_quality": 0.0, "line_count": 84.0})

    assert classifier.confidence_decision(narrative) == (
        False,
        "narrative_page_rejection",
    )


def test_segmenter_splits_a_scan_with_a_clear_central_seam() -> None:
    image = np.full((480, 1000, 3), 245, dtype=np.uint8)
    for x_offset in (55, 555):
        cv2.rectangle(image, (x_offset, 45), (x_offset + 355, 425), (220, 220, 220), 2)
        for y in range(100, 350, 36):
            cv2.line(image, (x_offset + 40, y), (x_offset + 280, y), (50, 50, 50), 3)
    image[:, 480:520] = 255
    segments = PageSegmenter(DEFAULT_CONFIG.segmenter).segment(image)

    assert len(segments) == 2
    assert abs(segments[0].image.shape[1] - segments[1].image.shape[1]) < 120
    assert segments[0].safe_image is not None
    assert segments[0].safe_image.shape[1] > segments[0].image.shape[1]
    assert segments[1].safe_image is not None
    assert segments[1].safe_image.shape[1] > segments[1].image.shape[1]


def test_segmenter_splits_a_stacked_scan_with_a_horizontal_seam() -> None:
    image = np.full((1000, 480, 3), 245, dtype=np.uint8)
    for y_offset in (55, 555):
        cv2.rectangle(image, (45, y_offset), (435, y_offset + 355), (220, 220, 220), 2)
        for x in range(90, 370, 36):
            cv2.line(image, (x, y_offset + 40), (x, y_offset + 280), (50, 50, 50), 3)
    image[480:520, :] = 255
    segments = PageSegmenter(DEFAULT_CONFIG.segmenter).segment(image)

    assert len(segments) == 2
    assert abs(segments[0].image.shape[0] - segments[1].image.shape[0]) < 120


def test_segmenter_detects_a_weak_but_full_width_page_boundary() -> None:
    image = np.full((1000, 700, 3), 225, dtype=np.uint8)
    image[500:, :] = (110, 145, 140)
    for y in range(80, 430, 58):
        cv2.line(image, (80, y), (560, y), (70, 70, 70), 3)
    for y in range(580, 920, 58):
        cv2.line(image, (100, y), (600, y), (70, 70, 70), 3)
    segments = PageSegmenter(DEFAULT_CONFIG.segmenter).segment(image)

    assert len(segments) == 2
    assert 430 < segments[0].image.shape[0] < 570


def test_classifier_returns_all_visual_scores_without_reading_text() -> None:
    image = _synthetic_data_page()
    scores, overlay = DataPageClassifier(DEFAULT_CONFIG.classifier).classify(image)

    assert 0.0 <= scores.portrait <= 100.0
    assert 0.0 <= scores.mrz_texture <= 100.0
    assert 0.0 <= scores.layout <= 100.0
    assert 0.0 <= scores.text_density <= 100.0
    assert 0.0 <= scores.final <= 100.0
    assert overlay.shape == image.shape


def test_pdf_loader_renders_pages_to_numpy(tmp_path: Path) -> None:
    pdf_path = tmp_path / "input.pdf"
    document = fitz.open()
    page = document.new_page(width=400, height=250)
    page.draw_rect(fitz.Rect(30, 20, 370, 230), color=(0, 0, 0), fill=(0.9, 0.9, 0.9))
    document.save(pdf_path)
    document.close()

    pages = load_pages(pdf_path, DEFAULT_CONFIG.loader)

    assert len(pages) == 1
    assert isinstance(pages[0].image, np.ndarray)
    assert pages[0].image.size > 0


def test_low_confidence_page_is_preserved_for_review(tmp_path: Path) -> None:
    input_path = tmp_path / "blank.png"
    cv2.imencode(".png", np.full((400, 700, 3), 255, dtype=np.uint8))[1].tofile(str(input_path))

    result = ExtractionPipeline().process(input_path, tmp_path / "output")

    page = result.page_results[0]
    assert page.status == "low_confidence"
    assert page.output_path is None
    assert (page.debug_dir / "log.json").exists()
    assert list((tmp_path / "output" / "failed_cases").iterdir())


def test_normalizer_corrects_confident_skew_and_adds_padding_without_cropping() -> None:
    image = np.full((600, 900, 3), 255, dtype=np.uint8)
    for y in range(100, 520, 45):
        cv2.line(image, (80, y), (820, y), (20, 20, 20), 3)
    matrix = cv2.getRotationMatrix2D((450, 300), 4.0, 1.0)
    skewed = cv2.warpAffine(image, matrix, (900, 600), borderValue=(255, 255, 255))

    result = Normalizer(DEFAULT_CONFIG.normalizer, DEFAULT_CONFIG.border_trim).normalize(skewed)

    assert abs(result.estimated_skew_degrees) >= 3.5
    assert abs(result.residual_skew_degrees) < 0.8
    assert result.deskew_degrees != 0.0
    assert result.padding_px >= DEFAULT_CONFIG.normalizer.minimum_padding_px
    assert result.image.shape[0] > skewed.shape[0]
    assert result.image.shape[1] > skewed.shape[1]


def test_normalizer_skips_small_angle_and_falls_back_from_edge_risk() -> None:
    config = replace(DEFAULT_CONFIG.normalizer, enable_deskew=False)
    risky = np.full((300, 420, 3), 255, dtype=np.uint8)
    cv2.rectangle(risky, (0, 70), (180, 240), (20, 20, 20), -1)
    parent = np.full((340, 500, 3), 255, dtype=np.uint8)
    cv2.rectangle(parent, (50, 90), (230, 260), (20, 20, 20), -1)

    result = Normalizer(config, DEFAULT_CONFIG.border_trim).normalize(risky, [("parent_roi", parent)])

    assert result.deskew_degrees == 0.0
    assert result.fallback_level == "parent_roi"
    assert result.ocr_safe


def test_document_roi_expands_when_dark_content_touches_a_crop_edge() -> None:
    image = np.full((700, 900, 3), 255, dtype=np.uint8)
    cv2.rectangle(image, (180, 100), (720, 560), (225, 225, 225), -1)
    cv2.putText(image, "MRZ-LINE-ONE", (230, 430), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (20, 20, 20), 3)
    cv2.putText(image, "MRZ-LINE-TWO", (230, 485), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (20, 20, 20), 3)
    analyzer = DocumentAnalyzer(DEFAULT_CONFIG.document)

    x0, y0, x1, y1 = analyzer._expand_while_content_touches_edge(image, 160, 80, 740, 432)

    assert (x0, y0, x1) == (160, 80, 740)
    assert y1 > 500


def test_document_roi_detects_significant_content_outside_candidate() -> None:
    image = np.full((700, 900, 3), 255, dtype=np.uint8)
    cv2.rectangle(image, (150, 80), (750, 520), (225, 225, 225), -1)
    cv2.rectangle(image, (150, 535), (750, 665), (185, 185, 185), -1)
    for y in range(545, 650, 30):
        cv2.putText(image, "OUTSIDE CONTENT", (170, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (20, 20, 20), 3)
    analyzer = DocumentAnalyzer(DEFAULT_CONFIG.document)

    density = analyzer._outside_ink_density(image, Rect(130, 60, 640, 480))

    assert density > DEFAULT_CONFIG.document.maximum_outside_ink_density


def _synthetic_data_page() -> np.ndarray:
    image = np.full((720, 1080, 3), 245, dtype=np.uint8)
    cv2.rectangle(image, (55, 45), (1025, 675), (200, 200, 200), 2)
    portrait = np.random.default_rng(42).normal(120, 38, size=(230, 175)).clip(0, 255).astype(np.uint8)
    image[110:340, 95:270] = cv2.cvtColor(portrait, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(image, (95, 110), (270, 340), (25, 25, 25), 2)
    for index, y in enumerate(range(115, 430, 44), start=1):
        cv2.putText(
            image,
            f"FIELD {index}  VALUE", (350, y), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (35, 35, 35), 2
        )
    for y in (555, 590):
        for x in range(95, 970, 18):
            cv2.rectangle(image, (x, y), (x + 9, y + 18), (20, 20, 20), -1)
    return image
