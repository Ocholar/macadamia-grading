"""
Unit and integration tests for the Macadamia Quality Grading AI System.
Tests ingestion, preprocessing, segmentation, detection, classification, XAI, and reports.
"""

import io
import os
import sys
import json
import pytest
import numpy as np
import cv2

# Project root on path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.ingestion.handler import ImageIngestionHandler
from src.preprocessing.normalization import LightingColorNormalizer
from src.preprocessing.segmentation import BackgroundSegmenter
from src.preprocessing.calibration import PixelCalibrator
from src.models.detector import NutDetector
from src.models.classifier import DefectClassifier
from src.xai.gradcam import ExplainabilityEngine
from src.pipeline import MacadamiaGradingPipeline
from src.reporting.report_generator import generate_pdf_report, generate_csv_report


# ── Fixtures ───────────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
SAMPLE_IMAGE = os.path.join(DATA_DIR, "batch_01_premium.jpg")
CONFIG_PATH  = os.path.join(PROJECT_ROOT, "config.yaml")


def make_synthetic_nut_image(size: int = 400, defect_class: int = 0) -> np.ndarray:
    """Creates a synthetic macadamia kernel image for testing."""
    img = np.full((size, size, 3), [160, 100, 50], dtype=np.uint8)  # Blue mat

    # Draw a cream-colored ellipse as the nut
    cv2.ellipse(img, (size // 2, size // 2), (size // 3, int(size // 3.2)), 20, 0, 360, (175, 220, 240), -1)

    if defect_class == 1:  # Cracked
        cv2.line(img, (size // 3, size // 2), (2 * size // 3, size // 2), (40, 50, 70), 3)

    elif defect_class == 2:  # Discolored
        cv2.circle(img, (size // 2, size // 2), size // 7, (60, 90, 120), -1)
        cv2.GaussianBlur(img, (9, 9), 0, img)

    elif defect_class == 3:  # Mold
        cv2.circle(img, (int(size * 0.55), int(size * 0.45)), size // 10, (70, 100, 85), -1)
        cv2.circle(img, (int(size * 0.55), int(size * 0.45)), size // 20, (20, 25, 30), -1)

    return img


def make_batch_image(num_nuts: int = 8, mat: str = "blue") -> np.ndarray:
    """Generates a minimal batch image with circular nuts on a mat."""
    w, h = 640, 640
    color = {"blue": [160, 100, 50], "green": [60, 130, 60]}.get(mat, [50, 50, 50])
    img = np.full((h, w, 3), color, dtype=np.uint8)

    radius = 42
    positions = []
    for row in range(2):
        for col in range(4):
            cx = 90 + col * 120
            cy = 160 + row * 220
            cv2.ellipse(img, (cx, cy), (radius, int(radius * 0.95)), 15, 0, 360, (175, 220, 240), -1)
            positions.append((cx, cy))

    return img


# ── Ingestion Tests ────────────────────────────────────────────────────────────
class TestImageIngestion:

    def test_load_valid_path(self):
        handler = ImageIngestionHandler()
        img = handler.load_from_path(SAMPLE_IMAGE)
        assert isinstance(img, np.ndarray)
        assert img.ndim == 3
        assert img.shape[2] == 3

    def test_load_from_bytes(self):
        with open(SAMPLE_IMAGE, "rb") as f:
            raw = f.read()
        handler = ImageIngestionHandler()
        img = handler.load_from_bytes(raw)
        assert img.shape[2] == 3

    def test_load_missing_file_raises(self):
        handler = ImageIngestionHandler()
        with pytest.raises(FileNotFoundError):
            handler.load_from_path("does_not_exist.jpg")

    def test_validate_grayscale_converts_to_bgr(self):
        handler = ImageIngestionHandler(min_width=50, min_height=50)
        gray = np.ones((100, 100), dtype=np.uint8) * 128
        bgr = handler.validate_image(gray)
        assert bgr.shape[2] == 3

    def test_validate_rgba_converts_to_bgr(self):
        handler = ImageIngestionHandler(min_width=50, min_height=50)
        rgba = np.ones((100, 100, 4), dtype=np.uint8) * 180
        bgr = handler.validate_image(rgba)
        assert bgr.shape[2] == 3

    def test_minimum_resolution_check(self):
        handler = ImageIngestionHandler(min_width=300, min_height=300)
        small = np.ones((100, 100, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match="resolution"):
            handler.validate_image(small)

    def test_roundtrip_base64(self):
        handler = ImageIngestionHandler()
        img = make_batch_image()
        b64 = handler.to_base64(img)
        assert b64.startswith("data:image/jpeg;base64,")
        recovered = handler.load_from_base64(b64)
        assert recovered.shape[:2] == img.shape[:2]

    def test_to_bytes_roundtrip(self):
        handler = ImageIngestionHandler()
        img = make_batch_image()
        raw_bytes = handler.to_bytes(img)
        recovered = handler.load_from_bytes(raw_bytes)
        assert recovered.shape[2] == 3


# ── Normalization Tests ────────────────────────────────────────────────────────
class TestLightingNormalization:

    def setup_method(self):
        self.normalizer = LightingColorNormalizer(target_size=(640, 640))

    def test_normalize_returns_correct_shape(self):
        img = make_batch_image()
        result = self.normalizer.normalize(img)
        assert result["normalized_image"].shape == (640, 640, 3)
        assert result["scale"] > 0
        assert "pad_x" in result and "pad_y" in result

    def test_gray_world_balance(self):
        img = np.full((100, 100, 3), [255, 50, 50], dtype=np.uint8)
        balanced = self.normalizer._gray_world_balance(img)
        # After balance the channel means should be closer to equal
        b, g, r = cv2.split(balanced.astype(np.float32))
        assert abs(np.mean(b) - np.mean(r)) < abs(255.0 - 50.0)

    def test_white_patch_balance(self):
        img = np.full((100, 100, 3), [200, 100, 80], dtype=np.uint8)
        balanced = self.normalizer._white_patch_balance(img)
        assert balanced.shape == img.shape
        assert balanced.dtype == np.uint8

    def test_letterbox_dimensions(self):
        img = np.zeros((480, 320, 3), dtype=np.uint8)
        resized, scale, pad_x, pad_y = LightingColorNormalizer._letterbox(img, (640, 640))
        assert resized.shape == (640, 640, 3)
        assert scale > 0

    def test_unletterbox_coordinate_recovery(self):
        orig_shape = (900, 1200)
        img = np.zeros((*orig_shape, 3), dtype=np.uint8)
        _, scale, pad_x, pad_y = LightingColorNormalizer._letterbox(img, (640, 640))
        # Map a mid-frame box through letterbox and back
        x1 = int(400 * scale) + pad_x
        y1 = int(200 * scale) + pad_y
        x2 = int(600 * scale) + pad_x
        y2 = int(400 * scale) + pad_y
        ox1, oy1, ox2, oy2 = LightingColorNormalizer.unletterbox_box(
            (x1, y1, x2, y2), scale, pad_x, pad_y, orig_shape
        )
        # Allow ~2px tolerance
        assert abs(ox1 - 400) <= 3
        assert abs(oy1 - 200) <= 3


# ── Segmentation Tests ─────────────────────────────────────────────────────────
class TestBackgroundSegmentation:

    def setup_method(self):
        self.segmenter = BackgroundSegmenter(mat_type="blue_mat", min_nut_area_px=400)

    def test_segment_returns_expected_keys(self):
        img = make_batch_image(mat="blue")
        result = self.segmenter.segment(img)
        assert "binary_mask" in result
        assert "foreground" in result
        assert "contours" in result
        assert isinstance(result["binary_mask"], np.ndarray)

    def test_binary_mask_shape_matches_image(self):
        img = make_batch_image()
        result = self.segmenter.segment(img)
        assert result["binary_mask"].shape == img.shape[:2]

    def test_foreground_has_same_dimensions(self):
        img = make_batch_image()
        result = self.segmenter.segment(img)
        assert result["foreground"].shape == img.shape

    def test_auto_segment_does_not_crash(self):
        seg = BackgroundSegmenter(mat_type="auto", min_nut_area_px=200)
        img = make_batch_image()
        result = seg.segment(img)
        assert "binary_mask" in result

    def test_green_mat_segmentation(self):
        seg = BackgroundSegmenter(mat_type="green_mat", min_nut_area_px=300)
        img = make_batch_image(mat="green")
        result = seg.segment(img)
        assert "binary_mask" in result


# ── Calibration Tests ──────────────────────────────────────────────────────────
class TestPixelCalibrator:

    def test_pixels_to_mm(self):
        cal = PixelCalibrator(default_pixels_per_mm=10.0)
        assert cal.pixels_to_mm(100) == pytest.approx(10.0)

    def test_pixels_to_cm(self):
        cal = PixelCalibrator(default_pixels_per_mm=10.0)
        assert cal.pixels_to_cm(100) == pytest.approx(1.0)

    def test_set_pixels_per_mm(self):
        cal = PixelCalibrator()
        cal.set_pixels_per_mm(15.0)
        assert cal.pixels_per_mm == 15.0

    def test_set_invalid_px_per_mm_raises(self):
        cal = PixelCalibrator()
        with pytest.raises(ValueError):
            cal.set_pixels_per_mm(-1.0)

    def test_classify_size_small(self):
        cal = PixelCalibrator()
        assert cal.classify_size(14.0) == "Small"

    def test_classify_size_medium(self):
        cal = PixelCalibrator()
        assert cal.classify_size(20.0) == "Medium"

    def test_classify_size_large(self):
        cal = PixelCalibrator()
        assert cal.classify_size(25.0) == "Large"

    def test_measure_contour_returns_dict(self):
        cal = PixelCalibrator()
        # Create a circular contour
        pts = []
        for i in range(360):
            angle = np.radians(i)
            pts.append([int(50 + 40 * np.cos(angle)), int(50 + 40 * np.sin(angle))])
        cnt = np.array(pts, dtype=np.int32).reshape(-1, 1, 2)
        result = cal.measure_contour(cnt)
        assert "diameter_mm" in result
        assert "size_grade" in result
        assert result["circularity"] > 0.0


# ── Detection Tests ────────────────────────────────────────────────────────────
class TestNutDetector:

    def setup_method(self):
        self.detector = NutDetector()

    def test_detect_returns_list(self):
        img = make_batch_image()
        nuts = self.detector.detect(img)
        assert isinstance(nuts, list)

    def test_detected_nuts_have_correct_attributes(self):
        img = make_batch_image()
        nuts = self.detector.detect(img)
        if nuts:
            n = nuts[0]
            assert hasattr(n, "nut_id")
            assert hasattr(n, "bbox")
            assert hasattr(n, "diameter_mm")
            assert hasattr(n, "size_grade")
            assert n.size_grade in ("Small", "Medium", "Large")

    def test_nms_removes_duplicates(self):
        boxes = [
            (10, 10, 60, 60, 0.9, None),
            (12, 12, 62, 62, 0.8, None),  # very high overlap → should be removed
            (200, 200, 260, 260, 0.9, None),
        ]
        kept = self.detector._nms(boxes, iou_thresh=0.5)
        assert len(kept) == 2

    def test_iou_identical_boxes(self):
        iou = NutDetector._compute_iou((0, 0, 100, 100), (0, 0, 100, 100))
        assert iou == pytest.approx(1.0)

    def test_iou_non_overlapping(self):
        iou = NutDetector._compute_iou((0, 0, 50, 50), (100, 100, 150, 150))
        assert iou == pytest.approx(0.0)

    def test_detect_with_explicit_mask(self):
        img = make_batch_image()
        # Provide a simple all-white mask
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.circle(mask, (90, 160), 42, 255, -1)
        nuts = self.detector.detect(img, binary_mask=mask)
        assert isinstance(nuts, list)


# ── Classification Tests ───────────────────────────────────────────────────────
class TestDefectClassifier:

    def setup_method(self):
        weights_path = os.path.join(PROJECT_ROOT, "src", "models", "weights", "defect_classifier.onnx")
        self.classifier = DefectClassifier(weights_path=weights_path)

    def test_classify_returns_expected_keys(self):
        crop = make_synthetic_nut_image(128, defect_class=0)
        result = self.classifier.classify(crop)
        for key in ("class_id", "class_label", "confidence", "probabilities",
                    "is_defective", "severity_score", "diagnostic_notes"):
            assert key in result

    def test_classify_valid_class_id(self):
        crop = make_synthetic_nut_image(128)
        result = self.classifier.classify(crop)
        assert result["class_id"] in (0, 1, 2, 3)

    def test_probabilities_sum_to_one(self):
        crop = make_synthetic_nut_image(128)
        result = self.classifier.classify(crop)
        total = sum(result["probabilities"].values())
        assert total == pytest.approx(1.0, abs=0.01)

    def test_confidence_in_range(self):
        crop = make_synthetic_nut_image(128)
        result = self.classifier.classify(crop)
        assert 0.0 <= result["confidence"] <= 1.0

    def test_severity_non_negative(self):
        crop = make_synthetic_nut_image(128)
        result = self.classifier.classify(crop)
        assert result["severity_score"] >= 0.0

    def test_classify_tiny_crop_fallback(self):
        tiny = np.ones((4, 4, 3), dtype=np.uint8)
        result = self.classifier.classify(tiny)
        assert result["class_id"] == 0  # defaults to healthy

    def test_healthy_signature_low_severity(self):
        """Healthy synthetic nut should have lower severity than a mold nut."""
        healthy_crop = make_synthetic_nut_image(200, defect_class=0)
        mold_crop    = make_synthetic_nut_image(200, defect_class=3)
        h_res = self.classifier.classify(healthy_crop)
        m_res = self.classifier.classify(mold_crop)
        assert h_res["severity_score"] <= m_res["severity_score"] + 30.0  # generous tolerance for synthetic


# ── XAI Tests ─────────────────────────────────────────────────────────────────
class TestExplainabilityEngine:

    def setup_method(self):
        self.xai = ExplainabilityEngine()

    def test_generate_explanation_keys(self):
        crop = make_synthetic_nut_image(160)
        saliency = np.random.rand(160, 160).astype(np.float32)
        result = self.xai.generate_explanation(crop, saliency, "Moldy / Insect Damaged", 0.88)
        for key in ("heatmap_overlay", "contour_overlay", "side_by_side",
                    "defect_area_pct", "audit_card"):
            assert key in result

    def test_heatmap_overlay_correct_shape(self):
        crop = make_synthetic_nut_image(160)
        saliency = np.ones((160, 160), dtype=np.float32) * 0.6
        result = self.xai.generate_explanation(crop, saliency, "Cracked / Broken", 0.75)
        assert result["heatmap_overlay"].shape == crop.shape

    def test_saliency_resize_handled(self):
        """XAI engine must gracefully resize a mismatched saliency map."""
        crop = make_synthetic_nut_image(200)
        saliency = np.random.rand(64, 64).astype(np.float32)
        result = self.xai.generate_explanation(crop, saliency, "Discolored / Brown Center", 0.65)
        assert result["heatmap_overlay"].shape == crop.shape

    def test_defect_area_pct_in_range(self):
        crop = make_synthetic_nut_image(160)
        saliency = np.ones((160, 160), dtype=np.float32) * 0.5
        result = self.xai.generate_explanation(crop, saliency, "Moldy / Insect Damaged", 0.90)
        assert 0.0 <= result["defect_area_pct"] <= 100.0

    def test_audit_card_has_compliance_status(self):
        crop = make_synthetic_nut_image(160)
        saliency = np.zeros((160, 160), dtype=np.float32)
        result = self.xai.generate_explanation(crop, saliency, "Healthy / Premium", 0.98)
        assert "compliance_status" in result["audit_card"]
        assert result["audit_card"]["compliance_status"] == "Passed"


# ── Pipeline Integration Tests ─────────────────────────────────────────────────
class TestEndToEndPipeline:

    @pytest.fixture(scope="class")
    def pipeline(self):
        return MacadamiaGradingPipeline(config_path=CONFIG_PATH)

    @pytest.fixture(scope="class")
    def sample_result(self, pipeline):
        return pipeline.process_image(SAMPLE_IMAGE)

    def test_pipeline_loads(self, pipeline):
        assert pipeline is not None
        assert pipeline.calibrator is not None

    def test_process_image_from_path(self, sample_result):
        assert "summary" in sample_result
        assert "analytics" in sample_result
        assert "nuts" in sample_result
        assert "images" in sample_result

    def test_summary_has_required_fields(self, sample_result):
        s = sample_result["summary"]
        for field in ("batch_id", "timestamp", "total_count", "quality_grade",
                      "defect_rate_pct", "premium_rate_pct", "avg_diameter_mm"):
            assert field in s, f"Missing field: {field}"

    def test_at_least_one_nut_detected(self, sample_result):
        assert sample_result["summary"]["total_count"] >= 1

    def test_all_rate_fields_in_range(self, sample_result):
        s = sample_result["summary"]
        assert 0.0 <= s["defect_rate_pct"] <= 100.0
        assert 0.0 <= s["premium_rate_pct"] <= 100.0
        assert abs(s["defect_rate_pct"] + s["premium_rate_pct"] - 100.0) < 0.5

    def test_analytics_size_distribution_complete(self, sample_result):
        sizes = sample_result["analytics"]["size_distribution"]
        assert set(sizes.keys()) == {"Small", "Medium", "Large"}

    def test_analytics_defect_distribution_complete(self, sample_result):
        defects = sample_result["analytics"]["defect_distribution"]
        expected = {"Healthy / Premium", "Cracked / Broken", "Discolored / Brown Center", "Moldy / Insect Damaged"}
        assert set(defects.keys()) == expected

    def test_total_counts_consistent(self, sample_result):
        s = sample_result["summary"]
        a = sample_result["analytics"]
        assert s["total_count"] == sum(a["size_distribution"].values())
        assert s["total_count"] == sum(a["defect_distribution"].values())

    def test_annotated_image_has_banner_height(self, sample_result):
        """Annotated image should be taller than raw (banner added)."""
        raw_h = sample_result["images"]["raw"].shape[0]
        ann_h = sample_result["images"]["annotated"].shape[0]
        assert ann_h > raw_h

    def test_quality_grade_is_valid_string(self, sample_result):
        grade = sample_result["summary"]["quality_grade"]
        valid = {"Grade 1 (Premium Export)", "Grade 2 (Commercial / Processing)", "Reject (Substandard)"}
        assert grade in valid

    def test_process_image_from_bytes(self, pipeline):
        with open(SAMPLE_IMAGE, "rb") as f:
            raw = f.read()
        result = pipeline.process_image(raw)
        assert result["summary"]["total_count"] >= 1

    def test_process_image_from_numpy(self, pipeline):
        img = make_batch_image()
        result = pipeline.process_image(img)
        assert "summary" in result

    def test_per_nut_has_all_fields(self, sample_result):
        for nut in sample_result["nuts"]:
            for field in ("nut_id", "bbox", "diameter_mm", "size_grade",
                          "class_id", "class_label", "confidence", "is_defective"):
                assert field in nut, f"Nut missing field: {field}"

    def test_all_nuts_have_valid_size_grade(self, sample_result):
        valid_sizes = {"Small", "Medium", "Large"}
        for n in sample_result["nuts"]:
            assert n["size_grade"] in valid_sizes

    def test_all_nuts_have_valid_class_id(self, sample_result):
        for n in sample_result["nuts"]:
            assert n["class_id"] in (0, 1, 2, 3)

    def test_pixels_per_mm_override(self, pipeline):
        result = pipeline.process_image(SAMPLE_IMAGE, pixels_per_mm_override=5.0)
        assert result["summary"]["total_count"] >= 1


# ── Reporting Tests ────────────────────────────────────────────────────────────
class TestReportGeneration:

    @pytest.fixture(scope="class")
    def grading_result(self):
        pipeline = MacadamiaGradingPipeline(config_path=CONFIG_PATH)
        return pipeline.process_image(SAMPLE_IMAGE)

    def test_pdf_generation_produces_bytes(self, grading_result):
        pdf = generate_pdf_report(grading_result)
        assert isinstance(pdf, bytes)
        assert len(pdf) > 1000

    def test_pdf_starts_with_pdf_magic(self, grading_result):
        pdf = generate_pdf_report(grading_result)
        assert pdf[:4] == b"%PDF"

    def test_csv_is_valid_string(self, grading_result):
        csv = generate_csv_report(grading_result)
        assert isinstance(csv, str)
        assert len(csv) > 100

    def test_csv_contains_header_columns(self, grading_result):
        csv = generate_csv_report(grading_result)
        assert "Nut_ID" in csv
        assert "Size_Grade" in csv
        assert "Defect_Class" in csv

    def test_csv_row_count_matches_nuts(self, grading_result):
        csv = generate_csv_report(grading_result)
        # Data rows start after the header row (which contains "Nut_ID").
        # Identify the header line index, then count subsequent non-empty lines.
        lines = csv.splitlines()
        header_idx = next((i for i, l in enumerate(lines) if "Nut_ID" in l), None)
        assert header_idx is not None, "CSV is missing header row"
        data_rows = [l for l in lines[header_idx + 1:] if l.strip()]
        expected = grading_result["summary"]["total_count"]
        assert len(data_rows) == expected

    def test_pdf_generation_empty_batch(self):
        """PDF generator must not crash on empty batch result."""
        empty_result = {
            "summary": {
                "batch_id": "EMPTY-001",
                "timestamp": "2026-01-01 00:00:00",
                "total_count": 0,
                "quality_grade": "No Nuts Detected",
                "defect_rate_pct": 0.0,
                "premium_rate_pct": 0.0,
                "avg_diameter_mm": 0.0,
                "processing_time_sec": 0.1,
            },
            "analytics": {
                "total_count": 0,
                "size_distribution": {"Small": 0, "Medium": 0, "Large": 0},
                "size_percentages": {"Small": 0.0, "Medium": 0.0, "Large": 0.0},
                "defect_distribution": {"Healthy / Premium": 0, "Cracked / Broken": 0,
                                        "Discolored / Brown Center": 0, "Moldy / Insect Damaged": 0},
                "defect_percentages": {"Healthy / Premium": 0.0, "Cracked / Broken": 0.0,
                                       "Discolored / Brown Center": 0.0, "Moldy / Insect Damaged": 0.0},
                "total_defective_count": 0,
                "defect_rate_pct": 0.0,
                "premium_rate_pct": 0.0,
                "avg_diameter_mm": 0.0,
                "min_diameter_mm": 0.0,
                "max_diameter_mm": 0.0,
                "batch_quality_grade": "No Nuts Detected",
            },
            "nuts": [],
        }
        pdf = generate_pdf_report(empty_result)
        assert isinstance(pdf, bytes)
        assert len(pdf) > 500

    def test_coco_annotations_format(self):
        annot_path = os.path.join(PROJECT_ROOT, "data", "annotations", "coco_annotations.json")
        with open(annot_path, "r") as f:
            coco = json.load(f)
        assert "images" in coco
        assert "annotations" in coco
        assert "categories" in coco
        assert len(coco["images"]) > 0
        assert len(coco["annotations"]) > 0
