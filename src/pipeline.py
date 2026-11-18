"""
End-to-End Macadamia Quality Grading Pipeline.
Orchestrates image ingestion, preprocessing, segmentation, detection, sizing,
defect classification, XAI heatmaps, and batch analytics.
"""

import os
import time
import logging
from typing import Dict, Any, Optional, List, Union
import cv2
import numpy as np
import yaml

from .ingestion.handler import ImageIngestionHandler
from .preprocessing.normalization import LightingColorNormalizer
from .preprocessing.segmentation import BackgroundSegmenter
from .preprocessing.calibration import PixelCalibrator
from .models.detector import NutDetector, DetectedNut
from .models.classifier import DefectClassifier
from .xai.gradcam import ExplainabilityEngine

logger = logging.getLogger(__name__)


class MacadamiaGradingPipeline:
    """
    Unified end-to-end inference engine for optical macadamia quality grading.
    """

    def __init__(self, config_path: Optional[str] = None):
        # Resolve config
        if config_path is None:
            base_dir = os.path.dirname(os.path.dirname(__file__))
            config_path = os.path.join(base_dir, "config.yaml")

        self.config = self._load_config(config_path)

        # Initialize submodules from config
        self.ingestion = ImageIngestionHandler()

        prep_cfg = self.config.get("preprocessing", {})
        self.normalizer = LightingColorNormalizer(
            target_size=tuple(prep_cfg.get("input_dimensions", [640, 640])),
            clahe_clip_limit=prep_cfg.get("clahe", {}).get("clip_limit", 2.5),
            enable_white_balance=prep_cfg.get("white_balance", {}).get("enabled", True),
            white_balance_method=prep_cfg.get("white_balance", {}).get("method", "gray_world"),
        )

        seg_cfg = prep_cfg.get("background_segmentation", {})
        self.segmenter = BackgroundSegmenter(
            mat_type=seg_cfg.get("mat_type", "auto"),
            morphology_kernel_size=seg_cfg.get("morphology_kernel_size", 5),
            min_nut_area_px=seg_cfg.get("min_nut_area_px", 400),
            max_nut_area_px=seg_cfg.get("max_nut_area_px", 150000),
        )

        cal_cfg = self.config.get("calibration", {})
        self.calibrator = PixelCalibrator(
            default_pixels_per_mm=cal_cfg.get("default_pixels_per_mm", 12.0),
            unit=cal_cfg.get("unit", "mm"),
            size_categories=self.config.get("sizing", {}).get("categories"),
        )

        models_dir = os.path.join(os.path.dirname(__file__), "models", "weights")
        det_weights = os.path.join(models_dir, "nut_detector.onnx")
        cls_weights = os.path.join(models_dir, "defect_classifier.onnx")

        self.detector = NutDetector(
            weights_path=det_weights if os.path.exists(det_weights) else None,
            calibrator=self.calibrator,
        )

        self.classifier = DefectClassifier(
            weights_path=cls_weights if os.path.exists(cls_weights) else None,
        )

        xai_cfg = self.config.get("xai", {})
        self.xai = ExplainabilityEngine(
            alpha_overlay=xai_cfg.get("alpha_overlay", 0.55),
        )

    def process_image(
        self,
        image_input: Union[str, bytes, np.ndarray],
        pixels_per_mm_override: Optional[float] = None,
        mat_type_override: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Runs the full end-to-end grading analysis on a batch photo.

        Args:
            image_input: file path, raw byte stream, or BGR numpy array
            pixels_per_mm_override: optional custom calibration ratio
            mat_type_override: optional background mat color override ("auto", "blue_mat", "green_mat", etc.)

        Returns:
            Structured grading report dict including analytics, per-nut evaluations, and images.
        """
        start_time = time.time()

        # 1. Ingestion & Validation
        if isinstance(image_input, str):
            raw_image = self.ingestion.load_from_path(image_input)
        elif isinstance(image_input, bytes):
            raw_image = self.ingestion.load_from_bytes(image_input)
        elif isinstance(image_input, np.ndarray):
            raw_image = self.ingestion.validate_image(image_input)
        else:
            raise TypeError(f"Unsupported image_input type: {type(image_input)}")

        # Update calibration if requested
        if pixels_per_mm_override and pixels_per_mm_override > 0:
            self.calibrator.set_pixels_per_mm(pixels_per_mm_override)

        # 2. Lighting & Color Normalization
        norm_result = self.normalizer.normalize(raw_image)
        balanced_image = norm_result["color_balanced"]

        # 3. Background Segmentation
        seg_result = self.segmenter.segment(balanced_image, mat_type_override=mat_type_override)
        binary_mask = seg_result["binary_mask"]
        foreground = seg_result["foreground"]

        # 4. Nut Localization & Metric Sizing
        detected_nuts = self.detector.detect(balanced_image, binary_mask=binary_mask)

        # 5. Defect Classification & XAI Generation per Nut
        evaluated_nuts = []
        xai_defect_gallery = []

        for nut in detected_nuts:
            # Run multi-class defect classification with nut contour mask
            diag = self.classifier.classify(nut.crop, nut_mask=nut.crop_mask)

            # Generate XAI heatmap explanation
            xai_result = self.xai.generate_explanation(
                crop=nut.crop,
                saliency_map=diag["defect_saliency_map"],
                defect_label=diag["class_label"],
                confidence=diag["confidence"],
                diagnostic_notes=diag["diagnostic_notes"],
            )

            nut_summary = {
                "nut_id": nut.nut_id,
                "bbox": [int(nut.x1), int(nut.y1), int(nut.x2), int(nut.y2)],
                "diameter_mm": nut.diameter_mm,
                "diameter_cm": nut.diameter_cm,
                "size_grade": nut.size_grade,
                "class_id": diag["class_id"],
                "class_label": diag["class_label"],
                "confidence": diag["confidence"],
                "is_defective": diag["is_defective"],
                "defect_severity_pct": diag["severity_score"],
                "diagnostic_notes": diag["diagnostic_notes"],
                "crop": nut.crop,
                "heatmap_overlay": xai_result["heatmap_overlay"],
                "contour_overlay": xai_result["contour_overlay"],
                "xai_side_by_side": xai_result["side_by_side"],
            }
            evaluated_nuts.append(nut_summary)

            if diag["is_defective"]:
                xai_defect_gallery.append(nut_summary)

        # 6. Aggregate Batch Analytics
        analytics = self._compile_analytics(evaluated_nuts)

        # 7. Render Master Annotated Image
        annotated_image = self._render_annotated_batch(raw_image, evaluated_nuts, analytics)

        elapsed = round(time.time() - start_time, 3)

        return {
            "summary": {
                "batch_id": f"BATCH-{int(time.time())}",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "total_count": analytics["total_count"],
                "quality_grade": analytics["batch_quality_grade"],
                "defect_rate_pct": analytics["defect_rate_pct"],
                "premium_rate_pct": analytics["premium_rate_pct"],
                "avg_diameter_mm": analytics["avg_diameter_mm"],
                "processing_time_sec": elapsed,
            },
            "analytics": analytics,
            "nuts": evaluated_nuts,
            "xai_defects": xai_defect_gallery,
            "images": {
                "raw": raw_image,
                "preprocessed": balanced_image,
                "binary_mask": binary_mask,
                "foreground": foreground,
                "annotated": annotated_image,
            },
        }

    def _compile_analytics(self, nuts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculates sizing distributions, defect percentages, and commercial batch grade."""
        total = len(nuts)
        if total == 0:
            return {
                "total_count": 0,
                "size_distribution": {"Small": 0, "Medium": 0, "Large": 0},
                "size_percentages": {"Small": 0.0, "Medium": 0.0, "Large": 0.0},
                "defect_distribution": {"Healthy / Premium": 0, "Cracked / Broken": 0, "Discolored / Brown Center": 0, "Moldy / Insect Damaged": 0},
                "defect_percentages": {"Healthy / Premium": 0.0, "Cracked / Broken": 0.0, "Discolored / Brown Center": 0.0, "Moldy / Insect Damaged": 0.0},
                "total_defective_count": 0,
                "defect_rate_pct": 0.0,
                "premium_rate_pct": 0.0,
                "avg_diameter_mm": 0.0,
                "min_diameter_mm": 0.0,
                "max_diameter_mm": 0.0,
                "batch_quality_grade": "No Nuts Detected",
            }

        # Size distribution
        size_counts = {"Small": 0, "Medium": 0, "Large": 0}
        diameters = []
        for n in nuts:
            sg = n["size_grade"]
            size_counts[sg] = size_counts.get(sg, 0) + 1
            diameters.append(n["diameter_mm"])

        size_pct = {k: round((v / total) * 100.0, 1) for k, v in size_counts.items()}

        # Defect distribution
        defect_counts = {
            "Healthy / Premium": 0,
            "Cracked / Broken": 0,
            "Discolored / Brown Center": 0,
            "Moldy / Insect Damaged": 0,
        }
        for n in nuts:
            lbl = n["class_label"]
            defect_counts[lbl] = defect_counts.get(lbl, 0) + 1

        defect_pct = {k: round((v / total) * 100.0, 1) for k, v in defect_counts.items()}

        total_defective = total - defect_counts["Healthy / Premium"]
        defect_rate = round((total_defective / total) * 100.0, 1)
        premium_rate = round(100.0 - defect_rate, 1)

        # Commercial Grade Determination
        grade_cfg = self.config.get("grading_standards", {})
        g1_max_defects = grade_cfg.get("grade_1", {}).get("max_total_defects_pct", 5.0)
        g1_max_mold = grade_cfg.get("grade_1", {}).get("max_moldy_pct", 1.0)
        comm_max_defects = grade_cfg.get("commercial", {}).get("max_total_defects_pct", 15.0)

        moldy_pct = defect_pct.get("Moldy / Insect Damaged", 0.0)

        if defect_rate <= g1_max_defects and moldy_pct <= g1_max_mold:
            quality_grade = "Grade 1 (Premium Export)"
        elif defect_rate <= comm_max_defects:
            quality_grade = "Grade 2 (Commercial / Processing)"
        else:
            quality_grade = "Reject (Substandard)"

        return {
            "total_count": total,
            "size_distribution": size_counts,
            "size_percentages": size_pct,
            "defect_distribution": defect_counts,
            "defect_percentages": defect_pct,
            "total_defective_count": total_defective,
            "defect_rate_pct": defect_rate,
            "premium_rate_pct": premium_rate,
            "avg_diameter_mm": round(float(np.mean(diameters)), 1),
            "min_diameter_mm": round(float(np.min(diameters)), 1),
            "max_diameter_mm": round(float(np.max(diameters)), 1),
            "batch_quality_grade": quality_grade,
        }

    def _render_annotated_batch(
        self,
        image: np.ndarray,
        nuts: List[Dict[str, Any]],
        analytics: Dict[str, Any],
    ) -> np.ndarray:
        """Draws bounding boxes, sizing tags, defect flags, and top HUD metrics."""
        annotated = image.copy()
        h, w = annotated.shape[:2]

        # Defect color palette (BGR)
        palette = {
            "Healthy / Premium": (60, 179, 113),       # Sea Green
            "Cracked / Broken": (255, 144, 30),        # Steel Blue / Orange
            "Discolored / Brown Center": (0, 165, 255),# Amber
            "Moldy / Insect Damaged": (0, 0, 220),     # Vivid Crimson Red
        }

        # 1. Draw each nut bounding box and label
        for n in nuts:
            x1, y1, x2, y2 = n["bbox"]
            color = palette.get(n["class_label"], (0, 255, 0))
            is_defective = n["is_defective"]
            line_thickness = 3 if is_defective else 2

            # Bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, line_thickness)

            # Label text
            tag = f"#{n['nut_id']} {n['size_grade'][0]}:{n['diameter_mm']}mm"
            if is_defective:
                short_defect = n["class_label"].split("/")[0].strip()
                tag += f" | {short_defect} {n['confidence']:.0%}"

            # Label badge background
            (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
            cv2.rectangle(annotated, (x1, max(0, y1 - 20)), (x1 + tw + 6, y1), color, -1)
            cv2.putText(
                annotated,
                tag,
                (x1 + 3, max(12, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        # 2. Render Header HUD Banner
        banner_h = 56
        banner = np.zeros((banner_h, w, 3), dtype=np.uint8)

        # Determine banner accent color based on grade
        grade = analytics.get("batch_quality_grade", "")
        accent_color = (40, 140, 40) if "Grade 1" in grade else ((0, 140, 220) if "Grade 2" in grade else (30, 30, 190))
        banner[:] = (28, 28, 28)
        cv2.rectangle(banner, (0, banner_h - 4), (w, banner_h), accent_color, -1)

        hud_title = f"MACADAMIA QUALITY AI: {grade.upper()}"
        hud_metrics = (
            f"NUTS: {analytics['total_count']}  |  "
            f"PREMIUM: {analytics['premium_rate_pct']}%  |  "
            f"DEFECTS: {analytics['defect_rate_pct']}%  |  "
            f"AVG SIZE: {analytics['avg_diameter_mm']} mm"
        )

        cv2.putText(banner, hud_title, (14, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(banner, hud_metrics, (14, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

        return np.vstack([banner, annotated])

    @staticmethod
    def _load_config(path: str) -> Dict[str, Any]:
        """Loads and parses configuration YAML."""
        if not os.path.exists(path):
            logger.warning("Config file %s not found. Using defaults.", path)
            return {}
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
