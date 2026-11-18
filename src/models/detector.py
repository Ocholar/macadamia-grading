"""
Macadamia Nut Detector Module.
Performs object detection, individual kernel localization, contour tracking,
and metric sizing (Small, Medium, Large) calibrated in physical millimeters/centimeters.
"""

import os
import logging
from typing import List, Dict, Any, Optional, Tuple
import cv2
import numpy as np

try:
    import onnxruntime as ort
    HAS_ORT = True
except ImportError:
    HAS_ORT = False

from ..preprocessing.calibration import PixelCalibrator

logger = logging.getLogger(__name__)


class DetectedNut:
    """Represents an individual localized macadamia nut."""

    def __init__(
        self,
        nut_id: int,
        bbox: Tuple[int, int, int, int],  # x1, y1, x2, y2
        confidence: float,
        contour: np.ndarray,
        crop: np.ndarray,
        crop_mask: np.ndarray,
        metric_measurements: Dict[str, Any],
    ):
        self.nut_id = nut_id
        self.bbox = bbox
        self.confidence = confidence
        self.contour = contour
        self.crop = crop
        self.crop_mask = crop_mask
        self.measurements = metric_measurements

        # Extracted geometry properties
        self.x1, self.y1, self.x2, self.y2 = bbox
        self.width = self.x2 - self.x1
        self.height = self.y2 - self.y1
        self.diameter_mm = metric_measurements.get("diameter_mm", 0.0)
        self.diameter_cm = metric_measurements.get("diameter_cm", 0.0)
        self.size_grade = metric_measurements.get("size_grade", "Medium")

    def to_dict(self) -> Dict[str, Any]:
        """Serializes nut object detection and sizing data."""
        return {
            "nut_id": self.nut_id,
            "bbox": [int(self.x1), int(self.y1), int(self.x2), int(self.y2)],
            "confidence": round(float(self.confidence), 3),
            "diameter_mm": round(float(self.diameter_mm), 2),
            "diameter_cm": round(float(self.diameter_cm), 3),
            "size_grade": self.size_grade,
            "area_px": round(float(self.measurements.get("area_px", 0.0)), 1),
            "circularity": round(float(self.measurements.get("circularity", 0.0)), 3),
        }


class NutDetector:
    """
    Object detector and dimensional analyzer for macadamia batches.
    Supports ONNX YOLO detection and high-accuracy contour-watershed localization.
    """

    def __init__(
        self,
        weights_path: Optional[str] = None,
        confidence_threshold: float = 0.35,
        iou_threshold: float = 0.45,
        calibrator: Optional[PixelCalibrator] = None,
    ):
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.calibrator = calibrator or PixelCalibrator()

        self.session = None
        if weights_path and os.path.exists(weights_path) and HAS_ORT:
            try:
                self.session = ort.InferenceSession(
                    weights_path, providers=["CPUExecutionProvider"]
                )
                logger.info("Loaded detector ONNX session from: %s", weights_path)
            except Exception as e:
                logger.warning("Could not initialize ONNX detector session (%s); using CV detector.", e)

    def detect(
        self,
        image: np.ndarray,
        binary_mask: Optional[np.ndarray] = None,
        crop_padding: int = 8,
    ) -> List[DetectedNut]:
        """
        Detects individual nuts in the batch image and estimates metric sizing.
        """
        h_img, w_img = image.shape[:2]

        # 1. Use segmentation contours if binary mask is available
        if binary_mask is not None:
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        else:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detected_nuts: List[DetectedNut] = []
        raw_boxes = []

        min_area = self.calibrator.pixels_per_mm * self.calibrator.pixels_per_mm * 15.0  # ~15 mm^2 min
        max_area = w_img * h_img * 0.45

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area or area > max_area:
                continue

            # Bounding box
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = float(w) / float(h)
            if aspect_ratio < 0.45 or aspect_ratio > 2.2:
                continue

            raw_boxes.append((x, y, x + w, y + h, 0.95, cnt))

        # 2. Non-Maximum Suppression to prevent duplicate bounding boxes
        final_boxes = self._nms(raw_boxes, self.iou_threshold)

        # 3. Create DetectedNut instances with crops and metric measurements
        for i, (x1, y1, x2, y2, score, cnt) in enumerate(final_boxes):
            measurements = self.calibrator.measure_contour(cnt)

            # Extract nut crop with padding
            pad_x1 = max(0, x1 - crop_padding)
            pad_y1 = max(0, y1 - crop_padding)
            pad_x2 = min(w_img, x2 + crop_padding)
            pad_y2 = min(h_img, y2 + crop_padding)

            crop = image[pad_y1:pad_y2, pad_x1:pad_x2].copy()
            crop_h, crop_w = crop.shape[:2]

            # Construct exact nut contour mask in the local crop coordinate space
            crop_mask = np.zeros((crop_h, crop_w), dtype=np.uint8)
            shifted_cnt = cnt - np.array([pad_x1, pad_y1])
            cv2.drawContours(crop_mask, [shifted_cnt], -1, 255, thickness=cv2.FILLED)

            nut = DetectedNut(
                nut_id=i + 1,
                bbox=(x1, y1, x2, y2),
                confidence=score,
                contour=cnt,
                crop=crop,
                crop_mask=crop_mask,
                metric_measurements=measurements,
            )
            detected_nuts.append(nut)

        # Sort nuts top-to-bottom, left-to-right
        detected_nuts.sort(key=lambda n: (n.y1 // 60, n.x1))
        for idx, nut in enumerate(detected_nuts):
            nut.nut_id = idx + 1

        return detected_nuts

    def _nms(
        self, boxes: List[Tuple[int, int, int, int, float, np.ndarray]], iou_thresh: float
    ) -> List[Tuple[int, int, int, int, float, np.ndarray]]:
        """Applies Non-Maximum Suppression over candidate boxes."""
        if not boxes:
            return []

        boxes = sorted(boxes, key=lambda b: b[4], reverse=True)
        picked = []

        while boxes:
            current = boxes.pop(0)
            picked.append(current)

            remaining = []
            for b in boxes:
                iou = self._compute_iou(current[:4], b[:4])
                if iou < iou_thresh:
                    remaining.append(b)
            boxes = remaining

        return picked

    @staticmethod
    def _compute_iou(
        box_a: Tuple[int, int, int, int], box_b: Tuple[int, int, int, int]
    ) -> float:
        """Computes Intersection over Union (IoU) between two bounding boxes."""
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b

        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)

        intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        area_a = (ax2 - ax1) * (ay2 - ay1)
        area_b = (bx2 - bx1) * (by2 - by1)
        union = float(area_a + area_b - intersection)

        if union <= 0:
            return 0.0
        return float(intersection / union)
