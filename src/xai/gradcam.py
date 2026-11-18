"""
Explainability Engine (Grad-CAM & Feature Saliency Heatmaps).
Provides visual auditability and explainability for defective macadamia nuts,
highlighting exact crack fissures, mold spots, and internal browning zones.
"""

import logging
from typing import Dict, Any, Optional, Tuple, List
import cv2
import numpy as np

logger = logging.getLogger(__name__)


class ExplainabilityEngine:
    """
    Computes visual explanations (Grad-CAM overlays, defect localization contours,
    and audit verification cards) for quality assurance and compliance reporting.
    """

    def __init__(
        self,
        colormap: int = cv2.COLORMAP_JET,
        alpha_overlay: float = 0.55,
        threshold_saliency: float = 0.35,
    ):
        self.colormap = colormap
        self.alpha_overlay = alpha_overlay
        self.threshold_saliency = threshold_saliency

    def generate_explanation(
        self,
        crop: np.ndarray,
        saliency_map: np.ndarray,
        defect_label: str,
        confidence: float,
        diagnostic_notes: str = "",
    ) -> Dict[str, Any]:
        """
        Generates full explainability package for a single inspected nut.

        Args:
            crop: BGR image of the nut crop
            saliency_map: 2D float array in [0, 1] representing anomaly intensity
            defect_label: predicted classification label
            confidence: model classification confidence
            diagnostic_notes: textual explanation

        Returns:
            Dictionary with:
                - 'heatmap_overlay': BGR image with translucent Grad-CAM heatmap
                - 'contour_overlay': BGR image with highlighted defect boundary contours
                - 'side_by_side': combined horizontal composite image for dashboard inspection
                - 'defect_contours': list of detected anomaly contours
                - 'defect_area_pct': percentage of surface exhibiting defect
                - 'audit_card': formatted inspection summary dict
        """
        h, w = crop.shape[:2]

        # Ensure saliency map matches crop dimensions
        if saliency_map.shape[:2] != (h, w):
            saliency_map = cv2.resize(saliency_map, (w, h), interpolation=cv2.INTER_LINEAR)

        saliency_map = np.clip(saliency_map, 0.0, 1.0)

        # 1. Generate Colored Heatmap
        saliency_uint8 = np.uint8(255 * saliency_map)
        heatmap_color = cv2.applyColorMap(saliency_uint8, self.colormap)

        # 2. Alpha blend onto crop
        overlay = cv2.addWeighted(crop, 1.0 - self.alpha_overlay, heatmap_color, self.alpha_overlay, 0)

        # Suppress heatmap in pure background regions (outside nut boundary)
        gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        _, nut_mask = cv2.threshold(gray_crop, 15, 255, cv2.THRESH_BINARY)
        overlay[nut_mask == 0] = crop[nut_mask == 0]

        # 3. Defect Contours & Outlines
        _, defect_thresh = cv2.threshold(
            saliency_uint8, int(self.threshold_saliency * 255), 255, cv2.THRESH_BINARY
        )
        defect_thresh = cv2.bitwise_and(defect_thresh, defect_thresh, mask=nut_mask)

        contours, _ = cv2.findContours(defect_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour_overlay = crop.copy()

        # Draw red contours around the defect hot spots
        cv2.drawContours(contour_overlay, contours, -1, (0, 0, 255), 2)

        # Calculate affected surface area
        nut_area_px = max(1, np.count_nonzero(nut_mask))
        defect_area_px = np.count_nonzero(defect_thresh)
        defect_area_pct = round(float(defect_area_px / nut_area_px) * 100.0, 1)

        # 4. Create Side-by-Side Verification Banner
        side_by_side = self._create_side_by_side(crop, overlay, contour_overlay, defect_label, confidence)

        return {
            "heatmap_overlay": overlay,
            "contour_overlay": contour_overlay,
            "side_by_side": side_by_side,
            "defect_contours_count": len(contours),
            "defect_area_pct": defect_area_pct,
            "audit_card": {
                "defect_label": defect_label,
                "confidence": round(float(confidence), 3),
                "affected_area_pct": defect_area_pct,
                "notes": diagnostic_notes,
                "compliance_status": "Flagged Defective" if defect_label != "Healthy / Premium" else "Passed",
            },
        }

    def _create_side_by_side(
        self,
        crop: np.ndarray,
        heatmap: np.ndarray,
        contours: np.ndarray,
        defect_label: str,
        confidence: float,
    ) -> np.ndarray:
        """
        Assembles a side-by-side composite panel:
        [Raw Crop] | [Grad-CAM Heatmap] | [Defect Saliency Outline]
        """
        target_size = (160, 160)
        c1 = cv2.resize(crop, target_size, interpolation=cv2.INTER_LINEAR)
        c2 = cv2.resize(heatmap, target_size, interpolation=cv2.INTER_LINEAR)
        c3 = cv2.resize(contours, target_size, interpolation=cv2.INTER_LINEAR)

        # Add subtitle labels
        self._add_banner_label(c1, "1. Raw Crop")
        self._add_banner_label(c2, "2. XAI Heatmap")
        self._add_banner_label(c3, "3. Defect Contour")

        # Concatenate horizontally
        banner = np.hstack([c1, c2, c3])

        # Add header banner with defect title
        header_h = 32
        header = np.zeros((header_h, banner.shape[1], 3), dtype=np.uint8)

        # Red for defect, green for healthy
        is_healthy = "Healthy" in defect_label
        bg_color = (40, 140, 40) if is_healthy else (30, 30, 180)
        header[:] = bg_color

        text = f"{defect_label} ({confidence:.0%})"
        cv2.putText(header, text, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        return np.vstack([header, banner])

    @staticmethod
    def _add_banner_label(image: np.ndarray, label: str) -> None:
        """Draws small bottom caption on an inspection card."""
        h, w = image.shape[:2]
        cv2.rectangle(image, (0, h - 22), (w, h), (0, 0, 0), -1)
        cv2.putText(image, label, (6, h - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)
