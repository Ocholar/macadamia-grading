"""
Pixel-to-Metric Calibration Module.
Provides conversion from pixel measurements to physical millimeters and centimeters.
Supports manual calibration factors and automated reference marker detection.
"""

import logging
from typing import Dict, Any, Optional, Tuple
import cv2
import numpy as np

logger = logging.getLogger(__name__)


class PixelCalibrator:
    """
    Calibrates camera spatial resolution and converts pixel contours into metric measurements (mm/cm).
    """

    def __init__(
        self,
        default_pixels_per_mm: float = 12.0,
        unit: str = "mm",
        size_categories: Optional[Dict[str, Any]] = None,
    ):
        self.pixels_per_mm = float(default_pixels_per_mm)
        self.unit = unit
        self.size_categories = size_categories or {
            "small": {"name": "Small", "max_diameter_mm": 18.0},
            "medium": {"name": "Medium", "min_diameter_mm": 18.0, "max_diameter_mm": 22.0},
            "large": {"name": "Large", "min_diameter_mm": 22.0},
        }

    def set_pixels_per_mm(self, px_per_mm: float) -> None:
        """Sets the active pixel-to-millimeter ratio."""
        if px_per_mm <= 0:
            raise ValueError(f"pixels_per_mm must be positive, got {px_per_mm}")
        self.pixels_per_mm = float(px_per_mm)

    def calibrate_from_reference_circle(
        self,
        image: np.ndarray,
        known_diameter_mm: float = 20.0,
        min_radius_px: int = 20,
        max_radius_px: int = 150,
    ) -> Optional[float]:
        """
        Detects a circular calibration coin/token on the mat using Hough Circle Transform
        and computes the resulting pixels_per_mm ratio.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)

        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=100,
            param1=50,
            param2=30,
            minRadius=min_radius_px,
            maxRadius=max_radius_px,
        )

        if circles is not None:
            circles = np.uint16(np.around(circles))
            best_circle = circles[0][0]
            radius_px = float(best_circle[2])
            diameter_px = radius_px * 2.0
            computed_px_per_mm = diameter_px / known_diameter_mm
            self.pixels_per_mm = computed_px_per_mm
            logger.info("Calibrated scale: %.2f px/mm from reference circle", computed_px_per_mm)
            return computed_px_per_mm

        logger.warning("No reference calibration circle found; keeping default %.2f px/mm", self.pixels_per_mm)
        return None

    def pixels_to_mm(self, pixels: float) -> float:
        """Converts linear pixel dimension to millimeters."""
        return float(pixels / self.pixels_per_mm)

    def pixels_to_cm(self, pixels: float) -> float:
        """Converts linear pixel dimension to centimeters."""
        return float(self.pixels_to_mm(pixels) / 10.0)

    def measure_contour(self, contour: np.ndarray) -> Dict[str, Any]:
        """
        Computes metric dimensions and geometry for a segmented nut contour:
        - Area in mm^2
        - Equivalent circular diameter in mm
        - Ellipse fit (major and minor axes) in mm
        - Circularity / Roundness index
        - Sizing category (Small, Medium, Large)
        """
        area_px = float(cv2.contourArea(contour))
        perimeter_px = float(cv2.arcLength(contour, True))

        # Equivalent circular diameter: d = 2 * sqrt(area / pi)
        eq_diameter_px = 2.0 * np.sqrt(max(0.0, area_px) / np.pi)
        eq_diameter_mm = self.pixels_to_mm(eq_diameter_px)
        eq_diameter_cm = self.pixels_to_cm(eq_diameter_px)

        # Ellipse fit for major & minor axes (requires >= 5 points)
        if len(contour) >= 5:
            try:
                ellipse = cv2.fitEllipse(contour)
                (cx, cy), (d1, d2), angle = ellipse
                major_axis_px = max(d1, d2)
                minor_axis_px = min(d1, d2)
            except Exception:
                major_axis_px = eq_diameter_px
                minor_axis_px = eq_diameter_px
        else:
            major_axis_px = eq_diameter_px
            minor_axis_px = eq_diameter_px

        major_axis_mm = self.pixels_to_mm(major_axis_px)
        minor_axis_mm = self.pixels_to_mm(minor_axis_px)

        # Circularity metric: 4 * pi * area / perimeter^2 (1.0 = perfect circle)
        circularity = (4.0 * np.pi * area_px / (perimeter_px**2)) if perimeter_px > 0 else 0.0
        circularity = min(1.0, max(0.0, circularity))

        # Size category binning
        size_category = self.classify_size(eq_diameter_mm)

        return {
            "area_px": area_px,
            "area_mm2": round(area_px / (self.pixels_per_mm**2), 2),
            "diameter_mm": round(eq_diameter_mm, 2),
            "diameter_cm": round(eq_diameter_cm, 3),
            "major_axis_mm": round(major_axis_mm, 2),
            "minor_axis_mm": round(minor_axis_mm, 2),
            "circularity": round(circularity, 3),
            "size_grade": size_category,
        }

    def classify_size(self, diameter_mm: float) -> str:
        """Bins diameter into Small, Medium, Large based on configured limits."""
        small_max = self.size_categories["small"].get("max_diameter_mm", 18.0)
        medium_max = self.size_categories["medium"].get("max_diameter_mm", 22.0)

        if diameter_mm < small_max:
            return "Small"
        elif diameter_mm <= medium_max:
            return "Medium"
        else:
            return "Large"
