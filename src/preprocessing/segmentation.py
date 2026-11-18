"""
Background Segmentation Module.
Isolates macadamia nuts from standardized background mats (blue, green, black, white, or auto-detected).
Includes morphological refinement and Watershed separation for touching nuts.
"""

import logging
from typing import List, Optional, Tuple, Dict, Any
import cv2
import numpy as np

logger = logging.getLogger(__name__)


class BackgroundSegmenter:
    """
    Automated background subtraction and foreground nut segmentation.
    Supports standardized inspection mats (blue, green, dark, light) and auto-detection.
    """

    def __init__(
        self,
        mat_type: str = "auto",
        morphology_kernel_size: int = 5,
        min_nut_area_px: int = 400,
        max_nut_area_px: int = 150000,
    ):
        self.mat_type = mat_type.lower()
        self.kernel_size = morphology_kernel_size
        self.min_nut_area = min_nut_area_px
        self.max_nut_area = max_nut_area_px

    def segment(
        self, image: np.ndarray, mat_type_override: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Segments macadamia nuts from the background mat.

        Args:
            image: BGR input image
            mat_type_override: optional override for mat_type ("auto", "blue_mat", "green_mat", "dark_mat", "light_mat")

        Returns:
            Dictionary containing:
                - 'binary_mask': binary mask (255=nuts, 0=background)
                - 'foreground': masked BGR image with background blacked out
                - 'contours': list of contours for individual detected nut regions
                - 'segmented_components': labeled component mask from watershed
        """
        effective_mat = (mat_type_override or self.mat_type).lower()
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

        # 1. Compute initial background mask based on mat type
        if effective_mat == "blue_mat":
            # Typical blue inspection mat: Hue 95-135, Saturation > 50
            mat_mask = cv2.inRange(hsv, np.array([90, 45, 40]), np.array([140, 255, 255]))
            nut_mask = cv2.bitwise_not(mat_mask)
        elif effective_mat == "green_mat":
            # Typical green inspection mat: Hue 35-85, Saturation > 45
            mat_mask = cv2.inRange(hsv, np.array([35, 45, 40]), np.array([85, 255, 255]))
            nut_mask = cv2.bitwise_not(mat_mask)
        elif effective_mat == "dark_mat":
            # Dark/black mat: Low value
            _, nut_mask = cv2.threshold(lab[:, :, 0], 55, 255, cv2.THRESH_BINARY)
        elif effective_mat == "light_mat":
            # White/light mat: High value
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            _, nut_mask = cv2.threshold(gray, 215, 255, cv2.THRESH_BINARY_INV)
        else:
            # Auto-detection: Sample border pixels to identify background color
            nut_mask = self._auto_segment_border_background(image, hsv, lab)

        # 2. Morphological Cleanup (Opening removes salt noise, Closing fills internal nut textures)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.kernel_size, self.kernel_size))
        opened = cv2.morphologyEx(nut_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=2)

        # Fill internal holes inside nuts
        contour_fill = np.zeros_like(closed)
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area >= self.min_nut_area:
                cv2.drawContours(contour_fill, [cnt], -1, 255, thickness=cv2.FILLED)

        # 3. Watershed Separation for Touching Nuts
        watershed_mask, valid_contours = self._apply_watershed(image, contour_fill)

        # 4. Generate clean foreground image
        foreground = cv2.bitwise_and(image, image, mask=watershed_mask)

        return {
            "binary_mask": watershed_mask,
            "foreground": foreground,
            "contours": valid_contours,
            "total_contours": len(valid_contours),
        }

    def _auto_segment_border_background(
        self, image: np.ndarray, hsv: np.ndarray, lab: np.ndarray
    ) -> np.ndarray:
        """
        Samples the perimeter of the image to determine background color profile,
        then thresholds based on Euclidean color distance in LAB space.
        """
        h, w = image.shape[:2]
        border_width = max(6, min(h, w) // 40)

        # Collect border pixels
        top = lab[:border_width, :]
        bottom = lab[-border_width:, :]
        left = lab[:, :border_width]
        right = lab[:, -border_width:]

        border_pixels = np.vstack([
            top.reshape(-1, 3),
            bottom.reshape(-1, 3),
            left.reshape(-1, 3),
            right.reshape(-1, 3),
        ])

        bg_mean = np.median(border_pixels, axis=0)
        bg_std = np.std(border_pixels, axis=0)

        # Compute color distance from background
        diff = lab.astype(np.float32) - bg_mean.astype(np.float32)
        dist = np.sqrt(np.sum(diff**2, axis=2))

        # Threshold distance: areas significantly different from background are foreground
        threshold_val = max(22.0, float(np.mean(bg_std) * 2.8))
        _, nut_mask = cv2.threshold(dist.astype(np.uint8), int(threshold_val), 255, cv2.THRESH_BINARY)

        # Combine with Otsu threshold on grayscale gradient for edge sharpness
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # If background is bright, invert otsu
        if bg_mean[0] > 128:
            otsu = cv2.bitwise_not(otsu)

        # Logical OR to combine color contrast and intensity contrast
        combined = cv2.bitwise_or(nut_mask, otsu)
        return combined

    def _apply_watershed(
        self, image: np.ndarray, binary_mask: np.ndarray
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Separates touching macadamia kernels using distance transform and watershed.
        """
        # Distance transform
        dist_transform = cv2.distanceTransform(binary_mask, cv2.DIST_L2, 5)

        # Sure foreground area (centers of nuts)
        sure_fg_thresh = 0.38 * dist_transform.max() if dist_transform.max() > 0 else 0
        _, sure_fg = cv2.threshold(dist_transform, sure_fg_thresh, 255, 0)
        sure_fg = np.uint8(sure_fg)

        # Unknown boundary region
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        sure_bg = cv2.dilate(binary_mask, kernel, iterations=2)
        unknown = cv2.subtract(sure_bg, sure_fg)

        # Marker labeling
        _, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0

        # Run watershed
        markers_ws = cv2.watershed(image, markers)

        # Output mask where markers > 1 (excluding boundary markers == -1 and background == 1)
        separated_mask = np.zeros_like(binary_mask)
        separated_mask[markers_ws > 1] = 255

        # Filter contours by size criteria
        raw_contours, _ = cv2.findContours(separated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid_contours = []
        final_mask = np.zeros_like(binary_mask)

        for cnt in raw_contours:
            area = cv2.contourArea(cnt)
            if self.min_nut_area <= area <= self.max_nut_area:
                # Shape filter: Macadamias have circular/oval shape (circularity > 0.35)
                perimeter = cv2.arcLength(cnt, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * (area / (perimeter * perimeter))
                    if circularity >= 0.30:
                        valid_contours.append(cnt)
                        cv2.drawContours(final_mask, [cnt], -1, 255, thickness=cv2.FILLED)

        return final_mask, valid_contours
