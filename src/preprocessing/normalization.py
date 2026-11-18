"""
Lighting and Color Normalization Module.
Performs color constancy (white balance), CLAHE illumination correction,
and standardized aspect-ratio-preserving letterbox resizing.
"""

import logging
from typing import Tuple, Dict, Any
import cv2
import numpy as np

logger = logging.getLogger(__name__)


class LightingColorNormalizer:
    """
    Normalizes lighting variation, shadows, and color temperature across batches.
    Provides standard dimension resizing with coordinate mapping transforms.
    """

    def __init__(
        self,
        target_size: Tuple[int, int] = (640, 640),
        clahe_clip_limit: float = 2.5,
        clahe_tile_grid_size: Tuple[int, int] = (8, 8),
        enable_white_balance: bool = True,
        white_balance_method: str = "gray_world",
    ):
        self.target_size = target_size
        self.clahe = cv2.createCLAHE(
            clipLimit=clahe_clip_limit, tileGridSize=clahe_tile_grid_size
        )
        self.enable_white_balance = enable_white_balance
        self.wb_method = white_balance_method.lower()

    def normalize(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Executes full normalization sequence:
        1. White balance correction
        2. Contrast-Limited Adaptive Histogram Equalization on L-channel
        3. Letterbox scaling to target dimensions

        Returns:
            Dictionary containing:
                - 'normalized_image': resized and lighting-normalized image
                - 'color_balanced': image before resize (full resolution)
                - 'scale': scale factor used for letterbox
                - 'pad_x', 'pad_y': padding added for letterbox
                - 'original_shape': (height, width)
        """
        orig_h, orig_w = image.shape[:2]

        # 1. White Balance
        wb_img = image.copy()
        if self.enable_white_balance:
            if self.wb_method == "white_patch":
                wb_img = self._white_patch_balance(wb_img)
            else:
                wb_img = self._gray_world_balance(wb_img)

        # 2. Illumination Equalization (LAB color space)
        lab = cv2.cvtColor(wb_img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l_clahe = self.clahe.apply(l)
        lab_clahe = cv2.merge((l_clahe, a, b))
        normalized_full = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)

        # 3. Standardized Letterbox Resizing
        resized, scale, pad_x, pad_y = self._letterbox(normalized_full, self.target_size)

        return {
            "normalized_image": resized,
            "color_balanced": normalized_full,
            "scale": scale,
            "pad_x": pad_x,
            "pad_y": pad_y,
            "original_shape": (orig_h, orig_w),
        }

    def _gray_world_balance(self, image: np.ndarray) -> np.ndarray:
        """
        Applies Gray World white balancing:
        Assumes the average scene reflectance is neutral gray.
        """
        b, g, r = cv2.split(image.astype(np.float32))
        b_mean = np.mean(b) + 1e-5
        g_mean = np.mean(g) + 1e-5
        r_mean = np.mean(r) + 1e-5

        gray_target = (b_mean + g_mean + r_mean) / 3.0

        b = np.clip(b * (gray_target / b_mean), 0, 255)
        g = np.clip(g * (gray_target / g_mean), 0, 255)
        r = np.clip(r * (gray_target / r_mean), 0, 255)

        return cv2.merge([b, g, r]).astype(np.uint8)

    def _white_patch_balance(self, image: np.ndarray, percentile: float = 98.0) -> np.ndarray:
        """
        Applies White Patch Retinex white balance based on top bright percentiles.
        """
        b, g, r = cv2.split(image.astype(np.float32))
        b_max = np.percentile(b, percentile) + 1e-5
        g_max = np.percentile(g, percentile) + 1e-5
        r_max = np.percentile(r, percentile) + 1e-5

        b = np.clip(b * (255.0 / b_max), 0, 255)
        g = np.clip(g * (255.0 / g_max), 0, 255)
        r = np.clip(r * (255.0 / r_max), 0, 255)

        return cv2.merge([b, g, r]).astype(np.uint8)

    @staticmethod
    def _letterbox(
        image: np.ndarray,
        target_size: Tuple[int, int] = (640, 640),
        pad_color: Tuple[int, int, int] = (114, 114, 114),
    ) -> Tuple[np.ndarray, float, int, int]:
        """
        Resizes an image preserving aspect ratio with symmetric border padding.
        """
        target_w, target_h = target_size
        h, w = image.shape[:2]

        scale = min(target_w / w, target_h / h)
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))

        pad_x = (target_w - new_w) // 2
        pad_y = (target_h - new_h) // 2

        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((target_h, target_w, 3), pad_color, dtype=np.uint8)
        canvas[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized

        return canvas, scale, pad_x, pad_y

    @staticmethod
    def unletterbox_box(
        box: Tuple[float, float, float, float],
        scale: float,
        pad_x: int,
        pad_y: int,
        orig_shape: Tuple[int, int],
    ) -> Tuple[int, int, int, int]:
        """
        Transforms bounding box from letterbox coordinates back to original image space.
        """
        x1, y1, x2, y2 = box
        orig_h, orig_w = orig_shape

        orig_x1 = int(round(max(0, (x1 - pad_x) / scale)))
        orig_y1 = int(round(max(0, (y1 - pad_y) / scale)))
        orig_x2 = int(round(min(orig_w, (x2 - pad_x) / scale)))
        orig_y2 = int(round(min(orig_h, (y2 - pad_y) / scale)))

        return orig_x1, orig_y1, orig_x2, orig_y2
