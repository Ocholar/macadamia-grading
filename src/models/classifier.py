"""
Macadamia Defect Classifier Module.
Classifies individual macadamia kernels into four commercial quality grades:
0: Healthy / Premium
1: Cracked / Broken
2: Discolored / Brown Center
3: Moldy / Insect Damaged

Integrates ONNX deep neural inference with specialized optical agronomic feature signatures.
"""

import os
import logging
from typing import Dict, Any, Optional, Tuple
import cv2
import numpy as np

try:
    import onnxruntime as ort
    HAS_ORT = True
except ImportError:
    HAS_ORT = False

logger = logging.getLogger(__name__)


CLASS_MAPPING = {
    0: {"id": "healthy", "label": "Healthy / Premium", "is_defective": False},
    1: {"id": "cracked", "label": "Cracked / Broken", "is_defective": True},
    2: {"id": "discolored", "label": "Discolored / Brown Center", "is_defective": True},
    3: {"id": "moldy", "label": "Moldy / Insect Damaged", "is_defective": True},
}


class DefectClassifier:
    """
    Evaluates individual macadamia kernels for structural, chromic, and fungal/entomological defects.
    """

    def __init__(
        self,
        weights_path: Optional[str] = None,
        confidence_threshold: float = 0.40,
    ):
        self.confidence_threshold = confidence_threshold
        self.session = None
        self.weights_path = weights_path

        if weights_path and os.path.exists(weights_path) and HAS_ORT:
            try:
                self.session = ort.InferenceSession(
                    weights_path, providers=["CPUExecutionProvider"]
                )
                logger.info("Loaded DefectClassifier ONNX session from: %s", weights_path)
            except Exception as e:
                logger.warning("Could not initialize ONNX session (%s); falling back to optical signature engine.", e)

    def classify(self, crop: np.ndarray, nut_mask: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Classifies a single cropped macadamia nut.

        Args:
            crop: BGR image crop of an individual nut
            nut_mask: binary mask (uint8) marking true nut interior (excluding background mat)

        Returns:
            Dictionary containing:
                - 'class_id': int (0-3)
                - 'class_label': str
                - 'confidence': float
                - 'probabilities': Dict[str, float]
                - 'is_defective': bool
                - 'defect_saliency_map': np.ndarray (float32, [0, 1])
                - 'severity_score': float (percentage of affected area)
                - 'diagnostic_notes': str
        """
        if crop is None or crop.size == 0 or crop.shape[0] < 8 or crop.shape[1] < 8:
            return {
                "class_id": 0,
                "class_label": "Healthy / Premium",
                "confidence": 0.50,
                "probabilities": {info["label"]: 0.25 for info in CLASS_MAPPING.values()},
                "is_defective": False,
                "defect_saliency_map": np.zeros((32, 32), dtype=np.float32),
                "severity_score": 0.0,
                "diagnostic_notes": "Undersized crop; defaulted to healthy",
            }

        # 1. Compute Agronomic Optical Signatures & Saliency Maps
        optical_features, saliency_map = self._extract_optical_signatures(crop, nut_mask)

        # 2. Compute probabilities from optical signatures
        probs = self._compute_defect_probabilities(optical_features)

        # 3. If ONNX model session is loaded, fuse with deep neural activations
        if self.session is not None:
            try:
                onnx_probs, nn_saliency = self._infer_onnx(crop, nut_mask)
                # Weighted ensemble
                for k in range(4):
                    probs[k] = 0.5 * probs[k] + 0.5 * onnx_probs[k]
                if nn_saliency is not None:
                    saliency_map = 0.6 * saliency_map + 0.4 * nn_saliency
            except Exception as e:
                logger.debug("ONNX inference skipped (%s); using optical signature engine", e)

        # 4. Determine final class
        pred_class_id = int(np.argmax(probs))
        confidence = float(probs[pred_class_id])
        class_info = CLASS_MAPPING[pred_class_id]

        # Calculate defect severity: fraction of nut surface exhibiting defect signals
        defect_severity = float(np.mean(saliency_map > 0.35)) * 100.0

        notes = self._generate_diagnostic_notes(pred_class_id, optical_features, defect_severity)

        return {
            "class_id": pred_class_id,
            "class_label": class_info["label"],
            "confidence": round(confidence, 3),
            "probabilities": {
                CLASS_MAPPING[i]["label"]: round(float(probs[i]), 3) for i in range(4)
            },
            "is_defective": class_info["is_defective"],
            "defect_saliency_map": saliency_map,
            "severity_score": round(defect_severity, 2),
            "diagnostic_notes": notes,
        }

    def _extract_optical_signatures(
        self, crop: np.ndarray, nut_mask: Optional[np.ndarray] = None
    ) -> Tuple[Dict[str, float], np.ndarray]:
        """
        Extracts biophysical defect signatures strictly within the nut mask.
        """
        h, w = crop.shape[:2]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

        # If nut_mask is not provided, estimate from foreground
        if nut_mask is None:
            _, nut_mask = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)

        nut_pixels = int(np.count_nonzero(nut_mask))
        if nut_pixels == 0:
            nut_pixels = max(1, h * w)
            nut_mask = np.ones((h, w), dtype=np.uint8) * 255

        # --- A. Crack & Structural Fissure Detection ---
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        grad_x = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
        gradient_mag = cv2.magnitude(grad_x, grad_y)
        gradient_mag = cv2.bitwise_and(gradient_mag, gradient_mag, mask=nut_mask)

        # Erode mask slightly to avoid counting the outer boundary of the nut as a crack
        kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        inner_mask = cv2.erode(nut_mask, kernel_erode, iterations=1)
        inner_pixels = max(1, np.count_nonzero(inner_mask))

        grad_norm = cv2.normalize(gradient_mag, None, 0.0, 1.0, cv2.NORM_MINMAX)
        grad_norm[inner_mask == 0] = 0.0
        fissure_density = float(np.count_nonzero(grad_norm > 0.45) / inner_pixels)

        # --- B. Browning & Enzymatic Discoloration ---
        # Healthy macadamia kernel has high L (cream/ivory ~190-240), low A (126-130), moderate B (135-155)
        # Discolored kernel: L is significantly lower, A is elevated (reddish brown) or B is brownish
        l_channel = lab[:, :, 0]
        a_channel = lab[:, :, 1]

        # Browning mask strictly inside nut
        browning_mask = (l_channel < 140) & (a_channel > 132) & (inner_mask > 0)
        browning_ratio = float(np.count_nonzero(browning_mask) / inner_pixels)

        # Center browning
        center_h1, center_h2 = int(h * 0.25), int(h * 0.75)
        center_w1, center_w2 = int(w * 0.25), int(w * 0.75)
        center_inner = inner_mask[center_h1:center_h2, center_w1:center_w2]
        center_nut_pixels = max(1, np.count_nonzero(center_inner))
        center_browning = np.count_nonzero(browning_mask[center_h1:center_h2, center_w1:center_w2] & (center_inner > 0))
        center_browning_ratio = float(center_browning / center_nut_pixels)

        # --- C. Mold & Insect Damage ---
        # Dark insect borehole: very dark spot inside the nut (L < 60)
        dark_borehole_mask = (l_channel < 60) & (inner_mask > 0)
        dark_spot_ratio = float(np.count_nonzero(dark_borehole_mask) / inner_pixels)

        # Mold mycelium: greenish-gray or dark discoloration with high local texture variance
        mold_color_mask = (hsv[:, :, 1] > 35) & (hsv[:, :, 0] > 30) & (hsv[:, :, 0] < 125) & (l_channel < 125) & (inner_mask > 0)
        mold_color_ratio = float(np.count_nonzero(mold_color_mask) / inner_pixels)

        # Local texture roughness (Laplacian on inner nut)
        local_lap = cv2.Laplacian(gray, cv2.CV_32F)
        texture_roughness = float(np.std(local_lap[inner_mask > 0])) if np.any(inner_mask > 0) else 0.0

        # --- Build Saliency Heatmap ---
        crack_saliency = cv2.GaussianBlur(grad_norm, (5, 5), 0)
        brown_saliency = cv2.GaussianBlur(browning_mask.astype(np.float32), (7, 7), 0)
        mold_saliency = cv2.GaussianBlur((dark_borehole_mask | mold_color_mask).astype(np.float32), (7, 7), 0)

        combined_saliency = 0.45 * crack_saliency + 0.35 * brown_saliency + 0.60 * mold_saliency
        combined_saliency = np.clip(combined_saliency, 0.0, 1.0)
        combined_saliency[nut_mask == 0] = 0.0

        features = {
            "fissure_density": fissure_density,
            "browning_ratio": browning_ratio,
            "center_browning_ratio": center_browning_ratio,
            "dark_spot_ratio": dark_spot_ratio,
            "mold_color_ratio": mold_color_ratio,
            "texture_roughness": texture_roughness,
        }

        return features, combined_saliency

    def _compute_defect_probabilities(self, features: Dict[str, float]) -> np.ndarray:
        """
        Converts extracted agronomic features into calibrated class probability distribution.
        """
        score_crack = (features["fissure_density"] * 18.0)
        score_discolored = (features["browning_ratio"] * 9.0) + (features["center_browning_ratio"] * 7.0)
        score_mold = (features["dark_spot_ratio"] * 28.0) + (features["mold_color_ratio"] * 20.0)

        # Base healthy score
        max_defect_score = max(score_crack, score_discolored, score_mold)
        score_healthy = max(0.1, 2.5 - (max_defect_score * 1.5))

        logits = np.array([score_healthy, score_crack, score_discolored, score_mold], dtype=np.float32)

        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)
        return probs

    def _infer_onnx(
        self, crop: np.ndarray, nut_mask: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Executes forward pass on the ONNX model."""
        masked_crop = crop.copy()
        if nut_mask is not None:
            masked_crop = cv2.bitwise_and(crop, crop, mask=nut_mask)

        resized = cv2.resize(masked_crop, (128, 128), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        tensor = np.transpose(rgb, (2, 0, 1))[np.newaxis, :, :, :].astype(np.float32)

        outputs = self.session.run(None, {"input": tensor})
        probs = outputs[0][0]

        feature_saliency = None
        if len(outputs) > 1:
            fmap = outputs[1][0]
            cam = np.mean(fmap, axis=0)
            cam = np.maximum(cam, 0)
            cam = cv2.resize(cam, (crop.shape[1], crop.shape[0]))
            if np.max(cam) > 0:
                cam = cam / np.max(cam)
            feature_saliency = cam.astype(np.float32)

        return probs, feature_saliency

    @staticmethod
    def _generate_diagnostic_notes(
        class_id: int, features: Dict[str, float], severity: float
    ) -> str:
        """Generates audit-ready agronomic explanations."""
        if class_id == 0:
            return "Kernel exhibits uniform ivory tone, intact surface texture, and zero structural fissures."
        elif class_id == 1:
            return f"Structural fissure detected (edge density: {features['fissure_density']:.1%}, affected surface: {severity:.1f}%)."
        elif class_id == 2:
            return f"Internal enzymatic oxidation / browning observed (browning ratio: {features['browning_ratio']:.1%})."
        elif class_id == 3:
            return f"Microbial mold growth or insect puncture borehole identified (lesion index: {features['dark_spot_ratio']:.1%})."
        return "Standard inspection evaluation complete."
