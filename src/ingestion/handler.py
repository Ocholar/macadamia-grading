"""
Image Ingestion Handler.
Provides resilient ingestion from file paths, byte streams, base64 data, and live video capture.
"""

import base64
import logging
import os
from typing import Optional, Tuple, Union
import cv2
import numpy as np

logger = logging.getLogger(__name__)


class ImageIngestionHandler:
    """
    Ingests and validates images for the macadamia grading pipeline.
    Ensures images are returned as standard OpenCV BGR numpy arrays.
    """

    def __init__(self, min_width: int = 100, min_height: int = 100):
        self.min_width = min_width
        self.min_height = min_height

    def load_from_path(self, file_path: str) -> np.ndarray:
        """Loads an image from filesystem path."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Image file not found: {file_path}")

        image = cv2.imread(file_path, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Failed to decode image from path: {file_path}")

        return self.validate_image(image)

    def load_from_bytes(self, image_bytes: bytes) -> np.ndarray:
        """Decodes an image from raw bytes (e.g. from FastAPI or Streamlit)."""
        if not image_bytes:
            raise ValueError("Empty byte sequence provided for image ingestion.")

        np_arr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Failed to decode image from provided byte stream.")

        return self.validate_image(image)

    def load_from_base64(self, b64_string: str) -> np.ndarray:
        """Decodes an image from base64 string (supports data URI schemes)."""
        if "," in b64_string:
            # Strip data URI prefix like 'data:image/jpeg;base64,'
            b64_string = b64_string.split(",", 1)[1]

        image_bytes = base64.b64decode(b64_string)
        return self.load_from_bytes(image_bytes)

    def capture_from_camera(self, camera_index: int = 0, timeout_frames: int = 5) -> np.ndarray:
        """Captures a single frame from an attached webcam/USB video device."""
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            raise RuntimeError(f"Unable to access camera device index: {camera_index}")

        try:
            # Warm up sensor for auto-exposure & white balance stabilization
            for _ in range(timeout_frames):
                ret, frame = cap.read()
                if not ret:
                    raise RuntimeError("Failed to read video frame from camera stream.")
            return self.validate_image(frame)
        finally:
            cap.release()

    def validate_image(self, image: np.ndarray) -> np.ndarray:
        """Validates channels, dimensionality, and non-empty status."""
        if image is None:
            raise ValueError("Image array is None.")

        if not isinstance(image, np.ndarray):
            raise TypeError(f"Expected numpy.ndarray, got {type(image)}")

        if len(image.shape) == 2:
            # Grayscale to BGR
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif len(image.shape) == 3:
            if image.shape[2] == 4:
                # RGBA/BGRA to BGR
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
            elif image.shape[2] != 3:
                raise ValueError(f"Unsupported number of image channels: {image.shape[2]}")
        else:
            raise ValueError(f"Invalid image dimensions: {image.shape}")

        h, w = image.shape[:2]
        if w < self.min_width or h < self.min_height:
            raise ValueError(
                f"Image resolution {w}x{h} is lower than required minimum {self.min_width}x{self.min_height}"
            )

        return image

    @staticmethod
    def to_base64(image: np.ndarray, ext: str = ".jpg", quality: int = 90) -> str:
        """Encodes an OpenCV image to base64 data string."""
        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), quality] if ext.lower() in [".jpg", ".jpeg"] else []
        ret, buf = cv2.imencode(ext, image, encode_params)
        if not ret:
            raise RuntimeError("Failed to encode image to buffer.")
        b64 = base64.b64encode(buf).decode("utf-8")
        mime = "image/jpeg" if ext.lower() in [".jpg", ".jpeg"] else "image/png"
        return f"data:{mime};base64,{b64}"

    @staticmethod
    def to_bytes(image: np.ndarray, ext: str = ".jpg", quality: int = 90) -> bytes:
        """Encodes an OpenCV image to raw bytes."""
        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), quality] if ext.lower() in [".jpg", ".jpeg"] else []
        ret, buf = cv2.imencode(ext, image, encode_params)
        if not ret:
            raise RuntimeError("Failed to encode image to bytes.")
        return buf.tobytes()
