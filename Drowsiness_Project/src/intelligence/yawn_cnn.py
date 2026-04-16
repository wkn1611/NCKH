"""
TFLite CNN inference for yawn detection via mouth ROI crop-and-classify.
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# MediaPipe FaceMesh mouth region landmark indices.
# Selected to tightly bound the outer lip contour.
_MOUTH_IDX = [0, 13, 14, 17, 37, 39, 40, 61, 78, 80,
              81, 82, 84, 87, 88, 91, 95, 146, 178, 181,
              185, 191, 267, 269, 270, 291, 308, 310, 311,
              312, 314, 317, 318, 321, 324, 375, 402, 405,
              409, 415]

_INPUT_SIZE:     int   = 64
_YAWN_THRESHOLD: float = 0.65
_PAD_RATIO:      float = 0.20


class YawnDetectorCNN:
    """
    Crop-and-classify yawn detector backed by a quantized TFLite model.

    Crops the mouth ROI from the original BGR frame using MediaPipe landmark
    bounds, resizes it to (_INPUT_SIZE × _INPUT_SIZE), normalizes to [0,1],
    and runs single-pass TFLite inference.
    """

    def __init__(self, model_path: Path) -> None:
        """
        Args:
            model_path: Absolute path to the .tflite model file.

        Raises:
            RuntimeError: If the model cannot be loaded or tensors allocated.
        """
        self._interpreter = self._load_interpreter(model_path)
        self._input_details  = self._interpreter.get_input_details()
        self._output_details = self._interpreter.get_output_details()
        logger.info("[YawnDetectorCNN] Model loaded: %s", model_path.name)

    # ── Setup ──────────────────────────────────────────────────────────────────

    @staticmethod
    def _load_interpreter(model_path: Path):
        """Loads TFLite interpreter, preferring tflite_runtime on edge devices."""
        try:
            from tflite_runtime.interpreter import Interpreter
            logger.info("[YawnDetectorCNN] Using tflite_runtime.")
        except ImportError:
            from tensorflow.lite.python.interpreter import Interpreter
            logger.info("[YawnDetectorCNN] Falling back to tensorflow.lite.")

        try:
            interpreter = Interpreter(model_path=str(model_path))
            interpreter.allocate_tensors()
            return interpreter
        except Exception as exc:
            raise RuntimeError(
                f"[YawnDetectorCNN] Failed to load {model_path}: {exc}"
            ) from exc

    # ── Public API ─────────────────────────────────────────────────────────────

    def predict_yawn(
        self,
        frame_bgr: np.ndarray,
        landmarks: List[Tuple[float, float, float]],
    ) -> bool:
        """
        Crops the mouth ROI and runs yawn classification.

        Args:
            frame_bgr : Original BGR frame from VideoStream.
            landmarks : 468/478 normalized (x, y, z) tuples from FaceMeshResult.

        Returns:
            True if yawn probability exceeds threshold, False otherwise.
        """
        roi = self._crop_mouth(frame_bgr, landmarks)
        if roi is None:
            return False
        return self._infer(roi)

    # ── Internals ──────────────────────────────────────────────────────────────

    def _crop_mouth(
        self,
        frame: np.ndarray,
        landmarks: List[Tuple[float, float, float]],
    ) -> Optional[np.ndarray]:
        """Extracts and returns the padded mouth bounding-box crop."""
        h, w = frame.shape[:2]

        xs = [landmarks[i][0] * w for i in _MOUTH_IDX]
        ys = [landmarks[i][1] * h for i in _MOUTH_IDX]

        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)

        pad_x = (x_max - x_min) * _PAD_RATIO
        pad_y = (y_max - y_min) * _PAD_RATIO

        x1 = max(0, int(x_min - pad_x))
        y1 = max(0, int(y_min - pad_y))
        x2 = min(w, int(x_max + pad_x))
        y2 = min(h, int(y_max + pad_y))

        if x2 <= x1 or y2 <= y1:
            return None

        return frame[y1:y2, x1:x2]

    def _infer(self, roi: np.ndarray) -> bool:
        """Preprocesses the ROI, runs inference, and returns the yawn decision."""
        try:
            resized = cv2.resize(roi, (_INPUT_SIZE, _INPUT_SIZE))
            tensor  = (resized.astype(np.float32) / 255.0)[np.newaxis, ...]

            self._interpreter.set_tensor(self._input_details[0]["index"], tensor)
            self._interpreter.invoke()

            prob = float(self._interpreter.get_tensor(
                self._output_details[0]["index"]
            )[0][0])
            return prob > _YAWN_THRESHOLD
        except Exception as exc:
            logger.warning("[YawnDetectorCNN] Inference error: %s", exc)
            return False
