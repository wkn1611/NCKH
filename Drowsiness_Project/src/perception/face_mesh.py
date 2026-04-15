"""
MediaPipe FaceMesh wrapper — perception layer.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np

logger = logging.getLogger(__name__)

_mp_face_mesh = mp.solutions.face_mesh
_mp_drawing = mp.solutions.drawing_utils
_mp_drawing_styles = mp.solutions.drawing_styles

_MESH_CONFIG: dict = dict(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.65,
    min_tracking_confidence=0.55,
)

_LANDMARK_SPEC = _mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)
_CONNECT_SPEC = _mp_drawing.DrawingSpec(color=(0, 180, 255), thickness=1)

@dataclass
class FaceMeshResult:
    """Immutable result object from FaceMeshDetector."""
    detected: bool = False
    landmarks: List[Tuple[float, float, float]] = field(default_factory=list)
    _raw_landmarks: Any = field(default=None, repr=False)

class FaceMeshDetector:
    """Wrapper for MediaPipe FaceMesh."""

    def __init__(self) -> None:
        self._mesh: Optional[_mp_face_mesh.FaceMesh] = None
        self._init_mesh()

    def _init_mesh(self) -> None:
        try:
            self._mesh = _mp_face_mesh.FaceMesh(**_MESH_CONFIG)
            logger.info("FaceMeshDetector initialized.")
        except Exception as exc:
            raise RuntimeError(f"Cannot load MediaPipe FaceMesh: {exc}") from exc

    def close(self) -> None:
        if self._mesh is not None:
            self._mesh.close()
            self._mesh = None
            logger.info("FaceMeshDetector resources released.")

    def __enter__(self) -> "FaceMeshDetector":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    def process(self, frame_bgr: np.ndarray) -> FaceMeshResult:
        if self._mesh is None:
            return FaceMeshResult()

        if frame_bgr is None or frame_bgr.size == 0:
            return FaceMeshResult()

        frame_rgb: np.ndarray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False

        try:
            mp_result = self._mesh.process(frame_rgb)
        except Exception as exc:
            logger.error("Inference error: %s", exc)
            return FaceMeshResult()
        finally:
            frame_rgb.flags.writeable = True

        if not mp_result.multi_face_landmarks:
            return FaceMeshResult(detected=False)

        raw = mp_result.multi_face_landmarks[0]
        landmarks: List[Tuple[float, float, float]] = [(lm.x, lm.y, lm.z) for lm in raw.landmark]

        return FaceMeshResult(
            detected=True,
            landmarks=landmarks,
            _raw_landmarks=raw,
        )

    @staticmethod
    def draw_mesh(frame_bgr: np.ndarray, result: FaceMeshResult) -> np.ndarray:
        if not result.detected or result._raw_landmarks is None:
            return frame_bgr

        _mp_drawing.draw_landmarks(
            image=frame_bgr,
            landmark_list=result._raw_landmarks,
            connections=_mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=_LANDMARK_SPEC,
            connection_drawing_spec=_CONNECT_SPEC,
        )
        return frame_bgr
