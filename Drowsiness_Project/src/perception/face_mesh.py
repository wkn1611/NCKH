"""
perception/face_mesh.py
───────────────────────
MediaPipe FaceMesh wrapper — perception layer only.

Responsibility:
  Accept a BGR frame → run FaceMesh inference → return FaceMeshResult.
  No geometric math (EAR / MAR / Head Pose) is performed here; that
  lives exclusively in the extraction/ module.

Optimisation notes for Raspberry Pi 4:
  - model_complexity=0  : lightweight graph (~3 ms vs ~10 ms for level 1).
  - refine_landmarks=False : skips iris mesh, saves ~30% compute.
  - max_num_faces=1     : driver use-case — exactly one face expected.
  - writeable=False flag : prevents MediaPipe from defensive-copying the frame.
  - BGR→RGB via cvtColor: cheaper than np.flip on large arrays.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, List, Optional

import cv2
import mediapipe as mp
import numpy as np

logger = logging.getLogger(__name__)

# ── MediaPipe aliases ──────────────────────────────────────────────────────────
_mp_face_mesh     = mp.solutions.face_mesh
_mp_drawing       = mp.solutions.drawing_utils
_mp_drawing_styles = mp.solutions.drawing_styles

# ── FaceMesh tuning ────────────────────────────────────────────────────────────
_MESH_CONFIG: dict = dict(
    static_image_mode        = False, # Video mode: reuses tracking between frames.
    max_num_faces            = 1,     # One driver, one face — no extra overhead.
    refine_landmarks         = True,  # REQUIRED: enables iris landmarks (indices
                                      # 468-477), which are the ground truth points
                                      # for precise EAR calculation in extraction/.
    min_detection_confidence = 0.65,  # Permissive for low-light cabin conditions.
    min_tracking_confidence  = 0.55,  # Keep tracking alive before expensive re-detect.
)

# Drawing specs — defined once, reused every frame (zero allocation in hot path).
_LANDMARK_SPEC = _mp_drawing.DrawingSpec(color=(0, 255, 0),  thickness=1, circle_radius=1)
_CONNECT_SPEC  = _mp_drawing.DrawingSpec(color=(0, 180, 255), thickness=1)


# ── Value object ───────────────────────────────────────────────────────────────

@dataclass
class FaceMeshResult:
    """
    Immutable value object returned by FaceMeshDetector.process().

    Attributes:
        detected  : True if at least one face was found.
        landmarks : 468 (x, y, z) tuples in normalised image coords [0, 1].
                    Empty list when detected=False.
                    Access: landmarks[LANDMARK_IDX] → (x, y, z)
        _raw_landmarks : Internal — raw MediaPipe NormalizedLandmarkList kept
                         solely for draw_mesh(). Do not access directly.
    """
    detected:       bool       = False
    landmarks:      List[tuple] = field(default_factory=list)
    _raw_landmarks: Any        = field(default=None, repr=False)


# ── Detector class ─────────────────────────────────────────────────────────────

class FaceMeshDetector:
    """
    Thin, resource-managed wrapper around MediaPipe FaceMesh.

    Handles BGR→RGB conversion, the writeable-flag optimisation, and
    clean extraction of landmark data into a framework-agnostic result.

    Usage (context manager — preferred):
        with FaceMeshDetector() as detector:
            result = detector.process(frame)
            annotated = FaceMeshDetector.draw_mesh(frame, result)

    Usage (manual):
        detector = FaceMeshDetector()
        result   = detector.process(frame)
        detector.close()
    """

    def __init__(self) -> None:
        self._mesh: Optional[_mp_face_mesh.FaceMesh] = None
        self._init_mesh()

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    def _init_mesh(self) -> None:
        """
        Load the MediaPipe FaceMesh model.

        Raises:
            RuntimeError: If the model cannot be allocated — typically a
                          missing mediapipe installation or out-of-memory on
                          a constrained device.
        """
        try:
            self._mesh = _mp_face_mesh.FaceMesh(**_MESH_CONFIG)
            logger.info(
                "[FaceMeshDetector] Initialised — refine_landmarks=%s (478 pts), "
                "max_faces=%d, det_conf=%.2f, track_conf=%.2f.",
                _MESH_CONFIG["refine_landmarks"],
                _MESH_CONFIG["max_num_faces"],
                _MESH_CONFIG["min_detection_confidence"],
                _MESH_CONFIG["min_tracking_confidence"],
            )
        except Exception as exc:
            raise RuntimeError(
                f"[FaceMeshDetector] Cannot load MediaPipe FaceMesh: {exc}\n"
                "Ensure mediapipe==0.10.5 is installed for your platform (ARM/Mac)."
            ) from exc

    def close(self) -> None:
        """Explicitly release MediaPipe FaceMesh resources."""
        if self._mesh is not None:
            self._mesh.close()
            self._mesh = None
            logger.info("[FaceMeshDetector] Resources released.")

    def __enter__(self) -> "FaceMeshDetector":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    # ── Public API ─────────────────────────────────────────────────────────────

    def process(self, frame_bgr: np.ndarray) -> FaceMeshResult:
        """
        Run FaceMesh inference on a single BGR frame.

        Steps:
          1. Convert BGR → RGB (MediaPipe contract).
          2. Set writeable=False (prevents defensive copy inside MediaPipe).
          3. Run inference.
          4. Extract normalised (x, y, z) tuples for downstream geometry math.
          5. Retain the raw NormalizedLandmarkList for draw_mesh().

        Args:
            frame_bgr: uint8 BGR numpy array from CameraHandler.get_frame().

        Returns:
            FaceMeshResult with detected=True and 468 landmarks if a face
            was found, or detected=False with an empty list otherwise.
        """
        if self._mesh is None:
            logger.warning("[FaceMeshDetector] process() called before init.")
            return FaceMeshResult()

        if frame_bgr is None or frame_bgr.size == 0:
            return FaceMeshResult()

        # ── BGR → RGB (MediaPipe requires RGB input) ───────────────────────
        frame_rgb: np.ndarray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # Writeable=False: signals MediaPipe it can read without copying.
        # Documented optimisation for video pipelines.
        frame_rgb.flags.writeable = False

        try:
            mp_result = self._mesh.process(frame_rgb)
        except Exception as exc:
            logger.error("[FaceMeshDetector] Inference error: %s", exc)
            return FaceMeshResult()
        finally:
            frame_rgb.flags.writeable = True  # Restore for safety.

        if not mp_result.multi_face_landmarks:
            return FaceMeshResult(detected=False)

        # ── Extract first face (max_num_faces=1 guarantees exactly one) ────
        raw = mp_result.multi_face_landmarks[0]

        # Convert to plain Python tuples — isolates MediaPipe proto types from
        # the rest of the pipeline (extraction/ only sees (x, y, z) floats).
        landmarks: List[tuple] = [(lm.x, lm.y, lm.z) for lm in raw.landmark]

        return FaceMeshResult(
            detected=True,
            landmarks=landmarks,
            _raw_landmarks=raw,  # Kept for draw_mesh(); not for geometry math.
        )

    # ── Visualisation (perception-layer helper) ────────────────────────────────

    @staticmethod
    def draw_mesh(frame_bgr: np.ndarray, result: FaceMeshResult) -> np.ndarray:
        """
        Draw the face mesh tesselation overlay directly onto the frame.

        Mutates frame_bgr in-place and returns it for convenient chaining.
        Only draws the FaceMesh tesselation (not iris / attention mesh) since
        refine_landmarks=False was used during inference.

        Args:
            frame_bgr : The BGR frame to annotate (from CameraHandler).
            result    : FaceMeshResult returned by process().

        Returns:
            The same frame_bgr array, annotated in-place.
        """
        if not result.detected or result._raw_landmarks is None:
            return frame_bgr

        _mp_drawing.draw_landmarks(
            image           = frame_bgr,
            landmark_list   = result._raw_landmarks,
            connections     = _mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec  = _LANDMARK_SPEC,
            connection_drawing_spec = _CONNECT_SPEC,
        )
        return frame_bgr
