"""
intelligence/pose.py
────────────────────
Head Pose Estimation using MediaPipe landmarks and solvePnP.
"""

import cv2
import numpy as np
from typing import List, Tuple

class HeadPoseEstimator:
    """
    Computes 3D head rotation (Pitch, Yaw, Roll) by mapping 2D facial 
    landmarks onto a standard 3D generic face model.
    """

    def __init__(self) -> None:
        # Standard generic 3D face model points.
        # Order: Nose tip, Chin, Left eye (subject right), Right eye (subject left),
        # Left mouth (subject right), Right mouth (subject left)
        self.model_points = np.array([
            (0.0, 0.0, 0.0),             # 1: Nose tip
            (0.0, -330.0, -65.0),        # 152: Chin
            (-225.0, 170.0, -135.0),     # 33: Outer corner, Subject's right eye
            (225.0, 170.0, -135.0),      # 263: Outer corner, Subject's left eye
            (-150.0, -150.0, -125.0),    # 61: Outer corner, Subject's right mouth
            (150.0, -150.0, -125.0)      # 291: Outer corner, Subject's left mouth
        ], dtype=np.float64)
        
        self._camera_matrix = None
        self._dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    def _init_camera_matrix(self, frame_shape: Tuple[int, int]) -> None:
        if self._camera_matrix is None:
            h, w = frame_shape
            focal_length = w  # Approximation
            center = (w / 2.0, h / 2.0)
            self._camera_matrix = np.array([
                [focal_length, 0, center[0]],
                [0, focal_length, center[1]],
                [0, 0, 1]
            ], dtype=np.float64)

    def estimate(
        self, 
        landmarks: List[Tuple[float, float, float]], 
        frame_shape: Tuple[int, int]
    ) -> Tuple[float, float, float]:
        """
        Calculates Euler angles (Pitch, Yaw, Roll) in degrees.
        
        Args:
            landmarks: 468 landmarks mapped to [0,1] from FaceMesh.
            frame_shape: (height, width) of the image.
            
        Returns:
            (pitch, yaw, roll) floats.
        """
        self._init_camera_matrix(frame_shape)
        h, w = frame_shape
        
        # Extract the 6 key 2D points into absolute pixel coordinates
        image_points = np.array([
            (landmarks[1][0] * w, landmarks[1][1] * h),
            (landmarks[152][0] * w, landmarks[152][1] * h),
            (landmarks[33][0] * w, landmarks[33][1] * h),
            (landmarks[263][0] * w, landmarks[263][1] * h),
            (landmarks[61][0] * w, landmarks[61][1] * h),
            (landmarks[291][0] * w, landmarks[291][1] * h)
        ], dtype=np.float64)
        
        success, rvec, tvec = cv2.solvePnP(
            self.model_points, 
            image_points, 
            self._camera_matrix, 
            self._dist_coeffs, 
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        
        if not success:
            return 0.0, 0.0, 0.0
            
        # Decompose robustly
        rmat, _ = cv2.Rodrigues(rvec)
        proj_matrix = np.hstack((rmat, tvec))
        _, _, _, _, _, _, euler = cv2.decomposeProjectionMatrix(proj_matrix)
        
        pitch, yaw, roll = euler.flatten()
        return float(pitch), float(yaw), float(roll)

    def is_looking_forward(self, yaw: float, pitch: float) -> bool:
        """
        Validates if the user's head pose constitutes 'looking forward'.
        """
        return -15.0 <= yaw <= 15.0 and -15.0 <= pitch <= 15.0
