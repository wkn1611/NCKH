"""
extraction/ear.py
─────────────────
Calculates the Eye Aspect Ratio (EAR) using fast numpy vector operations.
"""

from typing import List, Tuple
import numpy as np

# Standard MediaPipe FaceMesh eye indices
# Left Eye (subject's right)
# 0: outer corner, 1: top left, 2: top right, 3: inner corner, 4: bottom right, 5: bottom left
LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]

# Right Eye (subject's left)
# 6: inner corner, 7: top right, 8: top left, 9: outer corner, 10: bottom left, 11: bottom right
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]

def calculate_ear(landmarks: List[Tuple[float, float, float]]) -> float:
    """
    Computes the Eye Aspect Ratio (EAR) for both eyes and returns the average.
    
    Uses fast, vectorized numpy.linalg.norm operations to compute the ratios
    without standard math loops, optimizing for edge deployment.
    
    Args:
        landmarks: List of 468 (x, y, z) tuples from FaceMeshResult.
        
    Returns:
        float: Average EAR of the left and right eyes.
    """
    if not landmarks or len(landmarks) < 468:
        return 0.0

    # Extract all 12 points into a single numpy array for batch processing
    # Indices 0-5 are left eye, 6-11 are right eye
    coords = np.array([
        landmarks[33], landmarks[160], landmarks[158], landmarks[133], landmarks[153], landmarks[144],
        landmarks[362], landmarks[385], landmarks[387], landmarks[263], landmarks[373], landmarks[380]
    ], dtype=np.float32)

    # Vectorized computation for both eyes simultaneously:
    # We pair left and right eye points to compute v1, v2, and h in parallel.
    # coords[[1, 7]] means [Left P2, Right P2]
    # coords[[5, 11]] means [Left P6, Right P6], etc.
    
    # Verticals: ||P2 - P6|| and ||P3 - P5||
    v1 = np.linalg.norm(coords[[1, 7]] - coords[[5, 11]], axis=1)
    v2 = np.linalg.norm(coords[[2, 8]] - coords[[4, 10]], axis=1)
    
    # Horizontals: ||P1 - P4||
    h = np.linalg.norm(coords[[0, 6]] - coords[[3, 9]], axis=1)
    
    # Compute EAR for both eyes (avoiding div by zero with 1e-6)
    ears = (v1 + v2) / (2.0 * h + 1e-6)
    
    # Return numerical average of both eyes
    return float(np.mean(ears))
