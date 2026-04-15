"""
perception/__init__.py
──────────────────────
Public API for the perception module.

Downstream modules import from here, not from submodules directly:
    from perception import CameraHandler, FaceMeshDetector, FaceMeshResult
"""

from .camera import VideoStream, CameraHandler
from .face_mesh import FaceMeshDetector, FaceMeshResult

__all__ = [
    "VideoStream",
    "CameraHandler",
    "FaceMeshDetector",
    "FaceMeshResult",
]
