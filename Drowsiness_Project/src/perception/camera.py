"""
Threaded video capture for zero-latency frame delivery.
"""

import threading
import logging
import cv2
import numpy as np
from typing import Optional

logger = logging.getLogger(__name__)

_WIDTH  = 480
_HEIGHT = 320
_FPS    = 30


class VideoStream:
    """
    Background-threaded capture that ensures the main loop always reads
    the freshest available frame without blocking on sensor I/O.
    """

    def __init__(self, src: int = 0) -> None:
        self._src = src
        self._cap: Optional[cv2.VideoCapture] = None
        self._frame: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self) -> "VideoStream":
        """Opens the device and starts the background capture thread."""
        self._cap = cv2.VideoCapture(self._src)

        if not self._cap.isOpened():
            raise RuntimeError(
                f"[VideoStream] Cannot open camera at index {self._src}."
            )

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  _WIDTH)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, _HEIGHT)
        self._cap.set(cv2.CAP_PROP_FPS,          _FPS)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)
        self._cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

        ret, frame = self._cap.read()
        if not ret:
            self._cap.release()
            raise RuntimeError(
                f"[VideoStream] Camera {self._src} opened but failed warm-up read."
            )

        self._frame = frame
        self._running = True
        self._thread = threading.Thread(
            target=self._update, name=f"vs-{self._src}", daemon=True
        )
        self._thread.start()

        actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logger.info("[VideoStream] Camera %d ready — %dx%d.", self._src, actual_w, actual_h)
        return self

    def _update(self) -> None:
        """Continuously grabs frames; runs on the background daemon thread."""
        while self._running:
            try:
                ret, frame = self._cap.read()
                if not ret:
                    logger.warning("[VideoStream] Dropped frame (ret=False).")
                    continue
                with self._lock:
                    self._frame = frame
            except Exception as exc:
                logger.error("[VideoStream] Capture error: %s. Stopping.", exc)
                self._running = False
                break

    def read(self) -> Optional[np.ndarray]:
        """Returns the most recent frame. Non-blocking."""
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    def stop(self) -> None:
        """Stops the capture thread and releases the device."""
        self._running = False
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        logger.info("[VideoStream] Camera %d released.", self._src)

    def __enter__(self) -> "VideoStream":
        return self.start()

    def __exit__(self, *_) -> None:
        self.stop()


# Keep CameraHandler as an alias for backward-compatibility with __init__.py
CameraHandler = VideoStream