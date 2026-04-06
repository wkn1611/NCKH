"""
perception/camera.py
────────────────────
OpenCV camera handler — optimised for Raspberry Pi 4 edge deployment.

Design philosophy (Senior Embedded):
  - Background thread decouples blocking I/O from the main inference loop.
    On the Pi, cv2.VideoCapture.read() stalls until the sensor delivers a
    frame. Without threading, the main loop pays that wait on every iteration,
    making 15 FPS impossible when inference itself costs ~40–60 ms.
  - CAP_PROP_BUFFERSIZE=1 ensures get_frame() always returns the LATEST frame,
    not one sitting stale in OpenCV's internal queue.
  - MJPG FOURCC offloads JPEG encode/decode to the camera chip, freeing
    precious Pi CPU cycles for MediaPipe and TFLite.
"""

import logging
import threading
from typing import Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ── Capture defaults ───────────────────────────────────────────────────────────
# Resolution locked to 640×480:
#   - High enough for MediaPipe landmark accuracy on a human face.
#   - Low enough that MediaPipe + TFLite inference fits inside the 20 FPS
#     budget on Raspberry Pi 4 (each frame ~50 ms total pipeline budget).
# FOURCC=MJPG: the Pi Camera Module v2 hardware encodes JPEG on-chip, cutting
#   USB/CSI bandwidth and freeing ARM CPU cycles for inference.
_DEFAULT_WIDTH:  int = 640    # Locked — do not increase without re-profiling.
_DEFAULT_HEIGHT: int = 480    # Locked — do not increase without re-profiling.
_DEFAULT_FPS:    int = 30     # Request 30 from sensor; pipeline targets 20 FPS.
_FOURCC:         str = "MJPG" # Hardware JPEG stream — critical on Pi 4.


class CameraHandler:
    """
    Non-blocking, threaded OpenCV camera handler.

    A background daemon thread continuously grabs the latest frame so that
    the main loop's call to get_frame() is always instantaneous — it never
    blocks on sensor I/O.

    Typical usage (context manager — preferred):
    ─────────────────────────────────────────────
        cam = CameraHandler(src=0)
        cam.open()
        try:
            frame = cam.get_frame()        # Optional[np.ndarray]
            if frame is not None:
                process(frame)
        finally:
            cam.release()

    Or with 'with' syntax:
        with CameraHandler(src=0) as cam:
            frame = cam.get_frame()

    Args:
        src:    Camera device index. 0 = default camera / Pi Camera via v4l2.
        width:  Requested capture width in pixels.
        height: Requested capture height in pixels.
        fps:    Requested capture frame rate.
    """

    def __init__(
        self,
        src:    int = 0,
        width:  int = _DEFAULT_WIDTH,
        height: int = _DEFAULT_HEIGHT,
        fps:    int = _DEFAULT_FPS,
    ) -> None:
        self._src:    int = src
        self._width:  int = width
        self._height: int = height
        self._fps:    int = fps

        self._cap:     Optional[cv2.VideoCapture] = None
        self._frame:   Optional[np.ndarray]       = None
        self._ret:     bool                       = False
        self._running: bool                       = False

        # Mutex guards _frame and _ret across the capture thread and callers.
        self._lock:   threading.Lock  = threading.Lock()
        self._thread: Optional[threading.Thread] = None

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    def open(self) -> None:
        """
        Open the camera device and start the background capture thread.

        Raises:
            RuntimeError: Camera device cannot be opened (hardware absent,
                          permission denied, or device already in use).
            RuntimeError: Camera opens but fails to deliver the first frame
                          (cable issue, unsupported format, etc.).
        """
        try:
            self._cap = cv2.VideoCapture(self._src)

            if not self._cap.isOpened():
                raise RuntimeError(
                    f"[CameraHandler] Cannot open camera at index {self._src}. "
                    "Verify the device is connected and not in use by another process."
                )

            # ── Performance tuning ─────────────────────────────────────────
            self._cap.set(
                cv2.CAP_PROP_FOURCC,
                cv2.VideoWriter_fourcc(*_FOURCC),   # Hardware MJPEG compression
            )
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self._width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
            self._cap.set(cv2.CAP_PROP_FPS,          self._fps)
            # Buffer=1: prevents OpenCV from queuing stale frames.
            # get_frame() always returns the freshest available image.
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        except RuntimeError:
            raise  # Re-raise our own descriptive error unchanged.
        except Exception as exc:
            raise RuntimeError(
                f"[CameraHandler] Unexpected error while opening camera {self._src}: {exc}"
            ) from exc

        # Warm-up read: guarantees _frame is not None immediately after open().
        self._ret, self._frame = self._cap.read()
        if not self._ret:
            self._cap.release()
            raise RuntimeError(
                f"[CameraHandler] Camera {self._src} opened but failed to "
                "capture the first frame. Check cable and V4L2 driver."
            )

        self._running = True
        self._thread = threading.Thread(
            target=self._capture_loop,
            name=f"camera-{self._src}-thread",
            daemon=True,  # Dies automatically if main process exits.
        )
        self._thread.start()

        actual_w   = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h   = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self._cap.get(cv2.CAP_PROP_FPS)
        logger.info(
            "[CameraHandler] Camera %d ready — %dx%d @ %.1f FPS "
            "(requested %dx%d @ %d FPS, codec=%s).",
            self._src, actual_w, actual_h, actual_fps,
            self._width, self._height, self._fps, _FOURCC,
        )

    def release(self) -> None:
        """
        Stop the capture thread and release the camera device.

        Safe to call multiple times (idempotent).
        Always call this in a finally block or use the context manager.
        """
        self._running = False

        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)
            if self._thread.is_alive():
                logger.warning(
                    "[CameraHandler] Capture thread for camera %d did not "
                    "stop within the timeout.",
                    self._src,
                )

        if self._cap is not None and self._cap.isOpened():
            self._cap.release()
            logger.info("[CameraHandler] Camera %d released.", self._src)

        self._cap    = None
        self._thread = None

    # ── Context manager support ────────────────────────────────────────────────

    def __enter__(self) -> "CameraHandler":
        self.open()
        return self

    def __exit__(self, *_) -> None:
        self.release()

    # ── Public API ─────────────────────────────────────────────────────────────

    def get_frame(self) -> Optional[np.ndarray]:
        """
        Return the most recent frame captured by the background thread.

        NON-BLOCKING — returns instantly; never stalls on sensor I/O.
        A defensive copy is returned so the caller can freely mutate the
        array without causing race conditions with the capture thread.

        Returns:
            np.ndarray : BGR uint8 array (H×W×3) when a valid frame exists.
            None       : If the camera has not yet delivered a frame, or if
                         the capture thread has encountered a fatal error.

        Example:
            frame = cam.get_frame()
            if frame is None:
                logger.warning("Frame unavailable, skipping iteration.")
                continue
            cv2.imshow("Preview", frame)   # ← passes ndarray directly
        """
        with self._lock:
            if self._frame is None or not self._ret:
                return None
            return self._frame.copy()

    @property
    def resolution(self) -> Tuple[int, int]:
        """Actual (width, height) of the live capture stream."""
        if self._cap is None or not self._cap.isOpened():
            return (self._width, self._height)
        return (
            int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )

    @property
    def is_open(self) -> bool:
        """True while the capture thread is running and the device is active."""
        return self._running

    # ── Internal ───────────────────────────────────────────────────────────────

    def _capture_loop(self) -> None:
        """
        Background daemon: continuously reads frames from the camera sensor.

        Writes each new frame into self._frame under the mutex.
        Exits gracefully when self._running is set to False by release().
        """
        while self._running:
            try:
                ret, frame = self._cap.read()
                with self._lock:
                    self._ret   = ret
                    self._frame = frame
            except Exception as exc:
                logger.error(
                    "[CameraHandler] Fatal error in capture loop: %s. "
                    "Stopping thread.",
                    exc,
                )
                with self._lock:
                    self._ret = False
                break

        logger.debug(
            "[CameraHandler] Capture loop for camera %d exited cleanly.",
            self._src,
        )

if __name__ == "__main__":
    import time
    import logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    print("[camera.py] Standalone camera test — press 'q' to exit.")
    cam = CameraHandler(src=0)
    try:
        cam.open()                   # ← must call open() before get_frame()
        prev = time.perf_counter()

        while True:
            frame = cam.get_frame()  # Optional[np.ndarray] — no tuple unpacking
            if frame is None:
                print("[WARN] No frame yet, retrying...")
                continue

            # FPS overlay
            now  = time.perf_counter()
            fps  = 1.0 / max(now - prev, 1e-6)
            prev = now
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            cv2.imshow("Camera Test — Drowsiness Project", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except RuntimeError as exc:
        print(f"[ERROR] Hardware: {exc}")
    finally:
        cam.release()
        cv2.destroyAllWindows()
        print("[camera.py] Camera released. Done.")