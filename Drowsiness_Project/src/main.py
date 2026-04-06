"""
src/main.py
───────────
Master entry point — Perception Pipeline (Camera + FaceMesh).

Supports two runtime modes controlled by the HEADLESS_MODE flag:

  HEADLESS_MODE = False  (default, local Mac/desktop)
    • Renders face mesh overlay and HUD via cv2.imshow.
    • Exit with the 'q' key in the display window.

  HEADLESS_MODE = True   (Raspberry Pi over SSH — no display server)
    • All cv2.imshow / cv2.waitKey / cv2.circle calls are skipped entirely.
    • FPS is printed to stdout once per second (no terminal spam).
    • Exit cleanly with Ctrl+C — KeyboardInterrupt is caught and the camera
      and MediaPipe resources are released before the process dies.

Next step → extraction/ module (EAR, MAR, Head Pose, ROI crop).
"""

import logging
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# ── Path setup ─────────────────────────────────────────────────────────────────
# Allows running as:  python src/main.py  from the Drowsiness_Project/ root.
_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from perception import CameraHandler, FaceMeshDetector, FaceMeshResult

# ══════════════════════════════════════════════════════════════════════════════
#  RUNTIME CONFIGURATION
#  ─────────────────────
#  Flip this flag to switch between desktop (GUI) and Pi (headless SSH) modes.
# ══════════════════════════════════════════════════════════════════════════════
HEADLESS_MODE: bool = False   # ← Set to True before deploying to Raspberry Pi.

# ── FPS logger interval ────────────────────────────────────────────────────────
_FPS_LOG_INTERVAL_SEC: float = 1.0   # Print FPS to console at most once/second.

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt = "%H:%M:%S",
)
logger = logging.getLogger("main")

# ── GUI constants (only referenced when HEADLESS_MODE = False) ─────────────────
_WINDOW_TITLE: str   = "Drowsiness Detection — Perception Test"
_FONT                = cv2.FONT_HERSHEY_SIMPLEX
_GREEN: tuple        = (0, 255, 0)
_RED:   tuple        = (0, 60, 220)
_WHITE: tuple        = (255, 255, 255)
_DARK:  tuple        = (30, 30, 30)


# ══════════════════════════════════════════════════════════════════════════════
#  GUI helpers  (only called when HEADLESS_MODE = False)
# ══════════════════════════════════════════════════════════════════════════════

def _draw_hud(frame: np.ndarray, fps: float, detected: bool) -> None:
    """
    Render a semi-transparent HUD bar at the top of the frame (in-place).

    Shows:  FPS: 19.4  |  FACE DETECTED ✓   (or  NO FACE ✗)
    """
    h, w = frame.shape[:2]

    # Semi-transparent dark bar — copy → blend → overwrite original
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 45), _DARK, -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    cv2.putText(frame, f"FPS: {fps:5.1f}", (10, 30),
                _FONT, 0.75, _WHITE, 2, cv2.LINE_AA)

    status_text  = "FACE DETECTED  v" if detected else "NO FACE  x"
    status_color = _GREEN if detected else _RED
    cv2.putText(frame, status_text, (w - 240, 30),
                _FONT, 0.70, status_color, 2, cv2.LINE_AA)


# ══════════════════════════════════════════════════════════════════════════════
#  Main pipeline
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    """
    Perception pipeline — camera capture + FaceMesh inference loop.

    Lifecycle:
      1. Open CameraHandler  (640×480, MJPG, threaded).
      2. Init FaceMeshDetector (478 landmarks, refine_landmarks=True).
      3. Loop:
           capture → detect → [annotate + display | headless log]
      4. Graceful shutdown on 'q' key (GUI) or Ctrl+C (headless SSH).
    """
    mode_label = "HEADLESS (SSH)" if HEADLESS_MODE else "GUI (desktop)"
    logger.info("Pipeline starting in %s mode.", mode_label)
    if HEADLESS_MODE:
        logger.info("Display disabled. Press Ctrl+C to stop.")
    else:
        logger.info("Press 'q' in the display window to exit.")

    cam:      CameraHandler    = CameraHandler(src=0)   # 640×480 locked in camera.py
    detector: FaceMeshDetector = FaceMeshDetector()

    # ── FPS tracking ───────────────────────────────────────────────────────────
    frame_count:    int   = 0
    fps:            float = 0.0
    loop_tick:      float = time.perf_counter()   # per-frame delta timer
    log_tick:       float = time.perf_counter()   # 1-second console log timer

    try:
        cam.open()

        # ── Main inference loop ────────────────────────────────────────────────
        while True:

            # 1. Capture ───────────────────────────────────────────────────────
            frame: Optional[np.ndarray] = cam.get_frame()
            if frame is None:
                logger.debug("Frame not ready, skipping.")
                continue

            # 2. Detect ────────────────────────────────────────────────────────
            result: FaceMeshResult = detector.process(frame)

            # 3. FPS calculation ───────────────────────────────────────────────
            now       = time.perf_counter()
            fps       = 1.0 / max(now - loop_tick, 1e-6)
            loop_tick = now
            frame_count += 1

            # 4a. GUI mode — annotate and display ──────────────────────────────
            if not HEADLESS_MODE:
                if result.detected:
                    FaceMeshDetector.draw_mesh(frame, result)
                _draw_hud(frame, fps, result.detected)
                cv2.imshow(_WINDOW_TITLE, frame)

                # 'q' key exits; waitKey(1) keeps the window responsive.
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    logger.info("Exit signal received ('q').")
                    break

            # 4b. Headless mode — print FPS once per second ────────────────────
            else:
                elapsed_since_log = now - log_tick
                if elapsed_since_log >= _FPS_LOG_INTERVAL_SEC:
                    avg_fps = frame_count / elapsed_since_log
                    face_status = "DETECTED" if result.detected else "NOT DETECTED"
                    print(
                        f"[{time.strftime('%H:%M:%S')}] "
                        f"FPS: {avg_fps:5.1f} | "
                        f"Face: {face_status} | "
                        f"Landmarks: {len(result.landmarks)}",
                        flush=True,
                    )
                    # Reset counters for the next 1-second window.
                    frame_count = 0
                    log_tick    = now

    except RuntimeError as exc:
        # Hardware-level failure: camera not found, MediaPipe OOM, etc.
        logger.error("Hardware error — %s", exc)
        sys.exit(1)

    except KeyboardInterrupt:
        # Ctrl+C over SSH — the expected exit path in headless mode.
        logger.info("KeyboardInterrupt received — shutting down.")

    finally:
        # Guaranteed cleanup regardless of exit path.
        cam.release()
        detector.close()
        if not HEADLESS_MODE:
            cv2.destroyAllWindows()
        logger.info("Pipeline shut down cleanly.")


if __name__ == "__main__":
    main()
