"""
src/main.py
───────────
Master entry point — Edge-to-Client Streaming Architecture.

Deployment model:
  ┌──────────────────────────────────────────────────────────────┐
  │  Raspberry Pi 4                                              │
  │                                                              │
  │  [Camera] → CameraHandler (thread)                          │
  │                  │                                           │
  │                  ▼                                           │
  │           FaceMeshDetector                                   │
  │                  │ annotated frame                           │
  │                  ▼                                           │
  │           MJPEGStreamer.push_frame()  ──► Flask :5000        │
  └──────────────────────────────────────────────────────────────┘
                                                  │
                              Wi-Fi (local network)│
                                                  ▼
                                 MacBook browser → http://<Pi-IP>:5000

Exit:  Ctrl+C in the SSH terminal triggers KeyboardInterrupt → clean teardown.

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
_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from perception import CameraHandler, FaceMeshDetector, FaceMeshResult
from extraction import calculate_ear
from intelligence import DrowsinessDetector, DrowsinessState
from utils import MJPEGStreamer, FPSMonitor

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt = "%H:%M:%S",
)
logger = logging.getLogger("main")

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

STREAM_HOST: str  = "0.0.0.0"   # Bind all interfaces — reachable from MacBook.
STREAM_PORT: int  = 5000
JPEG_QUALITY: int = 75           # Quality / bandwidth tradeoff (scale: 0–100).

# Console FPS log printed at most once per this interval (avoids terminal spam).
_FPS_LOG_INTERVAL_SEC: float = 1.0

# ── HUD drawing constants ──────────────────────────────────────────────────────
_FONT          = cv2.FONT_HERSHEY_SIMPLEX
_GREEN: tuple  = (0, 255, 0)
_RED:   tuple  = (0, 60, 220)
_WHITE: tuple  = (255, 255, 255)
_DARK:  tuple  = (20, 20, 20)


# ══════════════════════════════════════════════════════════════════════════════
#  HUD renderer
# ══════════════════════════════════════════════════════════════════════════════

def _draw_hud(frame: np.ndarray, fps: float, detected: bool, ear: float = 0.0, state_str: str = "WAITING") -> None:
    """
    Burn a semi-transparent status bar into the frame (in-place).

    Shows:  FPS: 19.4  |  STATE: AWAKE   |  EAR: 0.32
    The annotated frame is then JPEG-encoded and pushed to the streamer.
    """
    h, w = frame.shape[:2]

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 45), _DARK, -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    cv2.putText(frame, f"FPS: {fps:5.1f}", (10, 30),
                _FONT, 0.75, _WHITE, 2, cv2.LINE_AA)

    # Resolve temporal state color mapping
    if not detected:
        state_color = _RED
        display_text = "NO FACE"
    else:
        display_text = f"STATE: {state_str}"
        if state_str == "DROWSY":
            state_color = (0, 0, 255)      # Red (BGR)
        elif state_str == "AWAKE":
            state_color = (0, 255, 0)      # Green
        elif state_str == "BLINKING":
            state_color = (0, 165, 255)    # Orange
        else: # CALIBRATING
            state_color = (255, 255, 0)    # Cyan

    cv2.putText(frame, display_text, (w - 320, 30),
                _FONT, 0.65, state_color, 2, cv2.LINE_AA)
                
    if detected:
        cv2.putText(frame, f"EAR: {ear:.2f}", (w - 120, 30),
                    _FONT, 0.65, (0, 255, 255), 2, cv2.LINE_AA)


# ══════════════════════════════════════════════════════════════════════════════
#  Main pipeline
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    """
    Perception pipeline + MJPEG streaming loop.

    Lifecycle:
      1. Start MJPEGStreamer (Flask daemon thread → port 5000).
      2. Open CameraHandler (640×480, MJPG codec, background capture thread).
      3. Init FaceMeshDetector (478 landmarks, refine_landmarks=True).
      4. Loop:
           get_frame() → process() → draw_mesh() → _draw_hud() → push_frame()
      5. Graceful teardown on Ctrl+C.
    """
    # ── Initialise components ─────────────────────────────────────────────────
    streamer = MJPEGStreamer(
        host         = STREAM_HOST,
        port         = STREAM_PORT,
        jpeg_quality = JPEG_QUALITY,
    )
    cam      = CameraHandler(src=0)
    detector = FaceMeshDetector()
    daze_detector = DrowsinessDetector()

    # Start Flask first so the endpoint is ready before inference begins.
    streamer.start()
    logger.info(
        "Stream live → open http://<Pi-IP>:%d in your MacBook browser.",
        STREAM_PORT,
    )
    logger.info("Press Ctrl+C to stop.")

    # ── FPS tracking ──────────────────────────────────────────────────────────
    fps_monitor = FPSMonitor(window_size=30)
    frame_count: int   = 0
    log_tick:    float = time.perf_counter()

    try:
        cam.open()

        # ── Inference loop ────────────────────────────────────────────────────
        while True:

            # 1. Capture ───────────────────────────────────────────────────────
            frame: Optional[np.ndarray] = cam.get_frame()
            if frame is None:
                logger.debug("Frame not ready, skipping.")
                continue

            # 2. Detect & Extract ──────────────────────────────────────────────
            result: FaceMeshResult = detector.process(frame)
            
            ear = 0.0
            state_str = "WAITING"
            
            if result.detected:
                ear = calculate_ear(result.landmarks)

            # 3. Intelligence — Evaluate Temporal State ────────────────────────
            daze_state = daze_detector.update(ear, result.detected)
            state_str = daze_state.value

            # 4. Annotate — draw mesh then HUD bar ─────────────────────────────
            if result.detected:
                FaceMeshDetector.draw_mesh(frame, result)

            fps_monitor.update()
            fps = fps_monitor.get_fps()
            
            now = time.perf_counter()
            frame_count += 1

            _draw_hud(frame, fps, result.detected, ear, state_str)

            # 4. Stream — push JPEG-encoded annotated frame to all clients ─────
            streamer.push_frame(frame)

            # 5. Console FPS log (once per second, not every frame) ────────────
            elapsed = now - log_tick
            if elapsed >= _FPS_LOG_INTERVAL_SEC:
                avg_fps     = frame_count / elapsed
                face_status = "DETECTED" if result.detected else "NONE    "
                print(
                    f"[{time.strftime('%H:%M:%S')}] "
                    f"FPS: {avg_fps:5.1f} | "
                    f"Face: {face_status} | "
                    f"State: {state_str:11s} | "
                    f"EAR: {ear:.2f} | "
                    f"Landmarks: {len(result.landmarks):3d}",
                    flush=True,
                )
                frame_count = 0
                log_tick    = now

    except RuntimeError as exc:
        logger.error("Hardware error — %s", exc)
        sys.exit(1)

    except KeyboardInterrupt:
        # Ctrl+C over SSH — expected exit path on the Pi.
        logger.info("KeyboardInterrupt — shutting down.")

    finally:
        # Order matters: stop inference sources before releasing hardware.
        detector.close()
        cam.release()
        streamer.stop()
        logger.info("Pipeline shut down cleanly.")


if __name__ == "__main__":
    main()
