"""
Master entry point — Hybrid Optimized Pipeline for Raspberry Pi 4.
"""

import logging
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from perception import VideoStream, FaceMeshDetector, FaceMeshResult
from extraction import calculate_ear
from intelligence import DrowsinessDetector, HeadPoseEstimator, YawnDetectorCNN
from utils import MJPEGStreamer, FPSMonitor

_MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("main")

# ── Config ─────────────────────────────────────────────────────────────────────
STREAM_HOST:  str   = "0.0.0.0"
STREAM_PORT:  int   = 5000
JPEG_QUALITY: int   = 75
SKIP_FRAMES:  int   = 2        # Run extraction/intelligence every Nth frame only.
_LOG_INTERVAL: float = 1.0

# ── HUD constants ──────────────────────────────────────────────────────────────
_FONT  = cv2.FONT_HERSHEY_SIMPLEX
_WHITE: Tuple[int, int, int] = (255, 255, 255)
_RED:   Tuple[int, int, int] = (0, 60, 220)
_DARK:  Tuple[int, int, int] = (20, 20, 20)

_STATE_COLORS = {
    "DROWSY":      (0, 0, 255),
    "YAWNING":     (0, 100, 255),
    "MONITORING":  (0, 255, 0),
    "AWAKE":       (0, 255, 0),
    "BLINKING":    (0, 165, 255),
    "DISTRACTED":  (255, 0, 255),
}


def _draw_hud(
    frame: np.ndarray,
    fps: float,
    detected: bool,
    ear: float,
    state_str: str,
    baseline_ear: float,
    pitch: float,
    yaw: float,
    yawn_detected: bool = False,
) -> None:
    """Renders semi-transparent HUD status bar onto the frame in-place."""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 75), _DARK, -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    cv2.putText(frame, f"FPS: {fps:5.1f}", (10, 30), _FONT, 0.75, _WHITE, 2, cv2.LINE_AA)

    if not detected:
        cv2.putText(frame, "NO FACE", (w - 200, 30), _FONT, 0.65, _RED, 2, cv2.LINE_AA)
        return

    state_color = _STATE_COLORS.get(state_str, (255, 255, 0))
    cv2.putText(frame, f"STATE: {state_str}", (w - 320, 30), _FONT, 0.65, state_color, 2, cv2.LINE_AA)
    cv2.putText(frame, f"EAR: {ear:.2f}", (w - 120, 30), _FONT, 0.65, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Pose: P:{pitch:4.0f} Y:{yaw:4.0f}", (10, 60), _FONT, 0.55, _WHITE, 1, cv2.LINE_AA)
    if baseline_ear > 0.0:
        cv2.putText(frame, f"BASE: {baseline_ear:.2f}", (w - 120, 60), _FONT, 0.55, (0, 255, 0), 1, cv2.LINE_AA)
    yawn_label = "YAWNING" if yawn_detected else "Normal"
    yawn_color = (0, 100, 255) if yawn_detected else (180, 180, 180)
    cv2.putText(frame, f"Yawn: {yawn_label}", (10, 30 + (frame.shape[0] - 75)), _FONT, 0.6, yawn_color, 2, cv2.LINE_AA)


def main() -> None:
    """Runs the optimized hybrid perception-extraction-streaming loop."""
    streamer       = MJPEGStreamer(host=STREAM_HOST, port=STREAM_PORT, jpeg_quality=JPEG_QUALITY)
    stream         = VideoStream(src=0)
    detector       = FaceMeshDetector()
    daze_detector  = DrowsinessDetector()
    pose_estimator = HeadPoseEstimator()
    yawn_detector  = YawnDetectorCNN(_MODELS_DIR / "yawn_detector.tflite")
    fps_monitor    = FPSMonitor(window_size=30)

    streamer.start()
    stream.start()
    logger.info("Stream live on port %d.", STREAM_PORT)

    # Cached values reused on skipped frames
    ear:             float = 0.0
    pitch:           float = 0.0
    yaw:             float = 0.0
    roll:            float = 0.0
    looking_forward: bool  = False
    yawn_detected:   bool  = False
    state_str:       str   = "WAITING"
    result:          Optional[FaceMeshResult] = None

    frame_count: int   = 0
    log_tick:    float = time.perf_counter()

    try:
        while True:
            frame = stream.read()
            if frame is None:
                continue

            # ── FaceMesh runs every frame for smooth tracking ──────────────────
            result = detector.process(frame)

            # ── Extraction + Intelligence run every SKIP_FRAMES-th frame ──────
            if frame_count % SKIP_FRAMES == 0 and result.detected:
                ear = calculate_ear(result.landmarks)
                pitch, yaw, roll = pose_estimator.estimate(result.landmarks, frame.shape[:2])
                looking_forward  = pose_estimator.is_looking_forward(yaw, pitch)
                yawn_detected    = yawn_detector.predict_yawn(frame, result.landmarks)
                daze_state = daze_detector.update(ear, result.detected, looking_forward, yawn_detected)
                state_str  = daze_state.value
            elif not result.detected:
                yawn_detected = False
                daze_detector.update(0.0, False, False)
                state_str = daze_detector.state.value

            # ── Annotate ───────────────────────────────────────────────────────
            if result.detected:
                FaceMeshDetector.draw_mesh(frame, result)

            fps_monitor.update()
            fps = fps_monitor.get_fps()
            now = time.perf_counter()
            frame_count += 1

            _draw_hud(frame, fps, result.detected, ear, state_str,
                      daze_detector.baseline_ear, pitch, yaw, yawn_detected)
            streamer.push_frame(frame)

            # ── Console log (once per second) ─────────────────────────────────
            elapsed = now - log_tick
            if elapsed >= _LOG_INTERVAL:
                avg_fps     = frame_count / elapsed
                face_status = "DETECTED" if result.detected else "NONE    "
                print(
                    f"[{time.strftime('%H:%M:%S')}] "
                    f"FPS: {avg_fps:5.1f} | Face: {face_status} | "
                    f"State: {state_str:11s} | EAR: {ear:.2f} | "
                    f"Yawn: {'YES' if yawn_detected else 'NO ':3s} | "
                    f"Yaw: {yaw:4.0f} | Landmarks: {len(result.landmarks):3d}",
                    flush=True,
                )
                frame_count = 0
                log_tick    = now

    except RuntimeError as exc:
        logger.error("Hardware error: %s", exc)
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt triggered.")
    finally:
        detector.close()
        stream.stop()
        streamer.stop()


if __name__ == "__main__":
    main()
