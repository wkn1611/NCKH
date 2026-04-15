
import logging
import threading
import time
from typing import Generator, Optional

import cv2
import numpy as np
from flask import Flask, Response

logger = logging.getLogger(__name__)

# ── JPEG encoding quality ──────────────────────────────────────────────────────
# 75 = good visual quality @ ~25–40 KB/frame.
# At 20 FPS → ~600 KB/s — well within a 2.4 GHz Wi-Fi link.
_JPEG_QUALITY: int = 75

# ── MJPEG multipart boundary ──────────────────────────────────────────────────
_BOUNDARY: bytes = b"--drowsiness_frame\r\nContent-Type: image/jpeg\r\n\r\n"
_BOUNDARY_END: bytes = b"\r\n"

# ── HTML served at / ──────────────────────────────────────────────────────────
_INDEX_HTML: str = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title></title>
  <style>
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body {
      background: #0d0f14;
      color: #e2e8f0;
      font-family: 'Segoe UI', system-ui, sans-serif;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      min-height: 100vh;
      gap: 16px;
    }
    h1 {
      font-size: 1.25rem;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      color: #94a3b8;
    }
    .badge {
      font-size: 0.75rem;
      background: #1e293b;
      padding: 4px 12px;
      border-radius: 99px;
      color: #38bdf8;
      letter-spacing: 0.05em;
    }
    .frame-wrapper {
      border: 1px solid #1e293b;
      border-radius: 8px;
      overflow: hidden;
      box-shadow: 0 0 40px rgba(56, 189, 248, 0.08);
    }
    img {
      display: block;
      max-width: 95vw;
    }
    footer {
      font-size: 0.7rem;
      color: #475569;
    }
  </style>
</head>
<body>
  <h1>🚗 Driver Drowsiness Detection</h1>
  <span class="badge">LIVE &nbsp;●&nbsp; Raspberry Pi 4 · MJPEG</span>
  <div class="frame-wrapper">
    <img src="/video_feed" alt="Live camera feed">
  </div>
  <footer>Stream: multipart/x-mixed-replace · JPEG Q75 · 640×480</footer>
</body>
</html>"""


class MJPEGStreamer:
    """
    Thread-safe MJPEG streaming server.

    The perception loop owns the push side; Flask generators own the pull side.
    They communicate through a shared JPEG byte buffer protected by a
    threading.Condition, which lets generators sleep efficiently instead of
    spinning when no new frame is available.

    Usage:
        streamer = MJPEGStreamer(host="0.0.0.0", port=5000)
        streamer.start()           # launches Flask in a daemon thread
        ...
        streamer.push_frame(frame) # called each iteration of the main loop
        ...
        streamer.stop()            # signals Flask thread to exit (optional —
                                   # daemon=True handles it on process exit)
    """

    def __init__(
        self,
        host:         str = "0.0.0.0",
        port:         int = 5000,
        jpeg_quality: int = _JPEG_QUALITY,
    ) -> None:
        self._host:         str          = host
        self._port:         int          = port
        self._jpeg_quality: int          = jpeg_quality

        # Shared state between perception thread and Flask generators.
        self._frame_bytes: Optional[bytes]    = None
        self._condition:   threading.Condition = threading.Condition()

        self._app:    Flask                   = self._build_app()
        self._thread: Optional[threading.Thread] = None

    # ── Flask application ──────────────────────────────────────────────────────

    def _build_app(self) -> Flask:
        """Construct and configure the Flask application."""
        app = Flask(__name__)

        # Silence Flask's default werkzeug request logger — the Pi terminal
        # is reserved for our FPS logger, not HTTP noise.
        logging.getLogger("werkzeug").setLevel(logging.ERROR)

        @app.route("/")
        def index() -> Response:
            return Response(_INDEX_HTML, mimetype="text/html")

        @app.route("/video_feed")
        def video_feed() -> Response:
            return Response(
                self._generate(),
                mimetype="multipart/x-mixed-replace; boundary=drowsiness_frame",
            )

        return app

    # ── Frame generator (Flask route handler) ─────────────────────────────────

    def _generate(self) -> Generator[bytes, None, None]:
        """
        Yield JPEG frames in MJPEG multipart format.

        Blocks efficiently on threading.Condition.wait() — no busy-waiting.
        Exits when the client disconnects (GeneratorExit is raised by Flask).
        """
        try:
            while True:
                with self._condition:
                    # Wait up to 200 ms for a new frame before re-checking.
                    # The timeout prevents hanging forever if the camera stops.
                    frame_available = self._condition.wait(timeout=0.2)

                    if not frame_available or self._frame_bytes is None:
                        continue

                    payload = self._frame_bytes  # Read under condition lock.

                yield _BOUNDARY + payload + _BOUNDARY_END

        except GeneratorExit:
            logger.debug("[MJPEGStreamer] Client disconnected.")

    # ── Public API ─────────────────────────────────────────────────────────────

    def push_frame(self, frame: np.ndarray) -> None:
        """
        JPEG-encode a BGR frame and notify all waiting generator clients.

        Called from the perception thread on every loop iteration.
        Encoding happens here (once) regardless of client count.

        Args:
            frame: BGR uint8 ndarray from CameraHandler.get_frame().
        """
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, self._jpeg_quality]
        ok, buf = cv2.imencode(".jpg", frame, encode_params)
        if not ok:
            logger.warning("[MJPEGStreamer] JPEG encode failed, skipping frame.")
            return

        with self._condition:
            self._frame_bytes = buf.tobytes()
            self._condition.notify_all()  # Wake every connected client.

    def start(self) -> None:
        """
        Launch Flask in a background daemon thread.

        Raises:
            RuntimeError: If the server is already running.
        """
        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError("[MJPEGStreamer] Server is already running.")

        self._thread = threading.Thread(
            target=self._run_flask,
            name="mjpeg-server-thread",
            daemon=True,   # Automatically killed when main process exits.
        )
        self._thread.start()
        logger.info(
            "[MJPEGStreamer] Server started — http://%s:%d  "
            "(access from MacBook: http://<Pi-IP>:%d)",
            self._host, self._port, self._port,
        )

    def stop(self) -> None:
        """
        Gracefully signal the Flask thread to exit.

        In practice, daemon=True handles this automatically on process exit.
        Call explicitly only if you need to stop streaming mid-run.
        """
        # Flask/Werkzeug does not expose a clean shutdown API in simple mode.
        # The daemon thread will be killed when main() returns — this is the
        # intended lifecycle for embedded deployments.
        logger.info("[MJPEGStreamer] Server stopping (daemon thread will exit).")

    # ── Internal ───────────────────────────────────────────────────────────────

    def _run_flask(self) -> None:
        """Target for the Flask daemon thread."""
        try:
            self._app.run(
                host        = self._host,
                port        = self._port,
                threaded    = True,   # Handle multiple browser tabs concurrently.
                use_reloader= False,  # CRITICAL: reloader forks — breaks camera ownership.
                debug       = False,  # No debug mode on Pi (performance + stability).
            )
        except Exception as exc:
            logger.error("[MJPEGStreamer] Flask thread error: %s", exc)
