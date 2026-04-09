"""
utils/fps.py
────────────
Provides a moving-average FPS Monitor for stable performance profiling.
"""

import time
import collections

class FPSMonitor:
    """
    Tracks application frame rate using a moving average over a specified
    window of frames. This smoothing eliminates erratic jumps and provides
    a stable reading essential for time-based temporal logic (e.g. PERCLOS).
    """

    def __init__(self, window_size: int = 30) -> None:
        """
        Args:
            window_size: Number of frames to average over.
        """
        self._window_size = window_size
        self._frame_times = collections.deque(maxlen=window_size)
        self._last_tick = time.perf_counter()
        self._current_fps = 0.0

    def update(self) -> None:
        """
        Marks the end of a frame processing cycle and updates the moving average.
        Must be called exactly once per loop iteration.
        """
        now = time.perf_counter()
        dt = max(now - self._last_tick, 1e-6)
        
        self._frame_times.append(dt)
        self._last_tick = now

        # Compute moving average
        avg_time = sum(self._frame_times) / len(self._frame_times)
        self._current_fps = 1.0 / avg_time

    def get_fps(self) -> float:
        """
        Returns the current smoothed FPS value.
        """
        return self._current_fps
