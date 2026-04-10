"""
intelligence/logic.py
─────────────────────
Temporal state machine for Drowsiness Detection.
Tracks EAR over time, establishes dynamic baselines, and fires alarms.
"""

import time
import numpy as np
from enum import Enum

class DrowsinessState(Enum):
    CALIBRATING = "CALIBRATING"
    AWAKE = "AWAKE"
    BLINKING = "BLINKING"
    DROWSY = "DROWSY"

class DrowsinessDetector:
    """
    Time-based State Machine for driver monitoring.
    Uses continuous polling rather than frame-counting to evaluate state,
    ensuring consistent behavior across variable frame rates (e.g. Pi 4 FPS drops).
    """

    def __init__(
        self,
        calibration_time: float = 5.0,
        ear_drop_ratio: float = 0.6,
        alarm_time: float = 1.5
    ) -> None:
        """
        Args:
            calibration_time : Seconds required to establish BASELINE_EAR.
            ear_drop_ratio   : Multiplier (0.0-1.0) against baseline that triggers "closed".
            alarm_time       : Seconds eyes must remain closed before "DROWSY" triggers.
        """
        self.calibration_time = calibration_time
        self.ear_drop_ratio = ear_drop_ratio
        self.alarm_time = alarm_time
        
        self.state = DrowsinessState.CALIBRATING
        self.baseline_ear = 0.0
        
        # Calibration state
        self._calibration_start_time = 0.0
        self._calibration_ears = []
        
        # Temporal tracking
        self._eyes_closed_start_time = 0.0

    def update(self, ear: float, face_detected: bool) -> DrowsinessState:
        """
        Heartbeat of the intelligence module. 
        Evaluates the current EAR against temporal thresholds and updates the state.
        
        Args:
            ear           : Current Eye Aspect Ratio.
            face_detected : Whether the perception pipeline has a valid face lock.
            
        Returns:
            DrowsinessState representing the user's current condition.
        """
        now = time.time()
        
        # ── 1. Check Face Lock ────────────────────────────────────────────────
        if not face_detected:
            # If we lose the face mid-blink, freeze tracking so we don't accidentally
            # trigger or clear an alarm wrongly.
            self._eyes_closed_start_time = 0.0
            return self.state

        # ── 2. Calibration Phase ──────────────────────────────────────────────
        if self.state == DrowsinessState.CALIBRATING:
            if self._calibration_start_time == 0.0:
                self._calibration_start_time = now
                
            self._calibration_ears.append(ear)
            
            elapsed = now - self._calibration_start_time
            if elapsed >= self.calibration_time:
                if len(self._calibration_ears) > 0:
                    # Use 90th percentile to establish the baseline of "Wide Open"
                    # This naturally filters out any blinks that occurred during calibration.
                    self.baseline_ear = float(np.percentile(self._calibration_ears, 90))
                else:
                    self.baseline_ear = 0.3 # Fallback if array empty somehow
                
                self.state = DrowsinessState.AWAKE
                
            return self.state
            
        # ── 3. Monitored Phase (Temporal Logic) ───────────────────────────────
        threshold_ear = self.baseline_ear * self.ear_drop_ratio
        
        if ear < threshold_ear:
            # ── Eye is CLOSED ──
            if self._eyes_closed_start_time == 0.0:
                # Initial closure boundary
                self._eyes_closed_start_time = now
                self.state = DrowsinessState.BLINKING
            else:
                closed_duration = now - self._eyes_closed_start_time
                if closed_duration >= self.alarm_time:
                    self.state = DrowsinessState.DROWSY
                else:
                    self.state = DrowsinessState.BLINKING
        else:
            # ── Eye is OPEN ──
            self._eyes_closed_start_time = 0.0
            self.state = DrowsinessState.AWAKE
            
        return self.state
