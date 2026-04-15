"""
Temporal state machine for Drowsiness Detection.
"""

import time
import numpy as np
from enum import Enum

class DrowsinessState(Enum):
    WAITING = "WAITING"
    CALIBRATING = "CALIBRATING"
    MONITORING = "MONITORING"
    DROWSY = "DROWSY"
    DISTRACTED = "DISTRACTED"

class DrowsinessDetector:
    """
    Time-based State Machine for rigorous driver monitoring.
    Uses continuous dt polling to handle fluctuating edge-device frame rates.
    """

    def __init__(
        self,
        calibration_time: float = 15.0,
        ear_drop_ratio: float = 0.6,
        alarm_time: float = 1.5,
        distraction_time: float = 2.0
    ) -> None:
        self.calibration_time = calibration_time
        self.ear_drop_ratio = ear_drop_ratio
        self.alarm_time = alarm_time
        self.distraction_time = distraction_time
        
        self.state = DrowsinessState.WAITING
        self.baseline_ear = 0.0
        
        self._last_tick = 0.0
        self._calib_accumulated_time = 0.0
        self._calib_ears = []
        
        self._eyes_closed_time = 0.0
        self._distracted_time = 0.0

    def update(self, ear: float, face_detected: bool, looking_forward: bool) -> DrowsinessState:
        """Evaluates the current EAR against temporal thresholds and updates the state."""
        now = time.time()
        
        if self._last_tick == 0.0:
            self._last_tick = now
            return self.state
            
        dt = now - self._last_tick
        self._last_tick = now
        
        if not face_detected:
            self._eyes_closed_time = 0.0
            self._distracted_time = 0.0
            return self.state

        if self.baseline_ear == 0.0:
            if looking_forward:
                self.state = DrowsinessState.CALIBRATING
                self._calib_accumulated_time += dt
                self._calib_ears.append(ear)
                
                if self._calib_accumulated_time >= self.calibration_time:
                    if len(self._calib_ears) > 0:
                        self.baseline_ear = float(np.percentile(self._calib_ears, 90))
                    else:
                        self.baseline_ear = 0.3
                    self.state = DrowsinessState.MONITORING
            else:
                self.state = DrowsinessState.WAITING
            return self.state
            
        threshold_ear = self.baseline_ear * self.ear_drop_ratio
        
        if ear < threshold_ear:
            self._eyes_closed_time += dt
        else:
            self._eyes_closed_time = 0.0
            
        if not looking_forward:
            self._distracted_time += dt
        else:
            self._distracted_time = 0.0
            
        if self._eyes_closed_time >= self.alarm_time:
            self.state = DrowsinessState.DROWSY
        elif self._distracted_time >= self.distraction_time:
            self.state = DrowsinessState.DISTRACTED
        else:
            self.state = DrowsinessState.MONITORING
            
        return self.state
