"""
Adaptive Driver Calibration Module.
Calibrates personal baselines during the initial session window:
1. Baseline Open EAR: sets adaptive personalized eye closure threshold.
2. Baseline Resting MAR: sets customized yawn threshold.
3. Neutral Head Pose: accounts for camera angle / mounting position.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import time
import numpy as np


@dataclass
class CalibrationResult:
    """Consolidated calibrated thresholds and baselines."""
    completed: bool = False
    samples_count: int = 0
    baseline_ear_open: float = 0.28
    adaptive_ear_threshold: float = 0.20
    baseline_mar: float = 0.25
    adaptive_mar_threshold: float = 0.58
    baseline_pitch: float = 0.0
    baseline_yaw: float = 0.0
    status_message: str = "Uncalibrated (Using defaults)"


class DriverCalibrator:
    """
    Collects facial metrics over a calibration window (default 20 seconds, or 100 valid frames)
    to compute robust personal thresholds.
    """
    def __init__(self, target_duration_sec: float = 15.0, min_samples: int = 40):
        self.target_duration_sec = target_duration_sec
        self.min_samples = min_samples

        self.is_calibrating: bool = False
        self.start_time: Optional[float] = None
        self._ear_samples: List[float] = []
        self._mar_samples: List[float] = []
        self._pitch_samples: List[float] = []
        self._yaw_samples: List[float] = []

        self.result: CalibrationResult = CalibrationResult()

    def start(self, current_time: Optional[float] = None):
        """Initiates a fresh calibration session."""
        self.is_calibrating = True
        self.start_time = current_time
        self._ear_samples.clear()
        self._mar_samples.clear()
        self._pitch_samples.clear()
        self._yaw_samples.clear()
        self.result = CalibrationResult(status_message="Calibrating... Look forward at the road")

    def update(
        self,
        ear: Optional[float],
        mar: Optional[float],
        pitch: Optional[float],
        yaw: Optional[float],
        current_time: Optional[float] = None
    ) -> Tuple[bool, float, CalibrationResult]:
        """
        Records a frame's metrics during calibration.
        
        Returns:
            (is_done, progress_0_to_1, current_result)
        """
        if not self.is_calibrating:
            return True, 1.0, self.result

        now = current_time if current_time is not None else time.time()
        if self.start_time is None:
            self.start_time = now

        elapsed = now - self.start_time
        progress = min(1.0, elapsed / max(1.0, self.target_duration_sec))

        if ear is not None and 0.10 <= ear <= 0.45:
            self._ear_samples.append(ear)
        if mar is not None and 0.05 <= mar <= 0.60:
            self._mar_samples.append(mar)
        if pitch is not None and abs(pitch) < 40.0:
            self._pitch_samples.append(pitch)
        if yaw is not None and abs(yaw) < 40.0:
            self._yaw_samples.append(yaw)

        # Check for completion
        if elapsed >= self.target_duration_sec or len(self._ear_samples) >= 120:
            if len(self._ear_samples) >= self.min_samples:
                # Use 75th percentile of EAR as representative of natural open eyes
                base_ear = float(np.percentile(self._ear_samples, 70))
                # Adaptive threshold is 72% of natural open state
                adaptive_ear = float(np.clip(base_ear * 0.72, 0.16, 0.25))

                base_mar = float(np.median(self._mar_samples)) if self._mar_samples else 0.25
                adaptive_mar = float(np.clip(base_mar + 0.30, 0.52, 0.75))

                base_pitch = float(np.median(self._pitch_samples)) if self._pitch_samples else 0.0
                base_yaw = float(np.median(self._yaw_samples)) if self._yaw_samples else 0.0

                self.result = CalibrationResult(
                    completed=True,
                    samples_count=len(self._ear_samples),
                    baseline_ear_open=base_ear,
                    adaptive_ear_threshold=adaptive_ear,
                    baseline_mar=base_mar,
                    adaptive_mar_threshold=adaptive_mar,
                    baseline_pitch=base_pitch,
                    baseline_yaw=base_yaw,
                    status_message=f"Calibrated ({len(self._ear_samples)} frames). EAR threshold: {adaptive_ear:.3f}"
                )
            else:
                self.result = CalibrationResult(
                    completed=True,
                    status_message="Calibration completed with default fallback (low samples)."
                )

            self.is_calibrating = False
            return True, 1.0, self.result

        return False, progress, self.result
