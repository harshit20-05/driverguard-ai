"""
Yawn Detector using Mouth Aspect Ratio (MAR) from MediaPipe Face Mesh.
Calculates lip vertical-to-horizontal opening ratio and tracks sustained open duration
to accurately separate talking/smiling from deep physiological yawning.
"""

from dataclasses import dataclass
from typing import Optional, List
import time
import numpy as np


@dataclass
class YawnStatus:
    """Current yawn detection evaluation for a single frame."""
    mar: float = 0.0
    is_mouth_open: bool = False
    is_yawning: bool = False
    yawn_duration_s: float = 0.0
    total_yawns: int = 0
    new_yawn_event: bool = False  # True on the exact frame a full yawn is registered


class YawnDetector:
    """
    Evaluates mouth aperture using Mouth Aspect Ratio (MAR).
    Triggers a yawn alert when MAR exceeds threshold for a sustained window (typically >= 1.2 seconds).
    """

    # MediaPipe Face Mesh lip landmark indices
    _LIP_TOP = 13
    _LIP_BOTTOM = 14
    _LIP_LEFT = 78
    _LIP_RIGHT = 308
    _LIP_TOP_L = 81
    _LIP_BOT_L = 178
    _LIP_TOP_R = 311
    _LIP_BOT_R = 402

    def __init__(
        self,
        mar_threshold: float = 0.62,
        min_yawn_duration_sec: float = 1.2,
        cooldown_sec: float = 3.0
    ):
        self.mar_threshold = mar_threshold
        self.min_yawn_duration_sec = min_yawn_duration_sec
        self.cooldown_sec = cooldown_sec

        self.total_yawns: int = 0
        self.baseline_mar: Optional[float] = None

        self._yawn_start_time: Optional[float] = None
        self._last_yawn_event_time: float = -999.0
        self._is_yawn_active: bool = False

    def calculate_mar(self, landmarks_px: np.ndarray) -> float:
        """
        Calculates Mouth Aspect Ratio (MAR).
        
        Args:
            landmarks_px: Array of shape (N, 2) or (N, 3) in pixel coordinates
            
        Returns:
            MAR scalar (typically 0.15 - 0.35 when resting, 0.65+ during a yawn)
        """
        if landmarks_px is None or len(landmarks_px) <= max(self._LIP_RIGHT, self._LIP_BOT_R):
            return 0.0

        p_top = landmarks_px[self._LIP_TOP][:2]
        p_bot = landmarks_px[self._LIP_BOTTOM][:2]
        p_left = landmarks_px[self._LIP_LEFT][:2]
        p_right = landmarks_px[self._LIP_RIGHT][:2]

        p_top_l = landmarks_px[self._LIP_TOP_L][:2]
        p_bot_l = landmarks_px[self._LIP_BOT_L][:2]
        p_top_r = landmarks_px[self._LIP_TOP_R][:2]
        p_bot_r = landmarks_px[self._LIP_BOT_R][:2]

        # 3 vertical distances averaged
        d_vert_center = np.linalg.norm(p_top - p_bot)
        d_vert_left = np.linalg.norm(p_top_l - p_bot_l)
        d_vert_right = np.linalg.norm(p_top_r - p_bot_r)
        d_horiz = np.linalg.norm(p_left - p_right)

        if d_horiz <= 1e-6:
            return 0.0

        mar = (d_vert_center + d_vert_left + d_vert_right) / (3.0 * d_horiz)
        return float(mar)

    def update(self, landmarks_px: Optional[np.ndarray], current_time: Optional[float] = None) -> YawnStatus:
        """
        Updates detector state with facial landmarks of current frame.
        """
        now = current_time if current_time is not None else time.time()

        if landmarks_px is None:
            self._yawn_start_time = None
            self._is_yawn_active = False
            return YawnStatus(total_yawns=self.total_yawns)

        mar = self.calculate_mar(landmarks_px)
        
        # Adaptive threshold if baseline was calibrated
        active_threshold = self.mar_threshold
        if self.baseline_mar is not None:
            active_threshold = max(0.50, self.baseline_mar + 0.30)

        is_open = mar >= active_threshold
        yawn_duration = 0.0
        new_event = False

        if is_open:
            if self._yawn_start_time is None:
                self._yawn_start_time = now

            yawn_duration = now - self._yawn_start_time

            if yawn_duration >= self.min_yawn_duration_sec:
                if not self._is_yawn_active and (now - self._last_yawn_event_time > self.cooldown_sec):
                    self.total_yawns += 1
                    self._is_yawn_active = True
                    self._last_yawn_event_time = now
                    new_event = True
        else:
            self._yawn_start_time = None
            self._is_yawn_active = False

        return YawnStatus(
            mar=mar,
            is_mouth_open=is_open,
            is_yawning=self._is_yawn_active or (yawn_duration >= self.min_yawn_duration_sec),
            yawn_duration_s=yawn_duration,
            total_yawns=self.total_yawns,
            new_yawn_event=new_event
        )

    def reset(self):
        """Resets detector state."""
        self.total_yawns = 0
        self._yawn_start_time = None
        self._last_yawn_event_time = 0.0
        self._is_yawn_active = False
