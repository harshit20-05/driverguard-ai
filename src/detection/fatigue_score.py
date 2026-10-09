"""
Fatigue Score Calculator.
Computes a comprehensive 0-100 fatigue indicator using PERCLOS, blink rates, and event frequency.
Labeled strictly as an 'AI-derived fatigue indicator'.
"""

from dataclasses import dataclass
from typing import Deque, Dict, Any, Optional
from collections import deque
import time
import numpy as np


@dataclass
class FatigueMetrics:
    """Encapsulates fatigue calculation components."""
    score: float = 0.0
    level: str = "LOW"               # LOW, MODERATE, HIGH, CRITICAL
    perclos: float = 0.0             # % of closed-eye frames in window (0.0 to 1.0)
    blinks_per_minute: float = 0.0
    prolonged_closure_score: float = 0.0
    color: str = "#10B981"           # Hex color for UI display


class FatigueScoreCalculator:
    """
    Computes an integrated 0-100 fatigue score based on rolling window metrics:
    - PERCLOS: Percentage of eye closure time
    - Prolonged eye closures duration
    - Abnormal blink frequency
    - Drowsiness event penalties
    """
    def __init__(
        self,
        window_seconds: float = 60.0,
        weights: Optional[Dict[str, float]] = None
    ):
        self.window_seconds = window_seconds
        self.weights = weights or {
            "perclos": 0.40,
            "prolonged_closures": 0.35,
            "blink_frequency": 0.15,
            "event_severity": 0.10
        }

        # Rolling buffers of (timestamp, is_closed)
        self._frame_history: Deque[tuple[float, bool]] = deque()
        self.max_score_seen: float = 0.0

    def update(
        self,
        is_closed: bool,
        closure_duration_ms: float,
        blinks_per_min: float,
        recent_drowsy_events: int,
        recent_critical_events: int,
        current_time: Optional[float] = None,
        recent_yawns: int = 0,
        recent_nods: int = 0
    ) -> FatigueMetrics:
        """
        Updates the fatigue score calculator with the latest frame metrics.
        
        Returns:
            FatigueMetrics with score [0 - 100], level label, PERCLOS, and UI color.
        """
        now = current_time if current_time is not None else time.time()
        self._frame_history.append((now, is_closed))
        self._prune(now)

        # 1. PERCLOS (% of time eyes closed in rolling window)
        if self._frame_history:
            closed_count = sum(1 for _, closed in self._frame_history if closed)
            perclos = closed_count / len(self._frame_history)
        else:
            perclos = 0.0

        # Normal PERCLOS < 0.12. Fatigue PERCLOS > 0.20
        perclos_score = np.clip(perclos * 280.0, 0.0, 100.0)

        # 2. Prolonged Eye Closures
        # If eye closure reaches 2 seconds, closure score reaches 100
        prolonged_score = np.clip((closure_duration_ms / 2000.0) * 100.0, 0.0, 100.0)

        # 3. Blink Rate Anomaly
        # Normal relaxed driver: 12 - 25 blinks per minute
        # Drowsy drivers often display very low blink frequency (< 8) or frantic compensatory blinks (> 35)
        if blinks_per_min < 8:
            blink_penalty = (8 - blinks_per_min) * 6.0
        elif blinks_per_min > 35:
            blink_penalty = min(50.0, (blinks_per_min - 35) * 3.0)
        else:
            blink_penalty = 5.0
        blink_score = np.clip(blink_penalty, 0.0, 100.0)

        # 4. Event Frequency Penalty (including yawns and head nodding)
        event_score = np.clip(
            (recent_drowsy_events * 12.0) +
            (recent_critical_events * 30.0) +
            (recent_yawns * 10.0) +
            (recent_nods * 20.0),
            0.0, 100.0
        )

        # Weighted combination
        raw_score = (
            self.weights["perclos"] * perclos_score +
            self.weights["prolonged_closures"] * prolonged_score +
            self.weights["blink_frequency"] * blink_score +
            self.weights["event_severity"] * event_score
        )

        final_score = float(np.clip(raw_score, 0.0, 100.0))
        self.max_score_seen = max(self.max_score_seen, final_score)

        # Determine level and styling
        if final_score <= 30.0:
            level = "LOW"
            color = "#10B981"  # Emerald
        elif final_score <= 60.0:
            level = "MODERATE"
            color = "#EAB308"  # Amber
        elif final_score <= 80.0:
            level = "HIGH"
            color = "#F97316"  # Orange
        else:
            level = "CRITICAL"
            color = "#EF4444"  # Red

        return FatigueMetrics(
            score=final_score,
            level=level,
            perclos=perclos,
            blinks_per_minute=blinks_per_min,
            prolonged_closure_score=prolonged_score,
            color=color
        )

    def _prune(self, current_time: float):
        cutoff = current_time - self.window_seconds
        while self._frame_history and self._frame_history[0][0] < cutoff:
            self._frame_history.popleft()

    def reset(self):
        self._frame_history.clear()
        self.max_score_seen = 0.0
