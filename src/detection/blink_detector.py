"""
Blink Detector for distinguishing involuntary physiological blinks from prolonged eye closure.
Maintains blink counters, timestamps, durations, and frequency (blinks per minute).
"""

from dataclasses import dataclass
from typing import List, Optional
import time


@dataclass
class BlinkEvent:
    """Records a single completed blink event."""
    timestamp: float
    duration_ms: float


class BlinkDetector:
    """
    Detects and tracks natural eye blinks using temporal state transitions.
    """
    def __init__(
        self,
        min_duration_ms: float = 80.0,
        max_duration_ms: float = 380.0,
        history_window_seconds: float = 60.0
    ):
        self.min_duration_ms = min_duration_ms
        self.max_duration_ms = max_duration_ms
        self.history_window_seconds = history_window_seconds

        self.total_blinks: int = 0
        self.blink_history: List[BlinkEvent] = []

        self._is_closing: bool = False
        self._closure_start_time: Optional[float] = None

    def update(self, is_closed: bool, current_time: Optional[float] = None) -> bool:
        """
        Updates blink tracking with the current frame's eye closure state.
        
        Args:
            is_closed: True if eyes are currently detected as closed
            current_time: Current timestamp (seconds). Uses time.time() if None.
            
        Returns:
            True if a complete blink was just finished in this frame, False otherwise.
        """
        now = current_time if current_time is not None else time.time()
        completed_blink = False

        if is_closed:
            if not self._is_closing:
                # Transition: Open -> Closed
                self._is_closing = True
                self._closure_start_time = now
        else:
            if self._is_closing:
                # Transition: Closed -> Open
                self._is_closing = False
                if self._closure_start_time is not None:
                    duration_ms = (now - self._closure_start_time) * 1000.0
                    if self.min_duration_ms <= duration_ms <= self.max_duration_ms:
                        self.total_blinks += 1
                        self.blink_history.append(BlinkEvent(timestamp=now, duration_ms=duration_ms))
                        completed_blink = True
                self._closure_start_time = None

        self._prune_history(now)
        return completed_blink

    def _prune_history(self, current_time: float):
        """Removes blink events older than the rolling time window."""
        cutoff = current_time - self.history_window_seconds
        self.blink_history = [b for b in self.blink_history if b.timestamp >= cutoff]

    @property
    def blinks_per_minute(self) -> float:
        """Calculates rolling blink rate per minute over the history window."""
        if not self.blink_history:
            return 0.0
        return float(len(self.blink_history))

    def reset(self):
        """Resets all blink detector state."""
        self.total_blinks = 0
        self.blink_history.clear()
        self._is_closing = False
        self._closure_start_time = None
