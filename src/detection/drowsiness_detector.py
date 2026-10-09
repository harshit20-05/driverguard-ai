"""
Drowsiness Detection Engine and Finite State Machine.
Evaluates continuous eye closure durations using precise timestamps to transition between
AWAKE, BLINKING, DROWSY, and CRITICAL states.
"""

from enum import Enum
from dataclasses import dataclass
from typing import Optional, List
import time


class DriverState(str, Enum):
    AWAKE = "AWAKE"
    BLINKING = "BLINKING"
    DROWSY = "DROWSY"
    CRITICAL = "CRITICAL"


@dataclass
class DrowsinessEventRecord:
    """Record of a triggered drowsiness event."""
    start_time: float
    end_time: float
    duration_ms: float
    severity: str  # "WARNING" or "CRITICAL"


@dataclass
class DrowsinessStatus:
    """Current state evaluation for a single frame."""
    state: DriverState = DriverState.AWAKE
    closure_duration_ms: float = 0.0
    alert_triggered: bool = False
    alert_severity: str = "NONE"  # "NONE", "WARNING", "CRITICAL"
    alert_message: str = "Driver Alert"
    total_drowsy_events: int = 0
    total_critical_events: int = 0


class DrowsinessDetector:
    """
    Temporal state machine tracking driver alertness.
    Transitions:
        AWAKE -> BLINKING -> DROWSY -> CRITICAL
        CRITICAL / DROWSY -> AWAKE (upon stable eye reopening)
    """
    def __init__(
        self,
        blink_max_ms: float = 380.0,
        warning_threshold_ms: float = 1200.0,
        critical_threshold_ms: float = 2200.0,
        recovery_open_frames: int = 6
    ):
        self.blink_max_ms = blink_max_ms
        self.warning_threshold_ms = warning_threshold_ms
        self.critical_threshold_ms = critical_threshold_ms
        self.recovery_open_frames = recovery_open_frames

        self.current_state: DriverState = DriverState.AWAKE
        self.closure_start_time: Optional[float] = None
        self.open_frames_counter: int = 0

        self.total_drowsy_events: int = 0
        self.total_critical_events: int = 0
        self.event_history: List[DrowsinessEventRecord] = []

        self._active_event_start: Optional[float] = None
        self._active_event_severity: Optional[str] = None

    def update(self, is_closed: bool, current_time: Optional[float] = None) -> DrowsinessStatus:
        """
        Updates the drowsiness state machine based on the eye state of the current frame.
        
        Args:
            is_closed: True if eyes are currently classified as closed
            current_time: High-precision timestamp (seconds). Uses time.time() if None.
            
        Returns:
            DrowsinessStatus dataclass with current state, durations, and alerts.
        """
        now = current_time if current_time is not None else time.time()
        duration_ms = 0.0

        if is_closed:
            self.open_frames_counter = 0

            if self.closure_start_time is None:
                self.closure_start_time = now

            duration_ms = (now - self.closure_start_time) * 1000.0

            # State transition logic
            if duration_ms >= self.critical_threshold_ms:
                if self.current_state != DriverState.CRITICAL:
                    self.current_state = DriverState.CRITICAL
                    self.total_critical_events += 1
                    self._active_event_severity = "CRITICAL"
                    if self._active_event_start is None:
                        self._active_event_start = self.closure_start_time
            elif duration_ms >= self.warning_threshold_ms:
                if self.current_state not in (DriverState.DROWSY, DriverState.CRITICAL):
                    self.current_state = DriverState.DROWSY
                    self.total_drowsy_events += 1
                    self._active_event_severity = "WARNING"
                    if self._active_event_start is None:
                        self._active_event_start = self.closure_start_time
            elif duration_ms > 0:
                self.current_state = DriverState.BLINKING
        else:
            # Eyes are open
            self.open_frames_counter += 1

            if self.closure_start_time is not None:
                closure_duration = (now - self.closure_start_time) * 1000.0
                if self._active_event_start is not None and closure_duration >= self.warning_threshold_ms:
                    # Finalize logged event
                    self.event_history.append(DrowsinessEventRecord(
                        start_time=self._active_event_start,
                        end_time=now,
                        duration_ms=closure_duration,
                        severity=self._active_event_severity or "WARNING"
                    ))
                    self._active_event_start = None
                    self._active_event_severity = None

                self.closure_start_time = None

            # Smooth recovery back to AWAKE after consecutive open frames
            if self.open_frames_counter >= self.recovery_open_frames:
                self.current_state = DriverState.AWAKE

        # Determine alerts and UI messaging
        alert_triggered = False
        alert_severity = "NONE"
        alert_message = "Driver Alert"

        if self.current_state == DriverState.CRITICAL:
            alert_triggered = True
            alert_severity = "CRITICAL"
            alert_message = f"CRITICAL DROWSINESS! Eyes closed for {duration_ms / 1000.0:.1f}s!"
        elif self.current_state == DriverState.DROWSY:
            alert_triggered = True
            alert_severity = "WARNING"
            alert_message = f"WARNING: Drowsiness Detected ({duration_ms / 1000.0:.1f}s)"
        elif self.current_state == DriverState.BLINKING:
            alert_message = "Normal Blinking"

        return DrowsinessStatus(
            state=self.current_state,
            closure_duration_ms=duration_ms,
            alert_triggered=alert_triggered,
            alert_severity=alert_severity,
            alert_message=alert_message,
            total_drowsy_events=self.total_drowsy_events,
            total_critical_events=self.total_critical_events
        )

    def reset(self):
        """Resets all detector counters and states."""
        self.current_state = DriverState.AWAKE
        self.closure_start_time = None
        self.open_frames_counter = 0
        self.total_drowsy_events = 0
        self.total_critical_events = 0
        self.event_history.clear()
        self._active_event_start = None
        self._active_event_severity = None
