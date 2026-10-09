"""
Live Session Metrics Tracker.
Accumulates frame-by-frame telemetry and computes comprehensive statistical summaries.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
import time
import numpy as np

from .session_logger import SessionSummary


class SessionTracker:
    """
    Manages in-memory telemetry collection during an active monitoring session.
    """
    def __init__(self, session_id: Optional[str] = None):
        self.session_id = session_id or datetime.now().strftime("drive_%Y%m%d_%H%M%S")
        self.start_time = datetime.now()
        self.start_timestamp = time.time()
        self.is_active = True

        # Telemetry samples (logged periodically, e.g. every 0.5s or frame)
        self.samples: List[Dict[str, Any]] = []
        self.events: List[Dict[str, Any]] = []

        # Running tallies
        self.total_frames = 0
        self.blink_count = 0
        self.drowsiness_events = 0
        self.critical_events = 0
        self.yawn_events = 0
        self.nod_events = 0
        self.distraction_events = 0

        self.fatigue_scores: List[float] = []
        self.confidences: List[float] = []
        self.latencies_ms: List[float] = []
        self.fps_values: List[float] = []

    def record_event(self, event_type: str, severity: str, duration_ms: float = 0.0, details: str = ""):
        """Records an isolated safety event for timeline auditing."""
        now = time.time()
        elapsed = now - self.start_timestamp
        self.events.append({
            "timestamp": now,
            "elapsed_seconds": round(elapsed, 2),
            "event_type": event_type,
            "severity": severity,
            "duration_ms": round(duration_ms, 1),
            "details": details
        })
        if event_type == "YAWN":
            self.yawn_events += 1
        elif event_type == "HEAD_NOD":
            self.nod_events += 1
        elif "DISTRACTION" in event_type or "PHONE" in event_type:
            self.distraction_events += 1

    def record_frame(
        self,
        fatigue_score: float,
        eye_prob: float,
        is_closed: bool,
        drowsiness_state: str,
        blinks_per_min: float,
        confidence: float,
        latency_ms: float,
        fps: float,
        blink_occurred: bool = False,
        new_drowsy_event: bool = False,
        new_critical_event: bool = False
    ):
        """Records metrics for the current processed frame."""
        now = time.time()
        elapsed = now - self.start_timestamp

        self.total_frames += 1
        if blink_occurred:
            self.blink_count += 1
        if new_drowsy_event:
            self.drowsiness_events += 1
        if new_critical_event:
            self.critical_events += 1

        self.fatigue_scores.append(fatigue_score)
        self.confidences.append(confidence)
        self.latencies_ms.append(latency_ms)
        if fps > 0:
            self.fps_values.append(fps)

        # Store periodic sample (1 sample every 0.5s or on key event)
        if not self.samples or (elapsed - self.samples[-1]["elapsed_seconds"]) >= 0.5:
            self.samples.append({
                "timestamp": now,
                "elapsed_seconds": round(elapsed, 2),
                "fatigue_score": round(fatigue_score, 1),
                "eye_prob": round(eye_prob, 3),
                "is_closed": int(is_closed),
                "drowsiness_state": drowsiness_state,
                "blinks_per_min": round(blinks_per_min, 1),
                "latency_ms": round(latency_ms, 2),
                "fps": round(fps, 1)
            })

    def finish(self) -> SessionSummary:
        """Finalizes the session and produces a SessionSummary."""
        self.is_active = False
        end_time = datetime.now()
        duration_sec = time.time() - self.start_timestamp

        avg_fatigue = float(np.mean(self.fatigue_scores)) if self.fatigue_scores else 0.0
        max_fatigue = float(np.max(self.fatigue_scores)) if self.fatigue_scores else 0.0
        avg_conf = float(np.mean(self.confidences)) if self.confidences else 0.0
        avg_lat = float(np.mean(self.latencies_ms)) if self.latencies_ms else 0.0
        avg_fps = float(np.mean(self.fps_values)) if self.fps_values else 0.0

        return SessionSummary(
            session_id=self.session_id,
            start_time=self.start_time.isoformat(),
            end_time=end_time.isoformat(),
            duration_seconds=duration_sec,
            blink_count=self.blink_count,
            drowsiness_events=self.drowsiness_events,
            critical_events=self.critical_events,
            avg_fatigue_score=round(avg_fatigue, 1),
            max_fatigue_score=round(max_fatigue, 1),
            avg_confidence=round(avg_conf, 3),
            avg_latency_ms=round(avg_lat, 2),
            avg_fps=round(avg_fps, 1)
        )
