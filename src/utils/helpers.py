"""
General utility functions for DriverGuard AI.
Timing, FPS calculation, serialization, and sound notifications.
"""

import os
import sys
import time
import json
import threading
from typing import Any, Dict
import numpy as np


class FPSMeter:
    """Calculates smoothed frames per second."""
    def __init__(self, window_size: int = 30):
        self.window_size = window_size
        self.timestamps = []

    def tick(self) -> float:
        """Records a frame timestamp and returns smoothed FPS."""
        now = time.perf_counter()
        self.timestamps.append(now)
        if len(self.timestamps) > self.window_size:
            self.timestamps.pop(0)

        if len(self.timestamps) < 2:
            return 0.0

        elapsed = self.timestamps[-1] - self.timestamps[0]
        if elapsed <= 0:
            return 0.0
        return (len(self.timestamps) - 1) / elapsed


class LatencyTimer:
    """High-precision latency measurement in milliseconds."""
    def __init__(self):
        self._start_time = 0.0
        self.last_ms = 0.0

    def start(self):
        self._start_time = time.perf_counter()

    def stop(self) -> float:
        self.last_ms = (time.perf_counter() - self._start_time) * 1000.0
        return self.last_ms


def format_duration(seconds: float) -> str:
    """Formats seconds into HH:MM:SS string."""
    seconds = int(max(0, seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def calculate_fps(prev_time: float) -> tuple[float, float]:
    """Simple fps calculator returning (fps, new_time)."""
    curr = time.time()
    diff = curr - prev_time
    fps = 1.0 / diff if diff > 0 else 0.0
    return fps, curr


class NumpyJsonEncoder(json.JSONEncoder):
    """Custom JSON encoder for NumPy scalars and arrays."""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def safe_json_dump(data: Any, indent: int = 2) -> str:
    """Serializes data structure containing numpy or primitives into clean JSON string."""
    return json.dumps(data, cls=NumpyJsonEncoder, indent=indent)


_SOUND_THREAD_LOCK = threading.Lock()
_LAST_BEEP_TIME = 0.0


def trigger_alert_beep(frequency: int = 1000, duration_ms: int = 250, cooldown_sec: float = 1.5):
    """
    Plays an alert sound asynchronously without blocking the video stream.
    Safely handles Windows `winsound` or POSIX fallback.
    """
    global _LAST_BEEP_TIME
    now = time.time()
    if now - _LAST_BEEP_TIME < cooldown_sec:
        return

    _LAST_BEEP_TIME = now

    def _beep():
        try:
            if sys.platform == "win32":
                import winsound
                winsound.Beep(frequency, duration_ms)
            else:
                # Terminal bell for unix/macos fallback
                sys.stdout.write("\a")
                sys.stdout.flush()
        except Exception:
            pass  # Silent failure if sound card or driver is unavailable

    t = threading.Thread(target=_beep, daemon=True)
    t.start()
