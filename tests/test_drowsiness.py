"""
Unit tests for the temporal drowsiness detection state machine.
"""

import pytest
from src.detection.drowsiness_detector import DrowsinessDetector, DriverState


def test_awake_state_default():
    """Initial state should be AWAKE with zero closure duration."""
    detector = DrowsinessDetector()
    status = detector.update(is_closed=False, current_time=0.0)

    assert status.state == DriverState.AWAKE
    assert status.alert_triggered is False
    assert status.alert_severity == "NONE"


def test_short_closure_blinking():
    """A short closure under warning threshold should be BLINKING without alert."""
    detector = DrowsinessDetector(warning_threshold_ms=1200.0)

    detector.update(is_closed=True, current_time=1.0)
    status = detector.update(is_closed=True, current_time=1.2)  # 200 ms closure

    assert status.state == DriverState.BLINKING
    assert status.alert_triggered is False


def test_warning_threshold_trigger():
    """Prolonged closure exceeding warning threshold transitions to DROWSY."""
    detector = DrowsinessDetector(warning_threshold_ms=1200.0, critical_threshold_ms=2200.0)

    detector.update(is_closed=True, current_time=0.0)
    status = detector.update(is_closed=True, current_time=1.3)  # 1300 ms closure

    assert status.state == DriverState.DROWSY
    assert status.alert_triggered is True
    assert status.alert_severity == "WARNING"
    assert status.total_drowsy_events == 1


def test_critical_threshold_trigger():
    """Severe prolonged closure exceeding critical threshold transitions to CRITICAL."""
    detector = DrowsinessDetector(warning_threshold_ms=1200.0, critical_threshold_ms=2200.0)

    detector.update(is_closed=True, current_time=0.0)
    status = detector.update(is_closed=True, current_time=2.5)  # 2500 ms closure

    assert status.state == DriverState.CRITICAL
    assert status.alert_triggered is True
    assert status.alert_severity == "CRITICAL"
    assert status.total_critical_events == 1


def test_smooth_recovery_after_reopening():
    """Reopening eyes requires consecutive open frames before recovering to AWAKE."""
    detector = DrowsinessDetector(
        warning_threshold_ms=1200.0,
        critical_threshold_ms=2200.0,
        recovery_open_frames=3
    )

    detector.update(is_closed=True, current_time=0.0)
    detector.update(is_closed=True, current_time=1.5)  # Enters DROWSY

    # Reopening eyes: 1st frame
    s1 = detector.update(is_closed=False, current_time=1.6)
    assert s1.state == DriverState.DROWSY

    # 2nd frame
    s2 = detector.update(is_closed=False, current_time=1.7)
    assert s2.state == DriverState.DROWSY

    # 3rd frame: reaches recovery threshold
    s3 = detector.update(is_closed=False, current_time=1.8)
    assert s3.state == DriverState.AWAKE
