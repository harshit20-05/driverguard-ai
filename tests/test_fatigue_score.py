"""
Unit tests for the AI-derived fatigue score calculator.
"""

import pytest
from src.detection.fatigue_score import FatigueScoreCalculator


def test_fatigue_score_bounds():
    """Fatigue score should be strictly within [0.0, 100.0]."""
    calc = FatigueScoreCalculator()

    # Normal alert state
    m_low = calc.update(
        is_closed=False,
        closure_duration_ms=0.0,
        blinks_per_min=18.0,
        recent_drowsy_events=0,
        recent_critical_events=0,
        current_time=1.0
    )
    assert 0.0 <= m_low.score <= 30.0
    assert m_low.level == "LOW"

    # Severe drowsy state
    m_crit = calc.update(
        is_closed=True,
        closure_duration_ms=2500.0,
        blinks_per_min=4.0,
        recent_drowsy_events=3,
        recent_critical_events=2,
        current_time=2.0
    )
    assert 0.0 <= m_crit.score <= 100.0
    assert m_crit.level in ("HIGH", "CRITICAL")


def test_perclos_accumulation():
    """PERCLOS should reflect the proportion of closed frames in rolling window."""
    calc = FatigueScoreCalculator(window_seconds=10.0)

    # 5 closed frames out of 10
    for i in range(5):
        calc.update(is_closed=True, closure_duration_ms=200, blinks_per_min=15,
                    recent_drowsy_events=0, recent_critical_events=0, current_time=float(i))
    for i in range(5, 10):
        calc.update(is_closed=False, closure_duration_ms=0, blinks_per_min=15,
                    recent_drowsy_events=0, recent_critical_events=0, current_time=float(i))

    m = calc.update(is_closed=False, closure_duration_ms=0, blinks_per_min=15,
                    recent_drowsy_events=0, recent_critical_events=0, current_time=10.0)

    assert 0.35 <= m.perclos <= 0.60


def test_reset():
    """Reset should clear history and peak score."""
    calc = FatigueScoreCalculator()
    calc.update(is_closed=True, closure_duration_ms=2000, blinks_per_min=5,
                recent_drowsy_events=2, recent_critical_events=1, current_time=1.0)
    assert calc.max_score_seen > 0

    calc.reset()
    assert calc.max_score_seen == 0.0
    assert len(calc._frame_history) == 0
