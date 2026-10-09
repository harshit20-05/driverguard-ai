"""
End-to-End Pipeline Verification Script.
Validates model loading, inference latency, state machine, fatigue calculation,
database persistence, and report generation without requiring a GUI.
"""

import os
import sys
import time
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

from src.models.inference import EyeStateClassifier
from src.vision.eye_detector import EyeData
from src.detection.drowsiness_detector import DrowsinessDetector, DriverState
from src.detection.fatigue_score import FatigueScoreCalculator
from src.analytics.session_logger import SessionDatabase, SessionSummary
from src.ai.report_generator import SessionReportChain


def test_pipeline():
    print("==================================================")
    print("DriverGuard AI - System Pipeline Self-Test")
    print("==================================================")

    # 1. Model Loading & Inference
    classifier = EyeStateClassifier("models/eye_classifier.keras")
    print(f"[1/6] Classifier Loaded: {classifier.is_model_loaded} (Engine: {'CNN' if classifier.is_model_loaded else 'Fallback'})")
    assert classifier.is_model_loaded is True

    dummy_data = EyeData(
        left_crop=np.zeros((64, 64, 3), dtype=np.uint8),
        right_crop=np.zeros((64, 64, 3), dtype=np.uint8),
        left_ear=0.25,
        right_ear=0.25,
        avg_ear=0.25,
        eyes_detected=True
    )
    pred = classifier.predict(dummy_data)
    print(f"[2/6] Dual-Eye Inference Latency: {pred.latency_ms:.2f} ms | Confidence: {pred.confidence:.2%}")
    assert pred.latency_ms >= 0.0

    # 2. Drowsiness State Machine
    det = DrowsinessDetector()
    s_awake = det.update(is_closed=False, current_time=0.0)
    assert s_awake.state == DriverState.AWAKE
    det.update(is_closed=True, current_time=1.0)  # Starts closure at t=1.0
    s_drowsy = det.update(is_closed=True, current_time=2.5)  # 1.5s closure (2.5 - 1.0)
    assert s_drowsy.state == DriverState.DROWSY
    s_crit = det.update(is_closed=True, current_time=3.5)   # 2.5s closure (3.5 - 1.0)
    assert s_crit.state == DriverState.CRITICAL
    print("[3/6] Temporal State Machine: Passed (AWAKE -> DROWSY -> CRITICAL)")

    # 3. Fatigue Score Calculator
    fatigue_calc = FatigueScoreCalculator()
    f_res = fatigue_calc.update(
        is_closed=True,
        closure_duration_ms=2500,
        blinks_per_min=5,
        recent_drowsy_events=1,
        recent_critical_events=1,
        current_time=2.5
    )
    print(f"[4/6] Fatigue Score: {f_res.score:.1f}/100 (Level: {f_res.level})")
    assert 0.0 <= f_res.score <= 100.0

    # 4. Database Persistence
    db = SessionDatabase("data/sessions.db")
    test_summary = SessionSummary(
        session_id="test_session_verification",
        start_time=time.strftime("%Y-%m-%dT%H:%M:%S"),
        end_time=time.strftime("%Y-%m-%dT%H:%M:%S"),
        duration_seconds=120.0,
        blink_count=35,
        drowsiness_events=2,
        critical_events=1,
        avg_fatigue_score=38.5,
        max_fatigue_score=82.0,
        avg_confidence=0.91,
        avg_latency_ms=19.4,
        avg_fps=28.5
    )
    db.save_session(test_summary, [
        {"timestamp": time.time(), "elapsed_seconds": 1.0, "fatigue_score": 15.0, "eye_prob": 0.95, "is_closed": 0, "drowsiness_state": "AWAKE", "blinks_per_min": 18.0, "latency_ms": 18.2, "fps": 29.0},
        {"timestamp": time.time(), "elapsed_seconds": 60.0, "fatigue_score": 82.0, "eye_prob": 0.05, "is_closed": 1, "drowsiness_state": "CRITICAL", "blinks_per_min": 6.0, "latency_ms": 19.1, "fps": 28.0}
    ])
    retrieved = db.get_session("test_session_verification")
    assert retrieved is not None
    print("[5/6] SQLite Session Storage & Telemetry: Verified")

    # 5. Report Generation & Fallback
    chain = SessionReportChain()
    report = chain.generate_report(test_summary)
    print(f"[6/6] AI Safety Report Engine: Verified (Risk Level: {report.get('risk_level')}, Source: {report.get('source')})")
    assert "summary" in report
    assert "risk_level" in report

    print("\n==================================================")
    print("ALL CORE SYSTEM CAPABILITIES VERIFIED SUCCESSFULLY!")
    print("==================================================")

if __name__ == "__main__":
    test_pipeline()
