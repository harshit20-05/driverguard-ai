"""
Unit tests for new detection, audio, calibration, and hybrid classification features.
"""

import pytest
import numpy as np

from src.utils.audio_generator import SoundLibrary, AudioAlertManager
from src.vision.yawn_detector import YawnDetector, YawnStatus
from src.vision.head_pose import HeadPoseEstimator, HeadPoseStatus
from src.detection.calibration import DriverCalibrator
from src.models.hybrid_classifier import HybridEyeClassifier
from src.vision.eye_detector import EyeData


def test_sound_library_generation():
    """Verify synthetic WAV generation across all sound library presets."""
    for st in ["beep", "soft_chime", "double_beep", "siren", "rising"]:
        wav_bytes = SoundLibrary.generate_tone(st, volume=0.7)
        assert isinstance(wav_bytes, bytes)
        assert len(wav_bytes) > 1000  # valid WAV header + PCM payload
        # Standard WAV header check: starts with RIFF and has WAVE tag
        assert wav_bytes[:4] == b"RIFF"
        assert wav_bytes[8:12] == b"WAVE"


def test_audio_alert_manager():
    """Verify escalation and cooldown logic in AudioAlertManager."""
    mgr = AudioAlertManager(volume=0.8, repeat_interval_sec=1.0, enabled=True)
    html1 = mgr.trigger("WARNING", force=True)
    assert html1 is not None
    assert "data:audio/wav;base64," in html1

    # Immediate second call should be throttled by cooldown
    html2 = mgr.trigger("WARNING", force=False)
    assert html2 is None

    # Muted manager returns None
    mgr.enabled = False
    assert mgr.trigger("CRITICAL", force=True) is None


def test_yawn_detector_mar():
    """Verify MAR calculation and sustained yawn event transition."""
    detector = YawnDetector(mar_threshold=0.60, min_yawn_duration_sec=1.0)
    
    # Create fake landmarks (478, 3)
    landmarks = np.zeros((478, 3), dtype=np.float32)
    # Resting closed mouth: left=(100, 200), right=(200, 200), top=(150, 195), bot=(150, 205)
    landmarks[78] = [100, 200, 0]
    landmarks[308] = [200, 200, 0]
    landmarks[13] = [150, 195, 0]
    landmarks[14] = [150, 205, 0]
    landmarks[81] = [130, 196, 0]
    landmarks[178] = [130, 204, 0]
    landmarks[311] = [170, 196, 0]
    landmarks[402] = [170, 204, 0]

    mar_resting = detector.calculate_mar(landmarks)
    assert mar_resting < 0.25

    status = detector.update(landmarks, current_time=0.0)
    assert not status.is_yawning
    assert detector.total_yawns == 0

    # Simulate wide open mouth: top=(150, 160), bot=(150, 240) -> vertical = 80, horiz = 100
    landmarks[13] = [150, 160, 0]
    landmarks[14] = [150, 240, 0]
    landmarks[81] = [130, 165, 0]
    landmarks[178] = [130, 235, 0]
    landmarks[311] = [170, 165, 0]
    landmarks[402] = [170, 235, 0]

    mar_yawn = detector.calculate_mar(landmarks)
    assert mar_yawn >= 0.65

    # Frame 1: yawn start
    status1 = detector.update(landmarks, current_time=1.0)
    assert not status1.is_yawning  # Not sustained yet

    # Frame 2: sustained for 1.3s
    status2 = detector.update(landmarks, current_time=2.3)
    assert status2.is_yawning
    assert detector.total_yawns == 1


def test_driver_calibrator():
    """Verify calibration collects samples and computes adaptive thresholds."""
    calib = DriverCalibrator(target_duration_sec=2.0, min_samples=5)
    calib.start()
    assert calib.is_calibrating

    # Feed samples
    for t in np.linspace(0.0, 2.5, 10):
        done, progress, res = calib.update(ear=0.30, mar=0.22, pitch=2.0, yaw=-1.0, current_time=t)
        if done:
            break

    assert not calib.is_calibrating
    assert res.completed
    assert res.adaptive_ear_threshold < 0.30
    assert res.adaptive_ear_threshold >= 0.16
