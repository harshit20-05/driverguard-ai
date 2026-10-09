"""
Additional unit tests for HeadPoseEstimator and HybridEyeClassifier.
"""

import pytest
import numpy as np

from src.vision.head_pose import HeadPoseEstimator
from src.vision.eye_detector import EyeData
from src.models.inference import EyeStateClassifier, EyePrediction
from src.models.hybrid_classifier import HybridEyeClassifier


def test_head_pose_estimation():
    estimator = HeadPoseEstimator()
    landmarks = np.zeros((300, 3), dtype=np.float32)
    # Nose tip (1), Chin (152), Left Eye (33), Right Eye (263), Left Mouth (61), Right Mouth (291)
    landmarks[1] = [320, 240, 0]
    landmarks[152] = [320, 340, 0]
    landmarks[33] = [260, 200, 0]
    landmarks[263] = [380, 200, 0]
    landmarks[61] = [280, 290, 0]
    landmarks[291] = [360, 290, 0]

    pitch, yaw, roll = estimator.estimate_pose(landmarks, frame_width=640, frame_height=480)
    assert isinstance(pitch, float)
    assert isinstance(yaw, float)
    assert isinstance(roll, float)

    status = estimator.update(landmarks, frame_width=640, frame_height=480, current_time=0.0)
    assert not status.is_nodding
    assert not status.is_looking_away


def test_hybrid_eye_classifier():
    cnn = EyeStateClassifier(weights_path="nonexistent.keras", open_threshold=0.5)
    hybrid = HybridEyeClassifier(cnn_classifier=cnn, cnn_weight=0.6, ear_threshold=0.20)

    # Simulated open eye data
    eye_data = EyeData(
        left_crop=np.zeros((64, 64, 3), dtype=np.uint8),
        right_crop=np.zeros((64, 64, 3), dtype=np.uint8),
        left_bbox=(10, 10, 20, 20),
        right_bbox=(40, 10, 20, 20),
        left_ear=0.32,
        right_ear=0.32,
        eyes_detected=True
    )

    pred = hybrid.predict(eye_data)
    assert not pred.is_closed
    assert pred.hybrid_open_prob > 0.5
    assert pred.engine in ["GEOMETRIC_FALLBACK", "HYBRID_ENSEMBLE"]
