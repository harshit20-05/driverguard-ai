"""
Unit tests for Model evaluation logic and inference pipeline fallbacks.
"""

import numpy as np
import pytest
from src.models.evaluation import compute_metrics, load_model_metrics
from src.models.inference import EyeStateClassifier
from src.vision.eye_detector import EyeData


def test_compute_metrics_accuracy():
    """Validates metric calculations using synthetic ground truth and predictions."""
    y_true = np.array([0, 0, 1, 1, 1, 0, 1, 0])
    y_pred_prob = np.array([0.1, 0.2, 0.9, 0.8, 0.7, 0.8, 0.9, 0.3])  # 1 mistake on sample index 5

    metrics = compute_metrics(y_true, y_pred_prob, threshold=0.5)

    assert metrics["status"] == "Measured"
    assert metrics["accuracy"] == 0.875
    assert metrics["confusion_matrix"] is not None
    assert metrics["precision"] > 0
    assert metrics["recall"] > 0
    assert metrics["f1_score"] > 0


def test_unmeasured_metrics_default():
    """Unmeasured model metrics must return 'Not measured yet'."""
    metrics = load_model_metrics("non_existent_path.json")
    assert metrics["accuracy"] == "Not measured yet"
    assert metrics["precision"] == "Not measured yet"
    assert metrics["recall"] == "Not measured yet"
    assert metrics["f1_score"] == "Not measured yet"


def test_inference_geometric_fallback():
    """In the absence of a trained .keras weights file, classifier falls back gracefully to geometric baseline."""
    classifier = EyeStateClassifier(weights_path="models/non_existent.keras")
    assert classifier.is_model_loaded is False

    dummy_eye_data = EyeData(
        left_crop=np.zeros((30, 30, 3), dtype=np.uint8),
        right_crop=np.zeros((30, 30, 3), dtype=np.uint8),
        left_ear=0.28,
        right_ear=0.28,
        avg_ear=0.28,
        eyes_detected=True
    )

    pred = classifier.predict(dummy_eye_data)
    assert pred.engine == "GEOMETRIC_FALLBACK"
    assert pred.left_state == "OPEN"
    assert pred.latency_ms >= 0.0
