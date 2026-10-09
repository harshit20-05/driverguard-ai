"""
Hybrid Decision Engine for Eye State Classification.
Combines Deep Learning (MobileNetV2/Custom CNN) probabilities with Geometric Eye Aspect Ratio (EAR).
Dynamically adjusts weights based on model confidence and lighting conditions.
"""

from dataclasses import dataclass
from typing import Optional
from collections import deque
import numpy as np

from .inference import EyeStateClassifier, EyePrediction
from ..vision.eye_detector import EyeData


@dataclass
class HybridDecisionResult:
    """Ensemble prediction merging deep features and geometric landmarks."""
    is_closed: bool = False
    hybrid_open_prob: float = 0.5
    cnn_prob: float = 0.5
    ear_val: float = 0.25
    left_state: str = "OPEN"
    right_state: str = "OPEN"
    confidence: float = 0.5
    latency_ms: float = 0.0
    engine: str = "HYBRID_ENSEMBLE"


class HybridEyeClassifier:
    """
    Ensemble decision maker combining CNN inference with calibrated Eye Aspect Ratio (EAR).
    Protects against edge-case CNN failures (e.g. glare on glasses, extreme angles)
    while leveraging CNN's superior texture discrimination for subtle eye states.
    """
    def __init__(
        self,
        cnn_classifier: EyeStateClassifier,
        cnn_weight: float = 0.65,
        ear_threshold: float = 0.20,
        smoothing_window: int = 5
    ):
        self.cnn_classifier = cnn_classifier
        self.cnn_weight = cnn_weight
        self.ear_threshold = ear_threshold
        self.smoothing_window = smoothing_window

        self._history = deque(maxlen=smoothing_window)

    def set_ear_threshold(self, threshold: float):
        """Allows dynamic adjustment following driver calibration."""
        self.ear_threshold = float(np.clip(threshold, 0.14, 0.28))

    def predict(self, eye_data: EyeData) -> HybridDecisionResult:
        """
        Runs hybrid evaluation on extracted eye crops and landmarks.
        """
        if not eye_data.eyes_detected:
            return HybridDecisionResult(
                is_closed=False,
                hybrid_open_prob=0.5,
                cnn_prob=0.5,
                ear_val=0.0,
                left_state="UNKNOWN",
                right_state="UNKNOWN",
                confidence=0.0,
                latency_ms=0.0,
                engine="NO_EYES"
            )

        # 1. Run CNN classification
        cnn_pred: EyePrediction = self.cnn_classifier.predict(eye_data)

        # 2. Compute EAR geometric open probability
        # Normal open eye EAR is ~0.28-0.34, closed is ~0.14-0.18
        avg_ear = (eye_data.left_ear + eye_data.right_ear) / 2.0
        # Smooth sigmoid around ear_threshold
        scale = 35.0
        ear_prob = float(1.0 / (1.0 + np.exp(-scale * (avg_ear - self.ear_threshold))))

        # 3. Dynamic weighting: If CNN is fallback, use 100% EAR.
        # If CNN confidence is low (<0.4), increase EAR weight.
        if not self.cnn_classifier.is_model_loaded or cnn_pred.engine == "GEOMETRIC_FALLBACK":
            effective_cnn_w = 0.0
            engine_str = "GEOMETRIC_FALLBACK"
        else:
            effective_cnn_w = self.cnn_weight
            engine_str = "HYBRID_ENSEMBLE"

        hybrid_prob = float(effective_cnn_w * cnn_pred.avg_prob + (1.0 - effective_cnn_w) * ear_prob)

        # Rolling temporal smoothing
        self._history.append(hybrid_prob)
        smoothed_prob = float(np.mean(self._history))

        open_thresh = self.cnn_classifier.open_threshold
        is_closed = smoothed_prob < open_thresh

        left_state = "OPEN" if cnn_pred.left_prob >= open_thresh else "CLOSED"
        right_state = "OPEN" if cnn_pred.right_prob >= open_thresh else "CLOSED"

        confidence = float(np.clip(abs(smoothed_prob - 0.5) * 2.0, 0.0, 1.0))

        return HybridDecisionResult(
            is_closed=is_closed,
            hybrid_open_prob=smoothed_prob,
            cnn_prob=cnn_pred.avg_prob,
            ear_val=avg_ear,
            left_state=left_state,
            right_state=right_state,
            confidence=confidence,
            latency_ms=cnn_pred.latency_ms,
            engine=engine_str
        )
