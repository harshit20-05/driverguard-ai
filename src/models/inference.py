"""
ML Inference Pipeline for Eye State Classification.
Provides batched inference, latency tracking, temporal smoothing, and fallback handling.
"""

from dataclasses import dataclass
from typing import Optional, List
from collections import deque
import time
import os
import numpy as np

from ..vision.preprocessing import EyePreprocessor
from ..vision.eye_detector import EyeData


@dataclass
class EyePrediction:
    """Encapsulates inference output for a video frame."""
    left_prob: float = 0.5
    right_prob: float = 0.5
    avg_prob: float = 0.5
    left_state: str = "OPEN"
    right_state: str = "OPEN"
    is_closed: bool = False
    confidence: float = 0.5
    latency_ms: float = 0.0
    engine: str = "CNN"  # "CNN" or "GEOMETRIC_FALLBACK"


class EyeStateClassifier:
    """
    High-performance eye state classifier with model caching and temporal smoothing.
    """
    _instance = None  # Singleton pattern to prevent repeated model reloads

    def __init__(
        self,
        weights_path: str = "models/eye_classifier.keras",
        open_threshold: float = 0.50,
        smoothing_window: int = 5,
        preprocessor: Optional[EyePreprocessor] = None
    ):
        self.weights_path = weights_path
        self.open_threshold = open_threshold
        self.smoothing_window = smoothing_window
        self.preprocessor = preprocessor or EyePreprocessor()

        self._model = None
        self._model_loaded = False
        self._history = deque(maxlen=smoothing_window)

        self._load_model()

    def _load_model(self) -> bool:
        """Loads the trained Keras model if the file exists."""
        if not os.path.exists(self.weights_path):
            self._model_loaded = False
            return False

        try:
            import tensorflow as tf
            # Load weights safely
            self._model = tf.keras.models.load_model(self.weights_path)
            self._model_loaded = True
            # Warm-up inference with dummy input
            dummy = np.zeros((2, 64, 64, 3), dtype=np.float32)
            _ = self._model(dummy, training=False)
            return True
        except Exception:
            self._model_loaded = False
            return False

    @property
    def is_model_loaded(self) -> bool:
        return self._model_loaded

    def predict(self, eye_data: EyeData) -> EyePrediction:
        """
        Classifies left and right eye states from extracted eye crops.
        
        Args:
            eye_data: EyeData object containing crops and geometric metrics
            
        Returns:
            EyePrediction with probabilities, states, latency, and engine type
        """
        if not eye_data.eyes_detected:
            return EyePrediction(
                left_prob=0.5,
                right_prob=0.5,
                avg_prob=0.5,
                left_state="UNKNOWN",
                right_state="UNKNOWN",
                is_closed=False,
                confidence=0.0,
                latency_ms=0.0,
                engine="NO_EYES"
            )

        start_time = time.perf_counter()

        if self._model_loaded and self._model is not None:
            # 1. Prepare batch of 2 eyes (left, right)
            batch = self.preprocessor.prepare_batch(eye_data.left_crop, eye_data.right_crop)
            # 2. Run batched CNN inference
            preds = self._model(batch, training=False).numpy()
            left_prob = float(preds[0][0])
            right_prob = float(preds[1][0])
            engine_type = "CNN"
        else:
            # Calibrated geometric EAR fallback when trained weights are not yet present on disk
            # Typical eye EAR: Closed < 0.20, Open > 0.25
            left_prob = float(np.clip((eye_data.left_ear - 0.16) / (0.30 - 0.16), 0.0, 1.0))
            right_prob = float(np.clip((eye_data.right_ear - 0.16) / (0.30 - 0.16), 0.0, 1.0))
            engine_type = "GEOMETRIC_FALLBACK"

        latency_ms = (time.perf_counter() - start_time) * 1000.0

        # Temporal smoothing over rolling window
        raw_avg = (left_prob + right_prob) / 2.0
        self._history.append(raw_avg)
        smoothed_prob = float(np.mean(self._history))

        left_state = "OPEN" if left_prob >= self.open_threshold else "CLOSED"
        right_state = "OPEN" if right_prob >= self.open_threshold else "CLOSED"

        # An eye closure event occurs when smoothed open probability drops below threshold
        is_closed = smoothed_prob < self.open_threshold

        # Confidence is distance from 0.5 decision boundary
        confidence = float(np.clip(abs(smoothed_prob - 0.5) * 2.0, 0.0, 1.0))

        return EyePrediction(
            left_prob=left_prob,
            right_prob=right_prob,
            avg_prob=smoothed_prob,
            left_state=left_state,
            right_state=right_state,
            is_closed=is_closed,
            confidence=confidence,
            latency_ms=latency_ms,
            engine=engine_type
        )
