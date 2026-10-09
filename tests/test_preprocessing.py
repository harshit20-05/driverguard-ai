"""
Unit tests for image preprocessing and eye patch extraction.
"""

import numpy as np
import pytest
from src.vision.preprocessing import EyePreprocessor


def test_eye_preprocessor_shape():
    """Verifies that preprocessed eye crop matches expected (64, 64, 3) float32 tensor."""
    preprocessor = EyePreprocessor(target_size=(64, 64), apply_clahe=False)
    dummy_crop = np.random.randint(0, 255, (40, 50, 3), dtype=np.uint8)

    processed = preprocessor.preprocess_crop(dummy_crop)

    assert processed.shape == (64, 64, 3)
    assert processed.dtype == np.float32
    assert 0.0 <= np.min(processed) <= np.max(processed) <= 1.0


def test_eye_preprocessor_invalid_input():
    """Verifies graceful handling of empty or None crops."""
    preprocessor = EyePreprocessor(target_size=(64, 64))

    res_none = preprocessor.preprocess_crop(None)
    assert res_none.shape == (64, 64, 3)
    assert np.all(res_none == 0.0)

    res_empty = preprocessor.preprocess_crop(np.empty((0, 0, 3), dtype=np.uint8))
    assert res_empty.shape == (64, 64, 3)


def test_prepare_batch():
    """Verifies batched tensor shape (2, 64, 64, 3) for dual-eye inference."""
    preprocessor = EyePreprocessor(target_size=(64, 64))
    left = np.random.randint(0, 255, (30, 40, 3), dtype=np.uint8)
    right = np.random.randint(0, 255, (32, 42, 3), dtype=np.uint8)

    batch = preprocessor.prepare_batch(left, right)
    assert batch.shape == (2, 64, 64, 3)
    assert batch.dtype == np.float32


def test_clahe_enhancement():
    """Verifies CLAHE lighting adjustment runs without crashing."""
    preprocessor = EyePreprocessor(target_size=(64, 64), apply_clahe=True)
    low_light_crop = np.zeros((50, 50, 3), dtype=np.uint8) + 20

    enhanced = preprocessor.enhance_lighting(low_light_crop)
    assert enhanced.shape == (50, 50, 3)
