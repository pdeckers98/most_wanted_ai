"""Unit tests for frame preprocessor."""

import numpy as np
from src.capture.preprocessor import preprocess_frame


def test_preprocess_frame_basic():
    """Test basic frame preprocessing with standard input shape."""
    # Create a synthetic greyscale frame (1024, 1280)
    frame = np.random.randint(0, 256, (1024, 1280), dtype=np.uint8)

    result = preprocess_frame(frame)

    assert result.shape == (384, 480)
    assert result.dtype == np.uint8
    assert np.all(result >= 0) and np.all(result <= 255)


def test_preprocess_frame_with_channel_dim():
    """Test preprocessing when frame has shape (H, W, 1)."""
    frame = np.random.randint(0, 256, (1024, 1280, 1), dtype=np.uint8)

    result = preprocess_frame(frame)

    assert result.shape == (384, 480)
    assert result.dtype == np.uint8


def test_preprocess_frame_dtype_conversion():
    """Test that dtype is converted to uint8 if needed."""
    frame = np.random.randint(0, 256, (1024, 1280), dtype=np.uint8).astype(np.float32) / 255.0

    result = preprocess_frame(frame)

    assert result.dtype == np.uint8


def test_preprocess_frame_all_black():
    """Test preprocessing all-black frame."""
    frame = np.zeros((1024, 1280), dtype=np.uint8)

    result = preprocess_frame(frame)

    assert result.shape == (384, 480)
    assert np.all(result == 0)


def test_preprocess_frame_all_white():
    """Test preprocessing all-white frame."""
    frame = np.ones((1024, 1280), dtype=np.uint8) * 255

    result = preprocess_frame(frame)

    assert result.shape == (384, 480)
    assert np.all(result == 255)


def test_preprocess_frame_preserves_values():
    """Test that preprocessing preserves general value range."""
    frame = np.ones((1024, 1280), dtype=np.uint8) * 128

    result = preprocess_frame(frame)

    # When resizing uniform image, values should remain roughly the same
    assert np.abs(np.mean(result) - 128) < 5
