"""Frame preprocessing pipeline for resizing and greyscale conversion."""

import cv2
import numpy as np


def preprocess_frame(frame):
    """Preprocess a captured frame: resize to target dimensions and ensure greyscale.

    Args:
        frame: Input numpy array (greyscale from dxcam), shape (H, W) or (H, W, 1).

    Returns:
        numpy array: Preprocessed frame, shape (384, 480) uint8 greyscale.
    """
    # Handle case where dxcam returns (H, W, 1) instead of (H, W)
    if len(frame.shape) == 3 and frame.shape[2] == 1:
        frame = np.squeeze(frame, axis=2)

    # Ensure uint8
    if frame.dtype != np.uint8:
        frame = frame.astype(np.uint8)

    # Resize using INTER_AREA for downscaling (minimizes aliasing and artifacts)
    # Output shape will be (384, 480) which is height x width in OpenCV
    processed = cv2.resize(frame, (480, 384), interpolation=cv2.INTER_AREA)

    return processed
