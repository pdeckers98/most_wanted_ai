"""Frame grabber using dxcam for screen capture."""

import dxcam
import numpy as np


class FrameGrabber:
    """Acquires frames from screen using dxcam with region capture."""

    def __init__(self, region, target_fps=30):
        """Initialize the frame grabber.

        Args:
            region: Tuple (left, top, right, bottom) in screen coordinates.
            target_fps: Target frames per second (default 30).
        """
        self.region = region
        self.target_fps = target_fps
        self._camera = dxcam.create(output_color="GRAY")
        self._running = False

    def start(self):
        """Start capturing frames at target FPS."""
        if self._camera is not None:
            self._camera.start(region=self.region, target_fps=self.target_fps)
            self._running = True

    def stop(self):
        """Stop capturing frames."""
        if self._camera is not None and self._running:
            self._camera.stop()
            self._running = False

    def get_frame(self):
        """Get the latest captured frame.

        Returns:
            numpy.ndarray: Latest frame (greyscale), or None if no frame available.
        """
        if self._camera is None or not self._running:
            return None

        frame = self._camera.get_latest_frame()
        return frame

    def close(self):
        """Close the camera and release resources."""
        if self._camera is not None:
            self.stop()
            self._camera = None

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
