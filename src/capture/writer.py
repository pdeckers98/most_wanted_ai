"""Asynchronous frame writer with session management."""

import json
import queue
import threading
from datetime import datetime
from pathlib import Path

import numpy as np


class FrameWriter:
    """Writes frames to disk asynchronously using a background thread."""

    def __init__(self, session_dir, max_queue_size=300):
        """Initialize the frame writer.

        Args:
            session_dir: Directory to save frames for this session.
            max_queue_size: Maximum number of frames to buffer in queue.
        """
        self.session_dir = Path(session_dir)
        self.session_dir.mkdir(parents=True, exist_ok=True)

        self._frame_queue = queue.Queue(maxsize=max_queue_size)
        self._stop_event = threading.Event()
        self._writer_thread = threading.Thread(target=self._writer_loop, daemon=False)
        self._frame_count = 0
        self._start_time = datetime.now()

        self._writer_thread.start()

    def write(self, frame, frame_number):
        """Queue a frame for writing to disk (non-blocking).

        Args:
            frame: Numpy array to save.
            frame_number: Frame index for filename.

        Raises:
            queue.Full: If queue buffer is exhausted.
        """
        self._frame_queue.put((frame, frame_number), block=False)

    def _writer_loop(self):
        """Background thread loop that writes frames from queue to disk."""
        while True:
            try:
                item = self._frame_queue.get(timeout=0.5)
                if item is None:  # Sentinel value for shutdown
                    break

                frame, frame_number = item
                filename = self.session_dir / f"frame_{frame_number:06d}.npy"
                np.save(filename, frame)
                self._frame_count += 1

            except queue.Empty:
                # Check if stop was requested
                if self._stop_event.is_set():
                    break

    def flush(self):
        """Wait for all queued frames to be written."""
        self._frame_queue.join()

    def close(self):
        """Close the writer and save session metadata."""
        # Signal stop and wait for thread
        self._stop_event.set()
        self._frame_queue.put(None)  # Sentinel
        self._writer_thread.join(timeout=5.0)

        # Save session metadata
        end_time = datetime.now()
        duration = (end_time - self._start_time).total_seconds()

        metadata = {
            'session_start': self._start_time.isoformat(),
            'session_end': end_time.isoformat(),
            'duration_seconds': duration,
            'total_frames': self._frame_count,
            'output_width': 480,
            'output_height': 384,
        }

        metadata_path = self.session_dir / 'session_meta.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
