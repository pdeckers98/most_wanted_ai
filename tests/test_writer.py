"""Unit tests for frame writer."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.capture.writer import FrameWriter


def test_frame_writer_write_and_read():
    """Test writing frames and reading them back."""
    with tempfile.TemporaryDirectory() as tmpdir:
        writer = FrameWriter(tmpdir)

        # Write some synthetic frames
        frames = [
            np.random.randint(0, 256, (384, 480), dtype=np.uint8) for _ in range(5)
        ]

        for i, frame in enumerate(frames):
            writer.write(frame, i)

        # Flush and close
        writer.flush()
        writer.close()

        # Verify files were written
        session_path = Path(tmpdir)
        frame_files = list(session_path.glob("frame_*.npy"))
        assert len(frame_files) == 5

        # Verify frame content
        for i, frame in enumerate(frames):
            loaded = np.load(session_path / f"frame_{i:06d}.npy")
            assert np.array_equal(loaded, frame)


def test_frame_writer_session_metadata():
    """Test that session metadata is saved correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        writer = FrameWriter(tmpdir)

        # Write a few frames
        for i in range(3):
            frame = np.zeros((384, 480), dtype=np.uint8)
            writer.write(frame, i)

        writer.flush()
        writer.close()

        # Check metadata
        session_path = Path(tmpdir)
        metadata_file = session_path / "session_meta.json"
        assert metadata_file.exists()

        with open(metadata_file) as f:
            metadata = json.load(f)

        assert metadata['total_frames'] == 3
        assert metadata['output_width'] == 480
        assert metadata['output_height'] == 384
        assert 'session_start' in metadata
        assert 'session_end' in metadata
        assert 'duration_seconds' in metadata


def test_frame_writer_sequential_numbering():
    """Test that frames are numbered sequentially."""
    with tempfile.TemporaryDirectory() as tmpdir:
        writer = FrameWriter(tmpdir)

        for i in range(10):
            frame = np.ones((384, 480), dtype=np.uint8) * i
            writer.write(frame, i)

        writer.flush()
        writer.close()

        session_path = Path(tmpdir)
        frame_files = sorted(session_path.glob("frame_*.npy"))
        assert len(frame_files) == 10

        # Verify numbering
        for i, filepath in enumerate(frame_files):
            assert filepath.name == f"frame_{i:06d}.npy"


def test_frame_writer_context_manager():
    """Test using writer as context manager."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with FrameWriter(tmpdir) as writer:
            frame = np.zeros((384, 480), dtype=np.uint8)
            writer.write(frame, 0)
            writer.flush()

        # Verify file exists after context exit
        session_path = Path(tmpdir)
        assert (session_path / "frame_000000.npy").exists()
        assert (session_path / "session_meta.json").exists()


def test_frame_writer_creates_directory():
    """Test that writer creates session directory if it doesn't exist."""
    with tempfile.TemporaryDirectory() as tmpdir:
        session_dir = Path(tmpdir) / "nested" / "session"
        assert not session_dir.exists()

        with FrameWriter(str(session_dir)) as writer:
            frame = np.zeros((384, 480), dtype=np.uint8)
            writer.write(frame, 0)
            writer.flush()

        assert session_dir.exists()
        assert (session_dir / "frame_000000.npy").exists()
