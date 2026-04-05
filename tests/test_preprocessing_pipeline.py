"""Unit tests for the preprocessing pipeline."""

import json

import numpy as np
import pytest

from src.preprocessing.pipeline import (
    encode_inputs,
    filter_invalid_frames,
    load_session,
    normalize_frames,
    process_session,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_session(tmp_path, n_frames=5, blank_indices=None):
    """Write a synthetic session directory and return its path."""
    blank_indices = blank_indices or []
    for i in range(n_frames):
        frame = np.zeros((384, 480), dtype=np.uint8)
        if i not in blank_indices:
            frame[:] = 128
        np.save(tmp_path / f"frame_{i:06d}.npy", frame)

    inputs = [
        {"steer": round(0.1 * i, 4), "throttle": 1, "brake": 0}
        for i in range(n_frames)
    ]
    with open(tmp_path / "inputs.json", "w") as f:
        json.dump(inputs, f)

    return tmp_path


# ---------------------------------------------------------------------------
# load_session
# ---------------------------------------------------------------------------

def test_load_session_returns_correct_shapes(tmp_path):
    _make_session(tmp_path, n_frames=4)
    frames, inputs = load_session(tmp_path)
    assert frames.shape == (4, 384, 480)
    assert frames.dtype == np.uint8
    assert len(inputs) == 4


def test_load_session_missing_directory():
    with pytest.raises(FileNotFoundError):
        load_session("/nonexistent/path")


def test_load_session_missing_inputs_json(tmp_path):
    np.save(tmp_path / "frame_000000.npy", np.zeros((384, 480), dtype=np.uint8))
    with pytest.raises(FileNotFoundError, match="inputs.json"):
        load_session(tmp_path)


def test_load_session_frame_input_mismatch(tmp_path):
    np.save(tmp_path / "frame_000000.npy", np.zeros((384, 480), dtype=np.uint8))
    np.save(tmp_path / "frame_000001.npy", np.zeros((384, 480), dtype=np.uint8))
    inputs = [{"steer": 0.0, "throttle": 0, "brake": 0}]
    with open(tmp_path / "inputs.json", "w") as f:
        json.dump(inputs, f)
    with pytest.raises(ValueError, match="Frame count"):
        load_session(tmp_path)


def test_load_session_no_frames(tmp_path):
    with open(tmp_path / "inputs.json", "w") as f:
        json.dump([], f)
    with pytest.raises(FileNotFoundError, match="No frame files"):
        load_session(tmp_path)


# ---------------------------------------------------------------------------
# normalize_frames
# ---------------------------------------------------------------------------

def test_normalize_frames_range():
    frames = np.array([[[0, 128, 255]]], dtype=np.uint8)
    result = normalize_frames(frames)
    assert result.dtype == np.float32
    assert result.min() >= 0.0
    assert result.max() <= 1.0


def test_normalize_frames_zero():
    frames = np.zeros((3, 384, 480), dtype=np.uint8)
    assert np.all(normalize_frames(frames) == 0.0)


def test_normalize_frames_max():
    frames = np.full((2, 384, 480), 255, dtype=np.uint8)
    result = normalize_frames(frames)
    assert np.allclose(result, 1.0)


# ---------------------------------------------------------------------------
# encode_inputs
# ---------------------------------------------------------------------------

def test_encode_inputs_shape():
    inputs = [
        {"steer": 0.5, "throttle": 1, "brake": 0},
        {"steer": -0.3, "throttle": 0, "brake": 1},
    ]
    result = encode_inputs(inputs)
    assert result.shape == (2, 3)
    assert result.dtype == np.float32


def test_encode_inputs_values():
    inputs = [{"steer": 0.25, "throttle": 1, "brake": 0}]
    result = encode_inputs(inputs)
    assert result[0, 0] == pytest.approx(0.25)
    assert result[0, 1] == pytest.approx(1.0)
    assert result[0, 2] == pytest.approx(0.0)


def test_encode_inputs_steer_range():
    inputs = [
        {"steer": -1.0, "throttle": 0, "brake": 0},
        {"steer": 1.0, "throttle": 0, "brake": 0},
    ]
    result = encode_inputs(inputs)
    assert result[0, 0] == pytest.approx(-1.0)
    assert result[1, 0] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# filter_invalid_frames
# ---------------------------------------------------------------------------

def test_filter_removes_blank_frames():
    frames = np.zeros((5, 384, 480), dtype=np.float32)
    frames[1:] = 0.5  # frames 1-4 are valid
    inputs_arr = np.ones((5, 3), dtype=np.float32)

    f_out, i_out = filter_invalid_frames(frames, inputs_arr)
    assert len(f_out) == 4
    assert len(i_out) == 4


def test_filter_keeps_all_valid_frames():
    frames = np.full((3, 384, 480), 0.5, dtype=np.float32)
    inputs_arr = np.ones((3, 3), dtype=np.float32)

    f_out, i_out = filter_invalid_frames(frames, inputs_arr)
    assert len(f_out) == 3


def test_filter_removes_all_blank():
    frames = np.zeros((3, 384, 480), dtype=np.float32)
    inputs_arr = np.ones((3, 3), dtype=np.float32)

    f_out, i_out = filter_invalid_frames(frames, inputs_arr)
    assert len(f_out) == 0


def test_filter_alignment():
    """Filtered inputs must remain aligned with filtered frames."""
    frames = np.zeros((4, 384, 480), dtype=np.float32)
    frames[0] = 0.5   # valid, steer=0.1
    frames[2] = 0.5   # valid, steer=0.3
    inputs_arr = np.array(
        [[0.1, 1, 0], [0.2, 0, 1], [0.3, 1, 0], [0.4, 0, 0]],
        dtype=np.float32,
    )

    f_out, i_out = filter_invalid_frames(frames, inputs_arr)
    assert len(f_out) == 2
    assert i_out[0, 0] == pytest.approx(0.1)
    assert i_out[1, 0] == pytest.approx(0.3)


# ---------------------------------------------------------------------------
# process_session (integration)
# ---------------------------------------------------------------------------

def test_process_session_writes_files(tmp_path):
    session_dir = tmp_path / "session_test"
    session_dir.mkdir()
    _make_session(session_dir, n_frames=6)

    out_dir = tmp_path / "processed"
    n = process_session(session_dir, out_dir)

    assert n == 6
    assert (out_dir / "frames.npy").exists()
    assert (out_dir / "inputs.npy").exists()
    assert (out_dir / "dataset_meta.json").exists()


def test_process_session_output_shapes(tmp_path):
    session_dir = tmp_path / "session_shapes"
    session_dir.mkdir()
    _make_session(session_dir, n_frames=4)

    out_dir = tmp_path / "processed_shapes"
    process_session(session_dir, out_dir)

    frames = np.load(out_dir / "frames.npy")
    inputs_arr = np.load(out_dir / "inputs.npy")

    assert frames.shape == (4, 384, 480)
    assert frames.dtype == np.float32
    assert inputs_arr.shape == (4, 3)
    assert inputs_arr.dtype == np.float32


def test_process_session_filters_blanks(tmp_path):
    session_dir = tmp_path / "session_blanks"
    session_dir.mkdir()
    _make_session(session_dir, n_frames=5, blank_indices=[1, 3])

    out_dir = tmp_path / "processed_blanks"
    n = process_session(session_dir, out_dir)

    assert n == 3


def test_process_session_meta_content(tmp_path):
    session_dir = tmp_path / "session_meta"
    session_dir.mkdir()
    _make_session(session_dir, n_frames=3)

    out_dir = tmp_path / "processed_meta"
    process_session(session_dir, out_dir)

    with open(out_dir / "dataset_meta.json") as f:
        meta = json.load(f)

    assert meta["n_raw_frames"] == 3
    assert meta["n_processed_frames"] == 3
    assert meta["frame_shape"] == [384, 480]
    assert meta["input_columns"] == ["steer", "throttle", "brake"]
