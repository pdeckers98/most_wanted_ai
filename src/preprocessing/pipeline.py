"""Preprocessing pipeline: loads raw session data and produces a training dataset."""

import argparse
import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Output filenames inside each processed dataset directory
FRAMES_FILENAME = "frames.npy"
INPUTS_FILENAME = "inputs.npy"
META_FILENAME = "dataset_meta.json"

# Input dtype expected from the capture pipeline
_EXPECTED_FRAME_SHAPE = (384, 480)


def load_session(session_dir):
    """Load raw frames and inputs from a single capture session directory.

    Args:
        session_dir: Path to session directory containing frame_*.npy files
            and inputs.json.

    Returns:
        Tuple of (frames, inputs) where:
            frames: np.ndarray of shape (N, 384, 480) uint8
            inputs: list of N dicts with keys 'steer', 'throttle', 'brake'

    Raises:
        FileNotFoundError: If session_dir or inputs.json do not exist.
        ValueError: If frame count does not match input count.
    """
    session_dir = Path(session_dir)
    if not session_dir.exists():
        raise FileNotFoundError(f"Session directory not found: {session_dir}")

    frame_files = sorted(session_dir.glob("frame_*.npy"))
    if not frame_files:
        raise FileNotFoundError(f"No frame files found in: {session_dir}")

    inputs_path = session_dir / "inputs.json"
    if not inputs_path.exists():
        raise FileNotFoundError(f"inputs.json not found in: {session_dir}")

    with open(inputs_path) as f:
        inputs = json.load(f)

    if len(frame_files) != len(inputs):
        raise ValueError(
            f"Frame count ({len(frame_files)}) does not match "
            f"input count ({len(inputs)}) in {session_dir}"
        )

    frames = np.stack([np.load(fp) for fp in frame_files], axis=0)

    if frames.ndim != 3 or frames.shape[1:] != _EXPECTED_FRAME_SHAPE:
        raise ValueError(
            f"Unexpected frame shape {frames.shape[1:]}, "
            f"expected {_EXPECTED_FRAME_SHAPE}"
        )

    return frames, inputs


def normalize_frames(frames):
    """Normalize uint8 frames to float32 in [0, 1].

    Args:
        frames: np.ndarray of shape (N, H, W) uint8.

    Returns:
        np.ndarray of shape (N, H, W) float32.
    """
    return frames.astype(np.float32) / 255.0


def encode_inputs(inputs):
    """Encode a list of input dicts into a structured numpy array.

    Args:
        inputs: List of N dicts, each with keys 'steer' (float),
            'throttle' (int), 'brake' (int).

    Returns:
        np.ndarray of shape (N, 3) float32 with columns
        [steer, throttle, brake].
    """
    rows = [[entry["steer"], entry["throttle"], entry["brake"]]
            for entry in inputs]
    return np.array(rows, dtype=np.float32)


def filter_invalid_frames(frames, inputs_arr):
    """Remove frames where the image is completely black (likely a loading screen).

    Args:
        frames: np.ndarray of shape (N, H, W) float32.
        inputs_arr: np.ndarray of shape (N, 3) float32.

    Returns:
        Tuple of (filtered_frames, filtered_inputs_arr).
    """
    # A frame is considered valid if the mean pixel value is above a low threshold
    mean_per_frame = frames.mean(axis=(1, 2))
    valid_mask = mean_per_frame > 0.01
    n_removed = int((~valid_mask).sum())
    if n_removed > 0:
        logger.info("Removed %d blank/black frames.", n_removed)
    return frames[valid_mask], inputs_arr[valid_mask]


def build_sequences(frames, inputs_arr, stack_size=4):
    """Produce overlapping frame stacks aligned with their target inputs.

    Each output sample consists of `stack_size` consecutive frames. The label
    for a stack is the input recorded at the **last** frame of that stack.
    The first `stack_size - 1` frames of the session are dropped because there
    is not enough history to form a full window.

    Args:
        frames: np.ndarray of shape (N, H, W) float32.
        inputs_arr: np.ndarray of shape (N, 3) float32.
        stack_size: Number of consecutive frames per sample (default 4).

    Returns:
        Tuple of (stacked_frames, aligned_inputs) where:
            stacked_frames: np.ndarray of shape (N - stack_size + 1,
                stack_size, H, W) float32.
            aligned_inputs: np.ndarray of shape (N - stack_size + 1, 3)
                float32 — label is the input at the last frame of each stack.

    Raises:
        ValueError: If there are fewer frames than stack_size.
    """
    n = len(frames)
    if n < stack_size:
        raise ValueError(
            f"Session has {n} frames, need at least {stack_size} "
            f"to build sequences."
        )

    n_sequences = n - stack_size + 1
    h, w = frames.shape[1], frames.shape[2]

    stacked = np.empty((n_sequences, stack_size, h, w), dtype=np.float32)
    for i in range(n_sequences):
        stacked[i] = frames[i: i + stack_size]

    aligned_inputs = inputs_arr[stack_size - 1:]

    return stacked, aligned_inputs


def process_session(session_dir, output_dir):
    """Run the full preprocessing pipeline on a single capture session.

    Loads raw frames and inputs, normalizes, encodes, filters invalid frames,
    and writes the processed dataset to output_dir.

    Args:
        session_dir: Path to the raw capture session directory.
        output_dir: Path to directory where the processed dataset will be saved.
            Created if it does not exist.

    Returns:
        int: Number of frames in the processed dataset.
    """
    session_dir = Path(session_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading session: %s", session_dir)
    frames_raw, inputs_list = load_session(session_dir)
    n_raw = len(frames_raw)
    logger.info("Loaded %d frames.", n_raw)

    frames = normalize_frames(frames_raw)
    inputs_arr = encode_inputs(inputs_list)

    frames, inputs_arr = filter_invalid_frames(frames, inputs_arr)
    n_filtered = len(frames)
    logger.info(
        "After filtering: %d frames (removed %d).",
        n_filtered,
        n_raw - n_filtered,
    )

    frames, inputs_arr = build_sequences(frames, inputs_arr)
    n_sequences = len(frames)
    logger.info("Built %d sequences (stack_size=4).", n_sequences)

    np.save(output_dir / FRAMES_FILENAME, frames)
    np.save(output_dir / INPUTS_FILENAME, inputs_arr)

    meta = {
        "source_session": str(session_dir),
        "n_raw_frames": n_raw,
        "n_filtered_frames": n_filtered,
        "n_sequences": n_sequences,
        "frame_shape": list(frames.shape[1:]),
        "frame_dtype": str(frames.dtype),
        "input_columns": ["steer", "throttle", "brake"],
        "stack_size": 4,
    }
    with open(output_dir / META_FILENAME, "w") as f:
        json.dump(meta, f, indent=2)

    logger.info("Dataset written to: %s", output_dir)
    return n_sequences


def process_all_sessions(captures_dir, processed_dir):
    """Run the preprocessing pipeline over all sessions in captures_dir.

    Each session subdirectory in captures_dir is processed and written to a
    corresponding subdirectory in processed_dir.

    Args:
        captures_dir: Root directory containing raw session subdirectories.
        processed_dir: Root directory where processed datasets will be written.

    Returns:
        dict: Mapping of session name -> frame count for each processed session.
    """
    captures_dir = Path(captures_dir)
    processed_dir = Path(processed_dir)

    session_dirs = sorted(
        p for p in captures_dir.iterdir()
        if p.is_dir() and (p / "inputs.json").exists()
    )

    if not session_dirs:
        logger.warning("No sessions with inputs.json found in: %s", captures_dir)
        return {}

    results = {}
    for session_dir in session_dirs:
        out = processed_dir / session_dir.name
        try:
            n = process_session(session_dir, out)
            results[session_dir.name] = n
        except Exception:
            logger.exception("Failed to process session: %s", session_dir)

    total = sum(results.values())
    logger.info(
        "Processed %d sessions, %d total frames.", len(results), total
    )
    return results


def main():
    """CLI entry point for the preprocessing pipeline."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Preprocess NFS Most Wanted 2005 capture sessions."
    )
    parser.add_argument(
        "--captures-dir",
        type=Path,
        default=Path("data/captures"),
        help="Root directory of raw capture sessions (default: data/captures)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed"),
        help="Root directory for processed datasets (default: data/processed)",
    )
    parser.add_argument(
        "--session",
        type=str,
        default=None,
        help="Process a single session by name instead of all sessions",
    )
    args = parser.parse_args()

    if args.session:
        session_dir = args.captures_dir / args.session
        out_dir = args.output_dir / args.session
        n = process_session(session_dir, out_dir)
        print(f"Done. {n} frames written to {out_dir}")
    else:
        results = process_all_sessions(args.captures_dir, args.output_dir)
        total = sum(results.values())
        print(f"Done. {len(results)} sessions, {total} total frames.")


if __name__ == "__main__":
    main()
