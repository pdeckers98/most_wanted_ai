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


def _build_sequences_to_memmap(frames, inputs_arr, out_path, stack_size=4,
                               chunk_size=1000, filter_steering=False):
    """Build stacked sequences and write directly to disk via memmap to avoid OOM.

    Args:
        frames: np.ndarray of shape (N, H, W) float32.
        inputs_arr: np.ndarray of shape (N, 3) float32.
        out_path: Path where frames.npy will be written.
        stack_size: Number of consecutive frames per sample (default 4).
        chunk_size: Number of sequences to process at a time in RAM (default 1000).
        filter_steering: If True, keep only sequences where label
            (last frame input) has steering != 0.

    Returns:
        Tuple of (n_sequences, aligned_inputs) where:
            n_sequences: int — number of sequences produced
            aligned_inputs: np.ndarray of shape (n_sequences, 3) float32 — labels

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

    # Compute which sequences to keep (before memmap allocation)
    aligned_inputs_temp = inputs_arr[stack_size - 1:]
    if filter_steering:
        keep_mask = aligned_inputs_temp[:, 0] != 0  # steer column (index 0)
        n_to_keep = int(keep_mask.sum())
    else:
        keep_mask = None
        n_to_keep = n_sequences

    # Pre-allocate the output file on disk as a memmap
    memmap_file = np.lib.format.open_memmap(
        out_path, mode='w+', dtype=np.float32,
        shape=(n_to_keep, stack_size, h, w)
    )

    # Fill the memmap in chunks to avoid holding all stacks in RAM
    out_idx = 0
    for start in range(0, n_sequences, chunk_size):
        end = min(start + chunk_size, n_sequences)
        for i in range(start, end):
            if keep_mask is None or keep_mask[i]:
                memmap_file[out_idx] = frames[i:i + stack_size]
                out_idx += 1
        logger.info("Built sequences %d/%d (keeping %d).", end, n_sequences, out_idx)

    # Close/flush the memmap to disk
    del memmap_file

    if filter_steering:
        aligned_inputs = aligned_inputs_temp[keep_mask]
    else:
        aligned_inputs = aligned_inputs_temp

    return n_to_keep, aligned_inputs


def build_sequences(frames, inputs_arr, stack_size=4, filter_steering=False):
    """Produce overlapping frame stacks aligned with their target inputs.

    Each output sample consists of `stack_size` consecutive frames. The label
    for a stack is the input recorded at the **last** frame of that stack.
    The first `stack_size - 1` frames of the session are dropped because there
    is not enough history to form a full window.

    Args:
        frames: np.ndarray of shape (N, H, W) float32.
        inputs_arr: np.ndarray of shape (N, 3) float32.
        stack_size: Number of consecutive frames per sample (default 4).
        filter_steering: If True, keep only sequences where label
            (last frame input) has steering != 0.

    Returns:
        Tuple of (stacked_frames, aligned_inputs) where:
            stacked_frames: np.ndarray of shape (N_kept, stack_size, H, W)
                float32.
            aligned_inputs: np.ndarray of shape (N_kept, 3) float32 — label
                is the input at the last frame of each stack.

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

    aligned_inputs = inputs_arr[stack_size - 1:]

    if filter_steering:
        keep_mask = aligned_inputs[:, 0] != 0  # steer column (index 0)
        n_to_keep = int(keep_mask.sum())
    else:
        keep_mask = None
        n_to_keep = n_sequences

    stacked = np.empty((n_to_keep, stack_size, h, w), dtype=np.float32)
    out_idx = 0
    for i in range(n_sequences):
        if keep_mask is None or keep_mask[i]:
            stacked[out_idx] = frames[i: i + stack_size]
            out_idx += 1

    if filter_steering:
        aligned_inputs = aligned_inputs[keep_mask]

    return stacked, aligned_inputs


def process_session(session_dir, output_dir, chunk_size=1000, filter_steering=False):
    """Run the full preprocessing pipeline on a single capture session.

    Loads raw frames and inputs, normalizes, encodes, filters invalid frames,
    and writes the processed dataset to output_dir.

    Args:
        session_dir: Path to the raw capture session directory.
        output_dir: Path to directory where the processed dataset will be saved.
            Created if it does not exist.
        chunk_size: Number of sequences to process in RAM at a time (default 1000).
            Reduces peak memory usage for large sessions.
        filter_steering: If True, keep only sequences where steering label != 0.

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

    frames_out_path = output_dir / FRAMES_FILENAME
    n_sequences, inputs_arr = _build_sequences_to_memmap(
        frames, inputs_arr, frames_out_path, stack_size=4, chunk_size=chunk_size,
        filter_steering=filter_steering
    )
    logger.info("Built %d sequences (stack_size=4).", n_sequences)

    np.save(output_dir / INPUTS_FILENAME, inputs_arr)

    meta = {
        "source_session": str(session_dir),
        "n_raw_frames": n_raw,
        "n_filtered_frames": n_filtered,
        "n_sequences": n_sequences,
        "frame_shape": [4, 384, 480],
        "frame_dtype": "float32",
        "input_columns": ["steer", "throttle", "brake"],
        "stack_size": 4,
        "filter_steering": filter_steering,
    }
    with open(output_dir / META_FILENAME, "w") as f:
        json.dump(meta, f, indent=2)

    logger.info("Dataset written to: %s", output_dir)
    return n_sequences


def process_all_sessions(captures_dir, processed_dir, chunk_size=1000, filter_steering=False):
    """Run the preprocessing pipeline over all sessions in captures_dir.

    Each session subdirectory in captures_dir is processed and written to a
    corresponding subdirectory in processed_dir.

    Args:
        captures_dir: Root directory containing raw session subdirectories.
        processed_dir: Root directory where processed datasets will be written.
        chunk_size: Number of sequences to process in RAM at a time (default 1000).
        filter_steering: If True, keep only sequences where steering label != 0.

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
            n = process_session(
                session_dir, out, chunk_size=chunk_size,
                filter_steering=filter_steering
            )
            results[session_dir.name] = n
        except Exception:
            logger.exception("Failed to process session: %s", session_dir)

    total = sum(results.values())
    logger.info(
        "Processed %d sessions, %d total frames.", len(results), total
    )
    return results


def process_sessions_until_target(captures_dir, processed_dir, target_frames=20000,
                                  chunk_size=1000, filter_steering=False):
    """Process sessions sequentially until reaching target frame count or exhausting sessions.

    Processes sessions in order, stopping as soon as accumulated frames >= target_frames
    or all sessions are processed.

    Args:
        captures_dir: Root directory containing raw session subdirectories.
        processed_dir: Root directory where processed datasets will be written.
        target_frames: Target number of stacked frames to accumulate (default 20000).
        chunk_size: Number of sequences to process in RAM at a time (default 1000).
        filter_steering: If True, keep only sequences where steering label != 0.

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
    total_frames = 0

    for session_dir in session_dirs:
        out = processed_dir / session_dir.name
        try:
            n = process_session(
                session_dir, out, chunk_size=chunk_size,
                filter_steering=filter_steering
            )
            results[session_dir.name] = n
            total_frames += n
            logger.info(
                "Session %s: %d frames (total: %d/%d)",
                session_dir.name, n, total_frames, target_frames
            )

            if total_frames >= target_frames:
                logger.info("Reached target of %d frames. Stopping.", target_frames)
                break

        except Exception:
            logger.exception("Failed to process session: %s", session_dir)

    logger.info(
        "Processed %d sessions, %d total frames.", len(results), total_frames
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
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Sequences to process in RAM at a time (default: 1000)",
    )
    parser.add_argument(
        "--filter-steering",
        action="store_true",
        help="Keep only sequences where steering != 0 (default: False)",
    )
    parser.add_argument(
        "--target-frames",
        type=int,
        default=None,
        help=(
            "Target frame count. Stops after reaching target or exhausting "
            "sessions. If not set, processes all sessions."
        ),
    )
    args = parser.parse_args()

    if args.session:
        session_dir = args.captures_dir / args.session
        out_dir = args.output_dir / args.session
        n = process_session(
            session_dir, out_dir, chunk_size=args.chunk_size,
            filter_steering=args.filter_steering
        )
        print(f"Done. {n} frames written to {out_dir}")
    else:
        if args.target_frames:
            results = process_sessions_until_target(
                args.captures_dir, args.output_dir,
                target_frames=args.target_frames,
                chunk_size=args.chunk_size,
                filter_steering=args.filter_steering
            )
        else:
            results = process_all_sessions(
                args.captures_dir, args.output_dir,
                chunk_size=args.chunk_size,
                filter_steering=args.filter_steering
            )
        total = sum(results.values())
        print(f"Done. {len(results)} sessions, {total} total frames.")


if __name__ == "__main__":
    main()
