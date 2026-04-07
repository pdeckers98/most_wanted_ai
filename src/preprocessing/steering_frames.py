"""Extract and stratify steering frames from raw capture sessions."""

import argparse
import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

TARGET_STEERING_FRAMES = 20000


def load_all_sessions(captures_dir):
    """Load frames and inputs from all sessions in captures_dir.

    Args:
        captures_dir: Root directory containing session subdirectories.

    Returns:
        List of (frames, inputs) tuples, one per session.
    """
    captures_dir = Path(captures_dir)
    session_dirs = sorted(
        p for p in captures_dir.iterdir()
        if p.is_dir() and (p / "inputs.json").exists()
    )

    sessions = []
    for session_dir in session_dirs:
        try:
            frame_files = sorted(session_dir.glob("frame_*.npy"))
            with open(session_dir / "inputs.json") as f:
                inputs = json.load(f)

            if len(frame_files) != len(inputs):
                logger.warning(
                    "Frame count (%d) != input count (%d) in %s, skipping",
                    len(frame_files), len(inputs), session_dir.name
                )
                continue

            frames = np.stack(
                [np.load(fp) for fp in frame_files], axis=0
            )
            sessions.append((frames, inputs))
            logger.info(
                "Loaded session %s: %d frames", session_dir.name, len(frames)
            )
        except Exception as e:
            logger.error("Failed to load session %s: %s", session_dir.name, e)

    return sessions


def filter_steering_frames(frames, inputs):
    """Filter frames where steering != 0.

    Args:
        frames: np.ndarray of shape (N, 384, 480) uint8.
        inputs: List of N dicts with keys 'steer', 'throttle', 'brake'.

    Returns:
        Tuple of (steering_frames, steering_inputs) where both are filtered.
    """
    mask = np.array([inp["steer"] != 0 for inp in inputs])
    steering_frames = frames[mask]
    steering_inputs = [inp for i, inp in enumerate(inputs) if mask[i]]
    return steering_frames, steering_inputs


def stratify_by_steering_magnitude(frames, inputs, n_bins=5):
    """Stratify frames by steering magnitude (for balance across turn sizes).

    Args:
        frames: np.ndarray of shape (N, 384, 480).
        inputs: List of N dicts.
        n_bins: Number of steering magnitude bins.

    Returns:
        List of (frame, input) tuples, stratified by magnitude.
    """
    steers = np.array([inp["steer"] for inp in inputs])
    steer_mag = np.abs(steers)

    # Create bins by magnitude percentiles
    edges = np.percentile(steer_mag, np.linspace(0, 100, n_bins + 1))
    bins = np.digitize(steer_mag, edges)

    # Collect indices by bin
    bin_indices = {i: [] for i in range(1, n_bins + 2)}
    for idx, b in enumerate(bins):
        bin_indices[b].append(idx)

    # Stratified list: cycle through bins, taking one from each
    stratified = []
    max_bin_size = max(len(indices) for indices in bin_indices.values())

    for slot in range(max_bin_size):
        for b in range(1, n_bins + 2):
            if slot < len(bin_indices[b]):
                idx = bin_indices[b][slot]
                stratified.append((frames[idx], inputs[idx]))

    return stratified


def stratify_by_action_combo(stratified_list, target_count):
    """Further stratify by (throttle, brake) combinations.

    Args:
        stratified_list: List of (frame, input) tuples.
        target_count: Target number of frames (20000).

    Returns:
        List of (frame, input) tuples, stratified and sampled to target_count.
    """
    # Group by (throttle, brake) combo
    combos = {}
    for frame, inp in stratified_list:
        key = (int(inp["throttle"]), int(inp["brake"]))
        if key not in combos:
            combos[key] = []
        combos[key].append((frame, inp))

    logger.info("Found %d action combinations: %s", len(combos), list(combos.keys()))
    for key, items in combos.items():
        logger.info("  %s: %d frames", key, len(items))

    # If we have fewer frames than target, return all
    total = sum(len(items) for items in combos.values())
    if total <= target_count:
        logger.warning(
            "Total steering frames (%d) < target (%d), returning all",
            total, target_count
        )
        return stratified_list

    # Otherwise, sample proportionally from each combo
    frames_per_combo = {}
    remaining = target_count
    for key in combos.keys():
        share = int(len(combos[key]) / total * target_count)
        frames_per_combo[key] = min(share, len(combos[key]))
        remaining -= frames_per_combo[key]

    # Distribute remainder to largest combos
    sorted_keys = sorted(
        combos.keys(),
        key=lambda k: len(combos[k]),
        reverse=True
    )
    for key in sorted_keys:
        if remaining <= 0:
            break
        available = len(combos[key]) - frames_per_combo[key]
        take = min(available, remaining)
        frames_per_combo[key] += take
        remaining -= take

    # Sample from each combo
    result = []
    for key in combos.keys():
        items = combos[key]
        count = frames_per_combo[key]
        indices = np.random.choice(len(items), size=count, replace=False)
        result.extend([items[i] for i in indices])

    return result


def collect_steering_frames(captures_dir, output_dir, target_count=TARGET_STEERING_FRAMES):
    """Collect and stratify steering frames from all sessions.

    Args:
        captures_dir: Root directory of raw sessions (data/captures).
        output_dir: Output directory for steering_frames.
        target_count: Target number of frames to collect (default 20000).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading all sessions from %s", captures_dir)
    sessions = load_all_sessions(captures_dir)

    if not sessions:
        logger.error("No sessions loaded.")
        return

    # Combine all frames and inputs
    all_frames = []
    all_inputs = []
    for frames, inputs in sessions:
        all_frames.append(frames)
        all_inputs.extend(inputs)

    all_frames = np.vstack(all_frames)
    logger.info("Loaded %d total frames across all sessions", len(all_frames))

    # Filter to steering frames
    steering_frames, steering_inputs = filter_steering_frames(all_frames, all_inputs)
    logger.info("Found %d steering frames (steering != 0)", len(steering_frames))

    if len(steering_frames) == 0:
        logger.error("No steering frames found.")
        return

    # Stratify by magnitude
    stratified = stratify_by_steering_magnitude(steering_frames, steering_inputs, n_bins=5)
    logger.info("After magnitude stratification: %d frames", len(stratified))

    # Stratify by action combo and sample to target
    final = stratify_by_action_combo(stratified, target_count)
    logger.info("Final dataset: %d frames", len(final))

    # Save to disk
    logger.info("Writing frames to %s", output_dir)
    for idx, (frame, inp) in enumerate(final):
        frame_path = output_dir / f"frame_{idx:06d}.npy"
        np.save(frame_path, frame)

        if (idx + 1) % 5000 == 0:
            logger.info("Saved %d/%d frames", idx + 1, len(final))

    # Save inputs.json
    inputs_path = output_dir / "inputs.json"
    with open(inputs_path, "w") as f:
        json.dump([inp for _, inp in final], f)

    logger.info("Done. Saved %d frames and inputs.json to %s", len(final), output_dir)


def main():
    """CLI entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Extract and stratify steering frames for focused training."
    )
    parser.add_argument(
        "--captures-dir",
        type=Path,
        default=Path("data/captures"),
        help="Root directory of raw sessions (default: data/captures)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/steering_frames"),
        help="Output directory (default: data/steering_frames)",
    )
    parser.add_argument(
        "--target",
        type=int,
        default=TARGET_STEERING_FRAMES,
        help=f"Target frame count (default: {TARGET_STEERING_FRAMES})",
    )

    args = parser.parse_args()
    collect_steering_frames(args.captures_dir, args.output_dir, args.target)


if __name__ == "__main__":
    main()
