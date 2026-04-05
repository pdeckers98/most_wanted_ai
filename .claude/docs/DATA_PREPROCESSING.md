# Data Preprocessing

Documentation for cleaning, normalising, and preparing captured gameplay data
for training.

## Overview

Raw capture sessions (individual `.npy` frames + `inputs.json`) are transformed
into training-ready datasets locally before uploading to Vast.ai. This avoids
paying for GPU time during data preparation.

**Pipeline stages:**

```
Raw session (frame_*.npy + inputs.json)
    ↓ load_session()          — load frames (uint8) and input dicts
    ↓ normalize_frames()      — uint8 → float32 in [0, 1]
    ↓ encode_inputs()         — list of dicts → (N, 3) float32 [steer, throttle, brake]
    ↓ filter_invalid_frames() — remove blank/loading-screen frames (mean < 0.01)
    ↓ build_sequences()       — sliding window of 4 frames → (N, 4, 384, 480)
    ↓ write to disk           — frames.npy, inputs.npy, dataset_meta.json
```

Entry point: `src/preprocessing/pipeline.py`

## Running the Pipeline

```bash
# Process all sessions
python src/preprocessing/pipeline.py

# Process a single session
python src/preprocessing/pipeline.py --session session_20260405_143022

# Custom directories
python src/preprocessing/pipeline.py --captures-dir data/captures --output-dir data/processed
```

## Frame Stacking (Sequences)

The model takes a **stack of 4 consecutive frames** as input, giving it implicit
motion information (velocity, direction of travel).

`build_sequences()` produces overlapping windows:

- Input: `(N, 384, 480)` float32 frames
- Output: `(N-3, 4, 384, 480)` float32 — each sample is 4 consecutive frames
- Aligned inputs: `(N-3, 3)` — the input at the **last frame** of each window
  is used as the label

The first 3 frames of each session are discarded (not enough history to form a
full window). This loses ~0.1s of data per session — negligible.

## Output Format

Each processed session directory contains:

```
data/processed/
└── session_20260405_143022/
    ├── frames.npy          # (N, 4, 384, 480) float32 — 4-frame stacks
    ├── inputs.npy          # (N, 3) float32 — [steer, throttle, brake]
    └── dataset_meta.json   # provenance and shape metadata
```

`dataset_meta.json` schema:

```json
{
  "source_session": "data/captures/session_20260405_143022",
  "n_raw_frames": 108000,
  "n_processed_frames": 107950,
  "frame_shape": [4, 384, 480],
  "frame_dtype": "float32",
  "input_columns": ["steer", "throttle", "brake"]
}
```

## Cleaning

**Blank frame removal**: frames with mean pixel value below 0.01 (after
normalisation) are discarded. These occur during loading screens or pauses.
The corresponding input entry is also discarded to maintain alignment.

**No other cleaning is applied in v1.** Future candidates if data quality
issues arise:
- Outlier steering inputs (spikes > ±1.0 from controller noise)
- Duplicate consecutive frames (dxcam occasionally returns the same frame twice)

## Train / Validation Split

The split is performed at **training time**, not during preprocessing. The
preprocessed files contain the full dataset. The training script holds out the
last 10% of each session as validation (chronological split — do not shuffle
across sessions to avoid data leakage between temporally adjacent frames).

## Storage Estimates

| Stage | Size per 108k frames |
|-------|----------------------|
| Raw uint8 frames | ~19 GB |
| Processed float32 stacks (4-frame) | ~300 GB |

If volume space on Vast.ai is a constraint, keep processed data as uint8 and
normalise in the PyTorch DataLoader (`frame.float() / 255.0`). Update
`normalize_frames()` to return uint8 and move the division into the dataset
class.
