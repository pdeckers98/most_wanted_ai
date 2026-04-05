# Controller Input Logging

Documentation for capturing and logging gamepad input during gameplay.

## Overview

The input logging system captures gamepad state once per captured frame, producing a JSON file that is perfectly aligned with the `.npy` frame files written by the screen capture pipeline. Both systems are controlled by the same F9 toggle inside `recorder.py`.

**Key specifications:**
- **Input device**: Gamepad (Xbox-style controller via pygame joystick API)
- **Steering**: Continuous float in `[-1.0, 1.0]` (left stick X axis), with configurable deadzone
- **Throttle / Brake**: Discrete `0` or `1` derived from trigger axes (RT = throttle, LT = brake) via threshold comparison
- **Sync method**: One input snapshot per written frame — list index equals frame index
- **Output**: `inputs.json` in the session directory alongside `session_meta.json`

## Setup

### Dependencies

Install required packages:
```bash
pip install -r requirements.txt
```

`pygame>=2.5.0` is required for joystick support.

### Controller Configuration

Default axis mapping targets a standard Xbox-style controller on Windows:

| Control  | Source       | Pygame axis index |
|----------|--------------|-------------------|
| Steer    | Left stick X | 0                 |
| Throttle | Right trigger (RT) | 5           |
| Brake    | Left trigger (LT) | 4            |

If your controller maps axes differently, edit `ControllerConfig` defaults in `src/input/config.py` or extend the CLI with additional flags.

## Recording

Input logging is integrated into `recorder.py` and starts/stops with the same F9 hotkey as screen capture. No separate process is needed.

### Basic Usage

```bash
python src/capture/recorder.py
```

This captures both frames and controller input with default settings (first connected controller, index 0).

### Command-Line Options (Input-Related)

```bash
python src/capture/recorder.py [OPTIONS]

  --controller-index INT   Pygame joystick index to use (default 0)
  --no-controller          Disable input logging; record screen only
```

**Examples:**

```bash
# Use second connected controller
python src/capture/recorder.py --controller-index 1

# Screen capture only (no input log)
python src/capture/recorder.py --no-controller
```

## Output Format

`inputs.json` is a JSON array written at session end. Each element corresponds to one saved frame (list index == frame index):

```json
[
  {"steer": 0.4231, "throttle": 1, "brake": 0},
  {"steer": 0.4102, "throttle": 1, "brake": 0},
  {"steer": 0.0,    "throttle": 0, "brake": 1},
  ...
]
```

**Fields per entry:**

| Field    | Type  | Description                                           |
|----------|-------|-------------------------------------------------------|
| steer    | float | Left stick X in `[-1.0, 1.0]`; negative = left       |
| throttle | int   | `1` if RT axis > threshold (default 0.5), else `0`   |
| brake    | int   | `1` if LT axis > threshold (default 0.5), else `0`   |

**Session directory structure:**

```
data/captures/
└── session_20260405_143022/
    ├── frame_000000.npy
    ├── frame_000001.npy
    ├── ...
    ├── inputs.json        ← input log (one entry per frame)
    └── session_meta.json
```

## Loading for Training

```python
import json
import numpy as np
from pathlib import Path

session_dir = Path("data/captures/session_20260405_143022")

frame_files = sorted(session_dir.glob("frame_*.npy"))
frames = np.array([np.load(f) for f in frame_files])  # (N, 384, 480) uint8

with open(session_dir / "inputs.json") as f:
    inputs = json.load(f)  # list of N dicts

# Access frame 42 and its corresponding input
frame_42 = frames[42]
steer_42 = inputs[42]["steer"]
throttle_42 = inputs[42]["throttle"]
brake_42 = inputs[42]["brake"]
```

## Architecture

Input reading is embedded in the main capture loop inside `ScreenCapture._capture_loop()`:

```
Frame available (dxcam)
    ↓
ControllerReader.read()       ← pygame.event.pump() + axis reads
    ↓
preprocess_frame(frame)
    ↓
FrameWriter.write(frame, idx) ← non-blocking queue
    ↓
InputLogger.record(state)     ← append to in-memory list
    ↓
[session stop]
    ↓
InputLogger.save()            ← json.dump to inputs.json
```

The controller is sampled right after a valid frame is obtained, so steer/throttle/brake reflect the player's input at the moment of capture.

## Troubleshooting

**"No gamepad detected"**
- Ensure the controller is connected before starting `recorder.py`
- Verify pygame sees it: `python -c "import pygame; pygame.init(); pygame.joystick.init(); print(pygame.joystick.get_count())"`

**Wrong axis for steering or triggers**
- Run the following snippet with the game paused to print live axis values:
  ```python
  import pygame, time
  pygame.init(); pygame.joystick.init()
  j = pygame.joystick.Joystick(0); j.init()
  while True:
      pygame.event.pump()
      print([round(j.get_axis(i), 3) for i in range(j.get_numaxes())])
      time.sleep(0.1)
  ```
- Update `ControllerConfig` defaults in `src/input/config.py` accordingly.

**inputs.json is shorter than the frame count**
- This should not occur under normal operation; frames and inputs are incremented together.
- If it does, check for exceptions in the capture loop output.
