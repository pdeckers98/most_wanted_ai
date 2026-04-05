# Screen Capture

Documentation for game frame capture pipeline.

## Overview

The screen capture system uses **DXGI Desktop Duplication API** via `dxcam` to capture frames from the NFS Most Wanted 2005 game window running at 1280x1024. Frames are downscaled to 480x384 greyscale and saved as NumPy `.npy` files for training data.

**Key specifications:**
- **Capture method**: dxcam (DXGI Desktop Duplication)
- **Game resolution**: 1280x1024 (windowed)
- **Output resolution**: 480x384 (preserves 5:4 aspect ratio)
- **Output format**: Greyscale uint8 NumPy arrays (`.npy` files)
- **Target FPS**: 30 (configurable)
- **Per-frame latency**: ~2-3ms (dxcam grab + resize), well within 33ms budget
- **Storage**: ~180 KB per frame, ~5.3 MB/s at 30fps, ~9.5 GB per 30min session
- **GTX 1660 impact**: Game uses ~0.8GB VRAM; capture adds negligible load

## Setup

### Dependencies

Install required packages:
```bash
pip install -r requirements.txt
```

Required packages:
- `dxcam>=0.4.0` — DXGI screen capture
- `opencv-python-headless>=4.8.0` — Image resizing
- `keyboard>=0.13.5` — Global hotkey support
- `pywin32>=306` — Windows API access
- `numpy>=1.24.0` — Array handling
- `pytest>=7.0.0` — Testing

### Finding the Game Window

Before first use, identify the exact game window title:
```bash
python scripts/detect_window.py --search "Speed"
```

This prints all visible windows. Find the NFS Most Wanted entry and note the exact title string (including special characters like ™). Use this with the `--window-title` argument to `recorder.py`.

## Usage

### Basic Usage

```bash
python src/capture/recorder.py
```

This starts the capture system with default settings:
- Window: Search for "Need for Speed" in title
- FPS: 30
- Hotkey: F9
- Output: `data/captures/`

### Command-Line Options

```bash
python src/capture/recorder.py [OPTIONS]

Options:
  --fps FPS                 Target frames per second (default 30)
  --hotkey KEY              Hotkey to toggle recording (default F9)
  --output-dir DIR          Output directory for captures (default data/captures)
  --window-title SUBSTRING  Window title substring to search for
```

**Examples:**

```bash
# Capture at 60 FPS with different hotkey
python src/capture/recorder.py --fps 60 --hotkey F10

# Custom output directory
python src/capture/recorder.py --output-dir /mnt/fast_drive/nfs_data

# Use exact window title if auto-detection fails
python src/capture/recorder.py --window-title "Need for Speed™ Most Wanted"
```

### Recording Workflow

1. Start the recorder
2. Press the hotkey (default F9) to begin recording
3. You will hear a **high beep** and see "Recording started" in the console
4. Play NFS Most Wanted as normal
5. Press F9 again to stop recording
6. You will hear a **low beep** and see frame count and duration

**Status updates:** Every 5 seconds, the console prints the current frame count and estimated FPS.

### Output Structure

```
data/captures/
├── session_20260405_143022/
│   ├── frame_000000.npy        # First frame
│   ├── frame_000001.npy
│   ├── frame_000002.npy
│   └── ...
│   └── session_meta.json       # Metadata file
├── session_20260405_150530/
│   └── ...
```

**Session metadata** (`session_meta.json`):
```json
{
  "session_start": "2026-04-05T14:30:22.123456",
  "session_end": "2026-04-05T14:30:52.654321",
  "duration_seconds": 30.531,
  "total_frames": 915,
  "output_width": 480,
  "output_height": 384
}
```

### Loading Captured Frames

Load frames for training:

```python
import numpy as np
import torch
from pathlib import Path

session_dir = Path("data/captures/session_20260405_143022")
frame_files = sorted(session_dir.glob("frame_*.npy"))

# Load a single frame
frame = np.load(frame_files[0])  # shape: (384, 480) uint8 greyscale

# Load all frames into memory (caution: ~9.5 GB for 30min at 30fps)
frames = np.array([np.load(f) for f in frame_files])  # shape: (N, 384, 480)

# Convert to PyTorch tensor
tensor = torch.from_numpy(frames).float() / 255.0
```

## Performance

**Capture latency breakdown per frame at 30fps (33.3ms budget):**
- dxcam DXGI grab: ~1-2ms
- Greyscale conversion: ~0.5ms (built into dxcam)
- cv2.resize (1.3MP → 0.18MP): ~0.2ms
- Queue enqueue: ~0.01ms
- **Total: ~2-3ms per frame** (88% headroom)

**Storage efficiency:**
- Frame size: 480 × 384 × 1 byte = 184,320 bytes ≈ 180 KB
- At 30fps: 30 frames/sec × 180 KB = 5.4 MB/sec
- 30-minute session: 30 min × 60 sec × 5.4 MB = ~9.7 GB (without compression)

No compression is applied because the overhead isn't worth it at these throughputs.

## Architecture

**Three-thread model:**

1. **Main thread**
   - Hotkey listener (via `keyboard` library)
   - Frame consumption loop: grab → preprocess → enqueue
   
2. **dxcam internal thread**
   - Grabs frames at target FPS into ring buffer via DXGI Desktop Duplication
   
3. **Writer thread**
   - Pulls frames from bounded queue (300-frame buffer)
   - Calls `np.save()` to write `.npy` files sequentially

**Data flow:**
```
Game Window (1280x1024, color)
    ↓ [win32gui.GetClientRect + ClientToScreen]
    ↓ Screen-coordinate region computed once at startup
    ↓ [dxcam.start(region=..., fps=30, output_color="GRAY")]
    ↓ DXGI Desktop Duplication, greyscale at library level
    ↓ [camera.get_latest_frame()]
    ↓ np.ndarray (1024, 1280) uint8
    ↓ [cv2.resize(..., (480, 384), INTER_AREA)]
    ↓ np.ndarray (384, 480) uint8
    ↓ [writer.write(frame, idx)]
    ↓ Non-blocking queue enqueue
    ↓ [Writer thread: np.save()]
    ↓ disk/captures/session_*/frame_NNNNNN.npy
```

## Important Notes

### DPI Awareness

Windows 11 applies DPI scaling to window coordinates by default. The capture system calls `SetProcessDPIAware()` at startup to get accurate pixel-level coordinates. If you experience misaligned captures (shifted or cropped frames), ensure DPI scaling is disabled or that the game runs in native (non-scaled) mode.

### Window Movement

The capture region is computed from the game window's position at startup. **Do not move or resize the game window during recording**. If you do, subsequent frames will be misaligned. Plan to add dynamic region re-computation in a future version.

### Frame Rate Consistency

dxcam's internal thread handles FPS pacing with high-resolution timers. The actual captured FPS should match the target FPS closely. If you observe dropped frames or jitter, check system load and ensure no other heavy processes are running.

## Troubleshooting

**"No window found with title containing 'Need for Speed'"**
- Run `python scripts/detect_window.py --search "Speed"` to find the exact window title
- Use the exact title with `--window-title` flag
- Ensure the game is running and not minimized

**Misaligned or partial frames**
- Do not move the game window during recording
- Check Windows DPI scaling settings
- Run `scripts/detect_window.py` to verify window dimensions match 1280x1024

**dxcam returns None frames**
- This can happen during loading screens or pauses when the game isn't rendering
- The system skips None frames without writing to disk
- This is normal and expected behavior

**keyboard library conflicts**
- Some security software may block the low-level keyboard hook
- If hotkey doesn't work, check your antivirus/security software
- Alternative: Add a command-line flag to trigger recording without hotkey support (future enhancement)

**High CPU usage**
- dxcam is designed to be very lightweight but check background processes
- Ensure no other screen capture apps are running
- Try reducing FPS with `--fps 15` to see if it helps
