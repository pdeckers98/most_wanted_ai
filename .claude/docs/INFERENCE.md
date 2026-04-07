# Inference

Documentation for running the trained agent locally on the GTX 1660.

## Overview

`src/agent/inference.py` captures live frames, runs a forward pass through the
trained ResNet-18 policy, and writes actions to a **virtual Xbox controller**
via vgamepad (ViGEm). The physical controller is read by the script; the game
is driven entirely through the virtual device.

**Performance target**: forward pass < 33 ms (30 fps budget).
ResNet-18 on GTX 1660: ~2–4 ms per pass — well within budget.

## One-Time Setup

Install [ViGEm Bus Driver](https://github.com/nefarius/ViGEmBus/releases) if
not already present — vgamepad requires it.

When you first run the script a new virtual controller appears in Windows.
Open NFS MW's controller options and reassign controls to that device.
This only needs to be done once.

## Running

```bash
python src/agent/inference.py
# Custom capture region or checkpoint:
python src/agent/inference.py --right 1920 --bottom 1080 --checkpoint data/checkpoints/best_model.pt
```

| Flag | Default | Description |
|------|---------|-------------|
| `--checkpoint` | `data/checkpoints/best_model.pt` | Model to load |
| `--left/top/right/bottom` | `0 0 1280 1024` | Screen capture region (pixels) |
| `--fps` | `30` | Target inference FPS |
| `--device` | `cuda` | `cuda` or `cpu` |

## Controls

| Input | Effect |
|-------|--------|
| **F7** | Toggle agent on / off |
| **RB (hold)** | Override — passes your physical input through while held |

Audio feedback on toggle: high beep (880 Hz) = on, low beep (440 Hz) = off.

## Agent States

| State | What the game receives |
|-------|----------------------|
| Agent OFF | Your physical controller (mirrored via vgamepad) |
| Agent ON | Model predictions |
| Agent ON + RB held | Your physical controller (override) |

## Model Loading

```python
from src.agent.model import RacingAgent

model = RacingAgent(pretrained=False)
ckpt = torch.load("data/checkpoints/best_model.pt", map_location="cuda")
model.load_state_dict(ckpt["model_state_dict"])
model.eval().cuda()
```

Frame buffer: `collections.deque(maxlen=4)`. On startup the first frame is
duplicated to fill the buffer before the model sees a real 4-frame stack.

## VRAM Footprint

| Component | VRAM |
|-----------|------|
| NFS Most Wanted 2005 | ~0.8 GB |
| ResNet-18 weights (float32) | ~45 MB |
| Single forward pass activations | ~28 MB |
| **Total** | **< 1 GB** — no quantisation needed |
