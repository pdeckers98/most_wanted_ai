# Inference

Documentation for running the trained agent locally on the GTX 1660.

## Overview

The inference pipeline runs alongside the game on the same machine. It captures
frames via the screen capture pipeline, assembles a 4-frame stack, runs a
forward pass through the trained ResNet-18 model, and outputs controller
actions in real time.

**Performance target**: forward pass must complete in under 33ms (30fps budget).
ResNet-18 on GTX 1660 runs a single forward pass in ~2-4ms — well within budget.

## Hardware Constraints

- **VRAM**: 6GB shared between the game and the agent
  - NFS Most Wanted 2005 uses ~0.8GB VRAM
  - ResNet-18 in float32 weights: ~45MB
  - Activations for a single (1, 4, 384, 480) batch: ~28MB
  - **Total agent footprint**: well under 1GB — no quantisation needed for v1
- **Quantisation**: not required for v1; revisit if VRAM becomes tight or
  latency is too high

## Model Loading

```python
import torch
from src.agent.model import ResNet18Policy

model = ResNet18Policy()
checkpoint = torch.load("data/checkpoints/best_model.pt", map_location="cuda")
model.load_state_dict(checkpoint["model_state_dict"])
model.eval().cuda()
```

## Real-Time Agent Loop

```
[Screen capture: 4 most recent frames]
    ↓ assemble (4, 384, 480) float32 tensor, normalise /255
    ↓ model.forward(tensor.unsqueeze(0).cuda())
    ↓ steer (float), throttle (binary), brake (binary)
    ↓ send to virtual controller output
```

Frame buffer: keep a `collections.deque(maxlen=4)` updated each frame.
On the first 3 frames before the buffer is full, duplicate the earliest frame.

## Integration with Screen Capture

The inference loop reuses the same `FrameGrabber` and `preprocess_frame()`
from `src/capture/`. It does **not** write frames to disk — capture is
live only during inference.

## Optimization

For v1, no special optimisation is needed. If latency or VRAM becomes an issue
in later versions:

1. **torch.compile()** — drop-in speed improvement on PyTorch 2.x
2. **float16 weights** — halves VRAM, minimal accuracy impact
3. **ONNX + TensorRT** — maximum throughput, more complex pipeline

## Testing & Validation

Before running live against the game, validate offline:

```bash
python src/agent/evaluate.py --data-dir data/processed --checkpoint data/checkpoints/best_model.pt
```

Key metrics to check:
- **Steer MAE** — mean absolute error on held-out validation frames
- **Throttle/Brake accuracy** — binary classification accuracy
- **Action distribution** — compare predicted vs expert histograms to check
  for mode collapse (e.g. model always predicts steer=0)
