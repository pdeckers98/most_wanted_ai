# Training

Documentation for cloud-based training setup.

## Overview

Training runs on a Vast.ai GPU instance against a persistent volume that holds
the preprocessed dataset. The volume is populated locally before the GPU is
spun up to avoid paying for idle GPU time during upload.

**Two-phase plan:**

| Phase | Algorithm | Status |
|-------|-----------|--------|
| 1 | Behavioural Cloning (imitation learning) | First target |
| 2 | Offline RL (IQL or TD3+BC) | After BC baseline established |

BC is standard supervised learning — minimise the difference between the
model's predicted actions and the expert's recorded actions. It gives a
drivable baseline quickly and acts as a reference point for offline RL gains.

## Dataset

- **Source**: ~1 hour of expert gameplay captured via `recorder.py`
- **Raw size**: ~108,000 frames at 30fps
- **Processed size**: float32 `(N, 4, 384, 480)` frame stacks + `(N, 3)` inputs
- **Storage estimate**: ~108k × 4 × 384 × 480 × 4 bytes ≈ ~320 GB uncompressed
  — consider keeping uint8 stacks on disk and normalising in the dataloader if
  volume space is a constraint
- **Location on Vast.ai**: persistent volume, mounted at `/data`

## Model Architecture

Class definition: `src/agent/model.py` — `RacingAgent`. Both `train.py` and
`inference.py` import from there; the checkpoint is compatible with both.

**Backbone**: ResNet-18

- Pretrained on ImageNet (`torchvision.models.resnet18(pretrained=True)`)
- First conv layer replaced: `Conv2d(4, 64, kernel_size=7, stride=2, padding=3)`
  (4 input channels for frame stack, pretrained weights averaged across the
  original 3 channels to initialise)
- Global average pool output: 512-dim feature vector

**Action heads** (two separate linear layers on top of the 512-dim features):

| Head | Output | Loss |
|------|--------|------|
| Steer | scalar float | MSE |
| Throttle/Brake | 2-dim sigmoid | BCE |

**Total loss**: `L = MSE(steer) + λ * BCE(throttle, brake)`
— start with `λ = 1.0`, tune if one term dominates.

## Hyperparameters (starting point)

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Learning rate | 1e-4 |
| LR schedule | CosineAnnealingLR |
| Batch size | 64 |
| Epochs | 30 |
| Weight decay | 1e-4 |
| Steer loss weight (λ) | 1.0 |

## Cloud Setup

1. Create a persistent volume on Vast.ai and note its mount path (`/data`)
2. Upload processed dataset locally using Vast.ai CLI or rclone before renting GPU
3. Rent an instance (RTX 3090 / A5000 recommended — 24GB VRAM, good price/perf)
4. Mount the volume and run training

```bash
# On the Vast.ai instance
pip install -r requirements.txt
python src/agent/train.py --data-dir /data/processed --output-dir /data/checkpoints
```

## Monitoring

- Log training loss and validation loss per epoch to stdout + a `training_log.json`
- Save a checkpoint after every epoch: `checkpoint_epoch_{N:02d}.pt`
- Keep the best checkpoint by validation loss: `best_model.pt`

## Model Export

After training, export `best_model.pt` from the volume for local inference:

```bash
# Download from Vast.ai volume to local machine
rclone copy vast:/data/checkpoints/best_model.pt data/checkpoints/
```

The inference script (`src/agent/inference.py`) loads `best_model.pt` directly.
No ONNX export required for the first iteration — PyTorch inference on GTX 1660
is fast enough at this model size.
