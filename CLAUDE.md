# NFS Most Wanted 2005 Racing AI

An offline learning racing AI for NFS Most Wanted 2005. Agent trained in cloud, inference runs locally on GTX 1660 (6GB VRAM).

## Project Purpose

Hobby project developing an autonomous racing agent through offline RL, capturing real gameplay data and training models to drive competitively in NFS Most Wanted 2005.

## Tech Stack

- **Language**: Python 3.10+
- **Training framework**: PyTorch (raw, no high-level wrapper)
- **Phase 1 algorithm**: Behavioural Cloning (imitation learning) — baseline before offline RL
- **Phase 2 algorithm**: Offline RL (algorithm TBD, likely IQL or TD3+BC)
- **Model backbone**: ResNet-18 pretrained on ImageNet, first conv replaced for greyscale input
- **Input representation**: 4-frame stack → shape `(4, 384, 480)` per sample
- **Action head**: Hybrid — MSE loss on steer (continuous), BCE loss on throttle/brake (binary)
- **Dataset**: ~1 hour of expert gameplay, ~108,000 frames at 30fps
- **Inference hardware**: GTX 1660 (6GB VRAM) — game + agent run locally
- **Training hardware**: Vast.ai GPU instance with persistent volume (data pre-uploaded before spin-up)
- **Game interaction**: Screen capture (dxcam) + controller input (pygame)

## Project Structure

```
src/
├── capture/          # Screen capture pipeline
├── input/            # Controller input logging
├── preprocessing/    # Data cleaning & feature engineering
├── agent/            # Inference code (runs on GTX 1660)
└── utils/            # Helpers & telemetry
tests/                # Unit & integration tests
data/                 # Local datasets, model checkpoints
.claude/docs/         # See below
```

## Essential Commands

**Setup**: `pip install -r requirements.txt`

**Capture gameplay**: `python src/capture/record.py`

**Preprocess data**: `python src/preprocessing/pipeline.py`

**Run inference**: `python -m src.agent.inference`

**Train**: `python -m src.agent.train --data-dir /mnt/data/processed --output-dir /mnt/data/checkpoints`

**Test**: `pytest tests/`

**Lint**: `flake8 src/ tests/`

## Code Quality

This project uses **Flake8** for linting. Before presenting any code changes, I will verify the code passes Flake8 checks to ensure it meets the project's style and quality standards.

## Additional Documentation

See `.claude/docs/` for specialized guides:

- **`SCREEN_CAPTURE.md`** - Frame capture, preprocessing, FPS/resolution
- **`CONTROLLER_INPUT.md`** - Input logging, action mapping, synchronized recording
- **`DATA_PREPROCESSING.md`** - Data cleaning, feature engineering, train/val split
- **`TRAINING.md`** - Cloud training setup, offline RL algorithm details, hyperparameters
- **`INFERENCE.md`** - Model loading, real-time inference, GTX 1660 optimization

---

**For Claude:** Check relevant docs based on task (capture, preprocessing, training, or inference).
