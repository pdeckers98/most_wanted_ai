# NFS Most Wanted 2005 Racing AI

An offline learning racing AI for NFS Most Wanted 2005. Agent trained in cloud, inference runs locally on GTX 1660 (6GB VRAM).

## Project Purpose

Hobby project developing an autonomous racing agent through offline RL, capturing real gameplay data and training models to drive competitively in NFS Most Wanted 2005.

## Tech Stack

- **Language**: Python 3.10+
- **RL Algorithm**: TBD (offline learning approach)
- **Inference Hardware**: GTX 1660 (6GB VRAM) - game + agent run locally
- **Training**: Cloud-based (separate environment)
- **Game Interaction**: Screen capture + controller input

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

**Run inference**: `python src/agent/inference.py`

**Test**: `pytest tests/`

## Additional Documentation

See `.claude/docs/` for specialized guides:

- **`SCREEN_CAPTURE.md`** - Frame capture, preprocessing, FPS/resolution
- **`CONTROLLER_INPUT.md`** - Input logging, action mapping, synchronized recording
- **`DATA_PREPROCESSING.md`** - Data cleaning, feature engineering, train/val split
- **`TRAINING.md`** - Cloud training setup, offline RL algorithm details, hyperparameters
- **`INFERENCE.md`** - Model loading, real-time inference, GTX 1660 optimization

---

**For Claude:** Check relevant docs based on task (capture, preprocessing, training, or inference).
