"""
Behavioral Cloning hyperparameter search for NFS Most Wanted 2005 racing agent.

Iterates over a predefined grid of hyperparameter combinations. Each trial trains
for up to 50 epochs with early stopping. Supports HuberLoss for steering.

Usage:
    python -m src.agent.train \\
        --data-dir /mnt/data/processed --output-dir /mnt/data/checkpoints

Prerequisites:
    wandb login  # One-time setup to link W&B account
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Tuple, Dict, Any
from datetime import timedelta
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import wandb

from src.agent.model import RacingAgent


# ============================================================================
# Logging Setup
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Hyperparameter Grid
# ============================================================================
# Combinations chosen for fine-tuning ResNet-18 on ~100k driving frames.
# - lr: covers a decade around the standard 1e-4 BC starting point
# - weight_decay: 1e-4 (light) vs 1e-3 (stronger regularisation)
# - steer_loss: mse is the standard; huber is more robust to outlier steer values
# - huber_delta: 0.5 penalises large errors less aggressively than 1.0
# - steer_weight: upweights steer relative to throttle/brake BCE
# - batch_size: 256 standard; 512 exploits A100 headroom for smoother gradients

HPARAM_GRID = [
    # --- MSE steer loss ---
    {
        'lr': 1e-3, 'weight_decay': 1e-4,
        'steer_loss': 'mse', 'huber_delta': None,
        'steer_weight': 1.0, 'batch_size': 256,
    },
    {
        'lr': 3e-4, 'weight_decay': 1e-4,
        'steer_loss': 'mse', 'huber_delta': None,
        'steer_weight': 2.0, 'batch_size': 256,
    },
    {
        # Baseline — mirrors previous default
        'lr': 1e-4, 'weight_decay': 1e-4,
        'steer_loss': 'mse', 'huber_delta': None,
        'steer_weight': 1.0, 'batch_size': 256,
    },
    {
        'lr': 1e-4, 'weight_decay': 1e-3,
        'steer_loss': 'mse', 'huber_delta': None,
        'steer_weight': 2.0, 'batch_size': 256,
    },
    # --- Huber steer loss (delta=1.0) ---
    {
        'lr': 3e-4, 'weight_decay': 1e-4,
        'steer_loss': 'huber', 'huber_delta': 1.0,
        'steer_weight': 1.0, 'batch_size': 256,
    },
    {
        'lr': 1e-4, 'weight_decay': 1e-4,
        'steer_loss': 'huber', 'huber_delta': 1.0,
        'steer_weight': 2.0, 'batch_size': 256,
    },
    # --- Huber steer loss (delta=0.5) ---
    {
        'lr': 3e-4, 'weight_decay': 1e-4,
        'steer_loss': 'huber', 'huber_delta': 0.5,
        'steer_weight': 1.0, 'batch_size': 512,
    },
    {
        'lr': 1e-4, 'weight_decay': 1e-4,
        'steer_loss': 'huber', 'huber_delta': 0.5,
        'steer_weight': 5.0, 'batch_size': 256,
    },
]


# ============================================================================
# Dataset Class
# ============================================================================

class RacingDataset(Dataset):
    """
    PyTorch Dataset for preprocessed racing frames and actions.

    Loads 4-frame stacks and corresponding input actions from a session.
    Applies random train/val split to ensure validation set captures
    diversity across all race conditions.

    Args:
        frames_path: Path to frames.npy (shape: (N, 4, 384, 480))
        inputs_path: Path to inputs.npy (shape: (N, 3) [steer, throttle, brake])
        split: 'train' or 'val' — random 90/10 split
        normalize: If True, normalize frames to [0, 1]
        seed: Random seed for reproducible splits (default: 42)
    """

    def __init__(
        self,
        frames_path: Path,
        inputs_path: Path,
        split: str = 'train',
        normalize: bool = True,
        seed: int = 42
    ):
        assert split in ['train', 'val'], \
            f"split must be 'train' or 'val', got {split}"

        self.frames = np.load(frames_path)   # (N, 4, 384, 480) float32
        self.inputs = np.load(inputs_path)   # (N, 3) float32

        assert len(self.frames) == len(self.inputs), \
            f"Mismatch: {len(self.frames)} frames vs {len(self.inputs)} inputs"

        rng = np.random.RandomState(seed)
        n_samples = len(self.frames)
        val_size = int(n_samples * 0.1)
        val_indices = rng.choice(n_samples, size=val_size, replace=False)
        val_mask = np.zeros(n_samples, dtype=bool)
        val_mask[val_indices] = True

        mask = ~val_mask if split == 'train' else val_mask
        self.frames = self.frames[mask]
        self.inputs = self.inputs[mask]
        self.normalize = normalize
        logger.info(f"Loaded {split} split: {len(self.frames)} samples")

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        frame_stack = torch.from_numpy(self.frames[idx].copy()).float()
        if self.normalize:
            frame_stack = frame_stack / 255.0
        action = torch.from_numpy(self.inputs[idx].copy()).float()
        return frame_stack, action


# ============================================================================
# Loss helper
# ============================================================================

def build_steer_loss_fn(steer_loss: str, huber_delta: float) -> nn.Module:
    """Return the steer loss criterion for this trial."""
    if steer_loss == 'huber':
        return nn.HuberLoss(delta=huber_delta)
    return nn.MSELoss()


# ============================================================================
# Training / Validation steps
# ============================================================================

def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    steer_criterion: nn.Module,
    device: torch.device,
    steer_weight: float,
) -> Tuple[float, float, float]:
    model.train()
    steer_losses, action_losses, total_losses = [], [], []
    action_criterion = nn.BCELoss()

    for frames, actions in train_loader:
        frames = frames.to(device)
        actions = actions.to(device)

        steer_gt = actions[:, 0:1]
        throttle_brake_gt = actions[:, 1:3]

        optimizer.zero_grad()
        steer_pred, actions_pred = model(frames)

        steer_loss = steer_criterion(steer_pred, steer_gt)
        action_loss = action_criterion(actions_pred, throttle_brake_gt)
        total_loss = steer_weight * steer_loss + action_loss

        total_loss.backward()
        optimizer.step()

        steer_losses.append(steer_loss.item())
        action_losses.append(action_loss.item())
        total_losses.append(total_loss.item())

    return (
        float(np.mean(steer_losses)),
        float(np.mean(action_losses)),
        float(np.mean(total_losses)),
    )


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    steer_criterion: nn.Module,
    device: torch.device,
    steer_weight: float,
) -> Tuple[float, float, float]:
    model.eval()
    steer_losses, action_losses, total_losses = [], [], []
    action_criterion = nn.BCELoss()

    with torch.no_grad():
        for frames, actions in val_loader:
            frames = frames.to(device)
            actions = actions.to(device)

            steer_gt = actions[:, 0:1]
            throttle_brake_gt = actions[:, 1:3]

            steer_pred, actions_pred = model(frames)

            steer_loss = steer_criterion(steer_pred, steer_gt)
            action_loss = action_criterion(actions_pred, throttle_brake_gt)
            total_loss = steer_weight * steer_loss + action_loss

            steer_losses.append(steer_loss.item())
            action_losses.append(action_loss.item())
            total_losses.append(total_loss.item())

    return (
        float(np.mean(steer_losses)),
        float(np.mean(action_losses)),
        float(np.mean(total_losses)),
    )


# ============================================================================
# Single trial
# ============================================================================

def run_trial(
    trial_idx: int,
    hparams: Dict[str, Any],
    train_datasets,
    val_datasets,
    output_dir: Path,
    epochs: int,
    early_stop_patience: int,
    num_workers: int,
    device: torch.device,
    wandb_group: str,
) -> Dict[str, Any]:
    """
    Train one hyperparameter configuration and return a result summary dict.
    Saves the best checkpoint for this trial to output_dir.
    """
    lr = hparams['lr']
    weight_decay = hparams['weight_decay']
    steer_loss_type = hparams['steer_loss']
    huber_delta = hparams.get('huber_delta') or 1.0
    steer_weight = hparams['steer_weight']
    batch_size = hparams['batch_size']

    run_name = (
        f"trial{trial_idx:02d}"
        f"-{steer_loss_type}"
        f"-lr{lr:.0e}"
        f"-wd{weight_decay:.0e}"
        f"-sw{steer_weight}"
        f"-bs{batch_size}"
        + (f"-d{huber_delta}" if steer_loss_type == 'huber' else "")
    )

    logger.info(
        f"\n{'='*70}\n"
        f"Trial {trial_idx}/{len(HPARAM_GRID)}: {run_name}\n"
        f"{'='*70}"
    )

    # Dataloaders (batch_size may differ per trial)
    train_loader = DataLoader(
        ConcatDataset(train_datasets),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        ConcatDataset(val_datasets),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    model = RacingAgent(pretrained=True).to(device)
    steer_criterion = build_steer_loss_fn(steer_loss_type, huber_delta)

    optimizer = optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    wandb.init(
        project='nfs-most-wanted-2005',
        group=wandb_group,
        name=run_name,
        config={
            'lr': lr,
            'weight_decay': weight_decay,
            'steer_loss': steer_loss_type,
            'huber_delta': huber_delta if steer_loss_type == 'huber' else None,
            'steer_weight': steer_weight,
            'batch_size': batch_size,
            'epochs': epochs,
            'early_stop_patience': early_stop_patience,
        },
        reinit=True,
    )

    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_path = output_dir / f'{run_name}_best.pt'
    epoch_times = []
    stopped_epoch = epochs

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        train_steer, train_action, train_total = train_epoch(
            model, train_loader, optimizer, steer_criterion,
            device, steer_weight,
        )
        val_steer, val_action, val_total = validate(
            model, val_loader, steer_criterion, device, steer_weight,
        )
        scheduler.step()

        epoch_time = time.time() - t0
        epoch_times.append(epoch_time)
        avg_epoch_time = float(np.mean(epoch_times))
        remaining_str = str(
            timedelta(seconds=int((epochs - epoch) * avg_epoch_time))
        )

        logger.info(
            f"  Epoch {epoch:3d}/{epochs} | "
            f"Train: steer={train_steer:.4f} action={train_action:.4f} "
            f"total={train_total:.4f} | "
            f"Val: steer={val_steer:.4f} action={val_action:.4f} "
            f"total={val_total:.4f} | "
            f"LR={optimizer.param_groups[0]['lr']:.2e} | "
            f"{epoch_time:.1f}s | est. remaining: {remaining_str}"
        )

        wandb.log({
            'epoch': epoch,
            'train/steer_loss': train_steer,
            'train/action_loss': train_action,
            'train/total_loss': train_total,
            'val/steer_loss': val_steer,
            'val/action_loss': val_action,
            'val/total_loss': val_total,
            'learning_rate': optimizer.param_groups[0]['lr'],
            'epoch_time_seconds': epoch_time,
        })

        if val_total < best_val_loss:
            best_val_loss = val_total
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"  -> New best val_loss={best_val_loss:.4f} saved."
                        f" (checkpoint: {best_model_path})")
        else:
            epochs_no_improve += 1
            logger.info(
                f"  -> No improvement for {epochs_no_improve}"
                f"/{early_stop_patience} epochs."
            )
            if epochs_no_improve >= early_stop_patience:
                logger.info(
                    f"  Early stopping at epoch {epoch} "
                    f"(patience={early_stop_patience})."
                )
                stopped_epoch = epoch
                break

    # Upload best checkpoint to W&B Artifacts
    artifact = wandb.Artifact(
        name=run_name,
        type='model',
        metadata={
            'best_val_loss': float(best_val_loss),
            'stopped_epoch': stopped_epoch,
            **hparams,
        },
    )
    artifact.add_file(str(best_model_path))
    wandb.log_artifact(artifact)
    logger.info(f"  Artifact '{run_name}' uploaded to W&B.")

    wandb_url = wandb.run.get_url()
    wandb.finish()

    result = {
        'trial': trial_idx,
        'run_name': run_name,
        'best_val_loss': float(best_val_loss),
        'stopped_epoch': stopped_epoch,
        'checkpoint': str(best_model_path),
        'wandb_url': wandb_url,
        **hparams,
    }
    logger.info(
        f"Trial {trial_idx} done. best_val_loss={best_val_loss:.4f} "
        f"at epoch {stopped_epoch}."
    )
    return result


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Hyperparameter search for NFS Most Wanted 2005 BC agent'
    )
    parser.add_argument(
        '--data-dir', type=Path, default=Path('/mnt/data/processed'),
        help='Directory containing processed sessions',
    )
    parser.add_argument(
        '--output-dir', type=Path, default=Path('/mnt/data/checkpoints'),
        help='Directory to save checkpoints and logs (default: /mnt/data/checkpoints)',
    )
    parser.add_argument(
        '--epochs', type=int, default=50,
        help='Max epochs per trial (default: 50)',
    )
    parser.add_argument(
        '--early-stop-patience', type=int, default=7,
        help='Epochs without val improvement before stopping (default: 7)',
    )
    parser.add_argument(
        '--num-workers', type=int, default=8,
        help='DataLoader workers (default: 8)',
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Discover sessions
    data_dir = Path(args.data_dir)
    sessions = sorted([d for d in data_dir.iterdir() if d.is_dir()])
    if not sessions:
        logger.error(f"No session directories found in {data_dir}")
        return
    logger.info(f"Found {len(sessions)} sessions: {[s.name for s in sessions]}")

    # Load datasets once — shared across all trials
    train_datasets, val_datasets = [], []
    for session_dir in sessions:
        frames_path = session_dir / 'frames.npy'
        inputs_path = session_dir / 'inputs.npy'
        if not frames_path.exists() or not inputs_path.exists():
            logger.warning(f"Skipping {session_dir.name}: missing files")
            continue
        train_datasets.append(
            RacingDataset(frames_path, inputs_path, split='train')
        )
        val_datasets.append(
            RacingDataset(frames_path, inputs_path, split='val')
        )

    if not train_datasets:
        logger.error("No valid datasets loaded")
        return

    total_train = sum(len(d) for d in train_datasets)
    total_val = sum(len(d) for d in val_datasets)
    logger.info(
        f"Dataset: {total_train} train samples | {total_val} val samples"
    )

    wandb_group = f"hparam-search-{time.strftime('%Y%m%d-%H%M%S')}"
    logger.info(
        f"Starting {len(HPARAM_GRID)} trials | W&B group: {wandb_group}"
    )

    all_results = []
    for i, hparams in enumerate(HPARAM_GRID, start=1):
        result = run_trial(
            trial_idx=i,
            hparams=hparams,
            train_datasets=train_datasets,
            val_datasets=val_datasets,
            output_dir=args.output_dir,
            epochs=args.epochs,
            early_stop_patience=args.early_stop_patience,
            num_workers=args.num_workers,
            device=device,
            wandb_group=wandb_group,
        )
        all_results.append(result)

    # Summary
    all_results.sort(key=lambda r: r['best_val_loss'])
    logger.info("\n" + "="*70)
    logger.info("HYPERPARAMETER SEARCH COMPLETE — ranked by best_val_loss:")
    for rank, r in enumerate(all_results, start=1):
        logger.info(
            f"  #{rank:2d} val_loss={r['best_val_loss']:.4f} "
            f"ep={r['stopped_epoch']:3d}  {r['run_name']}"
        )
    logger.info("="*70)

    summary_path = args.output_dir / 'hparam_search_results.json'
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Full results saved to {summary_path}")


if __name__ == '__main__':
    main()
