"""
Behavioral Cloning training script for NFS Most Wanted 2005 racing agent.

Trains a ResNet-18 backbone with dual action heads (steer + throttle/brake)
using supervised learning on expert gameplay data. Logs metrics to Weights & Biases
for real-time loss visualization and analysis.

Usage:
    python src/agent/train.py --data-dir /mnt/data/processed --output-dir /mnt/data/checkpoints

Prerequisites:
    wandb login  # One-time setup to link W&B account
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import wandb
from torchvision import models


# ============================================================================
# Logging Setup
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


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
        assert split in ['train', 'val'], f"split must be 'train' or 'val', got {split}"

        # Load data
        self.frames = np.load(frames_path)  # (N, 4, 384, 480) float32
        self.inputs = np.load(inputs_path)  # (N, 3) float32

        assert len(self.frames) == len(self.inputs), \
            f"Mismatch: {len(self.frames)} frames vs {len(self.inputs)} inputs"

        # Random split: 90% train, 10% val
        rng = np.random.RandomState(seed)
        n_samples = len(self.frames)
        val_size = int(n_samples * 0.1)
        val_indices = rng.choice(n_samples, size=val_size, replace=False)
        val_mask = np.zeros(n_samples, dtype=bool)
        val_mask[val_indices] = True

        if split == 'train':
            mask = ~val_mask
        else:  # val
            mask = val_mask

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
# Model Architecture
# ============================================================================

class RacingAgent(nn.Module):
    """
    ResNet-18 backbone with dual action heads.

    Steer head: Linear(512, 1) → continuous steering [-1, 1]
    Throttle/Brake head: Linear(512, 2) → binary [throttle, brake]

    The first conv layer is modified to accept 4-channel greyscale input
    instead of 3-channel RGB. Pretrained ImageNet weights are averaged
    across the channel dimension to initialize the 4th channel.
    """

    def __init__(self):
        super().__init__()

        # Load pretrained ResNet-18
        self.backbone = models.resnet18(pretrained=True)

        # Replace first conv: 3 channels → 4 channels (greyscale stacks)
        original_conv = self.backbone.conv1
        self.backbone.conv1 = nn.Conv2d(
            4, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        # Initialize 4th channel with average of pretrained weights
        with torch.no_grad():
            self.backbone.conv1.weight[:, :3, :, :] = original_conv.weight
            self.backbone.conv1.weight[:, 3, :, :] = original_conv.weight.mean(dim=1)

        # Remove original fully connected layer
        self.backbone.fc = nn.Identity()

        # Dual action heads
        self.steer_head = nn.Linear(512, 1)
        self.action_head = nn.Linear(512, 2)  # throttle, brake

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, 4, 384, 480) frame stacks

        Returns:
            steer: (batch, 1) continuous steering
            actions: (batch, 2) throttle and brake (sigmoid for BCE)
        """
        features = self.backbone(x)  # (batch, 512)
        steer = self.steer_head(features)  # (batch, 1)
        actions = torch.sigmoid(self.action_head(features))  # (batch, 2)
        return steer, actions


# ============================================================================
# Training Loop
# ============================================================================

def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    steer_weight: float = 1.0
) -> Tuple[float, float, float]:
    """
    Train for one epoch.

    Returns:
        avg_steer_loss, avg_action_loss, avg_total_loss
    """
    model.train()
    steer_losses = []
    action_losses = []
    total_losses = []

    for frames, actions in train_loader:
        frames = frames.to(device)
        actions = actions.to(device)

        # Unpack actions: [steer, throttle, brake]
        steer_gt = actions[:, 0:1]  # (batch, 1)
        throttle_brake_gt = actions[:, 1:3]  # (batch, 2)

        optimizer.zero_grad()

        # Forward pass
        steer_pred, actions_pred = model(frames)

        # Loss computation
        steer_loss = nn.MSELoss()(steer_pred, steer_gt)
        action_loss = nn.BCELoss()(actions_pred, throttle_brake_gt)
        total_loss = steer_weight * steer_loss + action_loss

        # Backward pass
        total_loss.backward()
        optimizer.step()

        steer_losses.append(steer_loss.item())
        action_losses.append(action_loss.item())
        total_losses.append(total_loss.item())

    return (
        np.mean(steer_losses),
        np.mean(action_losses),
        np.mean(total_losses)
    )


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    steer_weight: float = 1.0
) -> Tuple[float, float, float]:
    """
    Validate for one epoch.

    Returns:
        avg_steer_loss, avg_action_loss, avg_total_loss
    """
    model.eval()
    steer_losses = []
    action_losses = []
    total_losses = []

    with torch.no_grad():
        for frames, actions in val_loader:
            frames = frames.to(device)
            actions = actions.to(device)

            steer_gt = actions[:, 0:1]
            throttle_brake_gt = actions[:, 1:3]

            steer_pred, actions_pred = model(frames)

            steer_loss = nn.MSELoss()(steer_pred, steer_gt)
            action_loss = nn.BCELoss()(actions_pred, throttle_brake_gt)
            total_loss = steer_weight * steer_loss + action_loss

            steer_losses.append(steer_loss.item())
            action_losses.append(action_loss.item())
            total_losses.append(total_loss.item())

    return (
        np.mean(steer_losses),
        np.mean(action_losses),
        np.mean(total_losses)
    )


# ============================================================================
# Main Training
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Train behavioral cloning agent for NFS Most Wanted 2005'
    )
    parser.add_argument(
        '--data-dir',
        type=Path,
        default=Path('/mnt/data/processed'),
        help='Directory containing processed sessions (default: /mnt/data/processed)'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('/mnt/data/checkpoints'),
        help='Directory to save checkpoints and logs (default: /mnt/data/checkpoints)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=256,
        help='Batch size (default: 256, tuned for H100 80GB with 16.6k dataset)'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=30,
        help='Number of epochs (default: 30)'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=1e-4,
        help='Learning rate (default: 1e-4)'
    )
    parser.add_argument(
        '--weight-decay',
        type=float,
        default=1e-4,
        help='Weight decay for AdamW (default: 1e-4)'
    )
    parser.add_argument(
        '--steer-weight',
        type=float,
        default=1.0,
        help='Weight for steer loss in total loss (default: 1.0)'
    )
    parser.add_argument(
        '--num-workers',
        type=int,
        default=8,
        help='Number of DataLoader workers (default: 8)'
    )

    args = parser.parse_args()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Discover sessions
    data_dir = Path(args.data_dir)
    sessions = sorted([d for d in data_dir.iterdir() if d.is_dir()])
    if not sessions:
        logger.error(f"No session directories found in {data_dir}")
        return

    logger.info(f"Found {len(sessions)} sessions: {[s.name for s in sessions]}")

    # Load datasets
    train_datasets = []
    val_datasets = []

    for session_dir in sessions:
        frames_path = session_dir / 'frames.npy'
        inputs_path = session_dir / 'inputs.npy'

        if not frames_path.exists() or not inputs_path.exists():
            logger.warning(f"Skipping {session_dir.name}: missing frames or inputs")
            continue

        train_ds = RacingDataset(frames_path, inputs_path, split='train')
        val_ds = RacingDataset(frames_path, inputs_path, split='val')

        train_datasets.append(train_ds)
        val_datasets.append(val_ds)

    if not train_datasets:
        logger.error("No valid datasets loaded")
        return

    # Combine datasets
    train_dataset = ConcatDataset(train_datasets)
    val_dataset = ConcatDataset(val_datasets)

    logger.info(
        f"Training set: {len(train_dataset)} samples "
        f"| Validation set: {len(val_dataset)} samples"
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # Initialize model
    model = RacingAgent().to(device)
    logger.info(
        f"Model parameters: {sum(p.numel() for p in model.parameters()):,}"
    )

    # Optimizer and scheduler
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Initialize Weights & Biases
    hparams = {
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'lr': args.lr,
        'weight_decay': args.weight_decay,
        'steer_weight': args.steer_weight,
        'num_workers': args.num_workers,
    }
    wandb.init(
        project='nfs-most-wanted-2005',
        config=hparams,
        name=f'bc-baseline-{args.batch_size}bs'
    )
    logger.info(f"Weights & Biases initialized | Run: {wandb.run.name}")

    # Training loop
    best_val_loss = float('inf')
    best_model_path = args.output_dir / 'best_model.pt'

    for epoch in range(1, args.epochs + 1):
        # Train
        train_steer_loss, train_action_loss, train_total_loss = train_epoch(
            model, train_loader, optimizer, device, args.steer_weight
        )

        # Validate
        val_steer_loss, val_action_loss, val_total_loss = validate(
            model, val_loader, device, args.steer_weight
        )

        # LR scheduling
        scheduler.step()

        # Logging
        logger.info(
            f"Epoch {epoch}/{args.epochs} | "
            f"Train: steer={train_steer_loss:.4f} action={train_action_loss:.4f} "
            f"total={train_total_loss:.4f} | "
            f"Val: steer={val_steer_loss:.4f} action={val_action_loss:.4f} "
            f"total={val_total_loss:.4f} | "
            f"LR={optimizer.param_groups[0]['lr']:.6f}"
        )

        # Weights & Biases logging
        wandb.log({
            'epoch': epoch,
            'train/steer_loss': train_steer_loss,
            'train/action_loss': train_action_loss,
            'train/total_loss': train_total_loss,
            'val/steer_loss': val_steer_loss,
            'val/action_loss': val_action_loss,
            'val/total_loss': val_total_loss,
            'learning_rate': optimizer.param_groups[0]['lr'],
        })

        # Save best model
        if val_total_loss < best_val_loss:
            best_val_loss = val_total_loss
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"Saved best model (val_loss={best_val_loss:.4f})")

    wandb.finish()

    # Save training metadata
    metadata = {
        'best_val_loss': float(best_val_loss),
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'weight_decay': args.weight_decay,
        'steer_weight': args.steer_weight,
        'train_samples': len(train_dataset),
        'val_samples': len(val_dataset),
        'wandb_run_id': wandb.run.id,
        'wandb_project': wandb.run.project,
    }
    metadata_path = args.output_dir / 'training_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Training complete. Best model: {best_model_path}")
    logger.info(f"Weights & Biases dashboard: {wandb.run.get_url()}")


if __name__ == '__main__':
    main()
