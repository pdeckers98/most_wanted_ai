"""ResNet-18 policy network for NFS Most Wanted 2005 racing agent."""

from typing import Tuple

import torch
import torch.nn as nn
from torchvision import models


class RacingAgent(nn.Module):
    """
    ResNet-18 backbone with dual action heads.

    Steer head:  Linear(512, 1) -> continuous steering in [-1, 1]
    Action head: Linear(512, 2) -> sigmoid -> [throttle, brake] in [0, 1]

    The first conv layer accepts 4-channel greyscale input instead of the
    standard 3-channel RGB. When pretrained=True the 4th channel is
    initialised from the mean of the pretrained 3-channel weights.
    """

    def __init__(self, pretrained: bool = False):
        super().__init__()

        weights = (
            models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        )
        self.backbone = models.resnet18(weights=weights)

        original_conv = self.backbone.conv1
        self.backbone.conv1 = nn.Conv2d(
            4, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        with torch.no_grad():
            self.backbone.conv1.weight[:, :3, :, :] = original_conv.weight
            self.backbone.conv1.weight[:, 3, :, :] = (
                original_conv.weight.mean(dim=1)
            )

        self.backbone.fc = nn.Identity()
        self.steer_head = nn.Linear(512, 1)
        self.action_head = nn.Linear(512, 2)  # [throttle, brake]

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, 4, 384, 480) normalised frame stacks

        Returns:
            steer:   (batch, 1) continuous steering (unclamped)
            actions: (batch, 2) throttle and brake after sigmoid
        """
        features = self.backbone(x)            # (batch, 512)
        steer = self.steer_head(features)      # (batch, 1)
        actions = torch.sigmoid(               # (batch, 2)
            self.action_head(features)
        )
        return steer, actions
