"""
architecture.py - Neural Network for Pacman Imitation Learning
"""

import torch
import torch.nn as nn


class PacmanNetwork(nn.Module):
    """
    Simple MLP to imitate the expert Pacman player.

    - Input: 24 features
    - Output: 5 logits (one per action)
    - Architecture: 24 -> 128 -> 128 -> 64 -> 5

    Why MLP and not CNN?
    Our input is a 1D vector of features, not a 2D image.
    CNNs exploit spatial structure (neighboring pixels).
    Our features have no spatial relationship to each other.
    """

    def __init__(self):
        super().__init__()

        # Architecture: Linear + BatchNorm + ReLU
        # BatchNorm = normalizes values to stabilize training
        self.net = nn.Sequential(
            nn.Linear(24, 128),
            nn.BatchNorm1d(128),  # Normalize for stability
            nn.ReLU(),

            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Linear(64, 5)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x: Tensor of shape (batch_size, 24) - normalized features

        Returns:
            Tensor of shape (batch_size, 5) - logits for each action
        """
        return self.net(x)
