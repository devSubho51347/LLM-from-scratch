"""
1.5 Feed-Forward Network (GELU, dimensionality expansion)
"""

import torch
import torch.nn as nn


class FeedForward(nn.Module):
    """
    Feed-forward network: Linear -> GELU -> Linear.
    Typically expands to d_ff = mult * d_model, then back.
    """
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, d_model)
        Returns: (B, T, d_model)
        """
        return self.net(x)
