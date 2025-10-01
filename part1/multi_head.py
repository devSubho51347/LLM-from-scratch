"""
1.4 Multi-Head Attention (with shape tracing)
"""

import torch
import torch.nn as nn
from single_head import SingleHeadAttention


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention layer.
    """
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_head = d_model // num_heads
        self.num_heads = num_heads

        self.heads = nn.ModuleList([
            SingleHeadAttention(d_model, self.d_head) for _ in range(num_heads)
        ])
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        x: (B, T, d_model)
        mask: (T, T) causal mask
        Returns: (B, T, d_model)
        """
        # Apply each head: list of (B, T, d_head)
        head_outputs = [head(x, mask) for head in self.heads]

        # Concatenate: (B, T, num_heads * d_head) = (B, T, d_model)
        concatenated = torch.cat(head_outputs, dim=-1)

        # Project back to d_model
        output = self.proj(concatenated)
        return output
