"""
1.6 Transformer Block (residuals + LayerNorm)
"""

import torch
import torch.nn as nn
from multi_head import MultiHeadAttention
from ffn import FeedForward


class TransformerBlock(nn.Module):
    """
    Single transformer block: MHA + FFN with residual connections and LayerNorm.
    """
    def __init__(self, d_model: int, num_heads: int, d_ff: int):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        x: (B, T, d_model)
        mask: (T, T) causal mask
        Returns: (B, T, d_model)
        """
        # Attention with residual + prenorm
        attn_out = self.attn(self.norm1(x), mask)
        x = x + attn_out  # Residual

        # FFN with residual + prenorm
        ffn_out = self.ffn(self.norm2(x))
        x = x + ffn_out  # Residual

        return x
