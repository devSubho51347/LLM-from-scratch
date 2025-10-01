"""
1.3 Single Attention Head (PyTorch)
"""

import torch
import torch.nn as nn


class SingleHeadAttention(nn.Module):
    """
    Single attention head with query, key, value projections.
    """
    def __init__(self, d_model: int, d_head: int):
        super().__init__()
        self.d_head = d_head
        self.query = nn.Linear(d_model, d_head)
        self.key = nn.Linear(d_model, d_head)
        self.value = nn.Linear(d_model, d_head)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        x: (B, T, d_model)
        mask: (T, T) for causal masking
        Returns: (B, T, d_head)
        """
        B, T, _ = x.shape

        q = self.query(x)  # (B, T, d_head)
        k = self.key(x)   # (B, T, d_head)
        v = self.value(x) # (B, T, d_head)

        # Attention scores: (B, T, T)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_head ** 0.5)

        # Apply mask if provided (causal masking)
        if mask is not None:
            attn_scores = attn_scores + mask.unsqueeze(0)

        # Softmax
        attn_weights = torch.softmax(attn_scores, dim=-1)

        # Weighted sum: (B, T, T) @ (B, T, d_head) -> (B, T, d_head)
        output = torch.matmul(attn_weights, v)
        return output
