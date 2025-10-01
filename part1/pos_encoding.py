"""
1.1 Positional Encodings: Learned vs Sinusoidal
"""

import torch
import torch.nn as nn
import math


class PositionalEncodingLearned(nn.Module):
    """
    Learned positional embeddings.
    """
    def __init__(self, d_model: int, max_seq_len: int):
        super().__init__()
        self.pe = nn.Embedding(max_seq_len, d_model)

    def forward(self, seq_len: int) -> torch.Tensor:
        positions = torch.arange(seq_len)
        return self.pe(positions)


class PositionalEncodingSinusoidal(nn.Module):
    """
    Sinusoidal positional embeddings (Vaswani et al.).
    """
    def __init__(self, d_model: int, max_seq_len: int = 10000):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        pe = torch.zeros(seq_len, self.d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * (-math.log(10000.0) / self.d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).to(x.device)
        return pe
