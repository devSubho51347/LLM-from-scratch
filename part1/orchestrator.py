"""
Part 1: Core Transformer Architecture
Orchestrator for building fundamental transformer components.
"""

import torch
import torch.nn as nn
from typing import Tuple


def main():
    """
    Run all substeps for Part 1.
    """
    print("Starting Part 1: Core Transformer Architecture")

    # 1.1 Positional embeddings (absolute learned vs. sinusoidal)
    print("1.1 Positional embeddings")
    pos_enc_abs, pos_enc_sin = positional_embeddings(seq_len=10, d_model=8)
    print(f"Absolute positional: shape {pos_enc_abs.shape}")
    print(f"Sinusoidal positional: shape {pos_enc_sin.shape}")

    # 1.2 Self-attention from first principles
    print("\n1.2 Self-attention from first principles")
    x = torch.randn(1, 10, 8)  # batch=1, seq=10, dim=8
    manual_attn = self_attention_manual(x, d_head=8)
    print(f"Manual attention output: shape {manual_attn.shape}")

    # 1.3 Building a single attention head in PyTorch
    print("\n1.3 Single attention head")
    head = SingleAttentionHead(d_model=8, d_head=8)
    head_out = head(x)
    print(f"Attention head output: shape {head_out.shape}")

    # 1.4 Multi-head attention
    print("\n1.4 Multi-head attention")
    multihead = MultiHeadAttention(d_model=8, num_heads=2, d_head=4)
    multi_out = multihead(x)
    print(f"Multi-head output: shape {multi_out.shape}")

    # 1.5 Feed-forward networks
    print("\n1.5 Feed-forward networks")
    ff = FeedForward(d_model=8, d_ff=32)
    ff_out = ff(x)
    print(f"Feed-forward output: shape {ff_out.shape}")

    # 1.6 Residual connections & LayerNorm
    print("\n1.6 Residual connections & LayerNorm")
    norm_module = nn.LayerNorm(8)
    residual_out = x + norm_module(x)  # Simple residual with norm
    print(f"Residual + LayerNorm output: shape {residual_out.shape}")

    # 1.7 Stacking into a full Transformer block
    print("\n1.7 Full Transformer block")
    block = TransformerBlock(d_model=8, num_heads=2, d_ff=32)
    block_out = block(x)
    print(f"Transformer block output: shape {block_out.shape}")

    print("\nPart 1 completed!")


# Placeholder functions - to be implemented with actual logic

def positional_embeddings(seq_len: int, d_model: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """1.1: Implement absolute learned and sinusoidal positional embeddings."""
    # Learned
    pos_enc_abs = nn.Embedding(seq_len, d_model)()

    # Sinusoidal
    pos_enc_sin = torch.zeros(seq_len, d_model)
    position = torch.arange(0, seq_len).unsqueeze(1).float()
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(torch.log(torch.tensor(10000.0)) / d_model))
    pos_enc_sin[:, 0::2] = torch.sin(position * div_term)
    pos_enc_sin[:, 1::2] = torch.cos(position * div_term)

    return pos_enc_abs, pos_enc_sin


def self_attention_manual(x: torch.Tensor, d_head: int) -> torch.Tensor:
    """1.2: Manual computation of self-attention."""
    B, T, C = x.shape
    query = x  # Simplified, same as key/value
    key = x
    value = x

    # Attention scores
    attn = (query @ key.transpose(-2, -1)) / (d_head ** 0.5)
    attn = nn.functional.softmax(attn, dim=-1)

    # Output
    out = attn @ value
    return out


class SingleAttentionHead(nn.Module):
    """1.3: Single attention head."""

    def __init__(self, d_model: int, d_head: int):
        super().__init__()
        self.query = nn.Linear(d_model, d_head)
        self.key = nn.Linear(d_model, d_head)
        self.value = nn.Linear(d_model, d_head)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        attn = (q @ k.transpose(-2, -1)) / (C ** 0.5)
        attn = nn.functional.softmax(attn, dim=-1)
        out = attn @ v
        return out


class MultiHeadAttention(nn.Module):
    """1.4: Multi-head attention."""

    def __init__(self, d_model: int, num_heads: int, d_head: int):
        super().__init__()
        self.heads = nn.ModuleList([
            SingleAttentionHead(d_model, d_head) for _ in range(num_heads)
        ])
        self.proj = nn.Linear(num_heads * d_head, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        return out


class FeedForward(nn.Module):
    """1.5: Feed-forward network with GELU."""

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    """1.7: Full transformer block with residual and LayerNorm."""

    def __init__(self, d_model: int, num_heads: int, d_ff: int):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, num_heads, d_model // num_heads)
        self.ff = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.norm1(self.attn(x))
        x = x + self.norm2(self.ff(x))
        return x


if __name__ == "__main__":
    main()
