"""
Tests: Correctness of tiny example vs PyTorch single-head
"""

import torch
import numpy as np
import pytest
from attn_numpy_demo import self_attention_numpy, softmax
from single_head import SingleHeadAttention


def test_softmax():
    """Test custom softmax matches PyTorch."""
    x = torch.randn(3, 4)
    np_x = x.numpy()
    custom = softmax(np_x, axis=-1)
    pytorch = torch.softmax(x, dim=-1).numpy()
    np.testing.assert_allclose(custom, pytorch, rtol=1e-5)


def test_self_attention_numpy_vs_pytorch():
    """Test NumPy attention matches simplified PyTorch version."""
    B, T, C = 1, 4, 3
    x = torch.randn(B, T, C)
    np_x = x.numpy()

    # NumPy version
    np_out = self_attention_numpy(np_x)

    # PyTorch version (without projections: q=k=v=x)
    attn_out = torch.zeros_like(x)
    for b in range(B):
        for t in range(T):
            query = x[b, t]  # (C,)
            scores = torch.sum(x[b] * query, dim=-1) / (C ** 0.5)  # (T,)
            weights = torch.softmax(scores, dim=-1)
            attn_out[b, t] = torch.sum(weights.unsqueeze(-1) * x[b], dim=0)

    np.testing.assert_allclose(np_out, attn_out.numpy(), rtol=1e-4)
    print("✅ NumPy attention matches PyTorch!")


def test_single_head_shape():
    """Test SingleHeadAttention produces correct output shape."""
    B, T, d_model, d_head = 2, 10, 64, 32
    head = SingleHeadAttention(d_model, d_head)
    x = torch.randn(B, T, d_model)
    out = head(x)
    assert out.shape == (B, T, d_head)


if __name__ == "__main__":
    test_softmax()
    test_self_attention_numpy_vs_pytorch()
    test_single_head_shape()
    print("All tests passed!")
