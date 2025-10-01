"""
1.2 Self-Attention from First Principles (NumPy Computation)
"""

import numpy as np


def self_attention_numpy(x: np.ndarray) -> np.ndarray:
    """
    Manual computation of self-attention with a tiny example.
    x: (batch, seq, embed_dim)
    Returns: (batch, seq, embed_dim)
    """
    B, T, C = x.shape
    # Simple setup: use x as q=k=v, no projections
    query = x
    key = x
    value = x

    # Attention scores: (B, T, T)
    attn_scores = np.matmul(query, key.transpose(0, 2, 1)) / np.sqrt(C)

    # Apply softmax along seq dimension
    attn_weights = softmax(attn_scores, axis=-1)

    # Weighted sum: (B, T, T) @ (B, T, C) -> (B, T, C)
    output = np.matmul(attn_weights, value)
    return output


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Stable softmax implementation.
    """
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def demo_attention_math():
    """
    Tiny example with manual calculations.
    """
    # B=1, T=3, C=2
    x = np.array([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]])
    print("Input x (B=1, T=3, C=2):")
    print(x)

    output = self_attention_numpy(x)
    print("\nSelf-Attention Output:")
    print(output)


if __name__ == "__main__":
    demo_attention_math()
