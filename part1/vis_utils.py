"""
Plotting Helpers (matrices & attention maps)
"""

import matplotlib.pyplot as plt
import numpy as np
import os


def plot_attention_heatmap(attn_weights: np.ndarray, head_idx: int, save_path: str = None):
    """
    Plots attention heatmap for a single head.
    attn_weights: (seq_len, seq_len) or (T, T)
    """
    seq_len = attn_weights.shape[0]
    plt.figure(figsize=(8, 6))
    plt.imshow(attn_weights, cmap='viridis', aspect='equal')
    plt.colorbar()
    plt.title(f'Attention Heatmap - Head {head_idx}')
    plt.xlabel('Key Position')
    plt.ylabel('Query Position')
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def plot_positional_encodings(pe: np.ndarray, title: str, save_path: str = None):
    """
    Visualizes positional encodings as a heatmap.
    pe: (seq_len, d_model)
    """
    plt.figure(figsize=(10, 6))
    plt.imshow(pe.T, cmap='bwr', aspect='auto')  # Transpose for (d_model, seq_len)
    plt.colorbar()
    plt.title(title)
    plt.xlabel('Sequence Position')
    plt.ylabel('Embedding Dimension')
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()
