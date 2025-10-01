"""
Demo: Visualize Multi-Head Attention (saves attention heatmaps per head in grid)
"""

import os
import torch
import matplotlib.pyplot as plt
from vis_utils import plot_attention_heatmap


def demo_visualize_attention(mha_model: torch.nn.Module, x: torch.Tensor, save_dir: str = "out"):
    """
    Saves attention heatmaps for each head in a grid.
    Assumes mha_model is MultiHeadAttention.
    """
    os.makedirs(save_dir, exist_ok=True)

    # Access heads (assuming mha_model.heads is accessible)
    num_heads = len(mha_model.heads)
    B, T, d_model = x.shape
    d_head = mha_model.d_head

    # Collect attention weights for each head
    attn_maps = []
    for head_idx, head in enumerate(mha_model.heads):
        # Manual attention computation to get weights
        q = head.query(x)
        k = head.key(x)
        v = head.value(x)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (d_head ** 0.5)
        attn_weights = torch.softmax(attn_scores, dim=-1)

        # Take first batch (B=1 for viz)
        attn_map = attn_weights[0].detach().cpu().numpy()
        attn_maps.append(attn_map)

    # Create grid of heatmaps
    fig, axes = plt.subplots(num_heads // 4 + 1, 4, figsize=(20, 5 * (num_heads // 4 + 1)))
    axes = axes.flatten()

    for head_idx, attn_map in enumerate(attn_maps):
        ax = axes[head_idx]
        im = ax.imshow(attn_map, cmap='viridis', aspect='equal')
        ax.set_title(f'Head {head_idx}')
        ax.set_xlabel('Key Pos')
        ax.set_ylabel('Query Pos')
        plt.colorbar(im, ax=ax)

    # Hide empty axes
    for i in range(len(attn_maps), len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    save_path = os.path.join(save_dir, "multi_head_attention_grid.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved attention grids to {save_path}")
