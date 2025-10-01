"""
Part 1: Core Transformer Architecture
Orchestrator - Runs demos/tests/visualizations for Part 1.
# Repository layout (Part 1)
#
#   part_1/
#     orchestrator.py               # runs demos/tests/visualizations for Part 1
#     pos_encoding.py               # 1.1 positional encodings (learned + sinusoidal)
#     attn_numpy_demo.py            # 1.2 self-attention math with tiny numbers (NumPy)
#     single_head.py                # 1.3 single attention head (PyTorch)
#     multi_head.py                 # 1.4 multi-head attention (with shape tracing)
#     ffn.py                        # 1.5 feed-forward network (GELU, width = mult*d_model)
#     block.py                      # 1.6 Transformer block (residuals + LayerNorm)
#     attn_mask.py                  # causal mask helpers
#     vis_utils.py                  # plotting helpers (matrices & attention maps)
#     demo_mha_shapes.py            # prints explicit matrix multiplications & shapes step-by-step
#     demo_visualize_multi_head.py  # saves attention heatmaps per head (grid)
#     out/                          # (created at runtime) images & logs live here
#     tests/
#       test_attn_math.py           # correctness: tiny example vs PyTorch single-head
#       test_causal_mask.py         # verifies masking behavior
#
# NOTE ON IMPORTS
# ----------------
# All imports are LOCAL. Run from inside `part_1/`.
# Example quickstart (CPU ok):
#   cd part_1
#   python orchestrator.py --visualize

"""

import argparse
import torch
from pos_encoding import PositionalEncodingLearned, PositionalEncodingSinusoidal
from attn_numpy_demo import self_attention_numpy
from single_head import SingleHeadAttention
from multi_head import MultiHeadAttention
from ffn import FeedForward
from block import TransformerBlock
from attn_mask import CausalMask
from vis_utils import plot_attention_heatmap
from demo_mha_shapes import demo_mha_shapes
from demo_visualize_multi_head import demo_visualize_attention


def main():
    parser = argparse.ArgumentParser(description="Part 1: Core Transformer Architecture Demo")
    parser.add_argument('--visualize', action='store_true', help='Generate visualizations and save to out/')
    args = parser.parse_args()

    print("🚀 Starting Part 1: Core Transformer Architecture")

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(42)

    # Demo parameters
    seq_len, d_model = 10, 64
    num_heads = 8
    d_ff = 256

    # === 1.1 Positional Encodings ===
    print("\n🔍 1.1 Positional Encodings (Learned vs Sinusoidal)")
    learned_enc = PositionalEncodingLearned(d_model, seq_len)
    learned = learned_enc(seq_len)

    sin_enc = PositionalEncodingSinusoidal(d_model)
    sinusoidal = sin_enc(torch.arange(seq_len).unsqueeze(0))

    print(f"Learned PE shape: {learned.shape}")
    print(f"Sinusoidal PE shape: {sinusoidal.shape}")

    # === 1.2 Self-Attention (NumPy Demo) ===
    print("\n🧮 1.2 Self-Attention from First Principles (NumPy)")
    x_tiny = torch.randn(1, 4, 3)  # B=1, T=4, C=3
    attn_out = self_attention_numpy(x_tiny)
    print(f"NumPy attention output shape: {attn_out.shape}")

    # === 1.3 Single Head Attention ===
    print("\n⚡ 1.3 Single Head Attention (PyTorch)")
    single_head = SingleHeadAttention(d_model=d_model, d_head=d_model//2).to(device)
    x = torch.randn(1, seq_len, d_model).to(device)
    sh_output = single_head(x)
    print(f"Single head output shape: {sh_output.shape}")

    # === 1.4 Multi-Head Attention ===
    print("\n🔀 1.4 Multi-Head Attention")
    print("   Running shape demo...")
    demo_mha_shapes(seq_len, d_model, num_heads)

    mha = MultiHeadAttention(d_model, num_heads).to(device)
    mha_output = mha(x)
    print(f"Multi-head output shape: {mha_output.shape}")

    if args.visualize:
        print(f"\n📊 Visualizing attention for {num_heads} heads...")
        demo_visualize_attention(mha, x, save_dir="out")

    # === 1.5 Feed-Forward Network ===
    print("\n🌐 1.5 Feed-Forward Network")
    ffn = FeedForward(d_model, d_ff).to(device)
    ffn_output = ffn(x)
    print(f"FFN output shape: {ffn_output.shape}")

    # === 1.6 Transformer Block ===
    print("\n🏗️ 1.6 Transformer Block")
    causal_mask = CausalMask(seq_len).get_mask().to(device)
    block = TransformerBlock(d_model, num_heads, d_ff).to(device)
    block_output = block(x, mask=causal_mask)
    print(f"Transformer block output shape: {block_output.shape}")

    print("\n✅ Part 1 Demo Complete!")
    if args.visualize:
        print("   Check 'out/' directory for visualizations.")


if __name__ == "__main__":
    main()
