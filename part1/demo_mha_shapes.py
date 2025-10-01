 """
Demo: Explicit Matrix Multiplications & Shapes in Multi-Head Attention
Prints step-by-step shapes and operations.
"""


import torch


def demo_mha_shapes(seq_len: int, d_model: int, num_heads: int):
    """
    Demonstrates MHA computations with explicit shapes.
    """
    print(f"\n🔍 Multi-Head Attention Shapes Demo")
    print(f"Input: seq_len={seq_len}, d_model={d_model}, num_heads={num_heads}")

    B = 2  # Batch size for demo
    d_head = d_model // num_heads

    # Simulate input
    x = torch.randn(B, seq_len, d_model)
    print(f"x: {x.shape}  (B=Batch, T=Seq, C=d_model)")

    # Projections (conceptual)
    print("\n📊 Head Projections:")
    print(f"Each head gets d_head = {d_head}")

    # For each head: q, k, v
    q_heads = torch.randn(B, seq_len, num_heads, d_head)
    k_heads = torch.randn(B, seq_len, num_heads, d_head)
    v_heads = torch.randn(B, seq_len, num_heads, d_head)

    print(f"q_heads: {q_heads.shape}  (reshaped to group by heads)")
    print(f"k_heads: {k_heads.shape}")
    print(f"v_heads: {v_heads.shape}")

    # Attention for each head
    print("\n🧮 Attention per Head:")
    for head in range(min(num_heads, 3)):  # Show first 3
        print(f"  Head {head}:")
        # Rearrange: (B, T, H, D) -> (B, H, T, D)
        q_h = q_heads[:, :, head, :]  # (B, T, D)
        k_h = k_heads[:, :, head, :]  # (B, T, D)
        v_h = v_heads[:, :, head, :]  # (B, T, D)

        print(f"    q_h: {q_h.shape}, k_h: {k_h.shape}, v_h: {v_h.shape}")

        # Scores: (B, T, T)
        attn_scores = torch.matmul(q_h, k_h.transpose(-2, -1)) / (d_head ** 0.5)
        print(f"    scores: {attn_scores.shape} = matmul(q_h, k_h.T)")

        # Softmax: normalize over seq dim (dim=-1)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        print(f"    weights: {attn_weights.shape} (softmax over T)")

        # Output: (B, T, D)
        attn_out = torch.matmul(attn_weights, v_h)
        print(f"    attn_out: {attn_out.shape} = matmul(weights, v_h)")

    # Concat head outputs
    head_outputs = [torch.randn(B, seq_len, d_head) for _ in range(num_heads)]  # Mock
    concat = torch.cat(head_outputs, dim=-1)
    print(f"\n🔗 Concat Heads: {concat.shape} = cat([{d_head}] x {num_heads}) -> {d_model}")

    # Final projection
    final_proj = torch.randn(B, seq_len, d_model)
    print(f"🎯 Final Projection: {final_proj.shape} (W_O projection)")

    print("\n✅ Shape demo complete!")
