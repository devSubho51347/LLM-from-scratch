"""
Tests: Verifies masking behavior.
"""

import torch
from attn_mask import CausalMask


def test_causal_mask_creation():
    """Test mask creation."""
    seq_len = 5
    mask = CausalMask(seq_len)
    expected = torch.tensor([
        [0.,    -float('inf'), -float('inf'), -float('inf'), -float('inf')],
        [0.,    0.,    -float('inf'), -float('inf'), -float('inf')],
        [0.,    0.,    0.,    -float('inf'), -float('inf')],
        [0.,    0.,    0.,    0.,    -float('inf')],
        [0.,    0.,    0.,    0.,    0.]
    ])
    torch.testing.assert_close(mask.get_mask(), expected, rtol=0, atol=0)
    print("✅ Mask creation correct!")


def test_apply_causal_mask():
    """Test applying mask to attention scores."""
    seq_len = 3
    mask = CausalMask(seq_len)
    attn_scores = torch.randn(1, seq_len, seq_len)  # (B=1, T, T)

    masked_scores = mask.apply_causal_mask(attn_scores)
    # Bottom triangle should be unchanged, upper triangle -inf
    # For seq_len=3:
    assert masked_scores[0, 0, 1] == float('-inf')
    assert masked_scores[0, 0, 2] == float('-inf')
    assert masked_scores[0, 1, 2] == float('-inf')
    # Diagonal and lower unchanged
    assert masked_scores[0, 0, 0] == attn_scores[0, 0, 0]
    assert masked_scores[0, 1, 0] == attn_scores[0, 1, 0]
    assert masked_scores[0, 1, 1] == attn_scores[0, 1, 1]
    assert masked_scores[0, 2, 0] == attn_scores[0, 2, 0]
    assert masked_scores[0, 2, 1] == attn_scores[0, 2, 1]
    assert masked_scores[0, 2, 2] == attn_scores[0, 2, 2]
    print("✅ Mask application correct!")


def test_batch_mask():
    """Test mask with batch dimension."""
    seq_len = 4
    mask = CausalMask(seq_len)
    attn_scores = torch.randn(2, seq_len, seq_len)  # Batch=2

    masked = mask.apply_causal_mask(attn_scores)
    # Check that for each batch, upper triangle is masked
    for b in range(2):
        # Upper triangle
        assert masked[b, 0, 1] == float('-inf')
        assert masked[b, 0, 2] == float('-inf')
        assert masked[b, 0, 3] == float('-inf')
        assert masked[b, 1, 2] == float('-inf')
        assert masked[b, 1, 3] == float('-inf')
        assert masked[b, 2, 3] == float('-inf')
        # Lower triangle and diagonal unchanged
        assert masked[b, 1, 0] == attn_scores[b, 1, 0]
        assert masked[b, 2, 1] == attn_scores[b, 2, 1]
        assert masked[b, 3, 2] == attn_scores[b, 3, 2]
    print("✅ Batch mask correct!")


if __name__ == "__main__":
    test_causal_mask_creation()
    test_apply_causal_mask()
    test_batch_mask()
    print("All causal mask tests passed!")
