"""
Causal Mask Helpers
"""

import torch


class CausalMask:
    """
    Generates causal attention mask for auto-regressive generation.
    """
    def __init__(self, seq_len: int):
        self.seq_len = seq_len
        self.mask = self._create_mask()

    def _create_mask(self) -> torch.Tensor:
        """
        Creates upper triangular mask: 0 on diagonal and below, -inf above.
        Shape: (seq_len, seq_len)
        """
        mask = torch.triu(torch.ones(self.seq_len, self.seq_len), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask

    def get_mask(self) -> torch.Tensor:
        """
        Returns the mask: (seq_len, seq_len)
        """
        return self.mask.clone()

    def apply_causal_mask(self, attn_scores: torch.Tensor) -> torch.Tensor:
        """
        Applies mask to attention scores.
        attn_scores: (B, T, T) or (T, T)
        """
        if attn_scores.dim() == 2:
            return attn_scores + self.mask
        elif attn_scores.dim() == 3:
            B, T, T = attn_scores.shape
            return attn_scores + self.mask.unsqueeze(0)
        else:
            raise ValueError("Invalid attn_scores shape")
