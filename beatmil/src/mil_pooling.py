"""
Gated Attention-based MIL Pooling.

Reference:
    Ilse, M., Tomczak, J. M., & Welling, M. (2018).
    Attention-based Deep Multiple Instance Learning. ICML 2018.
    arXiv:1802.04712

For ECG, a "bag" is one 10-second segment, and "instances" are the
B detected beats inside it. The pooling weights alpha_i indicate
which beats drove the bag prediction — providing built-in,
beat-level interpretability of rhythm-level decisions.

Important shapes:
    z: (batch, B_max, d)      beat embeddings, padded
    mask: (batch, B_max)      1 for valid beats, 0 for padding
    H: (batch, d)             pooled bag representation
    alpha: (batch, B_max)     attention weights (sum to 1 over valid beats)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedAttentionMIL(nn.Module):
    """Gated attention pooling over a variable-length set of instances.

    a_i  = w^T ( tanh(V z_i) * sigmoid(U z_i) )      [scalar score per instance]
    alpha = softmax(a)                                [over valid instances]
    H    = sum_i alpha_i * z_i

    The gating (sigmoid(U z_i)) lets the model down-weight uninformative
    beats more sharply than plain additive attention.
    """

    def __init__(self, d_in: int, d_hidden: int = 128, dropout: float = 0.1):
        super().__init__()
        self.V = nn.Linear(d_in, d_hidden, bias=False)
        self.U = nn.Linear(d_in, d_hidden, bias=False)
        self.w = nn.Linear(d_hidden, 1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        z: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            z: (B, N, d_in) beat embeddings
            mask: (B, N) 1 for valid beats, 0 for padding. If None, all valid.

        Returns:
            H: (B, d_in) bag representation
            alpha: (B, N) attention weights (sum to 1 across valid beats)
        """
        # gated attention scores
        v = torch.tanh(self.V(z))           # (B, N, H)
        u = torch.sigmoid(self.U(z))        # (B, N, H)
        gated = self.dropout(v * u)         # (B, N, H)
        scores = self.w(gated).squeeze(-1)  # (B, N)

        # mask padding (set to -inf so softmax → 0)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        alpha = torch.softmax(scores, dim=1)             # (B, N)
        # zero out alpha where mask is 0 (handles all-padding edge case)
        if mask is not None:
            alpha = alpha * mask
            # avoid div-by-zero if a bag is somehow all-padding
            alpha = alpha / (alpha.sum(dim=1, keepdim=True).clamp(min=1e-8))

        H = torch.einsum("bn,bnd->bd", alpha, z)         # (B, d_in)
        return H, alpha


# ---------- tests / smoke -------------------------------------------------

def _smoke_test() -> None:
    """Run quickly to verify shapes and grad flow."""
    torch.manual_seed(0)
    B, N, d = 4, 12, 32
    pool = GatedAttentionMIL(d_in=d, d_hidden=16)
    z = torch.randn(B, N, d, requires_grad=True)
    # last 3 beats of last bag are padding
    mask = torch.ones(B, N)
    mask[-1, -3:] = 0

    H, alpha = pool(z, mask)
    assert H.shape == (B, d), f"H shape {H.shape}"
    assert alpha.shape == (B, N), f"alpha shape {alpha.shape}"

    # padding positions must have zero attention
    assert torch.allclose(alpha[-1, -3:], torch.zeros(3), atol=1e-6), \
        f"padding attention not zero: {alpha[-1, -3:]}"

    # attention sums to 1 (over valid beats)
    sums = alpha.sum(dim=1)
    assert torch.allclose(sums, torch.ones(B), atol=1e-5), f"sums={sums}"

    # backward works
    H.sum().backward()
    assert z.grad is not None and z.grad.abs().sum() > 0, "no grad flow"
    print("[GatedAttentionMIL] smoke test passed.")


if __name__ == "__main__":
    _smoke_test()
