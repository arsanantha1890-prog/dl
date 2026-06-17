"""
Consistency loss between bag-level and pooled-beat-level predictions.

This is the core regulariser of the Beat-MIL contribution.

When beat-level labels are available (MIT-BIH samples), we train two
prediction paths:
    path A  (bag head):   p_bag    = softmax( MLP_bag( H ) )
                          where H = sum_i alpha_i * z_i  (MIL-pooled)
    path B  (pooled-beat): p_pooled = softmax( sum_i alpha_i * MLP_beat(z_i) )

If the model is internally consistent, these should agree: the bag-level
prediction should be the attention-weighted aggregation of the beat-level
predictions. We enforce this with a symmetric KL penalty.

Notes:
    - We use symmetric KL (Jensen-Shannon-like) so the gradient flows
      meaningfully to both heads, not just one.
    - We detach NEITHER side: both heads should converge to agree.
    - The loss is only computed for MIT-BIH samples in the batch (the
      caller masks others before averaging).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def symmetric_kl(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Symmetric KL = 0.5 * ( KL(p||q) + KL(q||p) ), batch-wise.

    Args:
        p: (B, K) probability distribution.
        q: (B, K) probability distribution.

    Returns:
        (B,) per-sample symmetric KL.
    """
    p = p.clamp(min=eps)
    q = q.clamp(min=eps)
    kl_pq = (p * (p.log() - q.log())).sum(dim=-1)
    kl_qp = (q * (q.log() - p.log())).sum(dim=-1)
    return 0.5 * (kl_pq + kl_qp)


def consistency_loss(
    bag_logits: torch.Tensor,          # (B, K)
    beat_logits: torch.Tensor,         # (B, N, K) per-beat class logits
    alpha: torch.Tensor,               # (B, N) MIL attention weights, sum to 1
    mask: torch.Tensor | None = None,  # (B, N), 1 valid, 0 padding
    sample_mask: torch.Tensor | None = None,  # (B,) 1 if beat-labels available
    eps: float = 1e-8,
) -> torch.Tensor:
    """Compute symmetric KL between bag prediction and alpha-pooled
    beat prediction. If sample_mask is given, only those samples
    contribute (others contribute zero).
    """
    # path A: bag head softmax
    p_bag = F.softmax(bag_logits, dim=-1)                              # (B, K)

    # path B: alpha-pooled beat logits, then softmax
    # zero-out padded beats so they don't contribute
    if mask is not None:
        alpha = alpha * mask
        alpha = alpha / (alpha.sum(dim=1, keepdim=True).clamp(min=eps))
    pooled_logits = torch.einsum("bn,bnk->bk", alpha, beat_logits)     # (B, K)
    p_pooled = F.softmax(pooled_logits, dim=-1)                        # (B, K)

    per_sample = symmetric_kl(p_bag, p_pooled)                         # (B,)

    if sample_mask is not None:
        per_sample = per_sample * sample_mask
        denom = sample_mask.sum().clamp(min=1.0)
        return per_sample.sum() / denom
    return per_sample.mean()


# ---------- smoke ---------------------------------------------------------

def _smoke_test() -> None:
    torch.manual_seed(0)
    B, N, K = 8, 10, 4
    bag_logits = torch.randn(B, K, requires_grad=True)
    beat_logits = torch.randn(B, N, K, requires_grad=True)
    alpha = torch.softmax(torch.randn(B, N), dim=1)
    mask = torch.ones(B, N)
    mask[3:, -2:] = 0  # last 2 beats padding for some bags

    # only half the batch has beat labels (e.g. MIT-BIH portion)
    sample_mask = torch.zeros(B); sample_mask[:4] = 1.0

    loss = consistency_loss(bag_logits, beat_logits, alpha, mask, sample_mask)
    assert torch.isfinite(loss), "loss not finite"
    loss.backward()
    assert bag_logits.grad is not None and bag_logits.grad.abs().sum() > 0
    assert beat_logits.grad is not None and beat_logits.grad.abs().sum() > 0

    # ablation: when bag and pooled match exactly, loss should be ~0
    aligned_bag = torch.zeros(B, K)
    aligned_bag[:, 0] = 5.0       # all confident on class 0
    aligned_beat = torch.zeros(B, N, K)
    aligned_beat[:, :, 0] = 5.0
    z = consistency_loss(aligned_bag, aligned_beat, alpha, mask, sample_mask)
    assert z.item() < 1e-3, f"loss should be tiny when aligned: {z.item()}"

    # ablation: when sample_mask is all zero, loss should be 0
    z0 = consistency_loss(bag_logits, beat_logits, alpha, mask,
                          sample_mask=torch.zeros(B))
    assert z0.item() == 0.0, f"loss should be zero when no labeled samples: {z0}"

    print(f"[Consistency] random loss = {loss.item():.4f}, "
          f"aligned loss = {z.item():.6f}")
    print("[Consistency] smoke test passed.")


if __name__ == "__main__":
    _smoke_test()
