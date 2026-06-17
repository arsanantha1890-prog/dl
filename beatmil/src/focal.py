"""Focal loss for the beat-level head."""

from __future__ import annotations
import torch
import torch.nn.functional as F


def focal_cross_entropy(
    logits: torch.Tensor,            # (N, K)
    targets: torch.Tensor,           # (N,)
    gamma: float = 2.0,
    alpha: torch.Tensor | None = None,  # (K,) class weights, or None
    reduction: str = "mean",
) -> torch.Tensor:
    log_p = F.log_softmax(logits, dim=-1)
    p = log_p.exp()
    K = logits.shape[-1]
    y = F.one_hot(targets, num_classes=K).float()
    pt = (p * y).sum(dim=-1)
    log_pt = (log_p * y).sum(dim=-1)
    focal = -((1.0 - pt) ** gamma) * log_pt
    if alpha is not None:
        a = (alpha[targets]).to(focal.dtype).to(focal.device)
        focal = a * focal
    if reduction == "mean":
        return focal.mean()
    elif reduction == "sum":
        return focal.sum()
    return focal
