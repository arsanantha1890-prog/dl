"""
Evidential Deep Learning for classification.

Reference:
    Sensoy, M., Kaplan, L., & Kandemir, M. (2018).
    Evidential Deep Learning to Quantify Classification Uncertainty.
    NeurIPS 2018. arXiv:1806.01768

Idea: instead of outputting softmax(logits), output evidence e_k >= 0
for each class. Set Dirichlet parameters alpha_k = e_k + 1, total
strength S = sum_k alpha_k. Then:

    expected class prob   p_k = alpha_k / S
    vacuity (uncertainty) u   = K / S

A sample with no evidence yields alpha = (1,...,1), S = K, u = 1
(maximum uncertainty). A confident prediction yields large evidence
on one class, S >> K, u << 1.

Loss: Bayes risk with squared error + variance term, plus a KL penalty
that pushes evidence for INCORRECT classes toward the uniform Dirichlet.
The KL weight is annealed over the first ~10 epochs.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------- evidential head -----------------------------------------------

class EvidentialHead(nn.Module):
    """Linear layer producing non-negative evidence per class.

    Output:  e = ReLU(W h + b)  with shape (B, K)
    Use the helper functions below to derive alpha, p, S, u.
    """

    def __init__(self, d_in: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(d_in, num_classes)
        self.num_classes = num_classes

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # ReLU keeps the design simple and stable. Sensoy et al. also
        # mention softplus and exp; we use ReLU as the most common choice.
        return F.relu(self.fc(h))


def evidence_to_alpha(evidence: torch.Tensor) -> torch.Tensor:
    return evidence + 1.0


def alpha_to_probs(alpha: torch.Tensor) -> torch.Tensor:
    S = alpha.sum(dim=-1, keepdim=True)
    return alpha / S


def vacuity(alpha: torch.Tensor) -> torch.Tensor:
    """u = K / S — epistemic uncertainty."""
    K = alpha.shape[-1]
    S = alpha.sum(dim=-1)
    return K / S


# ---------- loss ----------------------------------------------------------

def _kl_dirichlet_uniform(alpha: torch.Tensor) -> torch.Tensor:
    """KL( Dir(alpha) || Dir(1, 1, ..., 1) ), batch-wise."""
    K = alpha.shape[-1]
    S = alpha.sum(dim=-1, keepdim=True)
    # all-ones Dirichlet
    ones = torch.ones_like(alpha)
    # KL closed-form for Dirichlet
    # KL = log(Gamma(S)) - sum_k log(Gamma(alpha_k))
    #      - log(Gamma(K)) + sum_k log(Gamma(1))   [the latter term = 0]
    #      + sum_k (alpha_k - 1) * (digamma(alpha_k) - digamma(S))
    term1 = torch.lgamma(S.squeeze(-1)) - torch.lgamma(alpha).sum(dim=-1)
    term2 = -torch.lgamma(torch.tensor(float(K), device=alpha.device))
    term3 = ((alpha - 1.0) * (torch.digamma(alpha) - torch.digamma(S))).sum(dim=-1)
    return term1 + term2 + term3


def evidential_loss(
    evidence: torch.Tensor,        # (B, K)
    targets: torch.Tensor,         # (B,) int class indices
    kl_weight: float = 1.0,
    epsilon: float = 1e-8,
) -> dict[str, torch.Tensor]:
    """Bayes-risk MSE loss + annealed KL penalty.

    Sensoy 2018 eq. (5) + (10): expected sum of squared errors over
    the Dirichlet posterior, plus a KL term that drives evidence for
    incorrect classes toward 0.
    """
    B, K = evidence.shape
    alpha = evidence_to_alpha(evidence)             # (B, K)
    S = alpha.sum(dim=-1, keepdim=True)             # (B, 1)
    p = alpha / S                                   # (B, K)

    # one-hot targets
    y = F.one_hot(targets, num_classes=K).float()   # (B, K)

    # Bayes-risk squared error: E[(y - p)^2]
    err = (y - p).pow(2).sum(dim=-1)                                 # (B,)
    var = (alpha * (S - alpha) / (S * S * (S + 1))).sum(dim=-1)      # (B,)
    bayes_risk = err + var

    # KL on evidence of WRONG classes (alpha-tilde: keep correct class evidence)
    alpha_tilde = y + (1.0 - y) * alpha
    kl = _kl_dirichlet_uniform(alpha_tilde).clamp(min=0.0)

    total = bayes_risk + kl_weight * kl
    return {
        "loss": total.mean(),
        "bayes_risk": bayes_risk.mean().detach(),
        "kl": kl.mean().detach(),
        "p": p.detach(),
        "alpha": alpha.detach(),
    }


# ---------- inference helpers ---------------------------------------------

@torch.no_grad()
def predict_with_uncertainty(
    evidence: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Return predicted class, expected probability, and vacuity."""
    alpha = evidence_to_alpha(evidence)
    p = alpha_to_probs(alpha)
    u = vacuity(alpha)
    pred = p.argmax(dim=-1)
    return {"pred": pred, "p": p, "alpha": alpha, "u": u}


# ---------- smoke ---------------------------------------------------------

def _smoke_test() -> None:
    torch.manual_seed(0)
    B, K, d = 16, 4, 32
    head = EvidentialHead(d_in=d, num_classes=K)
    h = torch.randn(B, d)
    targets = torch.randint(0, K, (B,))

    e = head(h)
    assert e.shape == (B, K)
    assert (e >= 0).all(), "evidence must be non-negative"

    out = evidential_loss(e, targets, kl_weight=0.5)
    assert torch.isfinite(out["loss"]), "loss not finite"

    # quick optimisation step: loss should drop
    opt = torch.optim.Adam(head.parameters(), lr=1e-2)
    losses = []
    for _ in range(50):
        opt.zero_grad()
        e = head(h)
        out = evidential_loss(e, targets, kl_weight=0.5)
        out["loss"].backward()
        opt.step()
        losses.append(out["loss"].item())
    assert losses[-1] < losses[0], f"loss did not drop: {losses[0]:.3f} -> {losses[-1]:.3f}"

    info = predict_with_uncertainty(e)
    assert info["u"].shape == (B,)
    assert ((info["u"] >= 0) & (info["u"] <= 1)).all(), "vacuity out of [0,1]"

    # sanity: after fitting toy data, vacuity should be lower on average than at init
    print(f"[Evidential] loss {losses[0]:.3f} -> {losses[-1]:.3f}, "
          f"mean vacuity = {info['u'].mean():.3f}")
    print("[Evidential] smoke test passed.")


if __name__ == "__main__":
    _smoke_test()


# ---------- accuracy-first evidential training loss -----------------------

def evidential_ce_loss(
    evidence: torch.Tensor,        # (B, K)
    targets: torch.Tensor,         # (B,)
    class_weights: torch.Tensor | None = None,  # (K,)
    label_smoothing: float = 0.05,
    kl_weight: float = 0.0,
    eps: float = 1e-8,
) -> dict[str, torch.Tensor]:
    """Train the evidential head with CROSS-ENTROPY on the Dirichlet
    expected probability, instead of the Bayes-risk MSE. This optimizes
    accuracy directly (like a softmax model) while keeping the Dirichlet
    parameterization so uncertainty (vacuity u = K/S) is still available.

    This fixes the well-known accuracy gap of evidential Bayes-risk training.
    """
    B, K = evidence.shape
    alpha = evidence + 1.0
    S = alpha.sum(dim=-1, keepdim=True)
    p = alpha / S                                   # expected probability
    log_p = torch.log(p + eps)                      # (B, K)

    # cross-entropy with optional label smoothing and class weights
    y = F.one_hot(targets, num_classes=K).float()
    if label_smoothing > 0:
        y = y * (1 - label_smoothing) + label_smoothing / K
    ce = -(y * log_p).sum(dim=-1)                    # (B,)
    if class_weights is not None:
        w = class_weights[targets]
        ce = ce * w

    loss = ce.mean()
    if kl_weight > 0:
        # tiny KL toward uniform Dirichlet on wrong-class evidence, keeps u meaningful
        alpha_tilde = y + (1.0 - y) * alpha
        loss = loss + kl_weight * _kl_dirichlet_uniform(alpha_tilde).clamp(min=0).mean()

    return {"loss": loss, "p": p.detach(), "alpha": alpha.detach()}
