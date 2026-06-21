"""
Evaluation metrics for the resubmission.

This module exists to answer the two JCSSE reviewer points that are *not* about
data validity but about statistical rigor:

  * "No statistical significance testing" -> bootstrap confidence intervals on
    every headline metric, and McNemar's paired test between models so that
    "our model beats baseline X" becomes a claim with a p-value attached rather
    than a bare point estimate.

Everything here operates on a fixed array of test predictions/probabilities so
the numbers are reproducible from a saved checkpoint.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, cohen_kappa_score,
    roc_auc_score, confusion_matrix, recall_score,
)
from scipy.stats import binomtest, chi2


# =====================================================================
# Point-estimate metric bundle
# =====================================================================
@dataclass
class MetricBundle:
    accuracy: float
    macro_f1: float
    weighted_f1: float
    cohen_kappa: float
    macro_auroc: Optional[float]
    sensitivity_macro: float          # macro-averaged recall
    per_class_f1: dict
    per_class_auc: dict
    confusion: list                   # row = true, col = pred

    def to_dict(self):
        return asdict(self)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    y_prob: Optional[np.ndarray] = None,
                    label_names: Optional[list] = None) -> MetricBundle:
    """y_true,y_pred: (N,) int. y_prob: (N,C) softmax probs (optional, for AUC)."""
    n_classes = int(max(y_true.max(), y_pred.max())) + 1
    if label_names is None:
        label_names = [str(i) for i in range(n_classes)]

    per_f1 = f1_score(y_true, y_pred, average=None,
                      labels=list(range(n_classes)), zero_division=0)
    per_class_f1 = {label_names[i]: float(per_f1[i]) for i in range(n_classes)}

    per_class_auc, macro_auroc = {}, None
    if y_prob is not None:
        try:
            # one-vs-rest AUC per class; guard classes with a single label present
            aucs = []
            for c in range(n_classes):
                yc = (y_true == c).astype(int)
                if yc.min() == yc.max():
                    per_class_auc[label_names[c]] = None
                    continue
                a = roc_auc_score(yc, y_prob[:, c])
                per_class_auc[label_names[c]] = float(a)
                aucs.append(a)
            macro_auroc = float(np.mean(aucs)) if aucs else None
        except Exception:
            per_class_auc, macro_auroc = {}, None

    return MetricBundle(
        accuracy=float(accuracy_score(y_true, y_pred)),
        macro_f1=float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        weighted_f1=float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        cohen_kappa=float(cohen_kappa_score(y_true, y_pred)),
        macro_auroc=macro_auroc,
        sensitivity_macro=float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        per_class_f1=per_class_f1,
        per_class_auc=per_class_auc,
        confusion=confusion_matrix(y_true, y_pred,
                                   labels=list(range(n_classes))).tolist(),
    )


# =====================================================================
# Bootstrap confidence intervals
# =====================================================================
def bootstrap_ci(y_true: np.ndarray, y_pred: np.ndarray,
                 metric: str = "macro_f1", n_boot: int = 1000,
                 alpha: float = 0.05, seed: int = 42) -> dict:
    """Percentile bootstrap CI for a metric, resampling test instances with
    replacement. Returns {point, lower, upper, se}.

    Resampling is over the test set as collected (the test set's own class
    proportions are preserved in expectation), which is the standard reporting
    convention for a fixed evaluation set.
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)

    def _score(yt, yp):
        if metric == "accuracy":
            return accuracy_score(yt, yp)
        if metric == "macro_f1":
            return f1_score(yt, yp, average="macro", zero_division=0)
        if metric == "weighted_f1":
            return f1_score(yt, yp, average="weighted", zero_division=0)
        if metric == "cohen_kappa":
            return cohen_kappa_score(yt, yp)
        raise ValueError(f"unknown metric {metric}")

    point = _score(y_true, y_pred)
    stats = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        stats[b] = _score(y_true[idx], y_pred[idx])
    lo = float(np.percentile(stats, 100 * alpha / 2))
    hi = float(np.percentile(stats, 100 * (1 - alpha / 2)))
    return {"metric": metric, "point": float(point),
            "lower": lo, "upper": hi, "se": float(stats.std()),
            "n_boot": n_boot, "ci_level": 1 - alpha}


# =====================================================================
# McNemar's paired significance test between two models
# =====================================================================
def mcnemar_test(y_true: np.ndarray, pred_a: np.ndarray,
                 pred_b: np.ndarray, name_a: str = "A",
                 name_b: str = "B") -> dict:
    """Paired test on the SAME test set. Compares the discordant pairs:
        n_ab = A correct, B wrong ; n_ba = A wrong, B correct.
    Uses the exact binomial test (correct for any sample size); also reports the
    continuity-corrected chi-square for reference.
    """
    a_correct = (pred_a == y_true)
    b_correct = (pred_b == y_true)
    n_ab = int(np.sum(a_correct & ~b_correct))   # A right, B wrong
    n_ba = int(np.sum(~a_correct & b_correct))   # A wrong, B right
    n_disc = n_ab + n_ba

    if n_disc == 0:
        return {"model_a": name_a, "model_b": name_b,
                "n_a_right_b_wrong": n_ab, "n_a_wrong_b_right": n_ba,
                "p_value": 1.0, "p_value_chi2_cc": 1.0, "chi2_cc": 0.0,
                "better_model": "tie",
                "note": "no discordant pairs; models agree on every test sample"}

    # Exact two-sided binomial under H0: p(discordant favors A) = 0.5
    p_exact = binomtest(n_ab, n_disc, 0.5, alternative="two-sided").pvalue
    # Continuity-corrected chi-square (for reference / large-N reporting)
    chi2_cc = (abs(n_ab - n_ba) - 1) ** 2 / n_disc if n_disc > 0 else 0.0
    p_chi2 = float(chi2.sf(chi2_cc, df=1))

    better = name_a if n_ab > n_ba else (name_b if n_ba > n_ab else "tie")
    return {"model_a": name_a, "model_b": name_b,
            "n_a_right_b_wrong": n_ab, "n_a_wrong_b_right": n_ba,
            "p_value": float(p_exact), "p_value_chi2_cc": p_chi2,
            "chi2_cc": float(chi2_cc), "better_model": better}


# =====================================================================
# Pretty reporting
# =====================================================================
def format_metric_report(mb: MetricBundle, ci: Optional[dict] = None) -> str:
    lines = ["=" * 52, "TEST-SET METRICS", "=" * 52]
    lines.append(f"  Accuracy        {mb.accuracy:.4f}")
    if ci is not None:
        lines.append(f"  Macro-F1        {mb.macro_f1:.4f}  "
                     f"[95% CI {ci['lower']:.4f}, {ci['upper']:.4f}]")
    else:
        lines.append(f"  Macro-F1        {mb.macro_f1:.4f}")
    lines.append(f"  Weighted-F1     {mb.weighted_f1:.4f}")
    lines.append(f"  Cohen's kappa   {mb.cohen_kappa:.4f}")
    if mb.macro_auroc is not None:
        lines.append(f"  Macro-AUROC     {mb.macro_auroc:.4f}")
    lines.append(f"  Sensitivity     {mb.sensitivity_macro:.4f}")
    lines.append("  Per-class F1:")
    for k, v in mb.per_class_f1.items():
        auc = mb.per_class_auc.get(k)
        auc_s = f"  AUC {auc:.4f}" if isinstance(auc, float) else ""
        lines.append(f"      {k:>4}  F1 {v:.4f}{auc_s}")
    lines.append("  Confusion (row=true, col=pred):")
    for row in mb.confusion:
        lines.append("      " + "  ".join(f"{x:5d}" for x in row))
    lines.append("=" * 52)
    return "\n".join(lines)


def save_results(path: str, mb: MetricBundle, cis: Optional[dict] = None,
                 mcnemar: Optional[list] = None):
    payload = {"metrics": mb.to_dict()}
    if cis:
        payload["bootstrap_ci"] = cis
    if mcnemar:
        payload["mcnemar"] = mcnemar
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return path


if __name__ == "__main__":
    # Self-test on synthetic predictions with a known imbalance (N/S/V).
    rng = np.random.default_rng(0)
    N = 4000
    y_true = rng.choice([0, 1, 2], size=N, p=[0.85, 0.05, 0.10])

    # Model A: strong. Model B: weaker (more errors on S).
    def make_pred(y, err_rate, s_extra):
        yp = y.copy()
        flip = rng.random(len(y)) < err_rate
        yp[flip] = rng.choice([0, 1, 2], size=flip.sum())
        s_mask = (y == 1) & (rng.random(len(y)) < s_extra)
        yp[s_mask] = 0
        return yp

    pred_a = make_pred(y_true, 0.02, 0.10)
    pred_b = make_pred(y_true, 0.05, 0.30)

    # fake probabilities consistent-ish with preds
    def fake_prob(yp):
        p = np.full((len(yp), 3), 0.05)
        p[np.arange(len(yp)), yp] = 0.90
        p = p + rng.random(p.shape) * 0.05
        return p / p.sum(1, keepdims=True)

    mb = compute_metrics(y_true, pred_a, fake_prob(pred_a), ["N", "S", "V"])
    ci = bootstrap_ci(y_true, pred_a, "macro_f1", n_boot=500)
    print(format_metric_report(mb, ci))
    mc = mcnemar_test(y_true, pred_a, pred_b, "Proposed", "Baseline")
    print("\nMcNemar Proposed vs Baseline:")
    print(f"  A-right/B-wrong={mc['n_a_right_b_wrong']}, "
          f"A-wrong/B-right={mc['n_a_wrong_b_right']}")
    print(f"  better={mc['better_model']}  p={mc['p_value']:.2e}")
    assert mc["better_model"] == "Proposed"
    assert mc["p_value"] < 0.05
    print("\n[METRICS SELF-TEST PASSED]")
