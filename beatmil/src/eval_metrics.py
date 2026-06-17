"""Evaluation: classification metrics, bootstrap CIs, McNemar test,
calibration (ECE, Brier, reliability), selective prediction (AUARC,
risk-coverage), and per-class breakdowns.
"""

from __future__ import annotations
import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, cohen_kappa_score,
    confusion_matrix, brier_score_loss,
)
from scipy.stats import chi2


# ---------- core metrics -------------------------------------------------

def classification_metrics(preds, targets, probs, K=4):
    """Return dict of all classification metrics."""
    out = {}
    out["accuracy"] = float(accuracy_score(targets, preds))
    out["f1_macro"] = float(f1_score(targets, preds, average="macro", zero_division=0))
    out["f1_weighted"] = float(f1_score(targets, preds, average="weighted", zero_division=0))
    out["kappa"] = float(cohen_kappa_score(targets, preds))
    # per-class F1
    f1_per = f1_score(targets, preds, average=None, labels=list(range(K)), zero_division=0)
    out["f1_per_class"] = [float(x) for x in f1_per]
    # AUROC macro
    try:
        # one-hot targets for multi-class AUROC
        onehot = np.eye(K)[targets]
        out["auroc_macro"] = float(roc_auc_score(onehot, probs, average="macro", multi_class="ovr"))
    except Exception:
        out["auroc_macro"] = None
    out["confusion_matrix"] = confusion_matrix(targets, preds, labels=list(range(K))).tolist()
    return out


# ---------- bootstrap CI -------------------------------------------------

def bootstrap_ci(preds, targets, probs, metric_fn, n_resamples=1000, seed=42, ci=0.95):
    """Compute bootstrap CI for a metric_fn(preds, targets, probs) -> float."""
    rng = np.random.default_rng(seed)
    N = len(preds)
    vals = []
    for _ in range(n_resamples):
        idx = rng.integers(0, N, N)
        try:
            v = metric_fn(preds[idx], targets[idx], probs[idx])
            if np.isfinite(v):
                vals.append(v)
        except Exception:
            continue
    vals = np.array(vals)
    lo = float(np.percentile(vals, (1 - ci) / 2 * 100))
    hi = float(np.percentile(vals, (1 + ci) / 2 * 100))
    return {"mean": float(vals.mean()), "ci_low": lo, "ci_high": hi, "n": len(vals)}


def f1_macro_metric(preds, targets, probs):
    return f1_score(targets, preds, average="macro", zero_division=0)


def accuracy_metric(preds, targets, probs):
    return accuracy_score(targets, preds)


# ---------- McNemar test -------------------------------------------------

def mcnemar(preds_a, preds_b, targets):
    """Exact McNemar test on paired predictions a, b vs ground truth.

    Returns p-value (two-sided) for the null that a and b have equal accuracy.
    """
    a_correct = preds_a == targets
    b_correct = preds_b == targets
    # b01: a wrong, b correct;  b10: a correct, b wrong
    b01 = int(((~a_correct) & b_correct).sum())
    b10 = int((a_correct & (~b_correct)).sum())
    n = b01 + b10
    if n == 0:
        return 1.0
    # continuity-corrected chi-square
    stat = (abs(b01 - b10) - 1) ** 2 / n
    p = 1 - chi2.cdf(stat, df=1)
    return float(p), b01, b10


def holm_bonferroni(pvalues, alpha=0.05):
    """Return per-test alpha thresholds and reject decisions (Holm-Bonferroni)."""
    m = len(pvalues)
    order = np.argsort(pvalues)
    decisions = [False] * m
    thresholds = [0.0] * m
    for k, idx in enumerate(order):
        thresh = alpha / (m - k)
        thresholds[idx] = thresh
        if pvalues[idx] < thresh:
            decisions[idx] = True
        else:
            break  # once a test fails to reject, stop
    return decisions, thresholds


# ---------- calibration --------------------------------------------------

def expected_calibration_error(probs, targets, n_bins=15):
    """ECE: weighted gap between confidence and accuracy across bins."""
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies = (predictions == targets).astype(float)
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    N = len(probs)
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (confidences > lo) & (confidences <= hi)
        if mask.any():
            avg_conf = confidences[mask].mean()
            avg_acc = accuracies[mask].mean()
            ece += mask.sum() / N * abs(avg_conf - avg_acc)
    return float(ece)


def reliability_diagram_data(probs, targets, n_bins=15):
    """Return (bin_centers, confidences, accuracies, counts) for plotting."""
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies = (predictions == targets).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    centers, confs, accs, counts = [], [], [], []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (confidences > lo) & (confidences <= hi)
        if mask.any():
            centers.append((lo + hi) / 2)
            confs.append(float(confidences[mask].mean()))
            accs.append(float(accuracies[mask].mean()))
            counts.append(int(mask.sum()))
    return np.array(centers), np.array(confs), np.array(accs), np.array(counts)


def brier_multiclass(probs, targets, K=4):
    """Multiclass Brier score = mean squared error vs one-hot."""
    onehot = np.eye(K)[targets]
    return float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))


# ---------- selective prediction ----------------------------------------

def risk_coverage_curve(preds, targets, uncertainty, n_points=100):
    """Sort by uncertainty ascending. Compute (coverage, risk) curve."""
    order = np.argsort(uncertainty)        # confident first
    preds_s = preds[order]
    targets_s = targets[order]
    N = len(preds)
    coverages, risks, f1s = [], [], []
    for k in np.linspace(int(N * 0.05), N, n_points).astype(int):
        keep = slice(0, k)
        coverage = k / N
        errors = (preds_s[keep] != targets_s[keep]).mean()
        f1 = f1_score(targets_s[keep], preds_s[keep], average="macro", zero_division=0)
        coverages.append(coverage)
        risks.append(float(errors))
        f1s.append(float(f1))
    return np.array(coverages), np.array(risks), np.array(f1s)


def auarc(coverages, risks):
    """Area under risk-coverage curve (lower is better)."""
    return float(np.trapz(risks, coverages))


def selective_metrics_at_coverage(preds, targets, uncertainty, coverages=(0.7, 0.8, 0.9, 0.95)):
    order = np.argsort(uncertainty)
    preds_s = preds[order]; targets_s = targets[order]
    N = len(preds)
    out = {}
    for c in coverages:
        k = int(N * c)
        out[c] = {
            "accuracy": float((preds_s[:k] == targets_s[:k]).mean()),
            "f1_macro": float(f1_score(targets_s[:k], preds_s[:k], average="macro", zero_division=0)),
        }
    return out
