"""Generate all paper figures from saved evaluation outputs (.npz files).

Produces:
    fig2_confusion_matrix.png
    fig3_reliability_diagram.png
    fig4_risk_coverage.png
    fig5_xai_iou_bars.png

The architecture figure (Fig 1) is drawn separately in TikZ/Inkscape.
"""

from __future__ import annotations
from pathlib import Path
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# IEEE-friendly style
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 300,
    "savefig.bbox": "tight",
})


def fig2_confusion(cm: np.ndarray, class_names, out_path: Path):
    fig, ax = plt.subplots(figsize=(3.4, 3.0))
    cm_norm = cm / cm.sum(axis=1, keepdims=True)
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    K = len(class_names)
    ax.set_xticks(range(K)); ax.set_xticklabels(class_names)
    ax.set_yticks(range(K)); ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    for i in range(K):
        for j in range(K):
            ax.text(j, i, f"{cm_norm[i,j]:.2f}", ha="center", va="center",
                    color="white" if cm_norm[i, j] > 0.5 else "black", fontsize=8)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(out_path)
    plt.close(fig)


def fig3_reliability(centers, confs, accs, counts, ece: float, out_path: Path):
    fig, ax = plt.subplots(figsize=(3.4, 3.0))
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="Perfect calibration")
    ax.bar(centers, accs, width=0.06, alpha=0.7, edgecolor="black",
           color="#4477AA", label="Beat-MIL")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("Confidence"); ax.set_ylabel("Accuracy")
    ax.set_title(f"Reliability diagram (ECE = {ece:.3f})")
    ax.legend(loc="upper left")
    fig.savefig(out_path)
    plt.close(fig)


def fig4_risk_coverage(curves: dict, out_path: Path):
    """curves[model_name] = (coverages, risks, f1s)."""
    fig, ax = plt.subplots(figsize=(3.4, 3.0))
    for name, (cov, risk, _) in curves.items():
        ax.plot(cov, risk, label=name, linewidth=1.5)
    ax.set_xlabel("Coverage"); ax.set_ylabel("Selective risk (error rate)")
    ax.set_title("Risk–coverage curves")
    ax.legend(loc="upper left")
    fig.savefig(out_path)
    plt.close(fig)


def fig5_xai_bars(xai_summary: dict, out_path: Path):
    """xai_summary[method][region] = {iou_mean, iou_std, dice_mean, dice_std}."""
    regions = ["p", "qrs", "t"]
    methods = list(xai_summary.keys())
    x = np.arange(len(regions))
    w = 0.8 / len(methods)
    fig, ax = plt.subplots(figsize=(3.6, 3.0))
    for i, m in enumerate(methods):
        means = [xai_summary[m][r]["iou_mean"] for r in regions]
        stds  = [xai_summary[m][r]["iou_std"]  for r in regions]
        ax.bar(x + i * w - 0.4 + w / 2, means, w, yerr=stds, capsize=2, label=m)
    ax.set_xticks(x); ax.set_xticklabels(["P-wave", "QRS", "T-wave"])
    ax.set_ylabel("IoU")
    ax.set_title("Saliency vs. LUDB expert annotations")
    ax.legend(loc="upper left")
    fig.savefig(out_path)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", required=True, help="contains eval outputs")
    ap.add_argument("--out-dir", default="figures")
    args = ap.parse_args()
    rd = Path(args.results_dir); od = Path(args.out_dir); od.mkdir(exist_ok=True)

    # Fig 2: confusion matrix
    cm = np.array(json.loads((rd / "intra_db_beatmil.json").read_text())["confusion_matrix"])
    fig2_confusion(cm, ["N", "S", "V+F", "Q"], od / "fig2_confusion.png")

    # Fig 3: reliability diagram
    rel = np.load(rd / "reliability.npz")
    ece = float(json.loads((rd / "calibration.json").read_text())["ece"])
    fig3_reliability(rel["centers"], rel["confs"], rel["accs"], rel["counts"],
                     ece, od / "fig3_reliability.png")

    # Fig 4: risk-coverage
    curves = {}
    for f in rd.glob("riskcov_*.npz"):
        name = f.stem.replace("riskcov_", "")
        data = np.load(f)
        curves[name] = (data["cov"], data["risk"], data["f1"])
    if curves:
        fig4_risk_coverage(curves, od / "fig4_risk_coverage.png")

    # Fig 5: XAI bars
    xai_path = rd / "xai_summary.json"
    if xai_path.exists():
        xai = json.loads(xai_path.read_text())
        fig5_xai_bars(xai, od / "fig5_xai_iou.png")

    print(f"figures written to {od}")


if __name__ == "__main__":
    main()
