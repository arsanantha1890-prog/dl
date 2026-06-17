"""Pairwise McNemar tests: Beat-MIL vs each baseline, with Holm-Bonferroni
correction across the family of tests.

Run AFTER run_eval.py.
Run: python run_mcnemar.py
"""

from __future__ import annotations
import sys
import json
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from eval_metrics import mcnemar, holm_bonferroni


def main():
    OUT = Path.home() / "beatmil/outputs/eval"

    # Intra-DB comparison (the main table)
    beat_path = OUT / "preds_beatmil_intra-db.npz"
    if not beat_path.exists():
        print(f"[error] {beat_path} not found — run run_eval.py first")
        sys.exit(1)

    beat = np.load(beat_path)
    preds_b = beat["preds"]; targets = beat["targets"]

    results = {}
    baselines = ["resnet1d", "cnnlstm", "ecgformer"]
    pvals = []
    names_in_order = []
    for name in baselines:
        f = OUT / f"preds_{name}_intra-db.npz"
        if not f.exists():
            print(f"[skip] {f} not found")
            continue
        d = np.load(f)
        assert (d["targets"] == targets).all(), \
            f"target mismatch for {name} — was it trained on the same split?"
        p, b01, b10 = mcnemar(preds_b, d["preds"], targets)
        results[name] = {
            "p_value": p,
            "beatmil_correct_only": int(b10),
            "baseline_correct_only": int(b01),
        }
        pvals.append(p)
        names_in_order.append(name)

    # Holm-Bonferroni across the family
    decisions, thresholds = holm_bonferroni(pvals, alpha=0.05)
    for i, n in enumerate(names_in_order):
        results[n]["holm_threshold"] = thresholds[i]
        results[n]["reject_null_at_alpha_0.05"] = bool(decisions[i])

    (OUT / "mcnemar.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    print(f"\n[mcnemar] saved to {OUT / 'mcnemar.json'}")


if __name__ == "__main__":
    main()
