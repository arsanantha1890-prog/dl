"""Evaluate every trained checkpoint on its test split.

For each `checkpoints/.../best.pt`:
    1. Reconstruct the test dataset from the saved split manifest.
    2. Run model forward, collect predictions + probabilities + vacuity.
    3. Compute all metrics + bootstrap 95% CIs.
    4. Save per-checkpoint JSON to outputs/eval/.
    5. Save raw predictions (npz) for downstream McNemar tests.

Run: python run_eval.py
"""

from __future__ import annotations
import sys
import json
import pickle
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
from unified_dataset import UnifiedECGDataset
from mitbih_loader import MITBIHLoader
from cpsc import CPSC2018Loader
from ptbxl import PTBXLLoader
from beatmil import BeatMIL
from baselines import build_baseline
from train import collate, evaluate
from eval_metrics import (
    classification_metrics, bootstrap_ci, f1_macro_metric,
    expected_calibration_error, brier_multiclass,
    reliability_diagram_data, risk_coverage_curve, auarc,
    selective_metrics_at_coverage,
)


def main():
    CKPT_ROOT = Path.home() / "beatmil/checkpoints"
    DATA_ROOT = Path.home() / "beatmil/data"
    CACHE_DIR = Path.home() / "beatmil/outputs/cache"
    OUT_ROOT  = Path.home() / "beatmil/outputs/eval"
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    # Reload cached specs from all three databases
    specs_all = []
    for name in ["mitbih", "cpsc", "ptbxl"]:
        cache = CACHE_DIR / f"{name}_specs.pkl"
        if not cache.exists():
            print(f"[error] {cache} not found — run cache_specs.py first")
            sys.exit(1)
        with cache.open("rb") as f:
            specs_all.extend(pickle.load(f))
    print(f"[eval] {len(specs_all):,} total cached specs")

    db_loaders = {
        "mitbih": MITBIHLoader(DATA_ROOT / "mitbih"),
        "cpsc":   CPSC2018Loader(DATA_ROOT / "cpsc2018"),
        "ptbxl":  PTBXLLoader(DATA_ROOT / "ptbxl"),
    }
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[eval] device = {device}")

    for ckpt_path in sorted(CKPT_ROOT.rglob("best.pt")):
        ckpt_dir = ckpt_path.parent
        # use a unique name derived from the relative path
        rel = ckpt_dir.relative_to(CKPT_ROOT)
        name = "_".join(rel.parts) if rel.parts else "default"
        print(f"\n[eval] === {name} ===")

        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        args = ckpt["args"]
        model_name = args["model"]
        is_beatmil = (model_name == "beatmil")

        # rebuild model and load weights
        if is_beatmil:
            model = BeatMIL(num_classes=4)
        else:
            model = build_baseline(model_name, num_classes=4)
        model = model.to(device).eval()
        model.load_state_dict(ckpt["model"])

        # reconstruct test set from split manifest
        split = json.loads((ckpt_dir / "split.json").read_text())
        test_keys = {(s["database"], s["record_id"], s["window_start"])
                     for s in split["test"]}
        test_specs = [s for s in specs_all
                      if (s.database, s.record_id, s.window_start) in test_keys]
        if not test_specs:
            print(f"[warn] {name}: zero test samples reconstructed — skipping")
            continue
        test_ds = UnifiedECGDataset(test_specs, db_loaders, augment=False)
        test_dl = DataLoader(test_ds, batch_size=128, shuffle=False,
                             num_workers=4, collate_fn=collate, pin_memory=True)

        ev = evaluate(model, test_dl, device, is_beatmil)
        preds = ev["preds"]; targets = ev["targets"]
        probs = ev["probs"]; vac = ev["vacuity"]

        metrics = classification_metrics(preds, targets, probs)
        metrics["f1_macro_ci"] = bootstrap_ci(preds, targets, probs, f1_macro_metric)
        metrics["ece"] = expected_calibration_error(probs, targets)
        metrics["brier"] = brier_multiclass(probs, targets)
        metrics["n_test"] = int(len(preds))

        if is_beatmil:
            cov, risk, f1c = risk_coverage_curve(preds, targets, vac)
            metrics["auarc"] = auarc(cov, risk)
            metrics["selective"] = {str(k): v for k, v in
                selective_metrics_at_coverage(preds, targets, vac).items()}
            np.savez(OUT_ROOT / f"riskcov_{name}.npz", cov=cov, risk=risk, f1=f1c)
            # save reliability data only for the intra-db beatmil
            if "intra-db" in name:
                c, cf, ac, ct = reliability_diagram_data(probs, targets)
                np.savez(OUT_ROOT / "reliability.npz",
                         centers=c, confs=cf, accs=ac, counts=ct)
                (OUT_ROOT / "calibration.json").write_text(json.dumps(
                    {"ece": metrics["ece"], "brier": metrics["brier"]}, indent=2))

        (OUT_ROOT / f"{name}.json").write_text(json.dumps(metrics, indent=2))
        np.savez(OUT_ROOT / f"preds_{name}.npz",
                 preds=preds, targets=targets, probs=probs, vacuity=vac)
        print(f"[eval] {name}: macro F1 = {metrics['f1_macro']:.4f} "
              f"(CI [{metrics['f1_macro_ci']['ci_low']:.3f}, "
              f"{metrics['f1_macro_ci']['ci_high']:.3f}]) "
              f"acc = {metrics['accuracy']:.4f} "
              f"ECE = {metrics['ece']:.4f}")

    print(f"\n[eval] all outputs in {OUT_ROOT}")


if __name__ == "__main__":
    main()
