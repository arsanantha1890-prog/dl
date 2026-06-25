"""
run_baselines_and_mcnemar.py

Runs on a fresh GPU instance. Does three things:
  1. Trains cnn1d and cnnlstm baselines on the identical DS1 split used for
     the proposed model (cache_v4 is reused if present, else rebuilt).
  2. Tunes thresholds for both baselines on the val set.
  3. Runs McNemar's exact paired test: proposed vs cnn1d, proposed vs cnnlstm.

Prerequisites on the instance:
  - /workspace/beatmil/ contains: data_pipeline.py, metrics.py, baselines.py,
    proposed_model.py, train_v4.py
  - /workspace/beatmil/runs/proposed_v4/predictions_tuned.npz  (downloaded earlier)
  - MIT-BIH data at $DATA_DIR

Usage:
  cd /workspace/beatmil
  export DATA_DIR=/workspace/data/mitbih/mit-bih-arrhythmia-database-1.0.0
  python run_baselines_and_mcnemar.py --data_dir $DATA_DIR

Outputs (all in /workspace/beatmil/runs/):
  cnn1d_v4/metrics_tuned.json
  cnnlstm_v4/metrics_tuned.json
  proposed_v4/mcnemar.json        ← the statistical significance numbers
  final_results_summary.txt       ← everything you need for Table III
"""

from __future__ import annotations
import os, json, time, argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, confusion_matrix

import data_pipeline as dp
from metrics import (compute_metrics, bootstrap_ci,
                     format_metric_report, save_results, mcnemar_test)
from train_v4 import (FusionModelV4, build_all_splits,
                      evaluate, ECGDataset, collate, focal_loss)


# =====================================================================
# Threshold tuning (reused from earlier)
# =====================================================================
def tune_thresholds(y_true, y_prob, n_steps=200):
    thresholds = np.array([1/3, 1/3, 1/3])
    for _ in range(5):
        for c in range(3):
            best_f1, best_t = -1.0, thresholds[c]
            for t in np.linspace(0.0, 0.9, n_steps):
                tmp = thresholds.copy(); tmp[c] = t
                pred = (y_prob - tmp[np.newaxis, :]).argmax(1)
                mf1 = f1_score(y_true, pred, average="macro", zero_division=0)
                if mf1 > best_f1:
                    best_f1 = mf1; best_t = t
            thresholds[c] = best_t
    return thresholds


# =====================================================================
# Train one baseline model
# =====================================================================
def train_baseline(model_name, data, alpha, device, use_amp, args):
    os.makedirs(f"runs/{model_name}_v4", exist_ok=True)

    dl_tr = DataLoader(
        ECGDataset(data["X_train"], data["y_train"],
                   data["rr_train"], data["qrs_train"]),
        batch_size=args.batch_size, shuffle=True, collate_fn=collate,
        num_workers=4, pin_memory=use_amp, drop_last=True)
    dl_va = DataLoader(
        ECGDataset(data["X_val"], data["y_val"],
                   data["rr_val"], data["qrs_val"]),
        batch_size=args.batch_size, shuffle=False,
        collate_fn=collate, num_workers=4)
    dl_te = DataLoader(
        ECGDataset(data["X_test"], data["y_test"],
                   data["rr_test"], data["qrs_test"]),
        batch_size=args.batch_size, shuffle=False,
        collate_fn=collate, num_workers=4)

    model = FusionModelV4(model_name).to(device)
    n_p = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"\n[{model_name}] {n_p:.2f}M params")

    opt    = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    warmup = torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.1, total_iters=5)
    cosine = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=25, T_mult=2)
    sched  = torch.optim.lr_scheduler.SequentialLR(
        opt, [warmup, cosine], milestones=[5])

    best_f1, best_state, patience = -1.0, None, 0
    ckpt = f"runs/{model_name}_v4/best.pt"

    for epoch in range(1, args.epochs + 1):
        model.train(); t0 = time.time(); running = 0.0
        for x, rr, qrs, y in dl_tr:
            x, rr, qrs, y = (x.to(device), rr.to(device),
                              qrs.to(device), y.to(device))
            opt.zero_grad(set_to_none=True)
            if use_amp:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = focal_loss(model(x, rr, qrs), y, alpha)
            else:
                loss = focal_loss(model(x, rr, qrs), y, alpha)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); running += loss.item()
        sched.step()

        yv, pv, _ = evaluate(model, dl_va, device)
        val_f1  = f1_score(yv, pv, average="macro", zero_division=0)
        val_per = f1_score(yv, pv, average=None, labels=[0,1,2], zero_division=0)
        dt = time.time() - t0
        print(f"  [epoch {epoch:3d}] loss={running/max(1,len(dl_tr)):.4f}  "
              f"val_macroF1={val_f1:.4f}  "
              f"[N={val_per[0]:.3f} S={val_per[1]:.3f} V={val_per[2]:.3f}]  "
              f"({dt:.1f}s)", flush=True)

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            torch.save({"state_dict": best_state, "val_f1": best_f1,
                        "epoch": epoch, "model": model_name}, ckpt)
            patience = 0
        else:
            patience += 1
            if patience >= args.patience:
                print(f"  [early stop] {args.patience} epochs no improvement")
                break

    # Evaluate on test
    model.load_state_dict(best_state)
    y_true, y_pred_argmax, y_prob = evaluate(model, dl_te, device)

    # Argmax metrics
    mb_argmax = compute_metrics(y_true, y_pred_argmax, y_prob, ["N","S","V"])
    f1_argmax = f1_score(y_true, y_pred_argmax, average="macro", zero_division=0)
    f1_per    = f1_score(y_true, y_pred_argmax, average=None,
                         labels=[0,1,2], zero_division=0)
    print(f"\n[{model_name}] argmax: macro={f1_argmax:.4f}  "
          f"N={f1_per[0]:.3f} S={f1_per[1]:.3f} V={f1_per[2]:.3f}")

    # Val set for threshold tuning
    yv, _, prob_val = evaluate(model, dl_va, device)
    thresholds = tune_thresholds(yv, prob_val)
    print(f"[{model_name}] thresholds: N={thresholds[0]:.3f} "
          f"S={thresholds[1]:.3f} V={thresholds[2]:.3f}")

    y_pred_tuned = (y_prob - thresholds[np.newaxis, :]).argmax(1)
    f1_tuned = f1_score(y_true, y_pred_tuned, average="macro", zero_division=0)
    f1_tper  = f1_score(y_true, y_pred_tuned, average=None,
                        labels=[0,1,2], zero_division=0)
    print(f"[{model_name}] tuned:  macro={f1_tuned:.4f}  "
          f"N={f1_tper[0]:.3f} S={f1_tper[1]:.3f} V={f1_tper[2]:.3f}")

    mb_tuned = compute_metrics(y_true, y_pred_tuned, y_prob, ["N","S","V"])
    cis = {m: bootstrap_ci(y_true, y_pred_tuned, m, n_boot=1000)
           for m in ("accuracy","macro_f1","weighted_f1","cohen_kappa")}
    print(format_metric_report(mb_tuned, cis["macro_f1"]))

    save_results(f"runs/{model_name}_v4/metrics_tuned.json", mb_tuned, cis)
    np.savez(f"runs/{model_name}_v4/predictions_tuned.npz",
             y_true=y_true, y_pred=y_pred_tuned, y_prob=y_prob,
             thresholds=thresholds, label_names=np.array(["N","S","V"]))

    return mb_tuned, f1_tuned, f1_tper


# =====================================================================
# McNemar comparison
# =====================================================================
def run_mcnemar(results_by_model):
    """
    results_by_model: dict of model_name -> predictions_tuned.npz path
    First key is the anchor (proposed model).
    """
    preds = {}
    ref_y = None
    for name, path in results_by_model.items():
        d = np.load(path)
        preds[name] = d["y_pred"]
        if ref_y is None:
            ref_y = d["y_true"]
        else:
            assert np.array_equal(ref_y, d["y_true"]), \
                f"{name} has different y_true — cannot pair!"

    anchor = list(results_by_model.keys())[0]
    print("\n" + "="*60)
    print("McNEMAR PAIRED TEST (same DS2 test set, tuned predictions)")
    print("="*60)

    mcnemar_results = []
    for other in list(results_by_model.keys())[1:]:
        r = mcnemar_test(ref_y, preds[anchor], preds[other], anchor, other)
        mcnemar_results.append(r)
        sig = "SIGNIFICANT" if r["p_value"] < 0.05 else "n.s."
        print(f"  {anchor} vs {other}:")
        print(f"    better={r['better_model']}  p={r['p_value']:.3e} ({sig})")
        print(f"    proposed-right/baseline-wrong={r['n_a_right_b_wrong']}")
        print(f"    proposed-wrong/baseline-right={r['n_a_wrong_b_right']}")

    with open("runs/proposed_v4/mcnemar.json", "w") as f:
        json.dump(mcnemar_results, f, indent=2)
    print(f"\n[saved] runs/proposed_v4/mcnemar.json")
    return mcnemar_results


# =====================================================================
# Summary report for paper
# =====================================================================
def write_summary(all_results, mcnemar_results):
    lines = [
        "=" * 65,
        "FINAL RESULTS SUMMARY — paste into Table II and Table III",
        "=" * 65,
        "",
    ]
    for name, (mb, f1, f1per) in all_results.items():
        ci = mb.to_dict()
        lines += [
            f"Model: {name}",
            f"  Accuracy     : {mb.accuracy:.4f}",
            f"  Macro-F1     : {mb.macro_f1:.4f}",
            f"  Weighted-F1  : {mb.weighted_f1:.4f}",
            f"  Cohen kappa  : {mb.cohen_kappa:.4f}",
            f"  Macro-AUROC  : {mb.macro_auroc:.4f}" if mb.macro_auroc else "",
            f"  N F1         : {mb.per_class_f1.get('N', 0):.4f}",
            f"  S F1         : {mb.per_class_f1.get('S', 0):.4f}",
            f"  V F1         : {mb.per_class_f1.get('V', 0):.4f}",
            f"  Confusion    : {mb.confusion}",
            "",
        ]

    lines += ["", "McNEMAR RESULTS:", ""]
    for r in mcnemar_results:
        sig = "p < 0.05 SIGNIFICANT" if r["p_value"] < 0.05 else "n.s."
        lines.append(f"  {r['model_a']} vs {r['model_b']}: "
                     f"p={r['p_value']:.3e} ({sig})  "
                     f"better={r['better_model']}")

    report = "\n".join(lines)
    print("\n" + report)
    with open("runs/final_results_summary.txt", "w") as f:
        f.write(report)
    print("\n[saved] runs/final_results_summary.txt")


# =====================================================================
# Main
# =====================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--cache_dir",  default="./cache_v4")
    ap.add_argument("--epochs",     type=int,   default=100)
    ap.add_argument("--batch_size", type=int,   default=256)
    ap.add_argument("--patience",   type=int,   default=25)
    ap.add_argument("--window_size",type=int,   default=360)
    ap.add_argument("--seed",       type=int,   default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device  = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = (device == "cuda")
    print(f"[device] {device}")

    # Build / load cache
    cfg = dp.PipelineConfig(data_dir=args.data_dir,
                            window_size=args.window_size,
                            val_fraction=0.25, seed=args.seed)
    data    = build_all_splits(cfg, args.data_dir, args.cache_dir)
    y_train = data["y_train"]
    counts  = np.bincount(y_train, minlength=3)
    print(f"[data] train N={counts[0]} S={counts[1]} V={counts[2]}")

    raw_w    = counts.sum() / (3.0 * counts.astype(float))
    cls_w    = np.sqrt(raw_w); cls_w /= cls_w.mean()
    cls_w[1] = min(cls_w[1], 6.0 * cls_w[0])
    cls_w    = cls_w.astype(np.float32)
    alpha    = torch.tensor(cls_w, device=device)
    print(f"[balance] N={cls_w[0]:.3f} S={cls_w[1]:.3f} V={cls_w[2]:.3f}")

    os.makedirs("runs/proposed_v4", exist_ok=True)

    all_results = {}

    # Train baselines
    for model_name in ["cnn1d", "cnnlstm"]:
        mb, f1, f1per = train_baseline(
            model_name, data, alpha, device, use_amp, args)
        all_results[model_name] = (mb, f1, f1per)

    # Load proposed model results (predictions_tuned.npz must be present)
    proposed_path = "runs/proposed_v4/predictions_tuned.npz"
    assert os.path.exists(proposed_path), \
        f"Missing {proposed_path} — upload predictions_tuned.npz from your local machine first."
    d = np.load(proposed_path)
    # Reconstruct a minimal MetricBundle for the summary
    from metrics import MetricBundle
    import sklearn.metrics as skm
    yt = d["y_true"]; yp = d["y_pred"]; probs = d["y_prob"]
    per_f1 = skm.f1_score(yt,yp,average=None,labels=[0,1,2],zero_division=0)
    proposed_mb = compute_metrics(yt, yp, probs, ["N","S","V"])
    all_results = {"proposed": (proposed_mb, proposed_mb.macro_f1,
                                [proposed_mb.per_class_f1[c] for c in ["N","S","V"]]
                                ), **all_results}

    # McNemar
    pred_paths = {
        "proposed": "runs/proposed_v4/predictions_tuned.npz",
        "cnn1d":    "runs/cnn1d_v4/predictions_tuned.npz",
        "cnnlstm":  "runs/cnnlstm_v4/predictions_tuned.npz",
    }
    mcnemar_results = run_mcnemar(pred_paths)

    # Final summary
    write_summary(all_results, mcnemar_results)


if __name__ == "__main__":
    main()
