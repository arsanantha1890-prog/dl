"""
Post-hoc threshold tuning and checkpoint continuation.

Two independent tools:

1. tune_and_eval(): loads saved predictions/probabilities, finds per-class
   probability thresholds that maximise macro-F1 on the VALIDATION set, then
   re-evaluates on the TEST set with those thresholds. No retraining needed.
   This alone can push S F1 from ~0.28 to ~0.50+ when AUROC is already high.

2. continue_training(): resumes from best.pt with a lower LR for more epochs.
   Use when the model plateaued too early (patience triggered before minority
   classes converged).

Usage:
    # Threshold tuning only (fast, ~30 seconds):
    python tune_thresholds.py --mode tune --run_dir runs/proposed \
        --data_dir /workspace/data/mitbih/mit-bih-arrhythmia-database-1.0.0

    # Continue training from checkpoint:
    python tune_thresholds.py --mode continue --run_dir runs/proposed \
        --data_dir /workspace/data/mitbih/mit-bih-arrhythmia-database-1.0.0 \
        --epochs 40 --lr 1e-4

    # Both (continue then tune):
    python tune_thresholds.py --mode both --run_dir runs/proposed \
        --data_dir /workspace/data/mitbih/mit-bih-arrhythmia-database-1.0.0 \
        --epochs 40 --lr 1e-4
"""

from __future__ import annotations

import os
import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import data_pipeline as dp
from metrics import compute_metrics, bootstrap_ci, format_metric_report, save_results
from train import ECGWindowDataset, collate, build_model, evaluate, focal_loss_with_alpha


# =====================================================================
# Threshold tuning
# =====================================================================
def find_best_thresholds(y_true: np.ndarray, y_prob: np.ndarray,
                          n_classes: int = 3,
                          n_steps: int = 100) -> np.ndarray:
    """
    Grid-search per-class probability thresholds to maximise macro-F1.

    Strategy: for each class c, find the threshold t such that predicting
    class c when prob[:,c] > t maximises that class's F1, holding other
    thresholds fixed. We iterate this a few times (coordinate descent).

    Returns thresholds: (n_classes,) array.
    Default argmax corresponds to thresholds all = 1/n_classes ≈ 0.333.
    """
    from sklearn.metrics import f1_score

    thresholds = np.full(n_classes, 1.0 / n_classes)

    def predict_with_thresholds(prob, thresh):
        # Subtract threshold from each class prob, then take argmax.
        # This shifts the decision boundary per class.
        adjusted = prob - thresh[np.newaxis, :]
        return adjusted.argmax(axis=1)

    for iteration in range(5):  # coordinate descent rounds
        for c in range(n_classes):
            best_t, best_f1 = thresholds[c], -1.0
            for t in np.linspace(0.0, 0.9, n_steps):
                thresholds[c] = t
                preds = predict_with_thresholds(y_prob, thresholds)
                mf1 = f1_score(y_true, preds, average="macro", zero_division=0)
                if mf1 > best_f1:
                    best_f1 = mf1
                    best_t = t
            thresholds[c] = best_t

    return thresholds


def tune_and_eval(args):
    """
    Load val probabilities (re-run inference on val set), tune thresholds,
    then evaluate on test set with tuned thresholds.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg = dp.PipelineConfig(data_dir=args.data_dir, window_size=args.window_size,
                            wavelet_baseline=True, seed=42)
    data = dp.build_and_cache(cfg, cache_dir=args.cache_dir)
    label_names = list(data["label_names"])

    # Load best checkpoint
    ckpt = torch.load(os.path.join(args.run_dir, "best.pt"),
                      map_location=device, weights_only=False)
    model_name = ckpt.get("model", "proposed")
    model, is_mt = build_model(model_name)
    model.load_state_dict(ckpt["state_dict"])
    model = model.to(device)
    model.eval()
    print(f"[tune] loaded {model_name} checkpoint (val_f1={ckpt.get('val_f1', '?'):.4f} "
          f"at epoch {ckpt.get('epoch', '?')})")

    # Get val probabilities
    ds_va = ECGWindowDataset(data["X_val"], data["y_val"])
    dl_va = DataLoader(ds_va, batch_size=256, shuffle=False,
                       collate_fn=collate, num_workers=0)
    y_val, _, prob_val = evaluate(model, is_mt, dl_va, device)

    # Get test probabilities
    ds_te = ECGWindowDataset(data["X_test"], data["y_test"])
    dl_te = DataLoader(ds_te, batch_size=256, shuffle=False,
                       collate_fn=collate, num_workers=0)
    y_test, _, prob_test = evaluate(model, is_mt, dl_te, device)

    from sklearn.metrics import f1_score

    # Baseline: argmax (what the original eval does)
    pred_argmax = prob_test.argmax(1)
    f1_argmax = f1_score(y_test, pred_argmax, average="macro", zero_division=0)
    f1_per_argmax = f1_score(y_test, pred_argmax, average=None,
                             labels=[0,1,2], zero_division=0)
    print(f"\n[baseline argmax]  macro-F1={f1_argmax:.4f}  "
          f"[N={f1_per_argmax[0]:.3f} S={f1_per_argmax[1]:.3f} V={f1_per_argmax[2]:.3f}]")

    # Tune on val set
    print("[tune] searching thresholds on validation set...")
    thresholds = find_best_thresholds(y_val, prob_val, n_classes=3, n_steps=200)
    print(f"[tune] optimal thresholds: N={thresholds[0]:.3f}  "
          f"S={thresholds[1]:.3f}  V={thresholds[2]:.3f}")

    # Evaluate on test with tuned thresholds
    adjusted = prob_test - thresholds[np.newaxis, :]
    pred_tuned = adjusted.argmax(1)
    f1_tuned = f1_score(y_test, pred_tuned, average="macro", zero_division=0)
    f1_per_tuned = f1_score(y_test, pred_tuned, average=None,
                            labels=[0,1,2], zero_division=0)
    print(f"[tuned thresholds] macro-F1={f1_tuned:.4f}  "
          f"[N={f1_per_tuned[0]:.3f} S={f1_per_tuned[1]:.3f} V={f1_per_tuned[2]:.3f}]")
    print(f"[gain] macro-F1 {f1_argmax:.4f} -> {f1_tuned:.4f} "
          f"({(f1_tuned-f1_argmax)*100:+.2f}pp)")

    # Full metrics with tuned thresholds
    mb = compute_metrics(y_test, pred_tuned, prob_test, label_names)
    cis = {m: bootstrap_ci(y_test, pred_tuned, m, n_boot=1000)
           for m in ("accuracy", "macro_f1", "weighted_f1", "cohen_kappa")}
    print("\n" + format_metric_report(mb, cis["macro_f1"]))

    # Save
    out = {**mb.to_dict(), "thresholds": thresholds.tolist(),
           "bootstrap_ci": cis}
    out_path = os.path.join(args.run_dir, "metrics_tuned.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    np.savez(os.path.join(args.run_dir, "predictions_tuned.npz"),
             y_true=y_test, y_pred=pred_tuned, y_prob=prob_test,
             thresholds=thresholds, label_names=np.asarray(label_names))
    print(f"[saved] {out_path}")

    return thresholds, mb


# =====================================================================
# Continue training from checkpoint
# =====================================================================
def continue_training(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = (device == "cuda")

    cfg = dp.PipelineConfig(data_dir=args.data_dir, window_size=args.window_size,
                            wavelet_baseline=True, seed=42)
    data = dp.build_and_cache(cfg, cache_dir=args.cache_dir)
    label_names = list(data["label_names"])

    ds_tr = ECGWindowDataset(data["X_train"], data["y_train"])
    ds_va = ECGWindowDataset(data["X_val"],   data["y_val"])
    ds_te = ECGWindowDataset(data["X_test"],  data["y_test"])

    dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True,
                       collate_fn=collate, num_workers=args.workers,
                       pin_memory=use_amp, drop_last=True)
    dl_va = DataLoader(ds_va, batch_size=args.batch_size, shuffle=False,
                       collate_fn=collate, num_workers=0)
    dl_te = DataLoader(ds_te, batch_size=args.batch_size, shuffle=False,
                       collate_fn=collate, num_workers=0)

    # Load checkpoint
    ckpt_path = os.path.join(args.run_dir, "best.pt")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model_name = ckpt.get("model", "proposed")
    model, is_mt = build_model(model_name)
    model.load_state_dict(ckpt["state_dict"])
    model = model.to(device)
    start_f1 = ckpt.get("val_f1", 0.0)
    print(f"[continue] resuming {model_name} from epoch {ckpt.get('epoch','?')} "
          f"(val_f1={start_f1:.4f})")
    print(f"[continue] fine-tuning for {args.epochs} more epochs at lr={args.lr}")

    # Low LR optimiser — fine-tuning regime
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    cls_w = dp.compute_class_weights(data["y_train"], mode="sqrt")
    alpha = torch.as_tensor(cls_w, dtype=torch.float32, device=device)

    from sklearn.metrics import f1_score
    best_f1 = start_f1
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    patience = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        import time; t0 = time.time()
        running = 0.0
        for x1d, x2d, y in dl_tr:
            x1d = x1d.to(device, non_blocking=use_amp)
            x2d = x2d.to(device) if x2d is not None else None
            y = y.to(device)
            opt.zero_grad(set_to_none=True)
            if use_amp:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss, _ = focal_loss_with_alpha.__wrapped__(model, is_mt, x1d, x2d, y, alpha) \
                        if hasattr(focal_loss_with_alpha, '__wrapped__') else \
                        _step(model, is_mt, x1d, x2d, y, alpha)
            else:
                loss, _ = _step(model, is_mt, x1d, x2d, y, alpha)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            running += loss.item()
        sched.step()

        yv, pv, _ = evaluate(model, is_mt, dl_va, device)
        val_f1 = f1_score(yv, pv, average="macro", zero_division=0)
        val_per = f1_score(yv, pv, average=None, labels=[0,1,2], zero_division=0)
        dt = time.time() - t0
        print(f"[ft epoch {epoch:3d}] loss={running/max(1,len(dl_tr)):.4f}  "
              f"val_macroF1={val_f1:.4f}  "
              f"[N={val_per[0]:.3f} S={val_per[1]:.3f} V={val_per[2]:.3f}]  "
              f"lr={opt.param_groups[0]['lr']:.2e}  ({dt:.1f}s)", flush=True)

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save({"state_dict": best_state, "val_f1": best_f1,
                        "epoch": f"ft_{epoch}", "model": model_name}, ckpt_path)
            patience = 0
        else:
            patience += 1
            if patience >= args.patience:
                print(f"[early stop] no improvement for {args.patience} ft epochs")
                break

    # Final eval
    model.load_state_dict(best_state)
    y_true, y_pred, y_prob = evaluate(model, is_mt, dl_te, device)
    mb = compute_metrics(y_true, y_pred, y_prob, label_names)
    cis = {m: bootstrap_ci(y_true, y_pred, m, n_boot=1000)
           for m in ("accuracy", "macro_f1", "weighted_f1", "cohen_kappa")}
    print("\n" + format_metric_report(mb, cis["macro_f1"]))
    save_results(os.path.join(args.run_dir, "metrics.json"), mb, cis)
    np.savez(os.path.join(args.run_dir, "predictions.npz"),
             y_true=y_true, y_pred=y_pred, y_prob=y_prob,
             label_names=np.asarray(label_names))
    print(f"[saved] updated metrics.json and predictions.npz")
    return mb


def _step(model, is_mt, x1d, x2d, y, alpha):
    """Reuse the forward+loss logic from train.py without circular import."""
    if is_mt:
        out = model(x1d, x2d)
        cls_logits = out["cls_logits"]
        L_cls = focal_loss_with_alpha(cls_logits, y, alpha)
        det_target = (y > 0).long()
        L_det = F.cross_entropy(out["det_logits"], det_target)
        log_var_cls = model.log_var_cls.clamp(-2.0, 1.0)
        log_var_det = model.log_var_det.clamp(-2.0, 1.0)
        w_cls = torch.exp(-log_var_cls)
        w_det = torch.exp(-log_var_det)
        loss = (w_cls * L_cls + 0.5 * log_var_cls +
                0.3 * w_det * L_det + 0.5 * log_var_det)
        return loss, cls_logits
    logits = model(x1d)
    return focal_loss_with_alpha(logits, y, alpha), logits


# =====================================================================
# CLI
# =====================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["tune", "continue", "both"], default="tune")
    ap.add_argument("--run_dir", default="runs/proposed")
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--cache_dir", default="./cache")
    ap.add_argument("--window_size", type=int, default=3600)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    if args.mode in ("continue", "both"):
        continue_training(args)
    if args.mode in ("tune", "both"):
        tune_and_eval(args)


if __name__ == "__main__":
    main()
