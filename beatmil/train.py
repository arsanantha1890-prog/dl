"""
Training + evaluation harness for the BMEiCON 2026 resubmission.

One loop trains either the proposed model or any baseline on the identical
DS1/DS2 split, then evaluates on DS2 and saves:
  * best checkpoint (by val macro-F1, the early-stopping criterion),
  * a metrics JSON with bootstrap 95% CIs,
  * a predictions .npz (y_true, y_pred, y_prob) so McNemar tests can be run
    ACROSS models afterwards with compare_models().

Run (survives SSH disconnect):
    nohup python train.py --model proposed --data_dir /data/mitbih \
        --out runs/proposed --epochs 60 > runs/proposed.log 2>&1 &

    nohup python train.py --model cnn1d --data_dir /data/mitbih \
        --out runs/cnn1d --epochs 60 > runs/cnn1d.log 2>&1 &

Then compare:
    python train.py --compare runs/proposed runs/cnn1d runs/cnnlstm

The proposed model defaults to the 1-D path (x_2d=None) because that is the
cleaner headline model and far faster; pass --with_cwt to enable the ResNet-34
scalogram branch (scalograms are precomputed once and memmapped, per the
GPU-starvation lesson — never recomputed per epoch).
"""

from __future__ import annotations

import os
import sys
import json
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

import data_pipeline as dp
from metrics import (compute_metrics, bootstrap_ci, mcnemar_test,
                     format_metric_report, save_results)


# =====================================================================
# Dataset
# =====================================================================
class ECGWindowDataset(Dataset):
    """Returns (x_1d (1,T), x_2d (3,H,W) or empty, label).
    Scalograms, if provided, are a memmapped array aligned with X."""
    def __init__(self, X, y, scalograms=None):
        self.X = X
        self.y = y
        self.scal = scalograms

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        x1d = torch.from_numpy(np.ascontiguousarray(self.X[i])).float().unsqueeze(0)
        label = int(self.y[i])
        if self.scal is not None:
            x2d = torch.from_numpy(np.ascontiguousarray(self.scal[i])).float()
            return x1d, x2d, label
        return x1d, torch.empty(0), label


def collate(batch):
    x1d = torch.stack([b[0] for b in batch])
    labels = torch.tensor([b[2] for b in batch], dtype=torch.long)
    has_2d = batch[0][1].numel() > 0
    x2d = torch.stack([b[1] for b in batch]) if has_2d else None
    return x1d, x2d, labels


# =====================================================================
# Scalogram precompute (one-time, memmapped)
# =====================================================================
def precompute_scalograms(X, cfg, cache_path, verbose=True):
    """Compute CWT scalograms for every window once and memmap them to disk."""
    n = len(X)
    shape = (n, 3, cfg.n_scales, cfg.cwt_width)
    if os.path.exists(cache_path):
        if verbose:
            print(f"[cwt] loading memmap {cache_path}")
        return np.lib.format.open_memmap(cache_path, mode="r")
    if verbose:
        print(f"[cwt] precomputing {n} scalograms -> {cache_path}")
    mm = np.lib.format.open_memmap(cache_path, mode="w+", dtype=np.float32, shape=shape)
    t0 = time.time()
    for i in range(n):
        mm[i] = dp.generate_cwt_scalogram(X[i], cfg.n_scales, cfg.cwt_width, cfg.target_fs)
        if verbose and (i + 1) % 2000 == 0:
            el = time.time() - t0
            print(f"[cwt] {i+1}/{n}  ({el:.0f}s, {(i+1)/el:.0f}/s)", flush=True)
    mm.flush()
    return mm


# =====================================================================
# Model factory + unified loss
# =====================================================================
def build_model(name, num_classes=3):
    if name == "proposed":
        from proposed_model import ProposedModel
        return ProposedModel(num_classes=num_classes), True   # is_multitask
    from baselines import BASELINE_REGISTRY
    if name not in BASELINE_REGISTRY:
        raise ValueError(f"unknown model '{name}'. "
                         f"choices: proposed, {list(BASELINE_REGISTRY)}")
    return BASELINE_REGISTRY[name](num_classes=num_classes), False


def forward_and_loss(model, is_mt, x1d, x2d, labels, alpha):
    """Returns (loss, cls_logits). Routes through the right objective."""
    from baselines import focal_loss
    if is_mt:
        out = model(x1d, x2d)
        loss_d = model.compute_loss(out, {"cls_target": labels})
        return loss_d["loss"], out["cls_logits"]
    logits = model(x1d)
    return focal_loss(logits, labels, alpha=alpha), logits


# =====================================================================
# Train / eval loops
# =====================================================================
@torch.no_grad()
def evaluate(model, is_mt, loader, device):
    model.eval()
    all_logits, all_y = [], []
    for x1d, x2d, y in loader:
        x1d = x1d.to(device)
        x2d = x2d.to(device) if x2d is not None else None
        if is_mt:
            logits = model(x1d, x2d)["cls_logits"]
        else:
            logits = model(x1d)
        all_logits.append(logits.float().cpu())
        all_y.append(y)
    logits = torch.cat(all_logits).numpy()
    y_true = torch.cat(all_y).numpy()
    probs = torch.softmax(torch.from_numpy(logits), dim=-1).numpy()
    y_pred = probs.argmax(1)
    return y_true, y_pred, probs


def train_model(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = (device == "cuda")
    os.makedirs(args.out, exist_ok=True)

    cfg = dp.PipelineConfig(data_dir=args.data_dir, window_size=args.window_size,
                            wavelet_baseline=not args.no_wavelet, seed=args.seed)
    print(f"[cfg] {cfg.fingerprint()}  device={device}")
    dp.verify_split_integrity(cfg)

    data = dp.build_and_cache(cfg, cache_dir=args.cache_dir)
    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]
    X_test, y_test = data["X_test"], data["y_test"]
    print(f"[data] train={len(y_train)} val={len(y_val)} test={len(y_test)}")

    # Optional scalograms for the proposed model's 2D branch.
    scal_tr = scal_va = scal_te = None
    if args.model == "proposed" and args.with_cwt:
        scal_tr = precompute_scalograms(X_train, cfg, os.path.join(args.cache_dir, "scal_train.npy"))
        scal_va = precompute_scalograms(X_val, cfg, os.path.join(args.cache_dir, "scal_val.npy"))
        scal_te = precompute_scalograms(X_test, cfg, os.path.join(args.cache_dir, "scal_test.npy"))

    ds_tr = ECGWindowDataset(X_train, y_train, scal_tr)
    ds_va = ECGWindowDataset(X_val, y_val, scal_va)
    ds_te = ECGWindowDataset(X_test, y_test, scal_te)

    # Balanced sampling: weight each sample by inverse class frequency.
    cls_w = dp.compute_class_weights(y_train)
    sample_w = cls_w[y_train]
    sampler = WeightedRandomSampler(torch.as_tensor(sample_w, dtype=torch.double),
                                    num_samples=len(y_train), replacement=True)
    alpha = torch.as_tensor(cls_w, dtype=torch.float32, device=device)

    dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, sampler=sampler,
                       collate_fn=collate, num_workers=args.workers,
                       pin_memory=use_amp, drop_last=True)
    dl_va = DataLoader(ds_va, batch_size=args.batch_size, shuffle=False,
                       collate_fn=collate, num_workers=args.workers)
    dl_te = DataLoader(ds_te, batch_size=args.batch_size, shuffle=False,
                       collate_fn=collate, num_workers=args.workers)

    model, is_mt = build_model(args.model)
    model = model.to(device)
    from baselines import count_params
    print(f"[model] {args.model}  {count_params(model):.2f}M params  multitask={is_mt}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    warmup = torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.1, total_iters=5)
    cosine = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=20, T_mult=2)
    sched = torch.optim.lr_scheduler.SequentialLR(opt, [warmup, cosine], milestones=[5])

    best_f1, best_state, patience = -1.0, None, 0
    ckpt_path = os.path.join(args.out, "best.pt")

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        running = 0.0
        for x1d, x2d, y in dl_tr:
            x1d = x1d.to(device, non_blocking=use_amp)
            x2d = x2d.to(device) if x2d is not None else None
            y = y.to(device)
            opt.zero_grad(set_to_none=True)
            if use_amp:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss, _ = forward_and_loss(model, is_mt, x1d, x2d, y, alpha)
            else:
                loss, _ = forward_and_loss(model, is_mt, x1d, x2d, y, alpha)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            running += loss.item()
        sched.step()

        # Validation macro-F1 = early-stopping signal.
        yv, pv, _ = evaluate(model, is_mt, dl_va, device)
        from sklearn.metrics import f1_score
        val_f1 = f1_score(yv, pv, average="macro", zero_division=0)
        dt = time.time() - t0
        print(f"[epoch {epoch:3d}] loss={running/max(1,len(dl_tr)):.4f} "
              f"val_macroF1={val_f1:.4f} lr={opt.param_groups[0]['lr']:.2e} "
              f"({dt:.1f}s)", flush=True)

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save({"state_dict": best_state, "val_f1": best_f1,
                        "epoch": epoch, "model": args.model}, ckpt_path)
            patience = 0
        else:
            patience += 1
            if patience >= args.patience:
                print(f"[early stop] no val improvement for {args.patience} epochs")
                break

    # Restore best and evaluate on DS2 test.
    if best_state is not None:
        model.load_state_dict(best_state)
    y_true, y_pred, y_prob = evaluate(model, is_mt, dl_te, device)
    label_names = list(data["label_names"])
    mb = compute_metrics(y_true, y_pred, y_prob, label_names)
    cis = {m: bootstrap_ci(y_true, y_pred, m, n_boot=args.n_boot)
           for m in ("accuracy", "macro_f1", "weighted_f1", "cohen_kappa")}
    print("\n" + format_metric_report(mb, cis["macro_f1"]))

    save_results(os.path.join(args.out, "metrics.json"), mb, cis)
    np.savez(os.path.join(args.out, "predictions.npz"),
             y_true=y_true, y_pred=y_pred, y_prob=y_prob,
             label_names=np.asarray(label_names))
    print(f"\n[saved] {args.out}/metrics.json  {args.out}/predictions.npz  {ckpt_path}")
    return mb


# =====================================================================
# Cross-model McNemar comparison
# =====================================================================
def compare_models(run_dirs):
    """Load saved predictions from each run dir and run pairwise McNemar tests.
    The first run dir is treated as the 'proposed' anchor."""
    preds, names = {}, []
    ref_y = None
    for d in run_dirs:
        p = np.load(os.path.join(d, "predictions.npz"))
        name = os.path.basename(d.rstrip("/"))
        names.append(name)
        preds[name] = p["y_pred"]
        if ref_y is None:
            ref_y = p["y_true"]
        else:
            assert np.array_equal(ref_y, p["y_true"]), \
                f"{name} evaluated on a different test set — cannot pair!"

    print("=" * 60)
    print("McNEMAR PAIRWISE COMPARISON (same DS2 test set)")
    print("=" * 60)
    anchor = names[0]
    results = []
    for other in names[1:]:
        r = mcnemar_test(ref_y, preds[anchor], preds[other], anchor, other)
        results.append(r)
        sig = "significant" if r["p_value"] < 0.05 else "n.s."
        print(f"  {anchor} vs {other}: better={r['better_model']}  "
              f"p={r['p_value']:.2e} ({sig})  "
              f"[+{r['n_a_right_b_wrong']}/-{r['n_a_wrong_b_right']}]")
    with open(os.path.join(run_dirs[0], "mcnemar.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"[saved] {run_dirs[0]}/mcnemar.json")
    return results


# =====================================================================
# CLI
# =====================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--compare", nargs="+", default=None,
                    help="run dirs to McNemar-compare (first = anchor)")
    ap.add_argument("--model", default="proposed",
                    help="proposed | cnn1d | cnnlstm")
    ap.add_argument("--data_dir", default=None, help="dir with MIT-BIH .dat/.hea/.atr")
    ap.add_argument("--out", default="runs/exp")
    ap.add_argument("--cache_dir", default="./cache")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--window_size", type=int, default=3600)
    ap.add_argument("--with_cwt", action="store_true",
                    help="enable proposed model's ResNet-34 scalogram branch")
    ap.add_argument("--no_wavelet", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.compare:
        compare_models(args.compare)
        return
    if args.data_dir is None:
        ap.error("--data_dir is required for training")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    train_model(args)


if __name__ == "__main__":
    main()
