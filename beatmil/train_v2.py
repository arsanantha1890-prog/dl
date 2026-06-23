"""
train_v2.py — Stable training harness with RR-interval augmentation.

Key changes vs train.py:
1. RR interval features appended as extra channels alongside the ECG window.
   S beats are primarily identified by their PREMATURE timing (short pre-RR)
   and COMPENSATORY pause (long post-RR). Without this, a CNN looking at
   waveform morphology alone cannot reliably separate S from N on Lead II.
2. Class weights use a stronger S emphasis (factor 6x N) applied ONLY in the
   loss — no sampler stacking.
3. Longer patience (25 epochs) and more epochs (100) since S learning is slow.
4. Per-epoch confusion matrix printed so you can see exactly what's happening.

Run:
    nohup python train_v2.py --model proposed --data_dir $DATA_DIR \
        --out runs/proposed_v2 --epochs 100 --batch_size 256 \
        --window_size 360 > runs/proposed_v2.log 2>&1 &
"""

from __future__ import annotations
import os, json, time, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, confusion_matrix

import data_pipeline as dp
from metrics import compute_metrics, bootstrap_ci, format_metric_report, save_results


# =====================================================================
# RR interval extraction
# =====================================================================
def extract_rr_features(ann_samples, ann_symbols, target_sample, fs=360):
    """
    For the beat at target_sample, extract:
      pre_rr  : interval to previous beat (normalised by mean RR)
      post_rr : interval to next beat (normalised by mean RR)
      ratio   : pre_rr / post_rr  (S beats: short pre, long post → ratio << 1)
      local_mean_rr : mean of surrounding 5 RR intervals

    Returns a (4,) float32 array. Returns zeros if neighbours unavailable.
    """
    # Filter to beat annotations only
    beat_syms = set(dp.SYMBOL_TO_AAMI.keys())
    beat_samples = [s for s, sym in zip(ann_samples, ann_symbols) if sym in beat_syms]
    beat_samples = sorted(beat_samples)

    try:
        idx = beat_samples.index(target_sample)
    except ValueError:
        return np.zeros(4, dtype=np.float32)

    pre_rr  = (beat_samples[idx] - beat_samples[idx-1]) / fs if idx > 0 else 0.0
    post_rr = (beat_samples[idx+1] - beat_samples[idx]) / fs if idx < len(beat_samples)-1 else 0.0

    # Local mean RR (window of 5 surrounding beats)
    lo, hi = max(0, idx-2), min(len(beat_samples)-1, idx+3)
    local_rrs = [beat_samples[j+1] - beat_samples[j]
                 for j in range(lo, hi) if j+1 <= hi]
    mean_rr = float(np.mean(local_rrs)) / fs if local_rrs else 1.0

    pre_norm  = pre_rr  / mean_rr if mean_rr > 0 else 0.0
    post_norm = post_rr / mean_rr if mean_rr > 0 else 0.0
    ratio     = pre_norm / post_norm if post_norm > 0 else 0.0
    ratio     = min(ratio, 5.0)  # cap outliers

    return np.array([pre_norm, post_norm, ratio, mean_rr], dtype=np.float32)


def build_dataset_with_rr(records, cfg, data_dir):
    """Build windows + RR features for a list of record IDs."""
    import wfdb
    windows, labels, rr_feats = [], [], []
    counts = {n: 0 for n in dp.LABEL_NAMES_3CLASS}
    half = cfg.window_size // 2

    for rec_id in records:
        path = os.path.join(data_dir, str(rec_id))
        record = wfdb.rdrecord(path)
        ann    = wfdb.rdann(path, "atr")
        sig = dp._select_lead_ii(record)
        sig = dp.preprocess_signal(sig, cfg)

        for sample, symbol in zip(ann.sample, ann.symbol):
            aami = dp.SYMBOL_TO_AAMI.get(symbol)
            if aami is None: continue
            label = dp.AAMI_TO_LABEL_3CLASS.get(aami)
            if label is None: continue
            start, end = sample - half, sample + half
            if start < 0 or end > len(sig): continue
            win = dp.zscore(sig[start:end])
            if len(win) != cfg.window_size: continue

            rr = extract_rr_features(ann.sample, ann.symbol, sample, cfg.target_fs)
            windows.append(win)
            labels.append(label)
            rr_feats.append(rr)
            counts[dp.LABEL_NAMES_3CLASS[label]] += 1

    return (np.array(windows, dtype=np.float32),
            np.array(labels,  dtype=np.int64),
            np.array(rr_feats, dtype=np.float32),
            counts)


def build_all_splits(cfg, data_dir, cache_dir):
    """Build or load all splits with RR features."""
    os.makedirs(cache_dir, exist_ok=True)
    fp = cfg.fingerprint() + "_rr"
    cache = os.path.join(cache_dir, f"mitbih_rr_{fp}.npz")
    if os.path.exists(cache):
        print(f"[cache] loading {cache}")
        d = np.load(cache, allow_pickle=True)
        return {k: d[k] for k in d.files}

    train_recs, val_recs = dp.split_train_val_records(cfg)
    test_recs = dp.DS2_RECORDS
    print(f"[build] train={len(train_recs)} val={len(val_recs)} test={len(test_recs)}")

    Xtr, ytr, rr_tr, ctr = build_dataset_with_rr(train_recs, cfg, data_dir)
    Xva, yva, rr_va, cva = build_dataset_with_rr(val_recs,   cfg, data_dir)
    Xte, yte, rr_te, cte = build_dataset_with_rr(test_recs,  cfg, data_dir)

    print(f"[counts] train {ctr}  val {cva}  test {cte}")
    out = dict(X_train=Xtr, y_train=ytr, rr_train=rr_tr,
               X_val=Xva,   y_val=yva,   rr_val=rr_va,
               X_test=Xte,  y_test=yte,  rr_test=rr_te,
               label_names=np.array(dp.LABEL_NAMES_3CLASS))
    np.savez_compressed(cache, **out)
    print(f"[cache] wrote {cache}")
    return out


# =====================================================================
# Model with RR features fused at the head
# =====================================================================
class RRFusionModel(nn.Module):
    """
    Wraps any backbone (proposed model's 1D CNN branch or a baseline) and
    concatenates normalised RR interval features before the final classifier.

    This is the key addition: RR intervals are the primary discriminator for
    S beats (premature + compensatory pause). The CNN sees morphology; the
    RR branch sees timing. Together they can classify S reliably.
    """
    def __init__(self, backbone_name="proposed", num_classes=3, n_rr=4):
        super().__init__()
        self.backbone_name = backbone_name

        if backbone_name == "proposed":
            from proposed_model import CNN1DBranch, TemporalEncoder
            self.cnn = CNN1DBranch(d_out=256)
            self.temporal = TemporalEncoder(d_in=256, d_model=256)
            feat_dim = 256
        elif backbone_name == "cnn1d":
            from baselines import CNN1D_ECG
            # Use CNN1D as feature extractor (remove final head)
            b = CNN1D_ECG(num_classes=num_classes)
            self.cnn = nn.Sequential(*list(b.children())[:-1])  # drop head
            feat_dim = 256
            self.temporal = nn.Identity()
        else:
            from baselines import CNNLSTM_ECG
            b = CNNLSTM_ECG(num_classes=num_classes)
            self.cnn = nn.Sequential(b.features, nn.AdaptiveAvgPool1d(1), nn.Flatten())
            feat_dim = 128
            self.temporal = nn.Identity()

        # RR feature encoder (4 → 32)
        self.rr_enc = nn.Sequential(
            nn.Linear(n_rr, 32), nn.LayerNorm(32), nn.ReLU(),
            nn.Linear(32, 32), nn.ReLU(),
        )

        # Fused classifier
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim + 32, 256), nn.BatchNorm1d(256),
            nn.GELU(), nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x_ecg, x_rr):
        # ECG branch
        if self.backbone_name == "proposed":
            feat = self.cnn(x_ecg)       # (B, 256)
            feat = self.temporal(feat)   # (B, 256)
        else:
            feat = self.cnn(x_ecg)
            if isinstance(feat, dict):
                feat = feat["fused"]

        # RR branch
        rr_feat = self.rr_enc(x_rr)     # (B, 32)

        # Fuse and classify
        fused = torch.cat([feat, rr_feat], dim=-1)  # (B, feat+32)
        return self.classifier(fused)


# =====================================================================
# Dataset
# =====================================================================
class ECGRRDataset(Dataset):
    def __init__(self, X, y, rr):
        self.X = X; self.y = y; self.rr = rr

    def __len__(self): return len(self.X)

    def __getitem__(self, i):
        x = torch.from_numpy(np.ascontiguousarray(self.X[i])).float().unsqueeze(0)
        r = torch.from_numpy(np.ascontiguousarray(self.rr[i])).float()
        return x, r, int(self.y[i])


def collate(batch):
    x   = torch.stack([b[0] for b in batch])
    rr  = torch.stack([b[1] for b in batch])
    y   = torch.tensor([b[2] for b in batch], dtype=torch.long)
    return x, rr, y


# =====================================================================
# Focal loss
# =====================================================================
def focal_loss(logits, targets, alpha, gamma=2.0):
    ce = F.cross_entropy(logits, targets, weight=alpha,
                         reduction="none", label_smoothing=0.05)
    with torch.no_grad():
        pt = F.softmax(logits.float(), dim=-1).gather(
            1, targets.unsqueeze(1)).squeeze(1)
    return ((1 - pt) ** gamma * ce).mean()


# =====================================================================
# Eval
# =====================================================================
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_logits, all_y = [], []
    for x, rr, y in loader:
        x, rr = x.to(device), rr.to(device)
        logits = model(x, rr)
        all_logits.append(logits.float().cpu())
        all_y.append(y)
    logits = torch.cat(all_logits).numpy()
    y_true = torch.cat(all_y).numpy()
    probs  = torch.softmax(torch.from_numpy(logits), dim=-1).numpy()
    return y_true, probs.argmax(1), probs


# =====================================================================
# Training loop
# =====================================================================
def train_model(args):
    device  = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = (device == "cuda")
    os.makedirs(args.out, exist_ok=True)

    cfg = dp.PipelineConfig(data_dir=args.data_dir,
                            window_size=args.window_size,
                            val_fraction=0.25, seed=42)
    print(f"[cfg] window={cfg.window_size}  val_frac={cfg.val_fraction}  device={device}")
    dp.verify_split_integrity(cfg)

    data = build_all_splits(cfg, args.data_dir, args.cache_dir)
    y_train = data["y_train"]
    counts = np.bincount(y_train, minlength=3)
    print(f"[data] train={len(y_train)}  val={len(data['y_val'])}  test={len(data['y_test'])}")
    print(f"[data] N={counts[0]}  S={counts[1]}  V={counts[2]}")

    # Strong but bounded S weight: manually set so S≈6x N, V≈2x N
    # (sqrt of raw inv-freq, then hand-checked against MIT-BIH proportions)
    raw_w = counts.sum() / (3.0 * counts.astype(float))
    cls_w = np.sqrt(raw_w); cls_w /= cls_w.mean()
    # Hard cap: S never more than 6x N to avoid inversion
    cls_w[1] = min(cls_w[1], 6.0 * cls_w[0])
    cls_w = cls_w.astype(np.float32)
    print(f"[balance] weights: N={cls_w[0]:.3f}  S={cls_w[1]:.3f}  V={cls_w[2]:.3f}")
    alpha = torch.tensor(cls_w, device=device)

    ds_tr = ECGRRDataset(data["X_train"], y_train,          data["rr_train"])
    ds_va = ECGRRDataset(data["X_val"],   data["y_val"],    data["rr_val"])
    ds_te = ECGRRDataset(data["X_test"],  data["y_test"],   data["rr_test"])

    dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True,
                       collate_fn=collate, num_workers=4,
                       pin_memory=use_amp, drop_last=True)
    dl_va = DataLoader(ds_va, batch_size=args.batch_size, shuffle=False,
                       collate_fn=collate, num_workers=4)
    dl_te = DataLoader(ds_te, batch_size=args.batch_size, shuffle=False,
                       collate_fn=collate, num_workers=4)

    model = RRFusionModel(args.model, num_classes=3).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[model] RRFusion({args.model})  {n_params:.2f}M params")

    opt     = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    warmup  = torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.1, total_iters=5)
    cosine  = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=25, T_mult=2)
    sched   = torch.optim.lr_scheduler.SequentialLR(opt, [warmup, cosine], milestones=[5])

    best_f1, best_state, patience = -1.0, None, 0
    ckpt = os.path.join(args.out, "best.pt")

    for epoch in range(1, args.epochs + 1):
        model.train(); t0 = time.time(); running = 0.0
        for x, rr, y in dl_tr:
            x, rr, y = x.to(device), rr.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            if use_amp:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = focal_loss(model(x, rr), y, alpha)
            else:
                loss = focal_loss(model(x, rr), y, alpha)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); running += loss.item()
        sched.step()

        yv, pv, _ = evaluate(model, dl_va, device)
        val_f1    = f1_score(yv, pv, average="macro", zero_division=0)
        val_per   = f1_score(yv, pv, average=None, labels=[0,1,2], zero_division=0)
        cm        = confusion_matrix(yv, pv, labels=[0,1,2])
        dt = time.time() - t0
        print(f"[epoch {epoch:3d}] loss={running/max(1,len(dl_tr)):.4f}  "
              f"val_macroF1={val_f1:.4f}  "
              f"[N={val_per[0]:.3f} S={val_per[1]:.3f} V={val_per[2]:.3f}]  "
              f"lr={opt.param_groups[0]['lr']:.2e}  ({dt:.1f}s)", flush=True)

        # Print confusion matrix every 10 epochs
        if epoch % 10 == 0:
            print(f"  val confusion:\n{cm}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            torch.save({"state_dict": best_state, "val_f1": best_f1,
                        "epoch": epoch, "model": args.model}, ckpt)
            patience = 0
        else:
            patience += 1
            if patience >= args.patience:
                print(f"[early stop] {args.patience} epochs no improvement")
                break

    # Final test eval
    model.load_state_dict(best_state)
    y_true, y_pred, y_prob = evaluate(model, dl_te, device)
    label_names = list(data["label_names"])
    mb  = compute_metrics(y_true, y_pred, y_prob, label_names)
    cis = {m: bootstrap_ci(y_true, y_pred, m, n_boot=1000)
           for m in ("accuracy", "macro_f1", "weighted_f1", "cohen_kappa")}
    print("\n" + format_metric_report(mb, cis["macro_f1"]))
    save_results(os.path.join(args.out, "metrics.json"), mb, cis)
    np.savez(os.path.join(args.out, "predictions.npz"),
             y_true=y_true, y_pred=y_pred, y_prob=y_prob,
             label_names=np.array(label_names))
    print(f"[saved] {args.out}/")
    return mb


# =====================================================================
# CLI
# =====================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",       default="proposed",
                    choices=["proposed", "cnn1d", "cnnlstm"])
    ap.add_argument("--data_dir",    required=True)
    ap.add_argument("--out",         default="runs/exp")
    ap.add_argument("--cache_dir",   default="./cache_rr")
    ap.add_argument("--epochs",      type=int,   default=100)
    ap.add_argument("--batch_size",  type=int,   default=256)
    ap.add_argument("--lr",          type=float, default=1e-3)
    ap.add_argument("--patience",    type=int,   default=25)
    ap.add_argument("--window_size", type=int,   default=360)
    ap.add_argument("--workers",     type=int,   default=4)
    ap.add_argument("--n_boot",      type=int,   default=1000)
    ap.add_argument("--seed",        type=int,   default=42)
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    train_model(args)

if __name__ == "__main__":
    main()
