"""Training script for Beat-MIL and baselines.

Usage:
    python train.py --model beatmil --mode intra-db
    python train.py --model beatmil --mode lodo --held-out ptbxl
    python train.py --model resnet1d --mode intra-db
"""

from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from unified_dataset import (
    UnifiedECGDataset, patient_level_split, lodo_split, save_split_manifest,
    AAMI_CLASSES,
)
from beatmil import BeatMIL
from baselines import build_baseline
from evidential import evidential_loss, predict_with_uncertainty
from consistency import consistency_loss
from focal import focal_cross_entropy


# ---------- training step ------------------------------------------------

def beatmil_step(model, batch, optimizer, lambda_cons: float = 0.5,
                 kl_weight: float = 0.1):
    out = model(batch["x"], batch["beat_positions"])
    bag_out = evidential_loss(out["bag_evidence"], batch["bag_target"], kl_weight=kl_weight)
    L_bag = bag_out["loss"]

    # beat-level focal loss only on MIT-BIH samples
    mask = (batch["beat_targets"] >= 0)
    if mask.any():
        L_beat = focal_cross_entropy(
            out["beat_logits"][mask], batch["beat_targets"][mask].long(), gamma=2.0,
        )
    else:
        L_beat = torch.tensor(0.0, device=batch["x"].device)

    # consistency only on MIT-BIH
    L_cons = consistency_loss(
        bag_logits=torch.log(bag_out["p"] + 1e-8),
        beat_logits=out["beat_logits"],
        alpha=out["alpha"],
        mask=out["mask"],
        sample_mask=batch["has_beat_labels"],
    )

    loss = L_bag + L_beat + lambda_cons * L_cons
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    return {"loss": loss.item(), "bag": L_bag.item(),
            "beat": float(L_beat), "cons": L_cons.item()}


def baseline_step(model, batch, optimizer):
    out = model(batch["x"])
    logits = out["logits"]
    loss = F.cross_entropy(logits, batch["bag_target"], label_smoothing=0.1)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    return {"loss": loss.item()}


# ---------- evaluation ---------------------------------------------------

@torch.no_grad()
def evaluate(model, loader, device, is_beatmil: bool):
    model.eval()
    all_preds, all_targets, all_probs, all_vacuity = [], [], [], []
    for batch in loader:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}
        if is_beatmil:
            out = model(batch["x"], batch["beat_positions"])
            info = predict_with_uncertainty(out["bag_evidence"])
            all_probs.append(info["p"].cpu().numpy())
            all_vacuity.append(info["u"].cpu().numpy())
            all_preds.append(info["pred"].cpu().numpy())
        else:
            logits = model(batch["x"])["logits"]
            probs = F.softmax(logits, dim=-1)
            all_probs.append(probs.cpu().numpy())
            all_preds.append(probs.argmax(dim=-1).cpu().numpy())
            all_vacuity.append(np.zeros(probs.shape[0]))  # not applicable
        all_targets.append(batch["bag_target"].cpu().numpy())

    return {
        "preds": np.concatenate(all_preds),
        "targets": np.concatenate(all_targets),
        "probs": np.concatenate(all_probs),
        "vacuity": np.concatenate(all_vacuity),
    }


# ---------- collate function for variable-length bags --------------------

def collate(batch):
    return {
        "x": torch.stack([b["x"] for b in batch]),
        "beat_positions": torch.stack([b["beat_positions"] for b in batch]),
        "bag_target": torch.tensor([b["bag_target"] for b in batch], dtype=torch.long),
        "has_beat_labels": torch.tensor([b["has_beat_labels"] for b in batch], dtype=torch.float),
        "beat_targets": torch.stack([b["beat_targets"] for b in batch]),
        "database": [b["database"] for b in batch],
    }


# ---------- main loop ----------------------------------------------------

def build_loaders(specs_train, specs_val, loaders_per_db, batch_size, num_workers):
    import numpy as np
    from torch.utils.data import WeightedRandomSampler
    train_ds = UnifiedECGDataset(specs_train, loaders_per_db, augment=True)
    val_ds   = UnifiedECGDataset(specs_val,   loaders_per_db, augment=False)

    # Class-balanced sampling: inverse-frequency weights over bag labels.
    # Critical after the majority-vote fix, since most MIT-BIH windows are N.
    labels = np.array([s.bag_label for s in specs_train], dtype=np.int64)
    class_counts = np.bincount(labels, minlength=4).astype(np.float64)
    class_weights = 1.0 / (class_counts + 1e-6)
    sample_weights = class_weights[labels]
    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights).double(),
        num_samples=len(labels),
        replacement=True,
    )

    train_dl = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                          num_workers=num_workers, collate_fn=collate,
                          pin_memory=True, persistent_workers=(num_workers > 0),
                          prefetch_factor=(4 if num_workers > 0 else None))
    val_dl   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                          num_workers=num_workers, collate_fn=collate,
                          pin_memory=True, persistent_workers=(num_workers > 0),
                          prefetch_factor=(4 if num_workers > 0 else None))
    return train_dl, val_dl


def macro_f1(preds, targets, K=4):
    f1s = []
    for c in range(K):
        tp = ((preds == c) & (targets == c)).sum()
        fp = ((preds == c) & (targets != c)).sum()
        fn = ((preds != c) & (targets == c)).sum()
        prec = tp / (tp + fp + 1e-8)
        rec  = tp / (tp + fn + 1e-8)
        f1s.append(2 * prec * rec / (prec + rec + 1e-8))
    return float(np.mean(f1s))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["beatmil", "resnet1d", "cnnlstm", "ecgformer"])
    ap.add_argument("--mode", default="intra-db", choices=["intra-db", "lodo"])
    ap.add_argument("--held-out", default=None, choices=["mitbih", "cpsc", "ptbxl"])
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--out-dir", default="checkpoints")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lambda-cons", type=float, default=0.5)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[train] device={device}, model={args.model}, mode={args.mode}")

    # Build loaders for each DB
    from mitbih_loader import MITBIHLoader  # extracted into its own file in Step 11
    from cpsc import CPSC2018Loader
    from ptbxl import PTBXLLoader
    data_root = Path(args.data_root)
    db_loaders = {
        "mitbih": MITBIHLoader(data_root / "mitbih"),
        "cpsc":   CPSC2018Loader(data_root / "cpsc2018"),
        "ptbxl":  PTBXLLoader(data_root / "ptbxl"),
    }
    print("[train] listing samples per database...")
    all_specs = []
    for name, ld in db_loaders.items():
        s = ld.list_samples()
        print(f"  {name}: {len(s):,} samples")
        all_specs.extend(s)
    print(f"  total: {len(all_specs):,}")

    # Split
    if args.mode == "intra-db":
        split = patient_level_split(all_specs, ratios=(0.7, 0.15, 0.15), seed=args.seed)
    else:
        assert args.held_out, "--held-out required for lodo mode"
        split = lodo_split(all_specs, held_out_db=args.held_out)
    out_dir = Path(args.out_dir) / f"{args.model}_{args.mode}" / (args.held_out or "")
    out_dir.mkdir(parents=True, exist_ok=True)
    save_split_manifest(split, out_dir / "split.json")
    print(f"[train] split: train={len(split['train']):,} val={len(split['val']):,} test={len(split['test']):,}")

    train_dl, val_dl = build_loaders(split["train"], split["val"], db_loaders,
                                     args.batch_size, args.num_workers)

    # Build model
    if args.model == "beatmil":
        model = BeatMIL(num_classes=len(AAMI_CLASSES))
    else:
        model = build_baseline(args.model, num_classes=len(AAMI_CLASSES))
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[train] {args.model}: {n_params:.2f}M params")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=20, T_mult=2)
    is_beatmil = args.model == "beatmil"

    best_val_f1 = -1.0
    patience = 0
    history = []
    t0 = time.time()

    for epoch in range(args.epochs):
        model.train()
        ep_losses = []
        for batch in train_dl:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            if is_beatmil:
                info = beatmil_step(model, batch, opt, lambda_cons=args.lambda_cons)
            else:
                info = baseline_step(model, batch, opt)
            ep_losses.append(info["loss"])
        sched.step()

        # Validate
        ev = evaluate(model, val_dl, device, is_beatmil)
        val_f1 = macro_f1(ev["preds"], ev["targets"])
        val_acc = float((ev["preds"] == ev["targets"]).mean())
        mean_loss = float(np.mean(ep_losses))
        elapsed = time.time() - t0
        print(f"epoch {epoch:3d} | loss {mean_loss:.4f} | val_acc {val_acc:.4f} | "
              f"val_F1 {val_f1:.4f} | {elapsed:.0f}s")
        history.append({"epoch": epoch, "loss": mean_loss,
                        "val_acc": val_acc, "val_f1": val_f1})

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience = 0
            torch.save({"model": model.state_dict(), "args": vars(args), "epoch": epoch,
                        "val_f1": val_f1}, out_dir / "best.pt")
        else:
            patience += 1
            if patience >= args.patience:
                print(f"[train] early stop at epoch {epoch}")
                break

    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    print(f"[train] done. best val F1 = {best_val_f1:.4f}")
    print(f"[train] checkpoint: {out_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
