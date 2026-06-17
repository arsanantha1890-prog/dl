"""XAI evaluation: Grad-CAM on the Beat-MIL backbone, IoU/Dice against
LUDB expert P/QRS/T-wave annotations.

Approach:
    For each LUDB record (200 total), we run Beat-MIL forward over the
    10-sec window, register a Grad-CAM hook on the last backbone block,
    backprop the predicted-class score, average gradients over channels
    to get a 1D saliency along the backbone-time axis, upsample to
    signal time, threshold at the top-K percentile, and compute IoU/Dice
    against each anatomical mask (P, QRS, T).

We average over records and report per-region and overall IoU/Dice.
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

from beatmil import BeatMIL
from ludb import load_ludb, LUDBRecord


# ---------- Grad-CAM hook ------------------------------------------------

class GradCAM1D:
    """Grad-CAM for a 1D backbone. Hooks the output and gradient of the
    final residual block."""
    def __init__(self, model: BeatMIL):
        self.model = model
        # the last block of the backbone is .blocks[-1] (after the maxpools)
        target = model.backbone.blocks[-1]
        self._features = None
        self._gradients = None
        target.register_forward_hook(self._fwd_hook)
        target.register_full_backward_hook(self._bwd_hook)

    def _fwd_hook(self, module, inp, out):
        self._features = out.detach()

    def _bwd_hook(self, module, grad_in, grad_out):
        self._gradients = grad_out[0].detach()

    def saliency(self, x, beat_positions, target_class: int | None = None) -> np.ndarray:
        """Return upsampled saliency over signal time (T,)."""
        self.model.zero_grad()
        out = self.model(x, beat_positions)        # x: (1,1,T)
        evidence = out["bag_evidence"]              # (1, K)
        if target_class is None:
            target_class = int(evidence.argmax(dim=-1).item())
        score = evidence[0, target_class]
        score.backward()

        feats = self._features        # (1, C, T')
        grads = self._gradients       # (1, C, T')
        # channel weights = mean gradient over time
        w = grads.mean(dim=-1, keepdim=True)              # (1, C, 1)
        cam = F.relu((w * feats).sum(dim=1, keepdim=True))  # (1, 1, T')
        # upsample to signal time
        T_signal = x.shape[-1]
        cam_up = F.interpolate(cam, size=T_signal, mode="linear", align_corners=False)
        cam_np = cam_up.squeeze().cpu().numpy()
        # normalize
        if cam_np.max() > cam_np.min():
            cam_np = (cam_np - cam_np.min()) / (cam_np.max() - cam_np.min())
        return cam_np


# ---------- IoU / Dice ---------------------------------------------------

def iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = (a & b).sum()
    union = (a | b).sum()
    return float(inter / union) if union > 0 else 0.0


def dice(a: np.ndarray, b: np.ndarray) -> float:
    inter = (a & b).sum()
    s = a.sum() + b.sum()
    return float(2 * inter / s) if s > 0 else 0.0


# ---------- detect beats in LUDB and build BeatMIL input -----------------

def beat_positions_from_qrs_mask(qrs_mask: np.ndarray, backbone_down: int = 8) -> torch.Tensor:
    """Find QRS region centers, convert to backbone-time."""
    diff = np.diff(qrs_mask.astype(int))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    if len(ends) < len(starts):
        ends = np.concatenate([ends, [len(qrs_mask) - 1]])
    elif len(starts) < len(ends):
        starts = np.concatenate([[0], starts])
    centers = (starts + ends) // 2
    centers = centers[centers >= 0]
    bt = (centers // backbone_down).astype(np.int64)
    N_MAX = 20
    out = np.full(N_MAX, -1, dtype=np.int64)
    for i, v in enumerate(bt[:N_MAX]):
        out[i] = v
    return torch.from_numpy(out)


# ---------- main XAI evaluation -----------------------------------------

def evaluate_xai(
    model_ckpt: Path,
    ludb_root: Path,
    device: str = "cuda",
    threshold_percentile: float = 70.0,
    T_WINDOW: int = 3600,
):
    """Compute IoU/Dice between Grad-CAM saliency and LUDB P/QRS/T regions."""
    model = BeatMIL(num_classes=4).to(device).eval()
    state = torch.load(model_ckpt, map_location=device)
    model.load_state_dict(state["model"])
    cam = GradCAM1D(model)

    records = load_ludb(ludb_root)
    print(f"[xai] loaded {len(records)} LUDB records")

    results = {"p": {"iou": [], "dice": []},
               "qrs": {"iou": [], "dice": []},
               "t": {"iou": [], "dice": []}}

    for rec in records:
        sig = rec.signal
        # take the first 10-sec window
        if len(sig) < T_WINDOW:
            continue
        x = sig[:T_WINDOW]
        x_t = torch.from_numpy(x).float().unsqueeze(0).unsqueeze(0).to(device)
        x_t = (x_t - x_t.mean()) / (x_t.std() + 1e-6)
        bp = beat_positions_from_qrs_mask(rec.qrs_mask[:T_WINDOW]).unsqueeze(0).to(device)

        sal = cam.saliency(x_t, bp)
        # threshold saliency into a binary mask at top-X percentile
        thresh = np.percentile(sal, threshold_percentile)
        sal_mask = sal >= thresh

        for name, gt in [("p", rec.p_mask[:T_WINDOW]),
                         ("qrs", rec.qrs_mask[:T_WINDOW]),
                         ("t", rec.t_mask[:T_WINDOW])]:
            results[name]["iou"].append(iou(sal_mask, gt.astype(bool)))
            results[name]["dice"].append(dice(sal_mask, gt.astype(bool)))

    summary = {}
    for name in ["p", "qrs", "t"]:
        summary[name] = {
            "iou_mean": float(np.mean(results[name]["iou"])),
            "iou_std":  float(np.std(results[name]["iou"])),
            "dice_mean": float(np.mean(results[name]["dice"])),
            "dice_std":  float(np.std(results[name]["dice"])),
            "n": len(results[name]["iou"]),
        }
    return summary, results
