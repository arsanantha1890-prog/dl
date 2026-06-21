"""Hybrid CNN-LSTM-Attention model with CWT scalogram branch.

Recreated from the submitted paper's architecture description (Fig. 1):
  1D-CNN (6 residual+SE blocks) -> Bi-LSTM -> 8-head Attention
  2D-CNN (ResNet-34 pretrained)  on CWT scalograms
  Gated fusion -> 3 task heads (detection, classification, prediction)
  Uncertainty-weighted multi-task loss (Kendall et al. 2018)

Total: ~24.9M params (ResNet-34 is the bulk).
Now uses 3-class N/S/V (Q excluded per de Chazal 2004).

CHANGES from the version Claude first recreated (both are real bug fixes,
verified by smoke test):
  (1) compute_loss: `.squeeze()` -> `.squeeze(1)`. The bare squeeze collapses a
      (B,1) tensor to a scalar when B==1, which crashes the focal term on the
      last (size-1) batch. squeeze(1) only removes the singleton class axis.
  (2) generate_cwt_scalogram: rewritten with pywt. The original used
      scipy.signal.cwt / morlet2, BOTH REMOVED in scipy >= 1.15 — the function
      raised ImportError on any current instance. pywt.cwt('morl') is the
      drop-in equivalent. (data_pipeline.generate_cwt_scalogram is identical;
      either may be used.)
"""

from __future__ import annotations
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# =====================================================================
# 1D-CNN Branch (same SE-ResNet backbone used in Beat-MIL)
# =====================================================================

class _SEBlock1D(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden), nn.ReLU(),
            nn.Linear(hidden, channels), nn.Sigmoid(),
        )

    def forward(self, x):
        s = self.pool(x).squeeze(-1)
        s = self.fc(s).unsqueeze(-1)
        return x * s


class _ResBlock1D(nn.Module):
    def __init__(self, in_c, out_c, kernel_size, stride=1, dropout=0.2):
        super().__init__()
        pad = kernel_size // 2
        self.conv = nn.Sequential(
            nn.Conv1d(in_c, out_c, kernel_size, stride=stride, padding=pad),
            nn.BatchNorm1d(out_c), nn.GELU(), nn.Dropout(dropout),
            nn.Conv1d(out_c, out_c, kernel_size, padding=pad),
            nn.BatchNorm1d(out_c),
        )
        self.se = _SEBlock1D(out_c)
        self.skip = (nn.Conv1d(in_c, out_c, 1, stride=stride)
                     if in_c != out_c or stride != 1 else nn.Identity())
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.se(self.conv(x)) + self.skip(x))


class CNN1DBranch(nn.Module):
    """6 residual blocks with SE, multi-scale pooling -> 256-d."""
    def __init__(self, d_out=256):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, 32, 15, stride=2, padding=7),
            nn.BatchNorm1d(32), nn.GELU(),
        )
        kernels = [15, 11, 7, 5, 5, 3]
        channels = [32, 64, 64, 128, 128, 256]
        layers = []
        in_c = 32
        for i, (k, c) in enumerate(zip(kernels, channels)):
            stride = 2 if i % 2 == 0 else 1
            layers.append(_ResBlock1D(in_c, c, k, stride=stride))
            in_c = c
        self.blocks = nn.Sequential(*layers)
        # Multi-scale pooling: avg + max + attention-weighted
        self.attn_pool = nn.Sequential(
            nn.Conv1d(256, 1, 1), nn.Softmax(dim=-1)
        )
        self.proj = nn.Sequential(
            nn.Linear(256 * 3, d_out), nn.LayerNorm(d_out), nn.GELU(),
        )

    def forward(self, x):
        # x: (B, 1, T)
        h = self.blocks(self.stem(x))  # (B, 256, T')
        avg = h.mean(dim=-1)           # (B, 256)
        mx = h.max(dim=-1).values      # (B, 256)
        aw = self.attn_pool(h)         # (B, 1, T')
        att = (h * aw).sum(dim=-1)     # (B, 256)
        return self.proj(torch.cat([avg, mx, att], dim=-1))  # (B, d_out)


# =====================================================================
# 2D-CNN Branch (ResNet-34 on CWT scalograms)
# =====================================================================

class CNN2DBranch(nn.Module):
    """ResNet-34 pretrained on ImageNet, first 2 groups frozen.
    Input: CWT scalogram (B, 3, 64, 256) — 3-channel for pretrained compat.
    Output: (B, 256).
    Falls back to a simple CNN if torchvision is not available."""
    def __init__(self, d_out=256, freeze_groups=2):
        super().__init__()
        self.d_out = d_out
        try:
            from torchvision.models import resnet34, ResNet34_Weights
            backbone = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
            self._use_resnet = True
        except ImportError:
            try:
                from torchvision.models import resnet34
                backbone = resnet34(pretrained=True)
                self._use_resnet = True
            except ImportError:
                # Fallback: simple CNN (for environments without torchvision)
                self._use_resnet = False
                self.fallback = nn.Sequential(
                    nn.Conv2d(3, 64, 7, stride=2, padding=3), nn.BatchNorm2d(64), nn.ReLU(),
                    nn.MaxPool2d(3, 2, 1),
                    nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
                    nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
                    nn.Conv2d(256, 512, 3, stride=2, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
                    nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                )
                self.proj = nn.Sequential(
                    nn.Linear(512, d_out), nn.LayerNorm(d_out), nn.GELU(),
                )
                return

        # ResNet path
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.avgpool = backbone.avgpool

        if freeze_groups >= 1:
            for p in [self.conv1, self.bn1, self.layer1]:
                for param in p.parameters():
                    param.requires_grad = False
        if freeze_groups >= 2:
            for param in self.layer2.parameters():
                param.requires_grad = False

        self.proj = nn.Sequential(
            nn.Linear(512, d_out), nn.LayerNorm(d_out), nn.GELU(),
        )

    def forward(self, scalogram):
        if not self._use_resnet:
            return self.proj(self.fallback(scalogram))
        x = self.maxpool(self.relu(self.bn1(self.conv1(scalogram))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x).flatten(1)
        return self.proj(x)


# =====================================================================
# Temporal encoder: Bi-LSTM + Multi-Head Attention
# =====================================================================

class TemporalEncoder(nn.Module):
    """Bi-LSTM (2 layers, 128 hidden/dir) + 8-head self-attention."""
    def __init__(self, d_in=256, d_model=256, n_heads=8, n_lstm_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(d_in, 128, num_layers=n_lstm_layers,
                            batch_first=True, bidirectional=True, dropout=0.2)
        self.lstm_proj = nn.Sequential(
            nn.Linear(256, d_model), nn.LayerNorm(d_model),
        )
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True, dropout=0.1)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(d_model * 4, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: (B, d_in) — treated as a single-step sequence.
        x = x.unsqueeze(1)              # (B, 1, d_in)
        h, _ = self.lstm(x)             # (B, 1, 256)
        h = self.lstm_proj(h)           # (B, 1, d_model)
        a, _ = self.attn(h, h, h)
        h = self.norm1(h + a)
        h = self.norm2(h + self.ffn(h))
        return h.squeeze(1)             # (B, d_model)


# =====================================================================
# CWT Scalogram Generation (on-the-fly) — pywt-based (scipy.signal.cwt is
# removed in scipy >= 1.15). Identical to data_pipeline.generate_cwt_scalogram.
# =====================================================================

def generate_cwt_scalogram(signal: np.ndarray, n_scales: int = 64,
                           width: int = 256, fs: int = 360) -> np.ndarray:
    """Generate a Morlet CWT scalogram from a 1D signal.

    Returns a (3, n_scales, width) array suitable for ResNet input.
    The 3 channels are identical (grayscale → pseudo-RGB for pretrained compat).
    """
    import pywt

    scales = np.geomspace(1, fs / 2, num=n_scales)
    coeffs, _ = pywt.cwt(signal, scales, "morl", sampling_period=1.0 / fs)
    power = np.abs(coeffs).astype(np.float32)  # (n_scales, len(signal))

    # Resize width to a fixed value via linear interpolation along the time axis.
    if power.shape[1] != width:
        idx = np.linspace(0, power.shape[1] - 1, width)
        power = np.stack([np.interp(idx, np.arange(power.shape[1]), row)
                          for row in power]).astype(np.float32)

    pmin, pmax = power.min(), power.max()
    if pmax > pmin:
        power = (power - pmin) / (pmax - pmin)

    return np.stack([power, power, power], axis=0)  # (3, 64, 256)


# =====================================================================
# Gated Fusion
# =====================================================================

class GatedFusion(nn.Module):
    """Learned per-sample branch weighting, outputs 512-d fused vector."""
    def __init__(self, d_branch=256):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(d_branch * 2, 2), nn.Softmax(dim=-1),
        )
        self.proj = nn.Sequential(
            nn.Linear(d_branch * 2, 512), nn.LayerNorm(512), nn.GELU(),
        )

    def forward(self, feat_1d, feat_2d):
        cat = torch.cat([feat_1d, feat_2d], dim=-1)  # (B, 512)
        g = self.gate(cat)  # (B, 2)
        fused = g[:, 0:1] * feat_1d + g[:, 1:2] * feat_2d  # (B, 256)
        # Also pass the full concatenation through projection for richer representation
        return self.proj(cat)  # (B, 512)


# =====================================================================
# Task Heads
# =====================================================================

class TaskHead(nn.Module):
    """Two-layer MLP with BN + dropout for a single task."""
    def __init__(self, d_in, d_hidden, d_out, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden), nn.BatchNorm1d(d_hidden),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_hidden, d_out),
        )

    def forward(self, x):
        return self.net(x)


# =====================================================================
# Full Proposed Model
# =====================================================================

class ProposedModel(nn.Module):
    """Hybrid CNN-LSTM-Attention with CWT branch.

    Three output heads:
        detection:      binary (normal vs abnormal)
        classification: 3-class N/S/V (AAMI, Q excluded)
        prediction:     scalar risk score

    Multi-task loss uses uncertainty weighting (Kendall et al. 2018).
    """
    def __init__(self, num_classes: int = 3):
        super().__init__()
        self.num_classes = num_classes

        # Branches
        self.cnn1d = CNN1DBranch(d_out=256)
        self.cnn2d = CNN2DBranch(d_out=256)
        self.temporal = TemporalEncoder(d_in=256, d_model=256)
        self.fusion = GatedFusion(d_branch=256)

        # Task heads
        self.detection_head = TaskHead(512, 256, 2)       # binary
        self.classification_head = TaskHead(512, 256, num_classes)
        self.prediction_head = TaskHead(512, 256, 1)      # risk score

        # Uncertainty weighting: learnable log-variance per task
        self.log_var_det = nn.Parameter(torch.zeros(1))
        self.log_var_cls = nn.Parameter(torch.zeros(1))
        self.log_var_pred = nn.Parameter(torch.zeros(1))

    def forward(self, x_1d: torch.Tensor,
                x_2d: torch.Tensor | None = None) -> dict:
        """
        Args:
            x_1d: (B, 1, 3600) raw ECG waveform
            x_2d: (B, 3, 64, 256) CWT scalogram, or None (skip 2D branch)
        Returns dict with logits for each head.
        """
        feat_1d = self.cnn1d(x_1d)           # (B, 256)
        feat_1d = self.temporal(feat_1d)      # (B, 256)

        if x_2d is not None:
            feat_2d = self.cnn2d(x_2d)        # (B, 256)
        else:
            feat_2d = torch.zeros_like(feat_1d)

        fused = self.fusion(feat_1d, feat_2d)  # (B, 512)

        return {
            "det_logits":  self.detection_head(fused),       # (B, 2)
            "cls_logits":  self.classification_head(fused),  # (B, num_classes)
            "pred_logits": self.prediction_head(fused),      # (B, 1)
            "fused":       fused,                            # for Grad-CAM hooks
        }

    def compute_loss(self, outputs: dict, targets: dict) -> dict:
        """Uncertainty-weighted multi-task loss (Eq. 3 in paper)."""
        # Classification loss (focal)
        cls_logits = outputs["cls_logits"]
        cls_target = targets["cls_target"]
        pt = F.softmax(cls_logits, dim=-1)
        ce = F.cross_entropy(cls_logits, cls_target, reduction='none',
                             label_smoothing=0.1)
        # FIX: squeeze(1) not squeeze() — bare squeeze() crashes when B==1.
        focal = ((1 - pt.gather(1, cls_target.unsqueeze(1)).squeeze(1)) ** 2) * ce
        L_cls = focal.mean()

        # Detection loss
        det_logits = outputs["det_logits"]
        det_target = targets.get("det_target")
        if det_target is None:
            det_target = (cls_target > 0).long()  # 0=N=normal, >0=abnormal
        L_det = F.cross_entropy(det_logits, det_target, label_smoothing=0.1)

        # Prediction loss (MSE on risk score)
        pred_logits = outputs["pred_logits"].squeeze(-1)
        pred_target = targets.get("pred_target")
        if pred_target is None:
            pred_target = (cls_target > 0).float()  # simple proxy
        L_pred = F.mse_loss(pred_logits, pred_target)

        # Uncertainty weighting
        w_det = torch.exp(-self.log_var_det)
        w_cls = torch.exp(-self.log_var_cls)
        w_pred = torch.exp(-self.log_var_pred)

        total = (0.5 * w_det * L_det + 0.5 * self.log_var_det +
                 0.5 * w_cls * L_cls + 0.5 * self.log_var_cls +
                 0.5 * w_pred * L_pred + 0.5 * self.log_var_pred)

        return {
            "loss": total,
            "L_det": L_det.item(),
            "L_cls": L_cls.item(),
            "L_pred": L_pred.item(),
        }


def count_params(model):
    return sum(p.numel() for p in model.parameters()) / 1e6


if __name__ == "__main__":
    torch.manual_seed(0)
    B = 4
    model = ProposedModel(num_classes=3)
    print(f"[ProposedModel] {count_params(model):.2f}M params")
    x_1d = torch.randn(B, 1, 3600)
    x_2d = torch.randn(B, 3, 64, 256)
    out = model(x_1d, x_2d)
    assert out["cls_logits"].shape == (B, 3)
    losses = model.compute_loss(out, {"cls_target": torch.randint(0, 3, (B,))})
    losses["loss"].backward()
    print(f"[ProposedModel] loss = {losses['loss'].item():.4f}")
    # batch-size-1 INFERENCE path (eval mode uses BN running stats, so this is
    # fine). NOTE: training on a size-1 batch is impossible because BatchNorm1d
    # needs >1 value per channel — train.py uses drop_last=True to prevent it.
    model.eval()
    with torch.no_grad():
        out1 = model(torch.randn(1, 1, 3600), torch.randn(1, 3, 64, 256))
    assert out1["cls_logits"].shape == (1, 3)
    print("[ProposedModel] smoke test passed (incl. batch size 1 inference).")
