"""Baselines for fair comparison on identical splits and preprocessing.

All three take the 10-sec window (B, 1, T=3600) directly. They do NOT
use MIL — they output a single 4-class softmax per window. For CPSC/
PTB-XL samples (rhythm/recording bags) we use the bag label as the
sample label, which is the standard practice we critique in the paper.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------- B1: ResNet-1D (Hannun-style) ---------------------------------

class _ResBlock1D(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_c, out_c, 17, stride=stride, padding=8)
        self.bn1 = nn.BatchNorm1d(out_c)
        self.conv2 = nn.Conv1d(out_c, out_c, 17, padding=8)
        self.bn2 = nn.BatchNorm1d(out_c)
        self.proj = nn.Conv1d(in_c, out_c, 1, stride=stride) if (in_c != out_c or stride != 1) else nn.Identity()

    def forward(self, x):
        h = F.relu(self.bn1(self.conv1(x)))
        h = self.bn2(self.conv2(h))
        return F.relu(h + self.proj(x))


class ResNet1D(nn.Module):
    """A 34-layer-ish 1D residual network."""
    def __init__(self, num_classes: int = 4):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, 32, 17, stride=2, padding=8), nn.BatchNorm1d(32), nn.ReLU(),
        )
        layers = []
        channels = [32, 64, 128, 192, 256]
        for i in range(len(channels) - 1):
            for j in range(4):
                in_c = channels[i] if j == 0 else channels[i + 1]
                stride = 2 if j == 0 else 1
                layers.append(_ResBlock1D(in_c, channels[i + 1], stride=stride))
        self.blocks = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Linear(channels[-1], num_classes),
        )

    def forward(self, x, **_ignored):
        return {"logits": self.head(self.blocks(self.stem(x)))}


# ---------- B2: CNN-LSTM (Oh 2018) ---------------------------------------

class CNNLSTM(nn.Module):
    def __init__(self, num_classes: int = 4):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 64, 7, padding=3), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 5, padding=2), nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(128, 128, 3, padding=1), nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(2),
        )
        self.lstm = nn.LSTM(128, 64, num_layers=2, batch_first=True,
                            bidirectional=True, dropout=0.2)
        self.head = nn.Linear(128, num_classes)

    def forward(self, x, **_ignored):
        h = self.cnn(x)                       # (B, C, T')
        h = h.transpose(1, 2)                 # (B, T', C)
        _, (h_n, _) = self.lstm(h)
        # concat forward+backward last states from last layer
        feat = torch.cat([h_n[-2], h_n[-1]], dim=-1)
        return {"logits": self.head(feat)}


# ---------- B3: ECGformer (transformer encoder on patches) ----------------

class ECGformer(nn.Module):
    def __init__(self, num_classes: int = 4, patch_size: int = 36, d_model: int = 128,
                 n_heads: int = 8, n_layers: int = 4):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Linear(patch_size, d_model)
        T_PATCHES = 3600 // patch_size
        self.pos = nn.Parameter(torch.zeros(1, T_PATCHES + 1, d_model))
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=256,
            dropout=0.1, batch_first=True, activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, num_classes))
        nn.init.trunc_normal_(self.pos, std=0.02)
        nn.init.trunc_normal_(self.cls, std=0.02)

    def forward(self, x, **_ignored):
        # x: (B, 1, T) -> patches (B, N, P) -> tokens (B, N, d)
        B, _, T = x.shape
        N = T // self.patch_size
        patches = x.reshape(B, 1, N, self.patch_size).squeeze(1)  # (B, N, P)
        tokens = self.proj(patches)                                # (B, N, d)
        cls = self.cls.expand(B, -1, -1)                           # (B, 1, d)
        tokens = torch.cat([cls, tokens], dim=1) + self.pos        # (B, N+1, d)
        out = self.transformer(tokens)
        return {"logits": self.head(out[:, 0])}


def build_baseline(name: str, num_classes: int = 4) -> nn.Module:
    name = name.lower()
    if name == "resnet1d":
        return ResNet1D(num_classes)
    if name == "cnnlstm":
        return CNNLSTM(num_classes)
    if name == "ecgformer":
        return ECGformer(num_classes)
    raise ValueError(f"unknown baseline: {name}")


if __name__ == "__main__":
    import sys
    x = torch.randn(4, 1, 3600)
    for name in ["resnet1d", "cnnlstm", "ecgformer"]:
        m = build_baseline(name)
        n_params = sum(p.numel() for p in m.parameters()) / 1e6
        out = m(x)["logits"]
        assert out.shape == (4, 4), f"{name}: {out.shape}"
        print(f"[{name}] {n_params:.2f}M params, out shape {tuple(out.shape)} OK")
