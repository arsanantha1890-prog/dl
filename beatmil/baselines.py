"""
Baseline architectures for the SOTA comparison the JCSSE reviewer asked for.

The point of these is fairness: every baseline consumes the *same* (B,1,3600)
windows, is trained on the *same* DS1 records, and is evaluated on the *same*
DS2 test set as the proposed model. That removes the usual confound where a
"comparison" actually compares split protocols rather than architectures. Any
accuracy difference is therefore attributable to the model.

Two distinct, well-known designs are provided (not trimmed copies of the
proposed model, which would make the comparison circular):

  * CNN1D_ECG    — a deep 1-D residual CNN in the Hannun/Acharya lineage
                   (raw-waveform CNN, no recurrence, no attention).
  * CNNLSTM_ECG  — a CNN feature extractor feeding an LSTM, in the Oh et al.
                   lineage (the standard CNN-LSTM hybrid baseline).

Both expose the same interface: forward(x) -> logits (B, num_classes).
The training harness wraps them with a shared focal-loss objective so the only
thing that differs across runs is the architecture.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------
# Baseline 1: deep 1-D residual CNN (Hannun/Acharya lineage)
# ---------------------------------------------------------------------
class _BasicRes1D(nn.Module):
    def __init__(self, in_c, out_c, k=7, stride=1, dropout=0.2):
        super().__init__()
        pad = k // 2
        self.conv = nn.Sequential(
            nn.Conv1d(in_c, out_c, k, stride=stride, padding=pad),
            nn.BatchNorm1d(out_c), nn.ReLU(), nn.Dropout(dropout),
            nn.Conv1d(out_c, out_c, k, padding=pad), nn.BatchNorm1d(out_c),
        )
        self.skip = (nn.Conv1d(in_c, out_c, 1, stride=stride)
                     if in_c != out_c or stride != 1 else nn.Identity())
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(self.conv(x) + self.skip(x))


class CNN1D_ECG(nn.Module):
    """Deep 1-D residual CNN on raw single-lead ECG. ~ Hannun/Acharya style."""
    def __init__(self, num_classes=3):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, 32, 15, stride=2, padding=7),
            nn.BatchNorm1d(32), nn.ReLU(),
        )
        chans = [32, 64, 64, 128, 128, 256, 256]
        blocks = []
        in_c = 32
        for i, c in enumerate(chans):
            stride = 2 if i % 2 == 0 else 1
            blocks.append(_BasicRes1D(in_c, c, k=7, stride=stride))
            in_c = c
        self.blocks = nn.Sequential(*blocks)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.head(self.blocks(self.stem(x)))


# ---------------------------------------------------------------------
# Baseline 2: CNN-LSTM hybrid (Oh et al. lineage)
# ---------------------------------------------------------------------
class CNNLSTM_ECG(nn.Module):
    """CNN feature extractor -> Bi-LSTM -> classifier. Standard hybrid baseline."""
    def __init__(self, num_classes=3, lstm_hidden=128):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 32, 11, stride=2, padding=5), nn.BatchNorm1d(32), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 9, stride=1, padding=4), nn.BatchNorm1d(64), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 7, stride=1, padding=3), nn.BatchNorm1d(128), nn.ReLU(),
            nn.MaxPool1d(2),
        )
        self.lstm = nn.LSTM(128, lstm_hidden, num_layers=2, batch_first=True,
                            bidirectional=True, dropout=0.2)
        self.head = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        f = self.features(x)              # (B, 128, T')
        f = f.transpose(1, 2)            # (B, T', 128)
        out, _ = self.lstm(f)            # (B, T', 2*hidden)
        pooled = out.mean(dim=1)         # temporal average pooling
        return self.head(pooled)


# ---------------------------------------------------------------------
# Shared focal-loss objective so baselines train under the same loss
# the proposed model uses for its classification head.
# ---------------------------------------------------------------------
def focal_loss(logits, target, alpha=None, gamma=2.0, label_smoothing=0.1):
    """Multi-class focal loss with optional per-class alpha weighting."""
    ce = F.cross_entropy(logits, target, weight=alpha, reduction="none",
                         label_smoothing=label_smoothing)
    pt = F.softmax(logits, dim=-1).gather(1, target.unsqueeze(1)).squeeze(1)
    return (((1 - pt) ** gamma) * ce).mean()


def count_params(model):
    return sum(p.numel() for p in model.parameters()) / 1e6


BASELINE_REGISTRY = {
    "cnn1d": CNN1D_ECG,
    "cnnlstm": CNNLSTM_ECG,
}


if __name__ == "__main__":
    x = torch.randn(4, 1, 3600)
    for name, cls in BASELINE_REGISTRY.items():
        m = cls(num_classes=3)
        out = m(x)
        assert out.shape == (4, 3), f"{name}: {out.shape}"
        loss = focal_loss(out, torch.randint(0, 3, (4,)))
        loss.backward()
        print(f"[ok] {name:9s} {count_params(m):5.2f}M params  out={tuple(out.shape)}  loss={loss.item():.3f}")
    print("[BASELINES SMOKE TEST PASSED]")
