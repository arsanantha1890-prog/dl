"""
Beat-MIL: the full model.

Pipeline:
    Raw ECG segment (B, 1, T)
        -> 1D-CNN with SE blocks (per-segment feature map)
        -> R-peak-anchored beat extraction (B, N, d) beat embeddings
        -> per-beat Bi-LSTM + self-attention refinement
        -> three outputs:
              (a) beat-head logits per beat              (B, N, K)
              (b) MIL gated-attention pooled bag rep H   (B, d)
              (c) evidential bag head: evidence e(H)     (B, K)

Loss (called separately, see losses/):
    L_bag         = evidential_loss(e, y_bag) on every sample
    L_beat        = focal cross-entropy on (b) for MIT-BIH samples only
    L_consistency = symmetric KL between bag softmax and alpha-pooled
                    beat softmax, for MIT-BIH samples only

Notes for reviewers (and for us, to keep honest):
    - We drop the 2D-CWT scalogram branch from the prior architecture.
      Old ablation showed +0.2% intra-DB for +23M params; the cost is
      not justified. Kept as an ablation in Sec V.C.
    - The 2-layer Bi-LSTM is small (128/dir), and self-attention has
      8 heads of dim 32. The backbone fits comfortably in <4M params.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from mil_pooling import GatedAttentionMIL
from evidential import EvidentialHead


# ---------- backbone blocks ----------------------------------------------

class SEBlock1D(nn.Module):
    """Squeeze-and-Excitation channel attention for 1D feature maps."""
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden), nn.GELU(),
            nn.Linear(hidden, channels), nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, C, T)
        s = x.mean(dim=-1)                # (B, C)
        s = self.fc(s).unsqueeze(-1)      # (B, C, 1)
        return x * s


class ResidualConvBlock(nn.Module):
    def __init__(self, in_c: int, out_c: int, kernel: int, dropout: float = 0.2):
        super().__init__()
        pad = kernel // 2
        self.conv1 = nn.Conv1d(in_c, out_c, kernel, padding=pad)
        self.bn1 = nn.BatchNorm1d(out_c)
        self.conv2 = nn.Conv1d(out_c, out_c, kernel, padding=pad)
        self.bn2 = nn.BatchNorm1d(out_c)
        self.se = SEBlock1D(out_c)
        self.drop = nn.Dropout(dropout)
        self.proj = nn.Conv1d(in_c, out_c, 1) if in_c != out_c else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.gelu(self.bn1(self.conv1(x)))
        h = self.drop(h)
        h = self.bn2(self.conv2(h))
        h = self.se(h)
        return F.gelu(h + self.proj(x))


# ---------- backbone ------------------------------------------------------

class ECGBackbone(nn.Module):
    """1D-CNN producing a sequence of features the per-beat extractor pools from."""
    def __init__(self, base_channels: int = 32):
        super().__init__()
        kernels = [15, 11, 7, 5, 5, 3]
        channels = [base_channels, 64, 96, 128, 192, 256, 256]
        layers = []
        for i, k in enumerate(kernels):
            layers.append(ResidualConvBlock(channels[i], channels[i + 1], k))
            if i < 3:  # downsample by 2 in the first 3 blocks (T: 3600->450)
                layers.append(nn.MaxPool1d(2))
        # input proj
        self.stem = nn.Conv1d(1, base_channels, 7, padding=3)
        self.blocks = nn.Sequential(*layers)
        self.out_channels = channels[-1]  # 256

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, T)
        h = self.stem(x)
        return self.blocks(h)  # (B, 256, T')


# ---------- beat extractor ------------------------------------------------

class BeatExtractor(nn.Module):
    """Given (B, C, T') backbone features and per-sample R-peak indices,
    pool a small window around each R-peak to produce per-beat embeddings.
    """
    def __init__(self, channels: int, window: int = 16):
        super().__init__()
        self.window = window
        # additional refinement: small bidirectional LSTM over beats + self-attention
        self.beat_lstm = nn.LSTM(channels, 128, num_layers=2, batch_first=True,
                                 bidirectional=True, dropout=0.1)
        self.beat_attn = nn.MultiheadAttention(
            embed_dim=256, num_heads=8, dropout=0.1, batch_first=True,
        )
        self.ln = nn.LayerNorm(256)

    def forward(
        self,
        feats: torch.Tensor,         # (B, C, T')
        beat_positions: torch.Tensor,  # (B, N_max) indices into T', -1 = padding
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, C, Tp = feats.shape
        N = beat_positions.shape[1]
        half = self.window // 2

        # gather a window of size `window` around each beat position
        # we average within the window — cheap and robust
        idx = beat_positions.clamp(min=0)                          # (B, N)
        # build offsets [-half, ..., +half-1]
        offsets = torch.arange(-half, self.window - half,
                               device=feats.device)                 # (window,)
        # broadcast: (B, N, window) absolute indices
        abs_idx = idx.unsqueeze(-1) + offsets                       # (B, N, W)
        abs_idx = abs_idx.clamp(0, Tp - 1)
        # gather: feats (B, C, T') -> (B, C, N*W) -> (B, N, W, C)
        flat = abs_idx.reshape(B, -1)                                # (B, N*W)
        gathered = feats.gather(
            dim=2, index=flat.unsqueeze(1).expand(-1, C, -1),
        )                                                            # (B, C, N*W)
        gathered = gathered.reshape(B, C, N, self.window).permute(0, 2, 3, 1)
        beat_feats = gathered.mean(dim=2)                            # (B, N, C)

        # mask invalid beats (positions = -1 before clamp)
        mask = (beat_positions >= 0).float()                         # (B, N)

        # refine with Bi-LSTM and self-attention
        lstm_out, _ = self.beat_lstm(beat_feats)                     # (B, N, 256)
        # key_padding_mask: True = ignore
        kpm = mask == 0
        attn_out, _ = self.beat_attn(
            lstm_out, lstm_out, lstm_out, key_padding_mask=kpm
        )
        z = self.ln(lstm_out + attn_out)                             # (B, N, 256)
        return z, mask


# ---------- full Beat-MIL model -------------------------------------------

class BeatMIL(nn.Module):
    def __init__(self, num_classes: int = 4):
        super().__init__()
        self.backbone = ECGBackbone(base_channels=32)
        # the backbone outputs 256-channel features; we feed those into the
        # beat extractor whose LSTM produces 256-d beat embeddings
        self.beats = BeatExtractor(channels=self.backbone.out_channels)
        # MIL pooling and heads operate in 256-d space
        self.mil = GatedAttentionMIL(d_in=256, d_hidden=128)
        self.head_dropout = nn.Dropout(0.3)            # regularize heads
        self.beat_head = nn.Linear(256, num_classes)   # per-beat logits
        self.bag_head = EvidentialHead(d_in=256, num_classes=num_classes)
        self.num_classes = num_classes

    def forward(
        self,
        x: torch.Tensor,                # (B, 1, T) raw ECG window
        beat_positions: torch.Tensor,   # (B, N_max) beat indices in backbone-time
    ) -> dict[str, torch.Tensor]:
        feats = self.backbone(x)
        z, mask = self.beats(feats, beat_positions)
        H, alpha = self.mil(z, mask)

        beat_logits = self.beat_head(self.head_dropout(z))   # (B, N, K)
        evidence = self.bag_head(self.head_dropout(H))       # (B, K)

        return {
            "beat_logits": beat_logits,
            "bag_evidence": evidence,
            "bag_repr": H,
            "alpha": alpha,
            "mask": mask,
        }


# ---------- smoke ---------------------------------------------------------

def _smoke_test() -> None:
    torch.manual_seed(0)
    B, T, N = 2, 3600, 12
    model = BeatMIL(num_classes=4)
    x = torch.randn(B, 1, T)

    # backbone downsamples 3 times (3600 -> 450); pick beat positions in that range
    # in real code these come from R-peak detection scaled to backbone time
    beat_positions = torch.randint(0, 450, (B, N))
    beat_positions[1, -3:] = -1   # padding for last 3 beats of bag 1

    out = model(x, beat_positions)
    assert out["beat_logits"].shape == (B, N, 4)
    assert out["bag_evidence"].shape == (B, 4)
    assert out["alpha"].shape == (B, N)
    assert (out["bag_evidence"] >= 0).all(), "evidence must be non-negative"

    # backward through everything
    loss = out["bag_evidence"].sum() + out["beat_logits"].sum()
    loss.backward()

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[BeatMIL] params = {n_params/1e6:.2f}M, "
          f"alpha (bag 1) = {out['alpha'][1].detach().numpy().round(3)}")
    # padding positions should have zero attention
    assert torch.allclose(out["alpha"][1, -3:], torch.zeros(3), atol=1e-6), \
        "padding attention not zero"
    print("[BeatMIL] smoke test passed.")


if __name__ == "__main__":
    _smoke_test()
