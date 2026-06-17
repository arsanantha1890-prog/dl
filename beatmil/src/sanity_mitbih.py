"""Sanity check for the MIT-BIH loader.

Verifies:
    - All records load without crashing
    - Sample count is roughly 100k
    - Class N is dominant (~84%)
    - Saves a 4-panel figure showing one window per class

Run: python sanity_mitbih.py
"""

from __future__ import annotations
import sys
from pathlib import Path
from collections import Counter

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless safe
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from mitbih_loader import MITBIHLoader
from unified_dataset import UnifiedECGDataset, AAMI_CLASSES


def main():
    data_root = Path.home() / "beatmil/data/mitbih"
    if not data_root.exists():
        print(f"[error] {data_root} not found. Symlink your MIT-BIH directory there.")
        sys.exit(1)

    print(f"[sanity] loading from {data_root}")
    loader = MITBIHLoader(root=data_root)
    specs = loader.list_samples()
    print(f"[sanity] total MIT-BIH samples: {len(specs):,}")

    if len(specs) < 50_000:
        print(f"[warn] expected ~100k samples — only got {len(specs)}. "
              "Check MITBIH_TO_AAMI mapping and MLII lead detection.")

    dist = Counter(s.bag_label for s in specs)
    print("\n[sanity] class distribution:")
    for name, idx in AAMI_CLASSES.items():
        n = dist.get(idx, 0)
        pct = 100 * n / max(len(specs), 1)
        print(f"  {name} (idx {idx}): {n:,}  ({pct:.1f}%)")

    pct_N = 100 * dist.get(0, 0) / max(len(specs), 1)
    if not (75 < pct_N < 92):
        print(f"\n[warn] class N is {pct_N:.1f}% — expected ~84%. "
              "Possible mapping bug.")

    # Plot one sample per class
    ds = UnifiedECGDataset(specs, {"mitbih": loader}, augment=False)
    fig, axes = plt.subplots(4, 1, figsize=(12, 8))
    shown: set[int] = set()
    for i in range(min(len(ds), 200_000)):
        s = ds[i]
        cls = s["bag_target"]
        if cls in shown:
            continue
        ax = axes[len(shown)]
        ax.plot(s["x"][0].numpy(), linewidth=0.7)
        for bp in s["beat_positions"].numpy():
            if bp >= 0:
                # backbone time -> signal time approx
                ax.axvline(bp * 8, color="red", alpha=0.3, linewidth=0.5)
        cn = [k for k, v in AAMI_CLASSES.items() if v == cls][0]
        ax.set_title(f"Class {cn} (idx {cls}) — record {s['record_id']}")
        ax.set_xlim(0, len(s["x"][0]))
        shown.add(cls)
        if len(shown) == 4:
            break

    plt.tight_layout()
    out = Path.home() / "beatmil/figures/sanity_mitbih.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=120)
    print(f"\n[sanity] figure saved to {out}")
    print("[sanity] open it and verify R-peaks (red lines) align with QRS complexes.")


if __name__ == "__main__":
    main()
