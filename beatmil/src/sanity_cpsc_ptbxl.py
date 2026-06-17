"""Sanity check for CPSC 2018 and PTB-XL loaders.

These are slow because of R-peak detection. Limit to first ~200 records
each for the sanity pass; full load happens later via cache_specs.py.

Run: python sanity_cpsc_ptbxl.py
"""

from __future__ import annotations
import sys
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent))
from cpsc import CPSC2018Loader
from ptbxl import PTBXLLoader
from unified_dataset import AAMI_CLASSES


def check(name: str, loader, expected_min: int):
    print(f"\n[{name}] listing samples (this can take several minutes)...")
    specs = loader.list_samples()
    print(f"[{name}] total samples: {len(specs):,}")

    if len(specs) < expected_min:
        print(f"[warn] expected at least {expected_min:,} — got {len(specs)}.")

    dist = Counter(s.bag_label for s in specs)
    print(f"[{name}] class distribution:")
    for cls_name, idx in AAMI_CLASSES.items():
        n = dist.get(idx, 0)
        pct = 100 * n / max(len(specs), 1)
        print(f"  {cls_name} (idx {idx}): {n:,}  ({pct:.1f}%)")

    # Beats per bag — should be 6-15 typically for 10-sec windows
    if specs:
        n_beats = [len(s.beat_positions_signal) for s in specs[:100]]
        import numpy as np
        print(f"[{name}] beats/bag: mean={np.mean(n_beats):.1f}, "
              f"min={min(n_beats)}, max={max(n_beats)}")


def main():
    data_root = Path.home() / "beatmil/data"

    cpsc_path = data_root / "cpsc2018"
    if cpsc_path.exists():
        check("CPSC", CPSC2018Loader(root=cpsc_path), expected_min=10_000)
    else:
        print(f"[skip] CPSC dir not found: {cpsc_path}")

    ptbxl_path = data_root / "ptbxl"
    if ptbxl_path.exists():
        check("PTB-XL", PTBXLLoader(root=ptbxl_path), expected_min=5_000)
    else:
        print(f"[skip] PTB-XL dir not found: {ptbxl_path}")


if __name__ == "__main__":
    main()
