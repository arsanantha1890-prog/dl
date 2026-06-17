"""Cache the SampleSpec lists for MIT-BIH, CPSC 2018, and PTB-XL.

R-peak detection on CPSC and PTB-XL is slow (30-60 minutes total).
This caches everything once so subsequent runs are instant.

Run: python cache_specs.py
"""

from __future__ import annotations
import sys
import pickle
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from mitbih_loader import MITBIHLoader
from cpsc import CPSC2018Loader
from ptbxl import PTBXLLoader


def main():
    data_root = Path.home() / "beatmil/data"
    cache_dir = Path.home() / "beatmil/outputs/cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    tasks = [
        ("mitbih", MITBIHLoader,    data_root / "mitbih"),
        ("cpsc",   CPSC2018Loader,  data_root / "cpsc2018"),
        ("ptbxl",  PTBXLLoader,     data_root / "ptbxl"),
    ]

    for name, cls, path in tasks:
        cache_path = cache_dir / f"{name}_specs.pkl"
        if cache_path.exists():
            with cache_path.open("rb") as f:
                specs = pickle.load(f)
            print(f"[{name}] cached: {len(specs):,} samples (from {cache_path})")
            continue

        if not path.exists():
            print(f"[{name}] SKIP — directory not found: {path}")
            continue

        print(f"[{name}] building specs from {path}...")
        t0 = time.time()
        specs = cls(root=path).list_samples()
        elapsed = time.time() - t0
        with cache_path.open("wb") as f:
            pickle.dump(specs, f)
        print(f"[{name}] {len(specs):,} samples in {elapsed:.0f}s → {cache_path}")


if __name__ == "__main__":
    main()
