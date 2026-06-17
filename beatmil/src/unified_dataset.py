"""
Unified ECG dataset combining MIT-BIH, CPSC 2018, and PTB-XL at their
native label granularities.

Output schema (per sample dict):
    'x':                Tensor (1, T), T=3600  (lead II, resampled to 360 Hz)
    'beat_positions':   Tensor (N_max,) int   (indices into the BACKBONE-TIME
                                              feature axis; padding = -1)
    'bag_target':       int  (AAMI class 0..3)
    'has_beat_labels':  float (1.0 if MIT-BIH else 0.0)
    'beat_targets':     Tensor (N_max,) int   (-1 if no beat label or padding)
    'database':         str   ('mitbih' | 'cpsc' | 'ptbxl')
    'record_id':        str

Notes for whoever implements this for real:
    - For MIT-BIH, R-peak positions come from the wfdb annotation file.
    - For CPSC and PTB-XL, R-peaks are detected via Pan-Tompkins
      (e.g. `wfdb.processing.gqrs_detect` or `neurokit2.ecg_peaks`).
    - 'beat_positions' are in BACKBONE-TIME, not signal-time. The backbone
      downsamples by ~8 (three MaxPool1d(2) layers in the first three
      residual blocks), so a sample-time R-peak at index R maps to
      backbone-time index R // 8.
    - LUDB is loaded by a separate module (`ludb.py`) and is NEVER
      combined with these three.

This file is a scaffold — fill in TODO blocks during Day 2-4 of Week 1.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch.utils.data import Dataset

# ---------- AAMI label mapping ------------------------------------------

AAMI_CLASSES: dict[str, int] = {
    "N": 0,    # Normal
    "S": 1,    # Supraventricular ectopic
    "V": 2,    # Ventricular ectopic (+ Fusion folded in)
    "Q": 3,    # Unknown / paced
}
K = len(AAMI_CLASSES)

# MIT-BIH symbol -> AAMI class (de Chazal 2004 mapping)
MITBIH_TO_AAMI: dict[str, str] = {
    "N": "N", "L": "N", "R": "N", "e": "N", "j": "N",
    "A": "S", "a": "S", "J": "S", "S": "S",
    "V": "V", "E": "V", "F": "V",                 # F folded into V
    "/": "Q", "f": "Q", "Q": "Q",
}

# CPSC 2018 rhythm class -> AAMI bag class (paper-defined harmonisation)
CPSC_TO_AAMI: dict[str, str] = {
    "Normal":  "N",
    "AF":      "S",   # atrial fibrillation
    "I-AVB":   "N",   # 1st-degree AV block, normal beats with long PR
    "LBBB":    "N",   # conduction-modified normal — debatable, see Sec III.A
    "RBBB":    "N",
    "PAC":     "S",
    "PVC":     "V",
    "STD":     "N",   # ST depression — not arrhythmia per se
    "STE":     "N",   # ST elevation — not arrhythmia per se
}

# PTB-XL SCP-ECG diagnostic statement -> AAMI bag class
# Use the 'diagnostic_class' field (NORM / MI / STTC / CD / HYP) plus
# rhythm-specific codes (AFIB, AFLT, PVC, PAC, etc.).
PTBXL_TO_AAMI: dict[str, str] = {
    "NORM":  "N",
    "AFIB":  "S",
    "AFLT":  "S",
    "PAC":   "S",
    "SVTAC": "S",
    "PVC":   "V",
    "BIGU":  "V",      # bigeminy is PVC pattern
    "TRIGU": "V",
    "PACE":  "Q",
    # MI / STTC / CD / HYP are morphology diagnoses, not rhythm. We treat
    # the rhythm as N unless an explicit rhythm code overrides it.
}


# ---------- per-database loader (sketch) --------------------------------

@dataclass
class SampleSpec:
    record_id: str
    database: Literal["mitbih", "cpsc", "ptbxl"]
    window_start: int            # in samples @ 360 Hz
    bag_label: int               # AAMI class 0..3
    beat_positions_signal: np.ndarray  # (N,) int, sample-time indices
    beat_labels: np.ndarray | None     # (N,) int 0..3 OR None
    patient_id: str              # for inter-patient splitting


class BaseECGLoader:
    """Abstract base — each concrete loader returns a list of SampleSpec."""
    def __init__(self, root: Path):
        self.root = Path(root)

    def list_samples(self) -> list[SampleSpec]:
        raise NotImplementedError

    def load_window(self, spec: SampleSpec) -> np.ndarray:
        """Return raw lead-II samples for spec.window_start : +T, @ 360 Hz."""
        raise NotImplementedError


# ---------- MIT-BIH (beat granularity) ----------------------------------

class MITBIHLoader(BaseECGLoader):
    """One sample = one 10-sec window R-peak-centered on a target beat.
    The bag label = the AAMI class of the center beat; beat labels are
    available for every R-peak in the window."""
    def list_samples(self) -> list[SampleSpec]:
        # TODO Day 2:
        #   for each record in self.root.glob("*.dat"):
        #     load via wfdb.rdrecord -> resample 360 Hz -> lead II
        #     load via wfdb.rdann (extension="atr")
        #     for each annotation beat:
        #       map symbol via MITBIH_TO_AAMI; skip if no mapping
        #       window = [beat - 1800, beat + 1800]
        #       gather other beats falling inside window
        #       yield SampleSpec(...)
        raise NotImplementedError


# ---------- CPSC 2018 (rhythm granularity) ------------------------------

class CPSC2018Loader(BaseECGLoader):
    """One sample = one 10-sec sliding window (50% overlap) over the record.
    The bag label = the rhythm class of the entire recording. Beat positions
    are detected with Pan-Tompkins; beat labels are unavailable (set to -1)."""
    def list_samples(self) -> list[SampleSpec]:
        # TODO Day 3:
        #   load 'REFERENCE.csv' for rhythm labels
        #   for each record:
        #     load .mat, take lead II, resample to 360 Hz
        #     for window_start in range(0, len-T, T//2):  # 50% overlap
        #       detect R-peaks within the window
        #       yield SampleSpec(beat_labels=None, ...)
        raise NotImplementedError


# ---------- PTB-XL (recording granularity) ------------------------------

class PTBXLLoader(BaseECGLoader):
    """One sample = one 10-sec sliding window over the record. The bag label
    = recording-level AAMI class derived from SCP-ECG statements (rhythm code
    overrides morphology codes)."""
    def list_samples(self) -> list[SampleSpec]:
        # TODO Day 3:
        #   load 'ptbxl_database.csv' and 'scp_statements.csv'
        #   for each record:
        #     get scp_codes dict, find any rhythm code, map via PTBXL_TO_AAMI
        #     fall back to 'N' if only morphology codes (already conservative)
        #     load .dat via wfdb, lead II, resample to 360 Hz
        #     for window_start in range(0, len-T, T//2):
        #       detect R-peaks within the window
        #       yield SampleSpec(beat_labels=None, ...)
        raise NotImplementedError


# ---------- unified dataset ---------------------------------------------

T_WINDOW = 3600          # samples = 10 s at 360 Hz
BACKBONE_DOWNSAMPLE = 8  # three MaxPool1d(2) layers in backbone
N_MAX = 20               # max beats per 10-sec window (typical: 6-15)


def signal_to_backbone_time(idx: int) -> int:
    """Map sample-time R-peak index to backbone-time index."""
    return idx // BACKBONE_DOWNSAMPLE


class UnifiedECGDataset(Dataset):
    """Combines per-database SampleSpec lists into one Dataset for training."""

    def __init__(
        self,
        specs: list[SampleSpec],
        loaders: dict[str, BaseECGLoader],
        augment: bool = False,
        augment_fn=None,
    ):
        self.specs = specs
        self.loaders = loaders
        self.augment = augment
        self.augment_fn = augment_fn

    def __len__(self) -> int:
        return len(self.specs)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor | float | str]:
        spec = self.specs[i]
        loader = self.loaders[spec.database]

        # 1. load and (optionally) augment the raw 10-sec window
        x = loader.load_window(spec)                          # (T,)
        assert x.shape == (T_WINDOW,), f"bad window shape {x.shape}"
        # z-score normalisation (per-window)
        x = (x - x.mean()) / (x.std() + 1e-6)
        if self.augment and self.augment_fn is not None:
            x = self.augment_fn(x)
        x = torch.from_numpy(x).float().unsqueeze(0)          # (1, T)

        # 2. beat positions: map sample-time to backbone-time, then pad
        bt = [signal_to_backbone_time(p - spec.window_start)
              for p in spec.beat_positions_signal
              if 0 <= p - spec.window_start < T_WINDOW]
        bt = bt[:N_MAX]                                       # truncate
        beat_positions = torch.full((N_MAX,), -1, dtype=torch.long)
        for k, v in enumerate(bt):
            beat_positions[k] = v

        # 3. beat labels (MIT-BIH only); -1 elsewhere or for padded slots
        has_beat_labels = float(spec.beat_labels is not None)
        beat_targets = torch.full((N_MAX,), -1, dtype=torch.long)
        if spec.beat_labels is not None:
            n = min(len(spec.beat_labels), len(bt))
            beat_targets[:n] = torch.from_numpy(spec.beat_labels[:n].astype(np.int64))

        return {
            "x": x,
            "beat_positions": beat_positions,
            "bag_target": int(spec.bag_label),
            "has_beat_labels": has_beat_labels,
            "beat_targets": beat_targets,
            "database": spec.database,
            "record_id": spec.record_id,
            "patient_id": spec.patient_id,
        }


# ---------- patient-level splitting -------------------------------------

def patient_level_split(
    specs: list[SampleSpec],
    ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 42,
) -> dict[str, list[SampleSpec]]:
    """Split specs into train/val/test such that no patient appears in
    more than one partition. Splits within each database independently
    to maintain class balance per database."""
    import random
    rng = random.Random(seed)
    out: dict[str, list[SampleSpec]] = {"train": [], "val": [], "test": []}

    # group by (database, patient_id)
    by_db: dict[str, dict[str, list[SampleSpec]]] = {}
    for s in specs:
        by_db.setdefault(s.database, {}).setdefault(s.patient_id, []).append(s)

    for db, by_patient in by_db.items():
        patients = list(by_patient.keys())
        rng.shuffle(patients)
        n = len(patients)
        n_train = int(n * ratios[0])
        n_val = int(n * ratios[1])
        split = {
            "train": patients[:n_train],
            "val": patients[n_train:n_train + n_val],
            "test": patients[n_train + n_val:],
        }
        for part, plist in split.items():
            for p in plist:
                out[part].extend(by_patient[p])

    return out


def lodo_split(
    specs: list[SampleSpec],
    held_out_db: Literal["mitbih", "cpsc", "ptbxl"],
) -> dict[str, list[SampleSpec]]:
    """Leave-one-database-out: train+val on the two non-held-out databases
    (using their own patient-level splits), test on the held-out database."""
    in_specs = [s for s in specs if s.database != held_out_db]
    out_specs = [s for s in specs if s.database == held_out_db]
    train_val = patient_level_split(in_specs, ratios=(0.85, 0.15, 0.0))
    return {
        "train": train_val["train"],
        "val": train_val["val"],
        "test": out_specs,
    }


# ---------- helper: write a JSON split manifest -------------------------

def save_split_manifest(splits: dict[str, list[SampleSpec]], path: Path) -> None:
    manifest = {
        part: [{"database": s.database, "record_id": s.record_id,
                "patient_id": s.patient_id, "window_start": s.window_start}
               for s in plist]
        for part, plist in splits.items()
    }
    path.write_text(json.dumps(manifest, indent=2))
