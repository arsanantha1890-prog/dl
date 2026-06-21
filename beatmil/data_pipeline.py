"""
MIT-BIH inter-patient data pipeline (de Chazal et al. 2004 protocol).

This is the load-bearing correctness module for the resubmission. It produces
beat-level windows from the *only* database with genuine beat-level AAMI
annotations (MIT-BIH), under the inter-patient DS1/DS2 split, so that the
reported metrics measure real beat classification rather than agreement with a
label-propagation heuristic.

Design decisions and the reasons a reviewer would accept them:
  * DS1 (train+val) and DS2 (test) are the fixed de Chazal record lists. No
    record appears in both, so no patient leakage is possible by construction.
  * The 4 paced records (102, 104, 107, 217) are excluded per AAMI EC57, because
    paced beats are not the target population and inflate the "Q" class.
  * Class space defaults to N / S / V (3-class). F is merged into V (matching the
    "V+F" convention in the original paper); Q is dropped. This is the standard
    reduced AAMI task and the one that stabilised your training previously.
  * Each example is a WINDOW_SIZE-sample window centred on an R-peak, labelled by
    its centre beat. WINDOW_SIZE defaults to 3600 (10 s @ 360 Hz) to match the
    recreated ProposedModel input. Set WINDOW_SIZE=360 for a tighter, more
    conventional single-beat window if a reviewer prefers it.

The CWT scalogram generator here REPLACES the one in proposed_model.py, which
relies on scipy.signal.cwt / morlet2 — both removed in scipy >= 1.15.

Author: research pipeline, recreated/verified for BMEiCON 2026 resubmission.
"""

from __future__ import annotations

import os
import json
import hashlib
from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np
import wfdb
import pywt
from scipy.signal import butter, filtfilt


# =====================================================================
# de Chazal 2004 inter-patient split (the canonical record lists)
# =====================================================================
# 44 records total (the 48 MIT-BIH records minus the 4 paced ones).
DS1_RECORDS = [  # training pool (22 records)
    101, 106, 108, 109, 112, 114, 115, 116, 118, 119, 122,
    124, 201, 203, 205, 207, 208, 209, 215, 220, 223, 230,
]
DS2_RECORDS = [  # test (22 records) — NEVER used for training or model selection
    100, 103, 105, 111, 113, 117, 121, 123, 200, 202, 210,
    212, 213, 214, 219, 221, 222, 228, 231, 232, 233, 234,
]
PACED_EXCLUDED = [102, 104, 107, 217]  # excluded per AAMI EC57


# =====================================================================
# AAMI label harmonisation (beat symbol -> AAMI superclass)
# =====================================================================
SYMBOL_TO_AAMI = {
    # N — any beat that is normal or a bundle-branch / escape variant of normal
    "N": "N", "L": "N", "R": "N", "e": "N", "j": "N",
    # S — supraventricular ectopic
    "A": "S", "a": "S", "J": "S", "S": "S",
    # V — ventricular ectopic
    "V": "V", "E": "V",
    # F — fusion (ventricular + normal)
    "F": "F",
    # Q — unknown / paced / unclassifiable
    "/": "Q", "f": "Q", "Q": "Q",
}

# Final label space. F is folded into V; Q is excluded (mapped to None).
AAMI_TO_LABEL_3CLASS = {"N": 0, "S": 1, "V": 2, "F": 2, "Q": None}
LABEL_NAMES_3CLASS = ["N", "S", "V"]


# =====================================================================
# Config
# =====================================================================
@dataclass
class PipelineConfig:
    data_dir: str                      # directory with MIT-BIH .dat/.hea/.atr files
    window_size: int = 3600            # samples; 3600 = 10 s @ 360 Hz (matches model)
    target_fs: int = 360               # MIT-BIH native rate; no resampling needed
    bandpass_low: float = 0.5
    bandpass_high: float = 45.0
    val_fraction: float = 0.10         # carved out of DS1 *by record*, not by beat
    wavelet_baseline: bool = True      # remove baseline wander via wavelet detrend
    n_scales: int = 64                 # CWT scalogram height
    cwt_width: int = 256               # CWT scalogram width (resized)
    seed: int = 42

    def fingerprint(self) -> str:
        """Stable hash of config — used to invalidate the cache when settings change."""
        d = {k: v for k, v in asdict(self).items() if k != "data_dir"}
        return hashlib.md5(json.dumps(d, sort_keys=True).encode()).hexdigest()[:10]


# =====================================================================
# Signal preprocessing
# =====================================================================
def _butter_bandpass(sig: np.ndarray, fs: int, lo: float, hi: float) -> np.ndarray:
    nyq = 0.5 * fs
    b, a = butter(4, [lo / nyq, hi / nyq], btype="band")
    return filtfilt(b, a, sig).astype(np.float32)


def _wavelet_baseline_removal(sig: np.ndarray, wavelet: str = "db4",
                              level: int = 8) -> np.ndarray:
    """Remove low-frequency baseline wander by zeroing the coarsest approximation.

    This is the 'baseline corrected via wavelet decomposition' step from the paper.
    Level is capped to the signal length so short records don't error out.
    """
    max_level = pywt.dwt_max_level(len(sig), pywt.Wavelet(wavelet).dec_len)
    level = min(level, max_level)
    if level < 1:
        return sig
    coeffs = pywt.wavedec(sig, wavelet, level=level)
    coeffs[0] = np.zeros_like(coeffs[0])  # kill the approximation = remove baseline
    rec = pywt.waverec(coeffs, wavelet)
    return rec[: len(sig)].astype(np.float32)


def preprocess_signal(sig: np.ndarray, cfg: PipelineConfig) -> np.ndarray:
    """Full-record preprocessing: bandpass -> (optional) baseline removal.
    z-scoring is applied per-window later, not here."""
    sig = _butter_bandpass(sig, cfg.target_fs, cfg.bandpass_low, cfg.bandpass_high)
    if cfg.wavelet_baseline:
        sig = _wavelet_baseline_removal(sig)
    return sig


def zscore(window: np.ndarray) -> np.ndarray:
    mu = window.mean()
    sd = window.std()
    if sd < 1e-6:
        sd = 1.0
    return ((window - mu) / sd).astype(np.float32)


# =====================================================================
# CWT scalogram (pywt-based — replaces the broken scipy version)
# =====================================================================
def generate_cwt_scalogram(signal: np.ndarray, n_scales: int = 64,
                           width: int = 256, fs: int = 360) -> np.ndarray:
    """Morlet CWT scalogram, returned as (3, n_scales, width) pseudo-RGB.

    Uses pywt.cwt with the 'morl' wavelet. The three channels are identical so a
    pretrained ResNet-34 (3-channel input) can consume it directly.
    """
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
    return np.stack([power, power, power], axis=0)  # (3, n_scales, width)


# =====================================================================
# Record reading + windowing
# =====================================================================
def _select_lead_ii(record) -> np.ndarray:
    """Return lead II (MLII) if present, else the first channel.
    de Chazal uses modified limb lead II; most DS records carry MLII, but a few
    (e.g. 114) place it in a non-zero channel, so we select by name."""
    names = [s.upper() for s in record.sig_name]
    for target in ("MLII", "II", "ML2"):
        if target in names:
            return record.p_signal[:, names.index(target)].astype(np.float32)
    return record.p_signal[:, 0].astype(np.float32)


def extract_windows_from_record(rec_id: int, cfg: PipelineConfig):
    """Load one record, preprocess, and emit (window, label) pairs centred on beats.

    Returns: list of (window float32[window_size], label int), and a per-class
    Counter for bookkeeping.
    """
    path = os.path.join(cfg.data_dir, str(rec_id))
    record = wfdb.rdrecord(path)
    ann = wfdb.rdann(path, "atr")

    sig = _select_lead_ii(record)
    sig = preprocess_signal(sig, cfg)

    half = cfg.window_size // 2
    windows, labels = [], []
    counts = {name: 0 for name in LABEL_NAMES_3CLASS}

    for sample, symbol in zip(ann.sample, ann.symbol):
        aami = SYMBOL_TO_AAMI.get(symbol)
        if aami is None:
            continue  # non-beat annotation or symbol we don't map
        label = AAMI_TO_LABEL_3CLASS.get(aami)
        if label is None:
            continue  # Q dropped
        start, end = sample - half, sample + half
        if start < 0 or end > len(sig):
            continue  # incomplete window at record edge
        win = zscore(sig[start:end])
        if len(win) != cfg.window_size:
            continue
        windows.append(win)
        labels.append(label)
        counts[LABEL_NAMES_3CLASS[label]] += 1

    return windows, labels, counts


# =====================================================================
# Record-level train/val split (no beat leakage across val)
# =====================================================================
def split_train_val_records(cfg: PipelineConfig):
    """Carve a validation set out of DS1 BY RECORD so val beats never share a
    patient with train beats. Returns (train_records, val_records)."""
    rng = np.random.default_rng(cfg.seed)
    recs = DS1_RECORDS.copy()
    rng.shuffle(recs)
    n_val = max(1, int(round(len(recs) * cfg.val_fraction)))
    val = sorted(recs[:n_val])
    train = sorted(recs[n_val:])
    return train, val


# =====================================================================
# Build + cache the whole dataset
# =====================================================================
def build_split(records: list[int], cfg: PipelineConfig):
    """Process a list of records into stacked arrays."""
    all_w, all_y = [], []
    total_counts = {name: 0 for name in LABEL_NAMES_3CLASS}
    for rec_id in records:
        w, y, counts = extract_windows_from_record(rec_id, cfg)
        all_w.extend(w)
        all_y.extend(y)
        for k, v in counts.items():
            total_counts[k] += v
    X = np.asarray(all_w, dtype=np.float32)
    Y = np.asarray(all_y, dtype=np.int64)
    return X, Y, total_counts


def build_and_cache(cfg: PipelineConfig, cache_dir: str = "./cache",
                    verbose: bool = True) -> dict:
    """Build train/val/test arrays and cache them as a single .npz keyed by config.

    Returns a dict with X_train, y_train, X_val, y_val, X_test, y_test, and the
    record assignments. Reuses the cache if the config fingerprint matches.
    """
    os.makedirs(cache_dir, exist_ok=True)
    fp = cfg.fingerprint()
    cache_path = os.path.join(cache_dir, f"mitbih_dechazal_{fp}.npz")

    if os.path.exists(cache_path):
        if verbose:
            print(f"[cache] loading {cache_path}")
        d = np.load(cache_path, allow_pickle=True)
        return {k: d[k] for k in d.files}

    train_recs, val_recs = split_train_val_records(cfg)
    test_recs = DS2_RECORDS

    if verbose:
        print(f"[build] train records ({len(train_recs)}): {train_recs}")
        print(f"[build] val   records ({len(val_recs)}): {val_recs}")
        print(f"[build] test  records ({len(test_recs)}): {test_recs}")

    X_train, y_train, c_train = build_split(train_recs, cfg)
    X_val,   y_val,   c_val   = build_split(val_recs, cfg)
    X_test,  y_test,  c_test  = build_split(test_recs, cfg)

    out = {
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val, "y_val": y_val,
        "X_test": X_test, "y_test": y_test,
        "train_records": np.asarray(train_recs),
        "val_records": np.asarray(val_recs),
        "test_records": np.asarray(test_recs),
        "label_names": np.asarray(LABEL_NAMES_3CLASS),
        "config": np.asarray([json.dumps(asdict(cfg))], dtype=object),
    }
    np.savez_compressed(cache_path, **out)
    if verbose:
        print(f"[cache] wrote {cache_path}")
        print(f"[counts] train {c_train}  val {c_val}  test {c_test}")
    return out


# =====================================================================
# Integrity self-check (run this before you trust any result)
# =====================================================================
def verify_split_integrity(cfg: Optional[PipelineConfig] = None) -> bool:
    """Assert the properties a reviewer cares about. Returns True if all pass."""
    ok = True

    # 1. No record appears in both DS1 and DS2.
    overlap = set(DS1_RECORDS) & set(DS2_RECORDS)
    if overlap:
        print(f"[FAIL] DS1/DS2 overlap: {overlap}"); ok = False
    else:
        print("[ok] DS1 and DS2 are disjoint (no patient leakage by construction)")

    # 2. Paced records are excluded from both.
    bad = (set(DS1_RECORDS) | set(DS2_RECORDS)) & set(PACED_EXCLUDED)
    if bad:
        print(f"[FAIL] paced records present: {bad}"); ok = False
    else:
        print("[ok] paced records (102,104,107,217) excluded per AAMI EC57")

    # 3. Expected record count.
    n = len(DS1_RECORDS) + len(DS2_RECORDS)
    if n != 44:
        print(f"[FAIL] expected 44 records, got {n}"); ok = False
    else:
        print(f"[ok] 44 records total (22 DS1 + 22 DS2)")

    # 4. If a config is given, verify the val split is record-disjoint from train.
    if cfg is not None:
        train, val = split_train_val_records(cfg)
        if set(train) & set(val):
            print("[FAIL] train/val record overlap"); ok = False
        else:
            print(f"[ok] val carved by record ({len(val)} records), disjoint from train")

    print("[PASS] all integrity checks passed" if ok else "[FAILED] fix above before training")
    return ok


# =====================================================================
# Class weights for balanced training
# =====================================================================
def compute_class_weights(y: np.ndarray, n_classes: int = 3) -> np.ndarray:
    """Inverse-frequency weights, normalised to mean 1.0. Use for focal-loss alpha
    or a WeightedRandomSampler."""
    counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    counts[counts == 0] = 1.0
    w = counts.sum() / (n_classes * counts)
    return (w / w.mean()).astype(np.float32)


if __name__ == "__main__":
    # Integrity checks run without any data present.
    verify_split_integrity(PipelineConfig(data_dir="."))
