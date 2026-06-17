"""LUDB loader — used ONLY for XAI evaluation (Sec V.F of the paper).

LUDB has 200 records with expert annotations of P-wave onset/peak/offset,
QRS onset/peak/offset, and T-wave onset/peak/offset. We use lead II,
resample to 360 Hz, and build binary masks for each anatomical region.
These masks serve as IoU/Dice ground truth for Grad-CAM saliency.

LUDB IS NEVER COMBINED WITH TRAINING DATA.
"""

from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass
import numpy as np
import wfdb
from scipy.signal import resample_poly


@dataclass
class LUDBRecord:
    record_id: str
    signal: np.ndarray          # (T,) lead II at 360 Hz
    p_mask: np.ndarray          # (T,) bool: P-wave region
    qrs_mask: np.ndarray        # (T,) bool: QRS region
    t_mask: np.ndarray          # (T,) bool: T-wave region


def load_ludb(root: Path, max_records: int | None = None) -> list[LUDBRecord]:
    """Load LUDB records with anatomical boundary masks.

    LUDB structure (PhysioNet ludb/1.0.1):
        data/<record_id>.dat   — 12-lead signal at 500 Hz
        data/<record_id>.hea   — header
        data/<record_id>.<lead> — annotation file per lead (P, N, T markers)

    Each annotation file contains tuples (sample_index, symbol). Symbols:
        '(' = wave onset, ')' = wave offset, 'p' = P, 'N' = QRS, 't' = T.
    """
    data_dir = root / "data"
    if not data_dir.exists():
        data_dir = root  # tolerate alternative layouts
    record_ids = sorted({p.stem for p in data_dir.glob("*.dat")})
    if max_records:
        record_ids = record_ids[:max_records]

    records: list[LUDBRecord] = []
    native_fs, target_fs = 500, 360

    for rid in record_ids:
        try:
            rec = wfdb.rdrecord(str(data_dir / rid))
        except Exception:
            continue
        # find lead II
        try:
            idx = rec.sig_name.index("ii")
        except ValueError:
            try:
                idx = rec.sig_name.index("II")
            except ValueError:
                continue
        sig = rec.p_signal[:, idx].astype(np.float32)

        # annotation file for lead II
        ann_ext = rec.sig_name[idx]  # 'ii' or 'II'
        try:
            ann = wfdb.rdann(str(data_dir / rid), extension=ann_ext)
        except Exception:
            continue

        samples = np.array(ann.sample, dtype=np.int64)
        symbols = list(ann.symbol)

        # Build masks at native rate, then resample
        T = len(sig)
        p_mask = np.zeros(T, dtype=bool)
        qrs_mask = np.zeros(T, dtype=bool)
        t_mask = np.zeros(T, dtype=bool)

        # walk annotations: '(' marks onset; the next 'p'/'N'/'t' identifies wave;
        # then ')' marks offset. We pair these into spans.
        i = 0
        while i < len(symbols):
            if symbols[i] == "(":
                onset = samples[i]
                # find wave-type marker
                if i + 1 < len(symbols) and symbols[i + 1] in ("p", "N", "t"):
                    wave = symbols[i + 1]
                    # find next ')'
                    j = i + 2
                    while j < len(symbols) and symbols[j] != ")":
                        j += 1
                    if j < len(symbols):
                        offset = samples[j]
                        if wave == "p":
                            p_mask[onset:offset + 1] = True
                        elif wave == "N":
                            qrs_mask[onset:offset + 1] = True
                        elif wave == "t":
                            t_mask[onset:offset + 1] = True
                        i = j + 1
                        continue
            i += 1

        # Resample everything to 360 Hz
        sig_360 = resample_poly(sig, up=target_fs, down=native_fs).astype(np.float32)
        def resample_mask(m):
            new_T = len(sig_360)
            idx = np.linspace(0, len(m) - 1, new_T).astype(int)
            return m[idx]

        records.append(LUDBRecord(
            record_id=rid,
            signal=sig_360,
            p_mask=resample_mask(p_mask),
            qrs_mask=resample_mask(qrs_mask),
            t_mask=resample_mask(t_mask),
        ))

    return records
