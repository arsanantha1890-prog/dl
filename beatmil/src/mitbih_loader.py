"""MIT-BIH Arrhythmia Database loader — beat-level granularity.

FIXED: bag_label is now the MAJORITY class across all beats in the window,
not the center beat's class. This removes the contradiction with the
consistency loss (which aggregates over all beats) that caused Beat-MIL
to train to ~0.54 F1 previously.
"""

from __future__ import annotations
from pathlib import Path
import numpy as np

from unified_dataset import (
    BaseECGLoader, SampleSpec, MITBIH_TO_AAMI, AAMI_CLASSES, T_WINDOW,
)


class MITBIHLoader(BaseECGLoader):
    def list_samples(self) -> list[SampleSpec]:
        import wfdb
        specs: list[SampleSpec] = []
        record_files = sorted({p.stem for p in self.root.glob("*.dat")})

        for rec_id in record_files:
            rec_path = str(self.root / rec_id)
            try:
                rec = wfdb.rdrecord(rec_path)
                ann = wfdb.rdann(rec_path, extension="atr")
            except Exception:
                continue
            try:
                lead_idx = rec.sig_name.index("MLII")
            except ValueError:
                continue

            signal = rec.p_signal[:, lead_idx].astype(np.float32)
            beat_idx = np.array(ann.sample, dtype=np.int64)
            beat_sym = np.array(ann.symbol)

            keep = np.array([s in MITBIH_TO_AAMI for s in beat_sym])
            beat_idx = beat_idx[keep]
            beat_sym = beat_sym[keep]
            beat_aami = np.array([AAMI_CLASSES[MITBIH_TO_AAMI[s]] for s in beat_sym],
                                 dtype=np.int64)

            half = T_WINDOW // 2
            for center, center_label in zip(beat_idx, beat_aami):
                start = int(center - half)
                end = start + T_WINDOW
                if start < 0 or end > len(signal):
                    continue
                inside = (beat_idx >= start) & (beat_idx < end)
                window_beats = beat_idx[inside].astype(np.int64)
                window_labels = beat_aami[inside]
                if len(window_labels) == 0:
                    continue
                # FIX: majority vote, not center beat
                vals, counts = np.unique(window_labels, return_counts=True)
                majority_label = int(vals[np.argmax(counts)])
                specs.append(SampleSpec(
                    record_id=rec_id,
                    database="mitbih",
                    window_start=start,
                    bag_label=majority_label,
                    beat_positions_signal=window_beats,
                    beat_labels=window_labels,
                    patient_id=rec_id,
                ))
        return specs

    def load_window(self, spec: SampleSpec) -> np.ndarray:
        import wfdb
        rec = wfdb.rdrecord(str(self.root / spec.record_id))
        lead_idx = rec.sig_name.index("MLII")
        signal = rec.p_signal[:, lead_idx].astype(np.float32)
        return signal[spec.window_start : spec.window_start + T_WINDOW]
