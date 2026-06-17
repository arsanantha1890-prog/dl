"""CPSC 2018 loader — rhythm-level (bag) granularity.

Each .mat file contains a 12-lead recording at 500 Hz with one rhythm
label from REFERENCE.csv. We resample to 360 Hz, take lead II, and
slide a 10-sec window with 50% overlap. R-peaks within each window are
detected with neurokit2 to populate MIL beat positions; beat-level
labels are unavailable (set to None).
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.signal import resample_poly

from unified_dataset import (
    BaseECGLoader, SampleSpec, CPSC_TO_AAMI, AAMI_CLASSES, T_WINDOW,
)

# CPSC label codes (from REFERENCE.csv) — column "First_label" gives the dominant class.
# Codes 1-9 map to: 1=Normal, 2=AF, 3=I-AVB, 4=LBBB, 5=RBBB, 6=PAC, 7=PVC, 8=STD, 9=STE
CPSC_CODE_TO_NAME = {1: "Normal", 2: "AF", 3: "I-AVB", 4: "LBBB", 5: "RBBB",
                     6: "PAC", 7: "PVC", 8: "STD", 9: "STE"}


class CPSC2018Loader(BaseECGLoader):
    def list_samples(self) -> list[SampleSpec]:
        try:
            import neurokit2 as nk
        except ImportError as e:
            raise RuntimeError("Install neurokit2: pip install neurokit2") from e

        ref_path = self.root / "REFERENCE.csv"
        ref = pd.read_csv(ref_path)
        specs: list[SampleSpec] = []
        native_fs, target_fs = 500, 360

        for _, row in ref.iterrows():
            rec_id = row["Recording"]
            mat_path = self.root / f"{rec_id}.mat"
            if not mat_path.exists():
                continue
            data = loadmat(str(mat_path))
            # CPSC convention: signal in 'ECG' with shape (12, N) — lead II is index 1
            sig12 = data.get("ECG", data.get("val"))
            if sig12 is None:
                continue
            lead_ii = sig12[1].astype(np.float32)

            # Resample 500 -> 360 Hz
            lead_ii = resample_poly(lead_ii, up=target_fs, down=native_fs).astype(np.float32)

            # Map rhythm code -> AAMI bag class
            code = int(row["First_label"])
            name = CPSC_CODE_TO_NAME.get(code)
            if name is None or name not in CPSC_TO_AAMI:
                continue
            bag_label = AAMI_CLASSES[CPSC_TO_AAMI[name]]

            # Slide 10-sec window with 50% overlap; need at least one full window
            stride = T_WINDOW // 2
            for start in range(0, len(lead_ii) - T_WINDOW + 1, stride):
                window = lead_ii[start:start + T_WINDOW]
                # Detect R-peaks in this window with neurokit2
                try:
                    _, info = nk.ecg_peaks(window, sampling_rate=target_fs, correct_artifacts=True)
                    rpeaks = np.array(info["ECG_R_Peaks"], dtype=np.int64) + start
                except Exception:
                    continue
                if len(rpeaks) < 3:
                    continue  # too few beats — likely noise

                specs.append(SampleSpec(
                    record_id=str(rec_id),
                    database="cpsc",
                    window_start=int(start),
                    bag_label=int(bag_label),
                    beat_positions_signal=rpeaks,
                    beat_labels=None,                       # rhythm-level only
                    patient_id=str(rec_id),                 # CPSC: one record per patient
                ))
        return specs

    def load_window(self, spec: SampleSpec) -> np.ndarray:
        mat_path = self.root / f"{spec.record_id}.mat"
        data = loadmat(str(mat_path))
        sig12 = data.get("ECG", data.get("val"))
        lead_ii = sig12[1].astype(np.float32)
        lead_ii = resample_poly(lead_ii, up=360, down=500).astype(np.float32)
        return lead_ii[spec.window_start : spec.window_start + T_WINDOW]
