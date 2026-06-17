"""PTB-XL loader — recording-level (bag) granularity.

PTB-XL stores SCP-ECG diagnostic statements per recording. We use the
rhythm codes (AFIB, AFLT, PVC, ...) when present; otherwise default to
N. We resample 500 Hz -> 360 Hz, take lead II, and slide 10-sec windows
with 50% overlap. Beat-level labels are unavailable.
"""

from __future__ import annotations
from pathlib import Path
import ast
import numpy as np
import pandas as pd
import wfdb
from scipy.signal import resample_poly

from unified_dataset import (
    BaseECGLoader, SampleSpec, PTBXL_TO_AAMI, AAMI_CLASSES, T_WINDOW,
)


def scp_codes_to_aami(scp_codes: dict, scp_table: pd.DataFrame) -> str:
    """Resolve SCP-ECG codes to a single AAMI super-class.

    Priority: any rhythm code in PTBXL_TO_AAMI wins. Otherwise default to N.
    """
    for code in scp_codes:
        if code in PTBXL_TO_AAMI:
            return PTBXL_TO_AAMI[code]
    return "N"


class PTBXLLoader(BaseECGLoader):
    def __init__(self, root: Path, use_500hz: bool = True):
        super().__init__(root)
        self.records_dir = "records500" if use_500hz else "records100"
        self.native_fs = 500 if use_500hz else 100

    def list_samples(self) -> list[SampleSpec]:
        try:
            import neurokit2 as nk
        except ImportError as e:
            raise RuntimeError("Install neurokit2: pip install neurokit2") from e

        meta = pd.read_csv(self.root / "ptbxl_database.csv", index_col="ecg_id")
        scp_table = pd.read_csv(self.root / "scp_statements.csv", index_col=0)
        # parse scp_codes column from string dict
        meta["scp_codes"] = meta["scp_codes"].apply(lambda s: ast.literal_eval(s))

        specs: list[SampleSpec] = []
        target_fs = 360
        col = "filename_hr" if self.native_fs == 500 else "filename_lr"

        for ecg_id, row in meta.iterrows():
            rel_path = row[col]
            full_path = self.root / rel_path
            if not full_path.with_suffix(".dat").exists():
                continue
            try:
                rec = wfdb.rdrecord(str(full_path))
            except Exception:
                continue
            if "II" in rec.sig_name:
                lead_idx = rec.sig_name.index("II")
            elif "ii" in rec.sig_name:
                lead_idx = rec.sig_name.index("ii")
            else:
                continue
            lead_ii = rec.p_signal[:, lead_idx].astype(np.float32)
            lead_ii = resample_poly(lead_ii, up=target_fs, down=self.native_fs).astype(np.float32)

            # AAMI class from rhythm codes
            aami_name = scp_codes_to_aami(row["scp_codes"], scp_table)
            bag_label = AAMI_CLASSES[aami_name]

            # Patient ID: PTB-XL has a 'patient_id' column
            patient_id = str(row.get("patient_id", ecg_id))

            # PTB-XL records are 10 sec exactly — usually one window per record
            stride = T_WINDOW // 2
            for start in range(0, max(1, len(lead_ii) - T_WINDOW + 1), stride):
                window = lead_ii[start:start + T_WINDOW]
                if len(window) < T_WINDOW:
                    continue
                try:
                    _, info = nk.ecg_peaks(window, sampling_rate=target_fs, correct_artifacts=True)
                    rpeaks = np.array(info["ECG_R_Peaks"], dtype=np.int64) + start
                except Exception:
                    continue
                if len(rpeaks) < 3:
                    continue
                specs.append(SampleSpec(
                    record_id=str(ecg_id),
                    database="ptbxl",
                    window_start=int(start),
                    bag_label=int(bag_label),
                    beat_positions_signal=rpeaks,
                    beat_labels=None,
                    patient_id=patient_id,
                ))
        return specs

    def load_window(self, spec: SampleSpec) -> np.ndarray:
        meta = pd.read_csv(self.root / "ptbxl_database.csv", index_col="ecg_id")
        col = "filename_hr" if self.native_fs == 500 else "filename_lr"
        rel = meta.loc[int(spec.record_id), col]
        rec = wfdb.rdrecord(str(self.root / rel))
        lead_ii = rec.p_signal[:, rec.sig_name.index("II" if "II" in rec.sig_name else "ii")]
        lead_ii = resample_poly(lead_ii.astype(np.float32), up=360, down=self.native_fs).astype(np.float32)
        return lead_ii[spec.window_start : spec.window_start + T_WINDOW]
