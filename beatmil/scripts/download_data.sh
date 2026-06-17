#!/usr/bin/env bash
# Download all four ECG databases used in Beat-MIL:
#   1. MIT-BIH Arrhythmia Database  (~75 MB)
#   2. PTB-XL                       (~3 GB)
#   3. LUDB                         (~50 MB)
#   4. CPSC 2018                    (~2 GB)  — instructions only, not automatic
#
# Usage:
#   bash download_data.sh                # downloads everything except CPSC
#   DATA=/custom/path bash download_data.sh
#
# Re-runnable — skips datasets already present.

set -e
DATA="${DATA:-$HOME/beatmil/data}"
mkdir -p "$DATA"
cd "$DATA"

ts() { date "+%H:%M:%S"; }
log() { echo "[$(ts)] $1"; }

# ---------- MIT-BIH Arrhythmia DB ------------------------------
if [ ! -f "$DATA/mitbih/100.dat" ]; then
    log "downloading MIT-BIH Arrhythmia Database (~75 MB)"
    mkdir -p "$DATA/mitbih"
    cd "$DATA/mitbih"
    wget -q --show-progress -r -np -nH --cut-dirs=4 -R "index.html*" \
        https://physionet.org/files/mitdb/1.0.0/ -P ./
    # the wget puts files in $DATA/mitbih/files/mitdb/1.0.0/* — flatten:
    if [ -d files ]; then
        mv files/*/*/* . 2>/dev/null || mv files/* . 2>/dev/null || true
        rm -rf files
    fi
    cd "$DATA"
    log "  MIT-BIH: $(ls $DATA/mitbih/*.dat 2>/dev/null | wc -l) records"
else
    log "MIT-BIH already present — skip"
fi

# ---------- LUDB ----------------------------------------------
if [ ! -d "$DATA/ludb/data" ] || [ -z "$(ls $DATA/ludb/data/*.dat 2>/dev/null)" ]; then
    log "downloading LUDB (~50 MB)"
    mkdir -p "$DATA/ludb"
    cd "$DATA/ludb"
    wget -q --show-progress -r -np -nH --cut-dirs=4 -R "index.html*" \
        https://physionet.org/files/ludb/1.0.1/ -P ./
    if [ -d files ]; then
        # find the deepest directory containing .dat files and lift it
        mv files/*/* . 2>/dev/null || true
        rm -rf files
    fi
    cd "$DATA"
    log "  LUDB: $(ls $DATA/ludb/data/*.dat 2>/dev/null | wc -l) records"
else
    log "LUDB already present — skip"
fi

# ---------- PTB-XL --------------------------------------------
if [ ! -f "$DATA/ptbxl/ptbxl_database.csv" ]; then
    log "downloading PTB-XL (~3 GB — this is the slow one)"
    mkdir -p "$DATA/ptbxl"
    cd "$DATA/ptbxl"
    wget -q --show-progress -r -np -nH --cut-dirs=4 -R "index.html*" \
        https://physionet.org/files/ptb-xl/1.0.3/ -P ./
    if [ -d files ]; then
        mv files/*/*/* . 2>/dev/null || mv files/* . 2>/dev/null || true
        rm -rf files
    fi
    cd "$DATA"
    log "  PTB-XL: ready"
else
    log "PTB-XL already present — skip"
fi

# ---------- CPSC 2018 -----------------------------------------
if [ ! -f "$DATA/cpsc2018/REFERENCE.csv" ]; then
    cat <<'EOF'
============================================================
  CPSC 2018 manual download required
============================================================
  Auto-download not available — CPSC 2018 hosting moves around.
  Get the TrainingSet zip (~1 GB) from one of:

    Primary:   http://2018.icbeb.org/Challenge.html
    Mirror 1:  https://www.kaggle.com/datasets/bjoernjostein/china-12lead-ecg-challenge-database
    Mirror 2:  https://physionet.org/content/challenge-2020/  (subset)

  Extract so that you have:
      DATA/cpsc2018/A0001.mat
      DATA/cpsc2018/A0002.mat
      ...
      DATA/cpsc2018/REFERENCE.csv

  Then re-run this script (it will skip everything else).
============================================================
EOF
else
    log "CPSC 2018 already present — skip"
fi

echo
log "================================================================="
log " Download summary:"
log "  MIT-BIH:   $(ls $DATA/mitbih/*.dat 2>/dev/null | wc -l) records"
log "  CPSC 2018: $(ls $DATA/cpsc2018/*.mat 2>/dev/null | wc -l) records"
log "  PTB-XL:    $([ -f $DATA/ptbxl/ptbxl_database.csv ] && echo present || echo MISSING)"
log "  LUDB:      $(ls $DATA/ludb/data/*.dat 2>/dev/null | wc -l) records"
log "================================================================="
