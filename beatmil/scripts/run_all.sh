#!/usr/bin/env bash
# Beat-MIL: one-command end-to-end pipeline.
#
# Stages: 1.cache  2.beatmil  3.baselines  4.LODO  5.eval  6.McNemar
#         7.XAI    8.figures  9.tables    10.paper
#
# Usage:
#   bash run_all.sh
#   bash run_all.sh --skip-lodo --skip-paper
#   SRC=/path bash run_all.sh

set -e
SRC="${SRC:-$HOME/beatmil/src}"
LOGS="$HOME/beatmil/logs"
mkdir -p "$LOGS"
cd "$SRC"

SKIP_LODO=0; SKIP_PAPER=0
for arg in "$@"; do case $arg in
    --skip-lodo)  SKIP_LODO=1 ;;
    --skip-paper) SKIP_PAPER=1 ;;
esac done

log() { echo "[$(date '+%H:%M:%S')] $1"; }
log "============================================================"
log " Beat-MIL pipeline  (skip_lodo=$SKIP_LODO skip_paper=$SKIP_PAPER)"
log "============================================================"

log "STAGE  1/10: cache spec lists"
python cache_specs.py 2>&1 | tee "$LOGS/01_cache.log"

log "STAGE  2/10: Beat-MIL intra-DB"
python train.py --model beatmil --mode intra-db \
    --data-root "$HOME/beatmil/data" --out-dir "$HOME/beatmil/checkpoints" \
    --epochs 50 --batch-size 128 --num-workers 4 2>&1 | tee "$LOGS/02_beatmil.log"

log "STAGE  3/10: baselines intra-DB"
for m in resnet1d cnnlstm ecgformer; do
    log "   $m"
    python train.py --model "$m" --mode intra-db \
        --data-root "$HOME/beatmil/data" --out-dir "$HOME/beatmil/checkpoints" \
        --epochs 50 --batch-size 128 --num-workers 4 2>&1 | tee "$LOGS/03_${m}.log"
done

if [ "$SKIP_LODO" -eq 0 ]; then
    log "STAGE  4/10: LODO-PTB-XL"
    for m in beatmil resnet1d cnnlstm ecgformer; do
        log "   LODO: $m"
        python train.py --model "$m" --mode lodo --held-out ptbxl \
            --data-root "$HOME/beatmil/data" --out-dir "$HOME/beatmil/checkpoints" \
            --epochs 40 --batch-size 128 --num-workers 4 2>&1 | tee "$LOGS/04_${m}_lodo.log"
    done
else
    log "STAGE  4/10: SKIPPED (--skip-lodo)"
fi

log "STAGE  5/10: evaluate all checkpoints"
python run_eval.py 2>&1 | tee "$LOGS/05_eval.log"

log "STAGE  6/10: McNemar pairwise tests"
python run_mcnemar.py 2>&1 | tee "$LOGS/06_mcnemar.log"

log "STAGE  7/10: XAI on LUDB"
python run_xai.py 2>&1 | tee "$LOGS/07_xai.log"

log "STAGE  8/10: paper figures"
python figures.py --results-dir "$HOME/beatmil/outputs/eval" \
    --out-dir "$HOME/beatmil/figures" 2>&1 | tee "$LOGS/08_figures.log"

log "STAGE  9/10: LaTeX tables"
python tables.py 2>&1 | tee "$LOGS/09_tables.log"

if [ "$SKIP_PAPER" -eq 0 ]; then
    log "STAGE 10/10: compile paper.tex"
    bash "$HOME/beatmil/scripts/make_paper.sh" 2>&1 | tee "$LOGS/10_paper.log"
else
    log "STAGE 10/10: SKIPPED (--skip-paper)"
fi

log "============================================================"
log " ✓ Pipeline complete"
log "   PDF: $HOME/beatmil/paper/paper.pdf"
log "============================================================"
