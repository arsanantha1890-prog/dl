#!/usr/bin/env bash
# Beat-MIL robust pipeline runner.
# Uses nohup + unbuffered output so training SURVIVES terminal/SSH death.
# You can disconnect freely and check progress anytime with: bash check.sh
#
# Usage:
#   bash run_pipeline.sh                # full pipeline in background
#   bash run_pipeline.sh --skip-lodo    # skip LODO rotations

set -e
SRC="/workspace/beatmil/src"
LOGS="/workspace/beatmil/logs"
DATA="/workspace/beatmil/data"
CKPT="/workspace/beatmil/checkpoints"
mkdir -p "$LOGS" "$CKPT"
cd "$SRC"

SKIP_LODO=""
for a in "$@"; do [ "$a" = "--skip-lodo" ] && SKIP_LODO=1; done

# The whole pipeline runs inside one backgrounded bash, fully detached.
nohup bash -c '
set -e
cd /workspace/beatmil/src
export PYTHONUNBUFFERED=1
L=/workspace/beatmil/logs
ts(){ date "+%H:%M:%S"; }

echo "[$(ts)] STAGE 1/10: cache specs"
python -u cache_specs.py 2>&1 | tee $L/01_cache.log

echo "[$(ts)] STAGE 2/10: Beat-MIL intra-DB"
python -u train.py --model beatmil --mode intra-db \
    --data-root /workspace/beatmil/data --out-dir /workspace/beatmil/checkpoints \
    --epochs 50 --batch-size 128 --num-workers 8 2>&1 | tee $L/02_beatmil.log

echo "[$(ts)] STAGE 3/10: baselines intra-DB"
for m in resnet1d cnnlstm ecgformer; do
    echo "[$(ts)]   $m"
    python -u train.py --model $m --mode intra-db \
        --data-root /workspace/beatmil/data --out-dir /workspace/beatmil/checkpoints \
        --epochs 50 --batch-size 128 --num-workers 8 2>&1 | tee $L/03_${m}.log
done

if [ -z "'"$SKIP_LODO"'" ]; then
    echo "[$(ts)] STAGE 4/10: LODO-PTB-XL"
    for m in beatmil resnet1d cnnlstm ecgformer; do
        echo "[$(ts)]   LODO $m"
        python -u train.py --model $m --mode lodo --held-out ptbxl \
            --data-root /workspace/beatmil/data --out-dir /workspace/beatmil/checkpoints \
            --epochs 40 --batch-size 128 --num-workers 8 2>&1 | tee $L/04_${m}_lodo.log
    done
else
    echo "[$(ts)] STAGE 4/10: SKIPPED"
fi

echo "[$(ts)] STAGE 5/10: evaluate"
python -u run_eval.py 2>&1 | tee $L/05_eval.log
echo "[$(ts)] STAGE 6/10: McNemar"
python -u run_mcnemar.py 2>&1 | tee $L/06_mcnemar.log
echo "[$(ts)] STAGE 7/10: XAI"
python -u run_xai.py 2>&1 | tee $L/07_xai.log
echo "[$(ts)] STAGE 8/10: figures"
python -u figures.py --results-dir /workspace/beatmil/outputs/eval --out-dir /workspace/beatmil/figures 2>&1 | tee $L/08_figures.log
echo "[$(ts)] STAGE 9/10: tables"
python -u tables.py 2>&1 | tee $L/09_tables.log
echo "[$(ts)] STAGE 10/10: compile paper"
bash /workspace/beatmil/scripts/make_paper.sh 2>&1 | tee $L/10_paper.log

echo "[$(ts)] ====== PIPELINE COMPLETE ======"
echo "[$(ts)] PDF: /workspace/beatmil/paper/paper.pdf"
' > "$LOGS/master.log" 2>&1 &

PID=$!
echo "$PID" > "$LOGS/pipeline.pid"
echo "============================================================"
echo " Pipeline launched in background. PID = $PID"
echo " It will keep running even if you disconnect SSH."
echo ""
echo " Check progress:   bash /workspace/beatmil/check.sh"
echo " Watch live:       tail -f /workspace/beatmil/logs/master.log"
echo " Stop everything:  kill $PID"
echo "============================================================"
