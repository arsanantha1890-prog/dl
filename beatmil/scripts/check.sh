#!/usr/bin/env bash
# Check Beat-MIL pipeline progress without disturbing anything.
L=/workspace/beatmil/logs
echo "=== Pipeline process ==="
if [ -f "$L/pipeline.pid" ]; then
    PID=$(cat "$L/pipeline.pid")
    if ps -p "$PID" > /dev/null 2>&1; then
        echo "RUNNING (PID $PID)"
    else
        echo "NOT running (finished or stopped)"
    fi
fi
echo ""
echo "=== GPU ==="
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null || echo "nvidia-smi unavailable"
echo ""
echo "=== Last 12 lines of master log ==="
tail -12 "$L/master.log" 2>/dev/null || echo "no master log yet"
echo ""
echo "=== Best val F1 so far (each model) ==="
for f in /workspace/beatmil/checkpoints/*/history.json; do
    [ -f "$f" ] || continue
    name=$(basename $(dirname "$f"))
    python3 -c "import json; h=json.load(open('$f')); b=max(h,key=lambda x:x['val_f1']); print(f'  $name: best val_F1={b[\"val_f1\"]:.4f} (epoch {b[\"epoch\"]})')" 2>/dev/null
done
