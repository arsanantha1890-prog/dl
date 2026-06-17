#!/usr/bin/env bash
# Smoke test: verify environment + code + GPU in <2 minutes.
# Run this BEFORE the big run_all.sh to catch problems cheaply.
#
# Usage:  bash smoke_test.sh
set -e
cd "$(dirname "$0")"
SRC="${SRC:-$HOME/beatmil/src}"

echo "============================================================"
echo "  Beat-MIL smoke test"
echo "  src dir: $SRC"
echo "============================================================"

# 1. Environment
echo
echo "[1/4] Python environment"
python --version
python -c "
import torch
print(f'  PyTorch:   {torch.__version__}')
print(f'  CUDA:      {torch.version.cuda}')
print(f'  GPU avail: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  Device:    {torch.cuda.get_device_name(0)}')
    print(f'  SM:        {torch.cuda.get_device_capability(0)}')
"

# 2. Run the 5 unit smoke tests
echo
echo "[2/4] Module smoke tests"
cd "$SRC"
for mod in mil_pooling evidential consistency beatmil integration_test; do
    echo "  - $mod.py"
    python "$mod.py" | tail -1 | sed 's/^/      /'
done

# 3. Baselines smoke test
echo
echo "[3/4] Baselines smoke test"
python baselines.py | sed 's/^/      /'

# 4. GPU forward+backward
echo
echo "[4/4] GPU forward + backward pass"
python - <<'EOF'
import torch
from beatmil import BeatMIL
from integration_test import make_synthetic_batch, run_step

if not torch.cuda.is_available():
    print("  [skip] no GPU")
else:
    m = BeatMIL(num_classes=4).cuda()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
    b = make_synthetic_batch(B=16, N=10)
    b = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k,v in b.items()}
    losses = []
    for s in range(5):
        info = run_step(m, b, opt, lambda_cons=0.5)
        losses.append(info["total"])
    print(f"  loss trace: {losses[0]:.4f} -> {losses[-1]:.4f}")
    assert losses[-1] < losses[0], "loss did not decrease"
    mem = torch.cuda.memory_allocated(0) / 1e9
    print(f"  GPU memory used: {mem:.2f} GB")
    print("  [GPU] OK")
EOF

echo
echo "============================================================"
echo "  ✓ Smoke test passed. Safe to run run_all.sh"
echo "============================================================"
