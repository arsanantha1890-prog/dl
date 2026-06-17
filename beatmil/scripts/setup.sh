#!/usr/bin/env bash
# One-time environment setup for Beat-MIL.
# Run once per machine. Assumes conda or venv is already activated.
#
# Usage:
#   bash setup.sh                       # full setup
#   SRC=/custom/path bash setup.sh      # custom source dir
#   SKIP_TORCH=1 bash setup.sh          # if torch already installed

set -e
SRC="${SRC:-$HOME/beatmil}"

echo "================================================================="
echo " Beat-MIL one-time setup"
echo " Target: $SRC"
echo "================================================================="

# 1. directory tree
echo
echo "[1/5] creating directory tree"
mkdir -p "$SRC"/{src,configs,data,outputs,checkpoints,figures,logs,notebooks,paper}
mkdir -p "$SRC"/outputs/{cache,eval}
echo "  done."

# 2. .gitignore
if [ ! -f "$SRC/.gitignore" ]; then
    echo
    echo "[2/5] writing .gitignore"
    cat > "$SRC/.gitignore" <<'EOF'
data/
checkpoints/
logs/
outputs/
figures/
__pycache__/
*.pyc
*.npz
*.pkl
*.pt
.ipynb_checkpoints/
EOF
    echo "  done."
fi

# 3. PyTorch for RTX 5090 (sm_120)
if [ -z "$SKIP_TORCH" ]; then
    echo
    echo "[3/5] installing PyTorch (CUDA 12.8 wheels for RTX 5090)"
    pip install --upgrade pip
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
fi

# 4. other dependencies
echo
echo "[4/5] installing scientific stack"
pip install numpy scipy scikit-learn pandas matplotlib seaborn tqdm pyyaml
pip install wfdb neurokit2 tensorboard
echo "  done."

# 5. freeze requirements + verify GPU
echo
echo "[5/5] verifying GPU and freezing requirements"
pip freeze > "$SRC/requirements.txt"
python - <<'EOF'
import torch
print(f"  PyTorch: {torch.__version__}")
print(f"  CUDA built: {torch.version.cuda}")
print(f"  CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  Device: {torch.cuda.get_device_name(0)}")
    sm = torch.cuda.get_device_capability(0)
    print(f"  SM: {sm}")
    if sm[0] >= 12:
        print("  ✓ RTX 5090 / Blackwell stack OK")
    else:
        print(f"  ⚠ unexpected SM {sm}, expected (12, 0) for RTX 5090")
else:
    print("  ⚠ no GPU detected — training will be very slow on CPU")
EOF

echo
echo "================================================================="
echo " ✓ Setup complete."
echo
echo " Next steps:"
echo "   1. Copy the 17 Python files into $SRC/src/"
echo "   2. Symlink your datasets into $SRC/data/ (mitbih, cpsc2018, ptbxl)"
echo "   3. Download LUDB into $SRC/data/ludb/"
echo "   4. bash $SRC/src/smoke_test.sh"
echo "   5. bash $SRC/src/run_all.sh"
echo "================================================================="
