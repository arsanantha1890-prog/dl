# Beat-MIL: Complete Execution Walkthrough

One continuous runbook from empty machine to submitted PDF. Run each step in order. Don't skip the verification gates — they exist to catch problems while they're cheap.

Target: **BMEiCON 2026**, submission deadline **June 30, 2026**.

---

## Part 1: Environment

### Step 1 — Create the project tree

```bash
mkdir -p ~/beatmil/{src,configs,data,outputs,checkpoints,figures,logs,notebooks,paper}
cd ~/beatmil
git init
cat > .gitignore <<'EOF'
data/
checkpoints/
logs/
outputs/
__pycache__/
*.pyc
*.npz
.ipynb_checkpoints/
EOF
```

### Step 2 — Drop in all 13 code files

From your `beatmil_day1/` and this walkthrough's accompanying bundle, place these files in `~/beatmil/src/`:

```
mil_pooling.py        evidential.py       consistency.py       beatmil.py
unified_dataset.py    integration_test.py focal.py             baselines.py
mitbih_loader.py      cpsc.py             ptbxl.py             ludb.py
train.py              eval_metrics.py     xai.py               figures.py
```

Verify:
```bash
ls ~/beatmil/src/ | wc -l    # expect 16
```

### Step 3 — Python environment

```bash
conda create -n beatmil python=3.11 -y
conda activate beatmil
# OR: python3.11 -m venv ~/venvs/beatmil && source ~/venvs/beatmil/bin/activate
```

### Step 4 — Install PyTorch for RTX 5090 (sm_120)

```bash
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

### Step 5 — Verify GPU stack

```bash
python -c "
import torch
print('PyTorch:', torch.__version__)
print('CUDA built:', torch.version.cuda)
print('CUDA avail:', torch.cuda.is_available())
print('Device:', torch.cuda.get_device_name(0))
print('SM:', torch.cuda.get_device_capability(0))
"
```

Expected output includes `SM: (12, 0)` and `NVIDIA GeForce RTX 5090`. If you see `no kernel image available` or `SM: (9, 0)`, re-run Step 4 — you got the wrong wheel.

### Step 6 — Install remaining dependencies

```bash
pip install numpy scipy scikit-learn pandas matplotlib seaborn tqdm pyyaml
pip install wfdb neurokit2 tensorboard
pip install jupyter         # for sanity notebooks
pip freeze > ~/beatmil/requirements.txt
```

### Step 7 — Run the 5 smoke tests

```bash
cd ~/beatmil/src
for f in mil_pooling.py evidential.py consistency.py beatmil.py integration_test.py; do
    echo "=== $f ==="
    python $f || { echo "FAILED: $f"; exit 1; }
done
```

All five must print "smoke test passed" / "all checks passed". If anything fails, stop and debug — these are deterministic; success on CPU means the architecture is sound.

### Step 8 — GPU smoke test

```bash
cd ~/beatmil/src
python -c "
import torch
from beatmil import BeatMIL
from integration_test import make_synthetic_batch, run_step
m = BeatMIL(num_classes=4).cuda()
opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
b = make_synthetic_batch(B=32, N=12)
b = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k,v in b.items()}
for s in range(10):
    info = run_step(m, b, opt, lambda_cons=0.5)
    print(f'step {s}: loss={info[\"total\"]:.4f}')
print(f'GPU mem: {torch.cuda.memory_allocated(0)/1e9:.2f} GB')
"
```

Loss should drop monotonically. Memory usage should be 1-3 GB.

---

## Part 2: Data

### Step 9 — Locate or download the four databases

You already have three from the prior paper. LUDB is new.

```bash
cd ~/beatmil/data

# If old data exists, symlink it:
ln -s /path/to/old/data/mitbih    mitbih
ln -s /path/to/old/data/cpsc2018  cpsc2018
ln -s /path/to/old/data/ptbxl     ptbxl

# Verify structure:
ls mitbih/   | head      # expect 100.dat 100.hea 100.atr ... 234.dat
ls cpsc2018/ | head      # expect A0001.mat A0002.mat ... REFERENCE.csv
ls ptbxl/    | head      # expect ptbxl_database.csv records100/ records500/

# Download LUDB (new):
mkdir -p ludb && cd ludb
wget -r -np -nH --cut-dirs=4 -R "index.html*" https://physionet.org/files/ludb/1.0.1/ -P ./
# Verify: 200 records
ls data/ 2>/dev/null | grep -c '\.dat$'    # → 200
cd ..
```

### Step 10 — Sanity-check the MIT-BIH loader

```python
# ~/beatmil/src/sanity_mitbih.py
from pathlib import Path
from collections import Counter
from mitbih_loader import MITBIHLoader
from unified_dataset import AAMI_CLASSES

loader = MITBIHLoader(root=Path.home() / "beatmil/data/mitbih")
specs = loader.list_samples()
print(f"MIT-BIH samples: {len(specs):,}")
dist = Counter(s.bag_label for s in specs)
for name, idx in AAMI_CLASSES.items():
    print(f"  {name}: {dist.get(idx, 0):,} ({100*dist.get(idx,0)/len(specs):.1f}%)")
```

Expected: ~100k samples, N ~84%, V ~7%, S ~2.5%, Q ~6%.

### Step 11 — Sanity-check CPSC and PTB-XL

```python
# ~/beatmil/src/sanity_cpsc_ptbxl.py
from pathlib import Path
from collections import Counter
from cpsc import CPSC2018Loader
from ptbxl import PTBXLLoader
from unified_dataset import AAMI_CLASSES

for name, cls in [("cpsc", CPSC2018Loader), ("ptbxl", PTBXLLoader)]:
    loader = cls(root=Path.home() / f"beatmil/data/{name if name=='cpsc' else 'ptbxl'}")
    specs = loader.list_samples()
    dist = Counter(s.bag_label for s in specs)
    print(f"\n{name}: {len(specs):,} samples")
    for cn, idx in AAMI_CLASSES.items():
        print(f"  {cn}: {dist.get(idx, 0):,} ({100*dist.get(idx,0)/len(specs):.1f}%)")
```

Run:
```bash
cd ~/beatmil/src
python sanity_cpsc_ptbxl.py
```

CPSC will take several minutes (R-peak detection on thousands of records). PTB-XL longer. If either crashes, the most likely cause is a malformed record — wrap the inner loop in `try/except Exception: continue`.

### Step 12 — Pre-cache the spec lists

R-peak detection is slow. Cache the spec lists so you don't redo it every run.

```python
# ~/beatmil/src/cache_specs.py
import pickle
from pathlib import Path
from mitbih_loader import MITBIHLoader
from cpsc import CPSC2018Loader
from ptbxl import PTBXLLoader

root = Path.home() / "beatmil/data"
cache_dir = Path.home() / "beatmil/outputs/cache"
cache_dir.mkdir(parents=True, exist_ok=True)

for name, cls, sub in [("mitbih", MITBIHLoader, "mitbih"),
                       ("cpsc",   CPSC2018Loader, "cpsc2018"),
                       ("ptbxl",  PTBXLLoader,    "ptbxl")]:
    cache_path = cache_dir / f"{name}_specs.pkl"
    if cache_path.exists():
        print(f"{name}: cached")
        continue
    print(f"building {name} specs...")
    specs = cls(root=root / sub).list_samples()
    with cache_path.open("wb") as f:
        pickle.dump(specs, f)
    print(f"  {len(specs):,} saved to {cache_path}")
```

Run once — takes ~30-90 minutes total.

---

## Part 3: Training

### Step 13 — Run Beat-MIL intra-database

```bash
cd ~/beatmil/src
python train.py --model beatmil --mode intra-db \
    --data-root ~/beatmil/data \
    --out-dir ~/beatmil/checkpoints \
    --epochs 50 --batch-size 128 --num-workers 4
```

You'll see per-epoch validation F1 climbing. Expected first epoch ~0.55, plateau ~0.85-0.94 depending on convergence. Best checkpoint written to `~/beatmil/checkpoints/beatmil_intra-db/best.pt`. Total time: 3-4 hours on RTX 5090.

### Step 14 — Train the three baselines

```bash
for m in resnet1d cnnlstm ecgformer; do
    python train.py --model $m --mode intra-db \
        --data-root ~/beatmil/data \
        --out-dir ~/beatmil/checkpoints \
        --epochs 50 --batch-size 128 --num-workers 4
done
```

Total: ~9-12 hours. Run overnight.

### Step 15 — Run LODO rotations

The biggest single deliverable for novelty. Three rotations; each ~3 hours.

```bash
for db in ptbxl cpsc mitbih; do
    python train.py --model beatmil --mode lodo --held-out $db \
        --data-root ~/beatmil/data \
        --out-dir ~/beatmil/checkpoints \
        --epochs 40 --batch-size 128 --num-workers 4
done
```

If time runs short, do **only `--held-out ptbxl`** (the most challenging — different acquisition device and most diverse population). One LODO rotation is enough for the submission.

### Step 16 — Baseline LODO (at least one for fair comparison)

```bash
for m in resnet1d cnnlstm ecgformer; do
    python train.py --model $m --mode lodo --held-out ptbxl \
        --data-root ~/beatmil/data \
        --out-dir ~/beatmil/checkpoints \
        --epochs 40 --batch-size 128 --num-workers 4
done
```

---

## Part 4: Evaluation

### Step 17 — Per-checkpoint evaluation, save outputs

```python
# ~/beatmil/src/run_eval.py
"""Evaluate every checkpoint on its test split, save predictions+probs."""
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from unified_dataset import UnifiedECGDataset
from mitbih_loader import MITBIHLoader
from cpsc import CPSC2018Loader
from ptbxl import PTBXLLoader
from beatmil import BeatMIL
from baselines import build_baseline
from train import collate, evaluate
from eval_metrics import (classification_metrics, bootstrap_ci, f1_macro_metric,
                          expected_calibration_error, brier_multiclass,
                          reliability_diagram_data, risk_coverage_curve, auarc,
                          selective_metrics_at_coverage)
import pickle

CKPT_ROOT = Path.home() / "beatmil/checkpoints"
DATA_ROOT = Path.home() / "beatmil/data"
OUT_ROOT  = Path.home() / "beatmil/outputs/eval"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

# Load cached specs
specs_all = []
for name in ["mitbih", "cpsc", "ptbxl"]:
    with open(Path.home() / f"beatmil/outputs/cache/{name}_specs.pkl", "rb") as f:
        specs_all.extend(pickle.load(f))
db_loaders = {"mitbih": MITBIHLoader(DATA_ROOT/"mitbih"),
              "cpsc":   CPSC2018Loader(DATA_ROOT/"cpsc2018"),
              "ptbxl":  PTBXLLoader(DATA_ROOT/"ptbxl")}

device = "cuda"

def eval_one(ckpt_dir: Path):
    ckpt = torch.load(ckpt_dir / "best.pt", map_location=device)
    args = ckpt["args"]; model_name = args["model"]
    is_beatmil = (model_name == "beatmil")
    model = (BeatMIL(num_classes=4) if is_beatmil else build_baseline(model_name, 4)).to(device)
    model.load_state_dict(ckpt["model"]); model.eval()

    split = json.loads((ckpt_dir / "split.json").read_text())
    test_ids = {(s["database"], s["record_id"], s["window_start"]) for s in split["test"]}
    test_specs = [s for s in specs_all if (s.database, s.record_id, s.window_start) in test_ids]
    test_ds = UnifiedECGDataset(test_specs, db_loaders, augment=False)
    test_dl = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=4,
                         collate_fn=collate, pin_memory=True)

    ev = evaluate(model, test_dl, device, is_beatmil)
    preds = ev["preds"]; targets = ev["targets"]; probs = ev["probs"]; vac = ev["vacuity"]

    metrics = classification_metrics(preds, targets, probs)
    metrics["f1_macro_ci"] = bootstrap_ci(preds, targets, probs, f1_macro_metric)
    metrics["ece"] = expected_calibration_error(probs, targets)
    metrics["brier"] = brier_multiclass(probs, targets)
    if is_beatmil:
        cov, risk, f1 = risk_coverage_curve(preds, targets, vac)
        metrics["auarc"] = auarc(cov, risk)
        metrics["selective"] = selective_metrics_at_coverage(preds, targets, vac)
        np.savez(OUT_ROOT / f"riskcov_{ckpt_dir.name}.npz", cov=cov, risk=risk, f1=f1)
        c, cf, ac, ct = reliability_diagram_data(probs, targets)
        np.savez(OUT_ROOT / "reliability.npz", centers=c, confs=cf, accs=ac, counts=ct)
    out_path = OUT_ROOT / f"{ckpt_dir.name}.json"
    out_path.write_text(json.dumps(metrics, indent=2))

    # Save raw predictions for downstream McNemar tests
    np.savez(OUT_ROOT / f"preds_{ckpt_dir.name}.npz",
             preds=preds, targets=targets, probs=probs, vacuity=vac)
    print(f"[eval] {ckpt_dir.name}: macro F1 = {metrics['f1_macro']:.4f}")

for ckpt_dir in sorted(CKPT_ROOT.rglob("best.pt")):
    eval_one(ckpt_dir.parent)
```

Run:
```bash
cd ~/beatmil/src
python run_eval.py
```

This produces one JSON per checkpoint with all metrics + bootstrap CIs, plus risk-coverage and reliability data for Beat-MIL.

### Step 18 — Pairwise McNemar tests

```python
# ~/beatmil/src/run_mcnemar.py
"""McNemar Beat-MIL vs each baseline, Holm-Bonferroni correction."""
import json
import numpy as np
from pathlib import Path
from eval_metrics import mcnemar, holm_bonferroni

OUT = Path.home() / "beatmil/outputs/eval"

beat = np.load(OUT / "preds_beatmil_intra-db.npz")
preds_b = beat["preds"]; targets = beat["targets"]
results = {}
pvals = []
for name in ["resnet1d", "cnnlstm", "ecgformer"]:
    f = OUT / f"preds_{name}_intra-db.npz"
    if not f.exists(): continue
    d = np.load(f)
    p, b01, b10 = mcnemar(preds_b, d["preds"], targets)
    results[name] = {"p": p, "beatmil_correct_only": b10, "baseline_correct_only": b01}
    pvals.append(p)

decisions, thresholds = holm_bonferroni(pvals)
for i, name in enumerate(results):
    results[name]["holm_threshold"] = thresholds[i]
    results[name]["reject_null"] = decisions[i]

(OUT / "mcnemar.json").write_text(json.dumps(results, indent=2))
print(json.dumps(results, indent=2))
```

Run:
```bash
python run_mcnemar.py
```

### Step 19 — XAI evaluation on LUDB

```bash
cd ~/beatmil/src
python -c "
from pathlib import Path
from xai import evaluate_xai
import json

summary, _ = evaluate_xai(
    model_ckpt=Path.home()/'beatmil/checkpoints/beatmil_intra-db/best.pt',
    ludb_root=Path.home()/'beatmil/data/ludb',
)
out = Path.home()/'beatmil/outputs/eval/xai_summary.json'
out.write_text(json.dumps({'beatmil_gradcam': summary}, indent=2))
print(json.dumps({'beatmil_gradcam': summary}, indent=2))
"
```

Expected ranges: QRS IoU ~0.50-0.75, P-wave IoU ~0.15-0.40, T-wave IoU ~0.20-0.45. Reviewers don't expect perfection — they expect a number that's defensible and beats LIME/random baselines.

### Step 20 — Generate all figures

```bash
cd ~/beatmil/src
python figures.py --results-dir ~/beatmil/outputs/eval --out-dir ~/beatmil/figures
ls ~/beatmil/figures/
# expect: fig2_confusion.png  fig3_reliability.png  fig4_risk_coverage.png  fig5_xai_iou.png
```

Figure 1 (architecture diagram) is drawn separately. Open the `01_PAPER_OUTLINE.md` ASCII diagram and re-draw it cleanly in **draw.io** or **TikZ** (~1 hour); the existing diagram from your prior paper can be adapted.

---

## Part 5: Paper assembly

### Step 21 — Set up the IEEE LaTeX template

```bash
cd ~/beatmil/paper
wget https://www.ieee.org/content/dam/ieee-org/ieee/web/org/conferences/conference-template-letter.zip
unzip conference-template-letter.zip
# OR clone the conference-paper template from Overleaf
```

You'll have `IEEEtran.cls`, `conference.tex`, and bibliography file. Use **Overleaf** for collaborative editing (paste in your university account).

### Step 22 — Port the draft markdown to LaTeX

In `~/beatmil/paper/main.tex`, structure as:

```latex
\documentclass[conference]{IEEEtran}
\usepackage{amsmath,amssymb,graphicx,booktabs,multirow,subcaption}
\usepackage[hidelinks]{hyperref}

\title{Beat-MIL: Hierarchical Multi-Instance Learning with Evidential Selective Prediction for Cross-Database ECG Arrhythmia Detection}

\author{Abheeshta V Aradhya, Aishwarya JR, Anantha Rama S, Rashmi N Ugarakhod\\
Dept. of Electronics and Communication, PES University, Bangalore, India}

\begin{document}
\maketitle
\begin{abstract}
... paste from 01_abstract_intro_related.md ...
\end{abstract}

\section{Introduction}
... paste & adapt from 01_abstract_intro_related.md ...

\section{Related Work}
\section{Methods}
\section{Experimental Setup}
\section{Results}
\section{Discussion}
\section{Conclusion}
\bibliographystyle{IEEEtran}
\bibliography{refs}
\end{document}
```

Copy paragraph text from the markdown drafts in `~/beatmil/paper/`. Convert `[REF]` placeholders to `\cite{key}` and add entries to `refs.bib`.

### Step 23 — Build the main results table (Table II)

Open the JSON files in `~/beatmil/outputs/eval/`. For each model, read `f1_macro`, `f1_macro_ci`, `accuracy`, `auroc_macro`, `kappa`, `ece`. Fill in:

```latex
\begin{table}[t]
\centering
\caption{Intra-database classification performance. 95\% CIs from 1000 bootstrap resamples.}
\label{tab:main}
\begin{tabular}{lcccc}
\toprule
Model & Acc & Macro F1 (95\% CI) & AUROC & $\kappa$ \\
\midrule
ResNet-1D \cite{hannun2019} & .XXX & .XXX [.XXX, .XXX] & .XXX & .XXX \\
CNN-LSTM \cite{oh2018}      & .XXX & .XXX [.XXX, .XXX] & .XXX & .XXX \\
ECGformer \cite{ecgformer}  & .XXX & .XXX [.XXX, .XXX] & .XXX & .XXX \\
\textbf{Beat-MIL (ours)}    & \textbf{.XXX} & \textbf{.XXX [.XXX, .XXX]} & \textbf{.XXX} & \textbf{.XXX} \\
\bottomrule
\end{tabular}
\end{table}
```

### Step 24 — Build the LODO table (Table III)

```latex
\begin{table}[t]
\centering
\caption{Leave-one-database-out (LODO) generalization. Test on the held-out database.}
\begin{tabular}{lccc}
\toprule
Model & LODO-MIT-BIH & LODO-CPSC & LODO-PTB-XL \\
\midrule
ResNet-1D & .XXX & .XXX & .XXX \\
CNN-LSTM & .XXX & .XXX & .XXX \\
ECGformer & .XXX & .XXX & .XXX \\
\textbf{Beat-MIL} & \textbf{.XXX} & \textbf{.XXX} & \textbf{.XXX} \\
\bottomrule
\end{tabular}
\end{table}
```

### Step 25 — Build the selective prediction table (Table IV)

```latex
\begin{table}[t]
\centering
\caption{Beat-MIL selective prediction at varying coverage levels.}
\begin{tabular}{lcc}
\toprule
Coverage & Selective Acc & Selective Macro F1 \\
\midrule
100\% (no abstention) & .XXX & .XXX \\
95\% & .XXX & .XXX \\
90\% & .XXX & .XXX \\
80\% & .XXX & .XXX \\
70\% & .XXX & .XXX \\
\bottomrule
\end{tabular}
\end{table}
```

### Step 26 — Build the XAI table (Table V)

```latex
\begin{table}[t]
\centering
\caption{Quantitative XAI evaluation on LUDB (IoU $\pm$ std, n=200 records).}
\begin{tabular}{lccc}
\toprule
Method & P-wave & QRS & T-wave \\
\midrule
Grad-CAM (Beat-MIL) & .XX $\pm$ .XX & .XX $\pm$ .XX & .XX $\pm$ .XX \\
\bottomrule
\end{tabular}
\end{table}
```

### Step 27 — Build the comparative gap table

Place this in Section I (Introduction). It's the 60-second smell test magnet:

```latex
\begin{table}[t]
\centering
\caption{Beat-MIL vs prior multi-database ECG arrhythmia methods on four
critical axes.}
\begin{tabular}{lcccc}
\toprule
Method & \shortstack{Granularity\\aware} & \shortstack{Cross-DB\\(LODO)} & \shortstack{Calibrated\\uncertainty} & \shortstack{Quant.\\XAI} \\
\midrule
Acharya 2017 \cite{acharya2017}  & $\times$ & $\times$ & $\times$ & $\times$ \\
Hannun 2019 \cite{hannun2019}    & $\times$ & $\times$ & $\times$ & $\times$ \\
Oh 2018 \cite{oh2018}            & $\times$ & $\times$ & $\times$ & $\times$ \\
ECGformer 2024                   & $\times$ & $\times$ & $\times$ & $\times$ \\
Prior arch (JCSSE 2026)          & $\times$ & $\times$ & $\times$ & $\times$ \\
\textbf{Beat-MIL (ours)}         & \checkmark & \checkmark & \checkmark & \checkmark \\
\bottomrule
\end{tabular}
\end{table}
```

### Step 28 — Verify page count

```bash
cd ~/beatmil/paper
pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
pdfinfo main.pdf | grep Pages    # must say "Pages: 6" or fewer
```

If over 6 pages: tighten Sections II (Related Work) and VI (Discussion) first. Don't cut from Methods or Results.

### Step 29 — Polish the architecture figure (Figure 1)

In **draw.io** or TikZ, build:
- Input ECG block at top
- 1D-CNN backbone box
- Beat extractor box producing N beat embeddings
- MIL gated attention pooling (highlight this — it's the novelty)
- Beat-level head and Bag-level evidential head as parallel outputs
- Show the three label paths: beat (MIT-BIH only), bag (all), and consistency arrow back from bag to beat

Export as `~/beatmil/figures/fig1_architecture.pdf` (vector format for IEEE).

### Step 30 — Internal review pass

All four authors review independently:
- [ ] Abstract claims match results numerically
- [ ] All citations resolve in `refs.bib`
- [ ] No `\TODO`, no placeholder `.XXX`, no Lorem Ipsum
- [ ] Figure captions are self-contained (a reader who only reads captions understands the paper)
- [ ] Table headers match the metrics described in Methods
- [ ] Section V results match Section III claims

### Step 31 — Reproducibility check

```bash
cd ~/beatmil
# Repo public on GitHub:
git remote add origin git@github.com:Aishwarya-jr2003/beat-mil.git
git push -u origin main
git tag bmeicon-submission
git push origin bmeicon-submission

# README.md must contain:
#   - One-command reproduce: bash scripts/reproduce_all.sh
#   - Pinned package versions (requirements.txt)
#   - Exact commit hash on submission day
```

### Step 32 — Submit

1. Compile final `main.pdf`.
2. Embed all fonts: `pdffonts main.pdf` — check every font has "yes" under "emb".
3. Upload PDF + IEEE copyright form to the BMEiCON 2026 submission system.
4. Verify by logging out and logging back in.
5. **Submit at least 24 hours before the Jun 30 deadline.** Don't wait until midnight.

---

## Definition of done

A submission is ready when **every box** below is checked:

- [ ] PDF compiles cleanly in IEEEtran, exactly ≤6 pages including references.
- [ ] Tables II, III, IV, V are filled with real numbers and bootstrap CIs.
- [ ] Figures 1, 2, 3, 4, 5 present, at 300+ DPI.
- [ ] McNemar p-values reported for Beat-MIL vs each baseline.
- [ ] GitHub repo is public; the tagged commit reproduces every result.
- [ ] No `\TODO`, no `.XXX`, no fabricated citations.
- [ ] All four abstract claims have a backing table or figure.
- [ ] Author emails and affiliations match the submission system.

---

## If something breaks

- **CUDA `no kernel image`** → wrong PyTorch wheel; re-run Step 4 with `--index-url https://download.pytorch.org/whl/cu128`.
- **MIT-BIH count too low** → some records lack MLII; that's expected. ~46/48 records have it.
- **CPSC R-peak detection slow** → reduce to first 2000 records for sanity, then run the full cohort overnight.
- **PTB-XL crashes on a specific record** → wrap the inner loop in `try/except` and skip.
- **Val F1 stuck below 0.5 after 5 epochs** → likely a label-mapping bug. Compare against the class distribution from Step 10/11.
- **GPU OOM at batch 128** → drop to 64; or set `torch.backends.cudnn.benchmark = True` in `train.py`.
- **LUDB Grad-CAM gives near-zero IoU on QRS** → check that `beat_positions_from_qrs_mask` is producing valid backbone-time indices (positive integers, fewer than 450 after the 3 downsamples).
- **Page count > 6** → trim Related Work (move details to footnotes) and Discussion. Keep all of Methods and Results.

---

## Cut list (if behind schedule)

Drop top-to-bottom:

1. LODO rotations 2 and 3 — keep only LODO-PTB-XL.
2. Selective prediction at 95% coverage — keep 70/80/90.
3. ECGformer baseline — keep ResNet-1D + CNN-LSTM.
4. Attention saliency separate from Grad-CAM (one method is enough).
5. Confidence intervals only on Beat-MIL (not all baselines).

Add back during the Aug 2 - Sep 10 camera-ready window. Each item is independently recoverable.
