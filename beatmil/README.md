# Beat-MIL: Run Order

Three shell scripts orchestrate everything. Run them in this order:

```bash
# 1. ONE-TIME SETUP (per machine, ~5 minutes)
#    creates directories, installs PyTorch + dependencies, verifies GPU
bash setup.sh

# After setup:
#   - place all 22 .py files into ~/beatmil/src/
#   - symlink your datasets into ~/beatmil/data/{mitbih,cpsc2018,ptbxl}
#   - download LUDB into ~/beatmil/data/ludb/

# 2. SMOKE TEST (< 2 minutes)
#    catches setup/code/GPU problems cheaply
bash smoke_test.sh

# 3. FULL PIPELINE (15-30 hours depending on GPU and dataset sizes)
#    runs caching -> training -> evaluation -> figures
bash run_all.sh
#    or skip the (slow) LODO rotations:
bash run_all.sh --skip-lodo
```

After `run_all.sh` finishes, you have:
- `~/beatmil/checkpoints/<model>_<mode>/best.pt` — trained models
- `~/beatmil/outputs/eval/*.json` — all metrics with bootstrap CIs
- `~/beatmil/outputs/eval/mcnemar.json` — statistical tests
- `~/beatmil/outputs/eval/xai_summary.json` — LUDB IoU/Dice
- `~/beatmil/figures/fig{2,3,4,5}_*.png` — paper figures
- `~/beatmil/logs/*.log` — per-stage logs

Then follow steps 21-32 of `WALKTHROUGH.md` for paper assembly.

---

## What each Python file does

### Core architecture (immutable foundation — built and tested first)
- `mil_pooling.py` — gated attention MIL pooling (Ilse 2018)
- `evidential.py` — Dirichlet evidential output + selective prediction
- `consistency.py` — bag-vs-pooled-beat KL loss (the H-MIL regularizer)
- `beatmil.py` — full Beat-MIL model (3.01M params)
- `unified_dataset.py` — granularity-aware dataset glue + splits
- `focal.py` — focal cross-entropy for the beat head

### Data loaders
- `mitbih_loader.py` — MIT-BIH beat-level loader
- `cpsc.py` — CPSC 2018 rhythm-bag loader
- `ptbxl.py` — PTB-XL recording-bag loader
- `ludb.py` — LUDB XAI loader (never used in training)

### Baselines
- `baselines.py` — ResNet-1D, CNN-LSTM, ECGformer (3 models)

### Training
- `train.py` — main training entry point (Beat-MIL + baselines, intra-DB + LODO)

### Evaluation
- `eval_metrics.py` — metrics + bootstrap CI + McNemar + calibration + selective
- `xai.py` — Grad-CAM hook + IoU/Dice machinery

### Orchestration (the scripts run_all.sh chains together)
- `cache_specs.py` — pre-compute and cache spec lists
- `sanity_mitbih.py` — verify MIT-BIH loader, save sample figure
- `sanity_cpsc_ptbxl.py` — verify CPSC/PTB-XL loaders
- `run_eval.py` — evaluate every checkpoint, save metrics JSON + raw preds
- `run_mcnemar.py` — pairwise tests with Holm-Bonferroni
- `run_xai.py` — Grad-CAM IoU on LUDB
- `figures.py` — generate fig 2-5 from saved eval outputs

### Tests
- `integration_test.py` — end-to-end smoke test of the full Beat-MIL loss

---

## Manual override examples

Run just one piece, without the full pipeline:

```bash
# train Beat-MIL only (skip baselines, LODO, eval)
python train.py --model beatmil --mode intra-db \
    --data-root ~/beatmil/data --out-dir ~/beatmil/checkpoints

# train one baseline
python train.py --model resnet1d --mode intra-db \
    --data-root ~/beatmil/data --out-dir ~/beatmil/checkpoints

# only run a single LODO rotation
python train.py --model beatmil --mode lodo --held-out ptbxl \
    --data-root ~/beatmil/data --out-dir ~/beatmil/checkpoints

# re-run eval after retraining one model
python run_eval.py

# re-run McNemar after changing a baseline
python run_mcnemar.py

# regenerate figures only
python figures.py --results-dir ~/beatmil/outputs/eval \
                  --out-dir ~/beatmil/figures
```

---

## Definitions of done at each stage

| After                 | You should see                                                    |
|-----------------------|-------------------------------------------------------------------|
| `setup.sh`            | `SM: (12, 0)` in the output and `requirements.txt` written        |
| `smoke_test.sh`       | All 5 module tests pass + GPU forward/backward works              |
| `cache_specs.py`      | ~150k samples cached in `outputs/cache/*.pkl`                     |
| `sanity_mitbih.py`    | ~100k samples, class N ~84%, R-peak figure looks correct          |
| `train.py` (Beat-MIL) | val macro F1 in `history.json` ≥ 0.85 after ~30 epochs            |
| `run_eval.py`         | One JSON per checkpoint in `outputs/eval/`                        |
| `run_mcnemar.py`      | `mcnemar.json` with p-values < 0.05 for at least one baseline     |
| `run_xai.py`          | `xai_summary.json` with QRS IoU > 0.4                             |
| `figures.py`          | 4 PNG files in `figures/`                                         |

If any "you should see" doesn't match — stop and investigate before continuing. Errors compound.
