# MIT-BIH inter-patient pipeline — BMEiCON 2026 resubmission

This is the **valid spine** for the resubmission: beat-level classification on
the one database with genuine beat-level annotations (MIT-BIH), under the
de Chazal inter-patient DS1/DS2 split, with the statistical rigor the JCSSE
reviewer asked for. It deliberately does **not** propagate CPSC/PTB-XL
rhythm/recording labels down to beats — that propagation was the central
methodological flaw in the rejected submission.

## Files

| File | What it is | Status |
|---|---|---|
| `data_pipeline.py` | MIT-BIH loading, DS1/DS2 split, AAMI label mapping, windowing, preprocessing, CWT scalograms, integrity checks | tested on synthetic WFDB records |
| `metrics.py` | accuracy / F1 / κ / AUROC, **bootstrap 95% CIs**, **McNemar** | self-tested |
| `baselines.py` | two SOTA-lineage baselines (deep 1-D CNN; CNN-LSTM) on the identical split | smoke-tested |
| `train.py` | one training/eval loop for proposed + baselines; saves predictions for cross-model McNemar | end-to-end tested |
| `proposed_model.py` | your model, verbatim, with two real bug fixes applied | smoke-tested incl. B=1 inference |
| `requirements.txt` | deps | — |

Everything was run in here on synthetic data to verify shapes, the split's
no-leakage property, label dropping (Q and non-beat annotations), z-scoring,
the new CWT, the full training loop, early stopping, bootstrap CIs, and McNemar.
The numbers from your real run come from the GPU instance — none are fabricated.

## Two bug fixes already applied to `proposed_model.py`

1. **`compute_loss`: `.squeeze()` → `.squeeze(1)`.** Bare `squeeze()` collapses a
   `(B,1)` tensor to a scalar when `B==1`. Hardened.
2. **`generate_cwt_scalogram` rewritten with PyWavelets.** The original used
   `scipy.signal.cwt` / `morlet2`, **both removed in scipy ≥ 1.15** — it would
   have raised `ImportError` on your instance. `pywt.cwt('morl')` is the drop-in.

One operational constraint to know: the task heads use `BatchNorm1d`, which
**cannot train on a size-1 batch**. `train.py` sets `drop_last=True` on the
training loader so this can't happen; just don't remove that.

## How to run (on the Vast.ai instance)

```bash
pip install -r requirements.txt

# 0. Sanity check the split before trusting anything (no data needed)
python data_pipeline.py        # prints integrity checks, must say [PASS]

# 1. Train the proposed model (1-D path — the clean headline model)
nohup python train.py --model proposed --data_dir /data/mitbih \
    --out runs/proposed --epochs 60 > runs/proposed.log 2>&1 &

# 2. Train the baselines on the IDENTICAL split
nohup python train.py --model cnn1d   --data_dir /data/mitbih --out runs/cnn1d   --epochs 60 > runs/cnn1d.log   2>&1 &
nohup python train.py --model cnnlstm --data_dir /data/mitbih --out runs/cnnlstm --epochs 60 > runs/cnnlstm.log 2>&1 &

# 3. (optional ablation) proposed model WITH the ResNet-34 scalogram branch.
#    Scalograms are precomputed once and memmapped — never recomputed per epoch.
nohup python train.py --model proposed --with_cwt --data_dir /data/mitbih \
    --out runs/proposed_cwt --epochs 60 > runs/proposed_cwt.log 2>&1 &

# 4. Statistical comparison (paired McNemar on the same DS2 test set)
python train.py --compare runs/proposed runs/cnn1d runs/cnnlstm
```

`--data_dir` is the folder holding the MIT-BIH records (`100.dat/.hea/.atr`, …).
Download once with: `wfdb.dl_database('mitdb', '/data/mitbih')`.

Monitor a run: `tail -f runs/proposed.log` (per-epoch val macro-F1 is printed).

Each run writes `metrics.json` (with bootstrap CIs), `predictions.npz` (for
McNemar), and `best.pt` (checkpoint at best val macro-F1).

## Things to decide / sanity-check early (these affect the paper)

- **Window size.** Defaults to 3600 (10 s) to match your model's input and the
  LSTM/attention "rhythm context" argument. Each example is a 10-s window
  labelled by its centre beat. If a reviewer prefers a tighter single-beat
  window, set `--window_size 360`; the model accepts any length. Decide which
  framing you'll defend and state it explicitly in the methods.
- **The 2-D CWT branch earns its keep?** Run `runs/proposed` vs
  `runs/proposed_cwt` and McNemar them. In the original paper the branch added
  0.2% accuracy for ~88% of the parameters; if that gain isn't significant
  under McNemar, the honest headline model is 1-D-only and the ResNet branch
  becomes an ablation, not the centrepiece.
- **The prediction head has no real labels** (`pred_target = cls_target > 0`).
  With proper stats in the paper, a reviewer will ask why a "task" with
  synthetic targets is in the multi-task claim. Cleanest framing: a beat
  classifier with an auxiliary binary detection head, and be upfront that
  detection is a deterministic coarsening of classification.

## What still needs building (next, on your word)

- **LUDB quantitative XAI** (IoU/Dice of Grad-CAM vs expert P/QRS/T
  delineations). This is the strongest genuinely-novel contribution and is
  independent of everything above.
- **Calibration** analysis (ECE / reliability curve).
- The paper rewrite + response-to-reviewers letter, scoped honestly to
  "beat-level MIT-BIH under inter-patient evaluation."
