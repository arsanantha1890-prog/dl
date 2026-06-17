# Beat-MIL Code Scaffold

PyTorch 2.x. Target: runs on Day 1 of Week 1 with toy data, real data plugged in incrementally.

```
beatmil/
├── configs/
│   ├── base.yaml           # shared hyperparameters
│   ├── beatmil.yaml        # our model
│   ├── b1_resnet1d.yaml    # baselines
│   ├── b2_cnnlstm.yaml
│   ├── b3_ecgformer.yaml
│   ├── b4_ecgtransform.yaml
│   ├── b5_ecgbert.yaml
│   └── b6_prior.yaml
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── mitbih.py          # beat-level loader
│   │   ├── cpsc2018.py        # rhythm-bag loader
│   │   ├── ptbxl.py           # recording-bag loader
│   │   ├── ludb.py            # XAI eval only — never trained on
│   │   ├── unified.py         # combines all three with granularity tags
│   │   ├── splits.py          # patient-level splits + LODO rotations
│   │   ├── preprocessing.py   # bandpass, baseline, z-score, R-peak detection
│   │   └── augmentation.py    # online ECG augmentations
│   ├── models/
│   │   ├── __init__.py
│   │   ├── backbone.py        # 1D CNN + Bi-LSTM + self-attn → beat embeddings
│   │   ├── mil_pooling.py     # gated attention MIL (Ilse 2018)
│   │   ├── heads.py           # beat-head, bag-head, evidential output
│   │   ├── beatmil.py         # full Beat-MIL model
│   │   └── baselines/         # 6 baselines as submodules
│   │       ├── resnet1d.py
│   │       ├── cnnlstm.py
│   │       ├── ecgformer.py
│   │       ├── ecgtransform.py
│   │       ├── ecgbert.py
│   │       └── prior.py
│   ├── losses/
│   │   ├── focal.py
│   │   ├── evidential.py      # Sensoy 2018 Dirichlet loss
│   │   └── consistency.py     # KL between bag and pooled-beat predictions
│   ├── eval/
│   │   ├── metrics.py         # acc, F1, AUROC, ECE, Brier
│   │   ├── selective.py       # risk-coverage, AUARC, selective F1
│   │   ├── calibration.py     # reliability diagram, ECE
│   │   ├── stats.py           # bootstrap CI, McNemar, Holm-Bonferroni
│   │   ├── xai.py             # Grad-CAM, attention saliency
│   │   └── iou_ludb.py        # IoU/Dice vs LUDB P/QRS/T regions
│   ├── train.py               # main training loop
│   ├── eval_intra.py          # intra-DB inter-patient evaluation
│   ├── eval_lodo.py           # leave-one-database-out evaluation
│   ├── eval_selective.py      # selective prediction analysis
│   └── eval_xai.py            # XAI validation on LUDB
├── notebooks/
│   ├── 01_data_sanity.ipynb
│   ├── 02_model_smoke_test.ipynb
│   ├── 03_results_tables.ipynb
│   └── 04_figures.ipynb
├── scripts/
│   ├── download_data.sh
│   ├── run_all_baselines.sh
│   ├── run_lodo_rotations.sh
│   └── reproduce_all.sh        # one-command reproduce
├── tests/
│   ├── test_data_loaders.py
│   ├── test_mil_pooling.py
│   ├── test_evidential.py
│   └── test_smoke.py           # 1-batch end-to-end
├── README.md
├── requirements.txt
└── LICENSE
```

## Day 1 Goal

Smoke test: `python -m src.train --config configs/beatmil.yaml --smoke` runs one batch through the full Beat-MIL forward+backward+optimizer step with synthetic data, no NaN, no shape errors.

## Day 7 Goal (end of Week 1)

Real data loaders working for all three databases. Intra-DB training of Beat-MIL converges, gets a sensible macro F1 on validation.
```
