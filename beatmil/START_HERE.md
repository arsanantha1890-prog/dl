# Beat-MIL — Start Here

This is the master document. Read it once, end to end, before running anything.

**Target venue:** BMEiCON 2026 (IEEE conference, Fukuoka, Japan)
**Submission deadline:** June 30, 2026
**Page limit:** 6 pages, IEEE conference format
**Authors:** Abheeshta V Aradhya, Aishwarya JR, Anantha Rama S, Rashmi N Ugarakhod (PES University, Bangalore)

---

## What is Beat-MIL?

A hierarchical multi-instance learning framework for ECG arrhythmia detection that addresses three structural problems in the existing ECG-ML literature:

1. **Label-granularity mismatch.** Public ECG databases label data at different levels (MIT-BIH: per beat, CPSC: per rhythm episode, PTB-XL: per recording). Prior multi-database work silently coerces them to one level. We model the hierarchy explicitly via MIL.
2. **Within-database evaluation bias.** Prior work tests on the same database it trained on. We do leave-one-database-out (LODO) — the honest test of cross-source generalization.
3. **No uncertainty calibration.** Standard softmax can't say "I don't know." We add an evidential head producing per-prediction uncertainty for principled abstention.

Plus quantitative XAI validation against expert P-, QRS-, T-wave annotations from LUDB.

---

## Directory layout

After unzipping, you have:

```
beatmil/
├── START_HERE.md              ← you are here
├── WALKTHROUGH.md             ← detailed step-by-step (Parts 1-5)
├── README.md                  ← file index and "what does each file do"
├── requirements.txt           ← pinned Python dependencies
├── refs.bib                   ← BibTeX bibliography
├── paper.tex                  ← IEEE paper template (text pre-written)
├── scripts/                   ← orchestration shell scripts
│   ├── setup.sh               ← one-time environment setup
│   ├── download_data.sh       ← fetch MIT-BIH, PTB-XL, LUDB
│   ├── smoke_test.sh          ← 2-min sanity check
│   ├── run_all.sh             ← THE master pipeline (one command)
│   └── make_paper.sh          ← LaTeX -> PDF
├── src/                       ← Python source (22 modules)
│   ├── mil_pooling.py         ← gated attention MIL (Ilse 2018)
│   ├── evidential.py          ← Dirichlet evidential head (Sensoy 2018)
│   ├── consistency.py         ← H-MIL consistency loss (our novelty)
│   ├── beatmil.py             ← full Beat-MIL model (3.01M params)
│   ├── unified_dataset.py     ← granularity-aware dataset glue
│   ├── mitbih_loader.py       ← MIT-BIH beat-level loader
│   ├── cpsc.py                ← CPSC 2018 rhythm-bag loader
│   ├── ptbxl.py               ← PTB-XL recording-bag loader
│   ├── ludb.py                ← LUDB loader (XAI only)
│   ├── focal.py               ← focal CE for beat head
│   ├── baselines.py           ← ResNet-1D, CNN-LSTM, ECGformer
│   ├── train.py               ← main training entry point
│   ├── eval_metrics.py        ← metrics + bootstrap + McNemar + calibration
│   ├── xai.py                 ← Grad-CAM + LUDB IoU/Dice
│   ├── figures.py             ← generate paper figures
│   ├── tables.py              ← generate LaTeX tables from eval JSONs
│   ├── cache_specs.py         ← cache spec lists (one-time)
│   ├── sanity_mitbih.py       ← MIT-BIH loader sanity check
│   ├── sanity_cpsc_ptbxl.py   ← CPSC + PTB-XL sanity check
│   ├── run_eval.py            ← evaluate every checkpoint
│   ├── run_mcnemar.py         ← pairwise statistical tests
│   ├── run_xai.py             ← run Grad-CAM IoU on LUDB
│   └── integration_test.py    ← end-to-end smoke test
└── docs/                      ← strategy and paper drafts (markdown)
    ├── 00_MASTER_PLAN.md      ← strategic positioning
    ├── 01_PAPER_OUTLINE.md    ← section-by-section structure
    ├── 02_REBUTTAL_TABLE.md   ← reviewer-by-reviewer rebuttal
    ├── 03_SPRINT_4WEEK.md     ← 4-week sprint timeline
    ├── 01_abstract_intro_related.md ← drafted prose sections
    └── 02_methods_experiments.md    ← drafted methods section
```

---

## Five-minute orientation: how to run everything

The shortest path from zero to a compiled PDF is **four commands**:

```bash
# 1. Drop the zip in your home directory and unzip
cd ~ && unzip beatmil.zip
cd ~/beatmil

# 2. Set up Python environment + PyTorch + dependencies (5 min)
bash scripts/setup.sh

# 3. Download the four ECG databases (~3 GB, mostly waiting)
bash scripts/download_data.sh
# (For CPSC 2018 follow the on-screen manual instructions —
#  Kaggle mirror works fine)

# 4. Run the entire pipeline (15-30 hours including training)
bash scripts/run_all.sh
```

When `run_all.sh` finishes you have `paper.pdf` ready for submission.

**For impatience:** before committing to the 15-30 hour pipeline, run
`bash scripts/smoke_test.sh` (90 seconds) to catch any environment issues.

---

## What `run_all.sh` does, in detail

| Stage | What runs | Time on RTX 5090 |
|-------|-----------|------------------|
| 1 | `cache_specs.py` — pre-compute spec lists for the 3 training databases | ~30 min |
| 2 | `train.py beatmil intra-db` — train Beat-MIL on combined data | ~3-4 h |
| 3 | `train.py {resnet1d,cnnlstm,ecgformer} intra-db` — 3 baselines | ~9-12 h |
| 4 | LODO-PTB-XL rotation: Beat-MIL + 3 baselines | ~6 h |
| 5 | `run_eval.py` — metrics + bootstrap CIs for every checkpoint | ~15 min |
| 6 | `run_mcnemar.py` — pairwise tests with Holm-Bonferroni | <1 min |
| 7 | `run_xai.py` — Grad-CAM IoU on LUDB | ~10 min |
| 8 | `figures.py` — generate Figures 2-5 | <1 min |
| 9 | `tables.py` — generate all LaTeX tables from eval JSONs | <1 min |
| 10 | `make_paper.sh` — pdflatex + bibtex × 4 passes | ~30 s |

Total: 15-30 hours wall-clock. Run overnight at minimum.

**To save 6 hours:** `bash scripts/run_all.sh --skip-lodo` (submit with intra-DB results only)

**To skip paper compile:** `bash scripts/run_all.sh --skip-paper`

---

## What you get when it finishes

```
~/beatmil/
├── checkpoints/
│   ├── beatmil_intra-db/best.pt        ← trained Beat-MIL
│   ├── {resnet1d,cnnlstm,ecgformer}_intra-db/best.pt
│   └── */lodo_ptbxl/best.pt
├── outputs/eval/
│   ├── beatmil_intra-db.json           ← all metrics + bootstrap CIs
│   ├── *_intra-db.json                  ← baseline metrics
│   ├── mcnemar.json                     ← p-values, Holm thresholds
│   ├── xai_summary.json                 ← LUDB IoU/Dice
│   ├── calibration.json                 ← ECE, Brier
│   └── *.npz                            ← raw predictions, riskcov curves
├── figures/
│   ├── fig2_confusion.png
│   ├── fig3_reliability.png
│   ├── fig4_risk_coverage.png
│   └── fig5_xai_iou.png
└── paper/
    ├── paper.tex
    ├── refs.bib
    ├── tables/                          ← 12 .tex files auto-generated
    ├── figures/                         ← copied from above
    └── paper.pdf                        ← SUBMIT THIS
```

---

## Three checkpoints during the run

**Checkpoint A — after `smoke_test.sh`.** Confirms environment is healthy.
*Stop here* if you see GPU errors or import errors. Don't waste 12 hours of training on a broken setup.

**Checkpoint B — after Stage 2 (Beat-MIL intra-DB).**
Check `~/beatmil/checkpoints/beatmil_intra-db/history.json`. If validation
macro F1 plateaus below 0.80, something is wrong with the data pipeline —
likely a label-mapping bug. Run `python sanity_mitbih.py` to debug.

**Checkpoint C — after Stage 8 (figures).**
Eyeball the four figures. If `fig5_xai_iou.png` shows QRS IoU < 0.30, the
Grad-CAM hook may be on the wrong layer. The other figures should look
clean and self-explanatory.

---

## What you still have to do by hand (not automated)

1. **Figure 1 (architecture diagram).** Draw in draw.io or TikZ. Should
   show: 1D-CNN backbone → beat embeddings → gated MIL pooling → bag
   head (evidential) + beat head, with the consistency arrow between them.
   Adapt the existing figure from your prior paper. Place at
   `~/beatmil/paper/figures/fig1_architecture.pdf`.

2. **Read through `paper.pdf` once.** The text is pre-written but should
   be read for tone, flow, and any remaining `\input{...}` placeholders
   that didn't get filled (will show as `--` in the rendered PDF).

3. **Page count check.** If `paper.pdf` is >6 pages, the `make_paper.sh`
   output flags it. Trim Section II (Related Work) first; keep Methods
   and Results untouched.

4. **Make GitHub repo public.** Tag the submission commit:
   `git tag bmeicon-2026-submission && git push --tags`

5. **Submit to BMEiCON.** PDF + IEEE copyright form. **Submit 24 hours
   before deadline** (don't wait until June 30 23:59).

---

## If something breaks

See the "If something breaks" table at the bottom of `WALKTHROUGH.md`.
Most common issues:

- **CUDA `no kernel image`** → wrong PyTorch wheel. Re-run `scripts/setup.sh`.
- **`MITBIHLoader` returns < 50k samples** → check `MITBIH_TO_AAMI` in
  `unified_dataset.py`; `L` and `R` (LBBB, RBBB) must map to `N`.
- **`paper.pdf` not generated** → check `~/beatmil/logs/10_paper.log`
  and `/tmp/latex_pass1.log` for the actual LaTeX error.
- **Tables show `--`** → that eval JSON doesn't exist; check the
  corresponding training stage didn't fail silently.

---

## Quality bar for submission

The paper is ready when **every** box is checked:

- [ ] `paper.pdf` exists and is exactly ≤6 pages
- [ ] No `--` in any table (means an eval JSON is missing)
- [ ] All 5 figures present (Figure 1 hand-drawn, 2-5 auto-generated)
- [ ] McNemar p-values < 0.05 for at least one baseline comparison
- [ ] Bootstrap CIs present on every metric in Table II
- [ ] Abstract claims numerically match the tables
- [ ] GitHub repo is public; reproducibility tag in place
- [ ] At least one co-author has read end-to-end and signed off

---

## Why this paper should be accepted at BMEiCON

The original paper got 3/5/5 from JCSSE 2026 reviewers on technical
content. JCSSE is a broader-scope venue; BMEiCON is purpose-built for
biomedical signal processing where ECG-cardiology work is core scope.
The only damaging reviewer comment was "limited novelty," and Beat-MIL
addresses that with four genuine contributions (H-MIL, evidential head,
LODO, quantitative XAI). The second reviewer's label-granularity
criticism becomes the central methodological contribution rather than
an attack surface. The paper is honest about LODO performance gaps
(which is rare and reviewers appreciate it).

Estimated acceptance probability if executed cleanly: **80%+.**

---

## When to come back to me

Three natural checkpoints during your build:

1. **After `smoke_test.sh` passes.** Reply with the output or "smoke clean."
2. **After Stage 2 (Beat-MIL intra-DB training).** Share the validation F1.
3. **After Stage 10 (paper compiled).** Share `paper.pdf` if there's any
   tone or claim that worries you.

Outside these points, ping with concrete errors only (paste exact log).
Don't ping for general progress — that wastes your time.

Good luck. The plan is solid. Execute.
