# Beat-MIL: 4-Week Sprint to BMEiCON 2026

**Today:** Mon Jun 2, 2026
**Deadline:** Tue Jun 30, 2026, 23:59 (local)
**Notification:** Aug 1, 2026
**Camera-ready:** Sep 10, 2026 (6-week post-acceptance polish window)
**Conference:** Oct 10-13, Fukuoka, Japan

---

## Sprint Philosophy

What lands in the Jun 30 submission must be **defensible, not maximal**. Anything that would be merely "nice to have" is deferred to the camera-ready window. The submission must:

1. Make every claim it states verifiable from the released code on the day of submission.
2. Have all five tables/figures populated with real numbers — no `\TODO` in the submitted PDF.
3. Pass the "60-second smell test" — a reviewer skimming for 1 minute sees novelty, results, honesty.

**Hard rule:** if a deliverable is behind by Day 14 (end of Week 2), we drop it from the submission and add it to the camera-ready cut list, no exceptions.

---

## Week 1 — Data and Architecture (Jun 2 – 8)

Goal: by Sunday Jun 8, the full Beat-MIL forward+backward+optimizer step runs on real (not synthetic) MIT-BIH data in a Python notebook on the RTX 5090.

### Day 1 (Mon Jun 2) — Today
- [x] Code scaffold done: `mil_pooling.py`, `evidential.py`, `consistency.py`, `beatmil.py`, `integration_test.py` all run and smoke-tests pass.
- [ ] **Tonight:** Download datasets. MIT-BIH and PTB-XL via wfdb; CPSC 2018 via official mirror; LUDB via PhysioNet. Verify file counts. (~3 hours total, mostly waiting.)
- [ ] **Tonight:** Verify CUDA on the RTX 5090. `python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"`

### Day 2 (Tue Jun 3)
- [ ] Build `src/data/mitbih.py`: load records via `wfdb.rdrecord`, extract lead II, resample 360 Hz, R-peak windowing, AAMI label mapping.
- [ ] Patient-level split: 70/15/15. Save split as JSON for reproducibility.
- [ ] Sanity notebook: load 5 random samples, plot waveforms, verify R-peak alignment, verify class distribution matches Table I of original paper.

### Day 3 (Wed Jun 4)
- [ ] Build `src/data/cpsc2018.py`: rhythm-level loader. Each 10-sec window gets a *bag label* from the rhythm class. R-peaks via Pan-Tompkins to populate beat positions for the MIL bag.
- [ ] Build `src/data/ptbxl.py`: recording-level loader. SCP-ECG to AAMI mapping. Each 10-sec sub-window inherits the recording's AAMI label.
- [ ] Both loaders return the same dict structure as MIT-BIH but with `has_beat_labels=False` and `beat_targets=-1`.

### Day 4 (Thu Jun 5)
- [ ] Build `src/data/unified.py`: combines all three loaders into one `torch.utils.data.Dataset` with a `granularity` tag per sample.
- [ ] Augmentation module (`src/data/augmentation.py`): Gaussian noise, time-warp, amp scale, baseline wander, powerline. **Apply only at training time, not at eval.**
- [ ] LUDB loader (`src/data/ludb.py`) — read-only. Returns waveform + P/QRS/T boundary masks. **Never combined with training data.**

### Day 5 (Fri Jun 6)
- [ ] Full Beat-MIL training run on a single epoch with real data. Mixed batch. Log: total loss, bag loss, beat loss, consistency loss, mean vacuity.
- [ ] Verify all three losses are non-zero and decreasing.
- [ ] **GO/NO-GO Gate 1:** Does the validation macro F1 after 5 epochs exceed 0.50 (just better than majority class)?
   - YES → proceed to Week 2.
   - NO → debug data pipeline (most likely culprit) over weekend.

### Day 6 (Sat Jun 7) — buffer
- [ ] Buffer day. Fix whatever is broken from Day 5.
- [ ] If on track: start the LUDB IoU evaluation harness (`src/eval/iou_ludb.py`).

### Day 7 (Sun Jun 8) — buffer
- [ ] Buffer day.
- [ ] If on track: implement `src/eval/metrics.py` (acc, F1, per-class F1, AUROC, Cohen's κ, confusion matrix).
- [ ] Tag git: `git tag v0.1-data-ready`

---

## Week 2 — Training and Baselines (Jun 9 – 15)

Goal: by Sun Jun 15, Beat-MIL plus 3 baselines all trained and evaluated intra-DB. Numbers logged to a CSV.

### Day 8 (Mon Jun 9)
- [ ] Implement focal loss correctly (`src/losses/focal.py`). Plug into the beat-level head.
- [ ] Full training run: Beat-MIL, 50 epochs, early stopping. Log to TensorBoard/wandb.
- [ ] Should converge in ~3-4 hours on a single RTX 5090.

### Day 9 (Tue Jun 10)
- [ ] Baseline B1 — ResNet-1D (`src/models/baselines/resnet1d.py`). 34-layer with softmax. Train.
- [ ] Baseline B2 — CNN-LSTM (`src/models/baselines/cnnlstm.py`). Train.

### Day 10 (Wed Jun 11)
- [ ] Baseline B3 — ECGformer / lightweight transformer encoder (`src/models/baselines/ecgformer.py`). Train.
- [ ] Run all 3 baselines on the same intra-DB protocol with bootstrap CIs.

### Day 11 (Thu Jun 12)
- [ ] Build `src/eval/stats.py`: bootstrap CIs (n=1000) and McNemar's test.
- [ ] Generate **Table II** (main classification table) and **confusion matrix** for Beat-MIL.

### Day 12 (Fri Jun 13)
- [ ] Ablation study runs:
  - Beat-MIL full
  - −consistency loss
  - −evidential (softmax instead)
  - −MIL (treat all samples as beat-level, majority vote)
- [ ] Each ablation: ~3 hours.

### Day 13 (Sat Jun 14) — buffer
- [ ] Buffer. If on track: start writing Section V.A (Main Results) and V.C (Ablation) with real numbers.

### Day 14 (Sun Jun 15) — Cutoff Day
- [ ] **GO/NO-GO Gate 2:** Are Beat-MIL macro F1 (intra-DB) and 3 baselines complete with bootstrap CIs?
   - YES → proceed to Week 3 LODO + XAI.
   - NO → drop one baseline. Drop one ablation. **Move on regardless.**
- [ ] Tag git: `v0.2-intra-db-done`

---

## Week 3 — LODO and XAI (Jun 16 – 22)

Goal: by Sun Jun 22, one LODO rotation done, LUDB IoU evaluation done, calibration figures generated.

### Day 15 (Mon Jun 16)
- [ ] LODO rotation: train on MIT-BIH + CPSC, test on PTB-XL (the hardest rotation — PTB-XL has the most diverse recording conditions).
- [ ] Beat-MIL + 3 baselines, all on this rotation. ~12 hours wall-clock for all 4.

### Day 16 (Tue Jun 17)
- [ ] (Time permitting) second LODO rotation: train on MIT-BIH + PTB-XL, test on CPSC. **If short on time, skip this and report only 1 rotation in the submission.**
- [ ] Generate Table III (LODO results) — submission may have one column or three.

### Day 17 (Wed Jun 18)
- [ ] LUDB IoU evaluation. Run Beat-MIL on all 200 LUDB records, extract Grad-CAM saliency, compare to expert P/QRS/T masks.
- [ ] Compute IoU and Dice per anatomical region (P, QRS, T) and overall.
- [ ] Generate Figure 5 (IoU bar chart).

### Day 18 (Thu Jun 19)
- [ ] Calibration analysis: reliability diagram, ECE, Brier on intra-DB test set.
- [ ] Risk-coverage curve: vary the abstention threshold τ across [0.1, 0.9], compute selective macro F1 at each.
- [ ] Generate Figure 3 (reliability diagram) and Figure 4 (risk-coverage curve).

### Day 19 (Fri Jun 20)
- [ ] Selective prediction operating-point table: macro F1 at coverage ∈ {70%, 80%, 90%, 95%}.

### Day 20 (Sat Jun 21) — buffer
- [ ] Buffer. Catch up on anything that slipped.
- [ ] Begin Sections IV (Experimental Setup) and V.D-V.F (Calibration, Selective, XAI) writing.

### Day 21 (Sun Jun 22) — Cutoff
- [ ] **GO/NO-GO Gate 3:** Are all five figures and all four tables populated with real numbers?
   - YES → proceed to Week 4 writing.
   - NO → freeze experiments, write paper with what we have.
- [ ] Tag git: `v0.3-all-experiments-done`

---

## Week 4 — Writing, Polish, Submit (Jun 23 – 30)

Goal: by Tue Jun 30 noon, IEEE-formatted PDF submitted with all supporting material.

### Day 22 (Mon Jun 23)
- [ ] Convert paper-draft Markdown files (`paper/01_*.md`, `paper/02_*.md`) to LaTeX in the **IEEE conference template** (`IEEEtran.cls`).
- [ ] Fill in all the `\PLACEHOLDER{}` numbers from CSV experiment logs.
- [ ] Verify page count: must be ≤6 pages for BMEiCON.

### Day 23 (Tue Jun 24)
- [ ] Write Section V (Results) — the longest section, ~1200 words. Use the structure in `paper/02_methods_experiments.md`.
- [ ] Generate the **comparative gap table** for Section I (Beat-MIL vs prior work on 4 axes: granularity-aware / cross-DB eval / calibrated uncertainty / quantitative XAI). This is the "60-second smell test" magnet.

### Day 24 (Wed Jun 25)
- [ ] Write Section VI (Discussion) and Section VII (Conclusion). Be explicit about limitations.
- [ ] Polish architecture figure (Figure 1).
- [ ] Polish all caption text.

### Day 25 (Thu Jun 26)
- [ ] Internal review pass 1 (Abheeshta + Aishwarya + Anantha review independently, share comments).
- [ ] Address all comments. Re-read the BMEiCON 2026 author guidelines once more.

### Day 26 (Fri Jun 27)
- [ ] Final polish: references (BibTeX), figure resolution (>=300 dpi), no orphan/widow lines, no equations cut by column breaks.
- [ ] Run aspell/Grammarly. Verify no informal phrasing.
- [ ] Verify GitHub repo is **public** with README pointing to the submitted commit hash.

### Day 27 (Sat Jun 28) — buffer
- [ ] Buffer for last-minute issues. Anonymize repo if double-blind required.
- [ ] Verify all author affiliations and emails.

### Day 28 (Sun Jun 29)
- [ ] **Submit a day early.** Upload PDF, supplementary materials, copyright form to the BMEiCON submission system.
- [ ] Verify the upload via a logout/login cycle.

### Mon Jun 30
- [ ] Final buffer. Submit if not already done.

---

## Cut-list (in priority order — drop top-to-bottom if behind)

1. Second LODO rotation (Day 16) — submission can report 1 of 3.
2. CPSC LODO rotation (Day 16) — if even the second drops, only LODO-PTB-XL remains.
3. Selective prediction at 95% coverage — table can show 70/80/90 only.
4. Attention saliency IoU (separate from Grad-CAM IoU) — Grad-CAM alone suffices.
5. Baseline B3 (ECGformer) — fall back to ResNet-1D and CNN-LSTM only.
6. Ablation: -evidential — keep only -consistency and -MIL.

Each cut is recoverable in the camera-ready window. None of them invalidates the headline claims.

---

## What we add for camera-ready (Aug 2 - Sep 10)

After acceptance, in the 6-week polish window:
- Add B4, B5, B6 baselines (ECGTransForm, ECGBert, prior architecture).
- Run the remaining LODO rotations.
- Add attention saliency + LIME + Integrated Gradients to the XAI comparison.
- Expand ablation to the full 7-row table.
- Add per-class breakdowns and confusion matrices for each LODO rotation.
- Strengthen the comparative-gap table with more recent (2025-2026) ECG papers.

---

## Definition of Done (Jun 30 submission)

A submission is acceptable when:

- [ ] PDF compiles cleanly in IEEEtran, ≤6 pages including references.
- [ ] Tables I, II, III, IV are all populated with real numbers and bootstrap CIs.
- [ ] Figures 1 (architecture), 2 (confusion), 3 (reliability), 4 (risk-coverage), 5 (IoU) all present.
- [ ] McNemar p-values reported for Beat-MIL vs each baseline.
- [ ] GitHub repo is public, code reproduces every result with one command.
- [ ] No `\TODO`, no placeholder values, no fake citations.
- [ ] The four claims in the abstract are each backed by a table or figure.
