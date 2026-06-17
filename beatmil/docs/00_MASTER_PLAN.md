# Beat-MIL: Master Execution Plan

**Target:** 6-page IEEE conference paper, single RTX 5090, 6-week timeline.

---

## 1. The Honest Positioning Statement

We are NOT claiming MIL for ECG is new. Prior work exists (Wang et al. 2022, WSDL-AD 2022, Cross-modal MIL 2023). These trained on one annotation granularity at a time.

We ARE claiming a new, defensible contribution:

> **The first ECG framework to jointly learn from beat-level, rhythm-level, AND recording-level supervision simultaneously across heterogeneous databases, with calibrated evidential uncertainty enabling clinically meaningful selective prediction, and quantitative anatomical validation of explanations.**

This is a four-part novelty bundle. No single part is trivially attackable.

---

## 2. The Four Pillars of Contribution

### Pillar A: Hierarchical Multi-Granularity Multi-Instance Learning (H-MIL)
- **What's new:** Joint training under three label granularities at once, with a learned consistency constraint between levels.
- **Math:** Beat instances `b_i` aggregate via gated attention into rhythm bag `R_j`; rhythm bags aggregate into recording bag `D_k`. Loss = `L_beat (MIT-BIH only) + L_rhythm (CPSC) + L_record (PTB-XL) + λ·L_consistency`.
- **Why nobody can attack:** The closest prior work (Wang 2022) does sequential training. We do joint. We also add the consistency loss, which is genuinely new.

### Pillar B: Evidential Selective Prediction Head
- **What's new:** Replace softmax with Dirichlet evidential output (Sensoy 2018). At inference, compute vacuity `u = K/S` where `S = Σα_i`. Abstain when `u > τ`.
- **Reportable metrics:** Risk-coverage curve, AUARC (Area Under Risk-Coverage), selective accuracy at fixed coverage (e.g., 80%, 90%, 95%), ECE (expected calibration error), Brier score.
- **Why nobody can attack:** Evidential learning is well-established. We're the first to apply it for ECG selective prediction with clinical operating points. Reviewer 1's "prediction head not evaluated" complaint is now obsolete.

### Pillar C: Leave-One-Database-Out (LODO) Cross-Database Evaluation
- **What's new:** Three rotation experiments: train {MIT-BIH+CPSC} test PTB-XL; train {MIT-BIH+PTB-XL} test CPSC; train {CPSC+PTB-XL} test MIT-BIH.
- **Why this matters:** Every existing ECG paper trains and tests on the same database (or random splits within). LODO is the only test of true generalization.
- **Honest result:** LODO numbers will be lower than 97.6%. That's expected and that's the point — we're the first to report them honestly.

### Pillar D: Anatomical IoU for Explanation Validation
- **What's new:** Use LUDB (200 records, expert P/QRS/T boundary annotations) to quantitatively validate Grad-CAM and attention saliency.
- **Metrics:** IoU and Dice between (saliency map > threshold) regions and expert-annotated QRS complex spans. Reported per anatomical region (P-wave, QRS, T-wave).
- **Why nobody can attack:** It's a real quantitative metric grounded in cardiologist annotations. Replaces qualitative "look at the pretty heatmap" with numbers.

---

## 3. What We DROP from the Old Paper

| Old element | Why we drop it | What replaces it |
|---|---|---|
| The "prediction head" (24h recurrence risk) | Reviewer 1 correctly flagged it as fake — no longitudinal ground truth | Evidential uncertainty head with real clinical metrics |
| Heuristic re-labeling of CPSC/PTB-XL to beat level | Reviewer 2's killing blow | MIL bag-level supervision for CPSC/PTB-XL |
| Claim "validated on 3 datasets" | Misleading | "Trained with three label granularities, evaluated LODO" |
| Claim 97.6% as headline | Inflated by mixed-granularity labels | Honest per-protocol numbers: intra-DB, LODO, and selective |
| 2D CWT scalogram branch (maybe — see decision below) | Adds 23M parameters for 0.2% gain | TBD — keep as ablation only |

---

## 4. What We KEEP and ENHANCE

- 1D CNN backbone with SE blocks (works well, lean)
- Bi-LSTM temporal encoder
- 8-head self-attention
- Inter-patient splitting
- Grad-CAM (but now validated with IoU)

---

## 5. Six-Week Schedule

### Week 1: Foundation
- [ ] Day 1-2: Re-do dataset pipeline. Three loaders: MIT-BIH (beat), CPSC (rhythm-bag), PTB-XL (recording-bag). LUDB (held-out, never trained on).
- [ ] Day 3-4: Implement H-MIL architecture. Backbone → beat embeddings → gated attention pooling → bag prediction. Beat-level head only active on MIT-BIH samples.
- [ ] Day 5-7: Implement evidential head (Dirichlet output). Verify loss converges on toy data.

### Week 2: Training & Baselines
- [ ] Day 8-9: Train full Beat-MIL on combined data. Get baseline numbers.
- [ ] Day 10-14: Implement and train 6 baselines on identical data:
  - B1: ResNet-1D (original Hannun-style)
  - B2: CNN-LSTM (Oh 2018)
  - B3: ECGformer (Vaswani-style transformer encoder)
  - B4: ECGTransForm (Bi-directional transformer, 2024)
  - B5: ECGBert (CNN + BERT, 2024)
  - B6: Old version of our own paper (the ablation baseline)
  - All trained on identical splits, identical preprocessing, identical compute budget.

### Week 3: LODO + Statistical Rigor
- [ ] Day 15-18: Three LODO experiments (each ~12 hours training)
- [ ] Day 19-20: Bootstrap CIs (1000 resamples) for every reported metric
- [ ] Day 21: McNemar tests for all pairwise model comparisons. Holm-Bonferroni correction.

### Week 4: Uncertainty & Selective Prediction
- [ ] Day 22-23: Calibration analysis. Reliability diagrams. ECE. Brier.
- [ ] Day 24-25: Risk-coverage curves. AUARC.
- [ ] Day 26-28: Selective prediction at fixed operating points. Clinical interpretation table.

### Week 5: XAI Validation on LUDB
- [ ] Day 29-30: Load LUDB. Build IoU evaluation harness.
- [ ] Day 31-33: Generate Grad-CAM + attention saliency on LUDB records. Compute IoU/Dice vs P/QRS/T annotations.
- [ ] Day 34-35: Comparative XAI ablation: Grad-CAM vs attention vs LIME vs integrated gradients.

### Week 6: Writing & Polish
- [ ] Day 36-38: Write full paper (use section drafts as starting point)
- [ ] Day 39-40: Generate all figures (architecture, ROC, risk-coverage, IoU bars, confusion matrices)
- [ ] Day 41-42: Internal review, polish, format to IEEE conference template, submit.

---

## 6. Risk Register & Mitigations

| Risk | Probability | Mitigation |
|---|---|---|
| LODO numbers are too low to publish | Medium | Frame honestly; report the gap as a finding. Add database-adversarial loss as remediation. |
| MIL prior work is cited by reviewer | High | Pre-emptive: cite Wang 2022, WSDL-AD in related work, position H-MIL explicitly as joint not sequential |
| Evidential training unstable | Medium | Use ensemble fallback (Dirichlet via method of moments on softmax ensemble — see ArXiv 2604.06032) |
| LUDB only has sinus rhythm records | Low (confirmed has variety) | LUDB has multiple morphologies + diagnoses (200 records) |
| Page limit (6 pages) too tight for 4 contributions | High | Detailed supplementary material; main paper carries the narrative |
| Old GitHub repo with original authors gets cited and embarrasses us | Low | Update repo to match new method; old code archived in branch |

---

## 7. Authorship & Submission Notes

Same author list. Same affiliation (PES University, Bangalore). Possibly add a more senior co-author if available — strengthens credibility.

Target venues (pick one, all 6-page IEEE):
- IEEE BIBM 2026 (deadline ~Aug)
- IEEE EMBC 2026 (deadline typically Jan/Feb — may be past)
- IEEE BHI 2026 (Biomedical and Health Informatics, ~July)
- IEEE TENCON 2026
- IEEE HEALTHCOM 2026

Need to verify deadlines closer to submission time.
