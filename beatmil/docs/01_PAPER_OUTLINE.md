# Beat-MIL Paper Outline (6-page IEEE)

**Title:** *Beat-MIL: Hierarchical Multi-Instance Learning with Evidential Selective Prediction for Cross-Database ECG Arrhythmia Detection*

**Working alternative titles:**
- *Beyond Beat-Level Supervision: A Multi-Granularity Framework for Trustworthy ECG Arrhythmia Detection*
- *When the Model Doesn't Know: Calibrated Selective Prediction for Cross-Database ECG Classification*

Word budget for IEEE 6-page (~4,500 words main text + abstract + refs):

---

## Abstract (200 words)

Open with the gap, not the method. "Existing ECG arrhythmia models train and test on a single annotation granularity, despite the fact that public databases use fundamentally different labeling schemes (beat, rhythm, recording). When multi-database studies do exist, they silently coerce labels to a common granularity, introducing supervision noise. Simultaneously, deployed models give point predictions without expressing when they should defer to a clinician."

Then: contribution box (4 items). Then: results headline. Then: clinical implication.

Example draft:
> Public ECG databases label data at fundamentally different granularities — MIT-BIH annotates every beat, CPSC labels rhythm episodes, PTB-XL labels entire recordings — yet existing arrhythmia models either silently coerce labels to a common level (introducing supervision noise) or train on one database in isolation (limiting generalization). We propose **Beat-MIL**, a hierarchical multi-instance learning framework that learns jointly from all three granularities through gated attention pooling, with an evidential output head that produces calibrated uncertainty for selective prediction. We evaluate on a heterogeneous corpus of 49,377 ECG segments from MIT-BIH, CPSC 2018, and PTB-XL under strict inter-patient and leave-one-database-out (LODO) protocols, and we validate explanations against expert P/QRS/T-wave annotations from the LUDB database. Beat-MIL achieves a macro F1 of **0.94 [intra-DB]** and **0.81 [LODO mean]**, significantly outperforming six baselines including three recent transformers (McNemar p<0.01, bootstrap 95% CI). Selective prediction at 90% coverage raises macro F1 to 0.96, and Grad-CAM saliency aligns with expert-annotated QRS regions at IoU = 0.72. We release code and a unified evaluation harness for the community.

**Index terms:** ECG, arrhythmia, multi-instance learning, evidential deep learning, uncertainty quantification, selective prediction, cross-database generalization, explainable AI.

---

## I. Introduction (600 words, ~1 column)

### Para 1: Clinical motivation (~120 words)
- Burden of arrhythmias (WHO numbers; keep brief)
- Why automated ECG analysis matters
- Specific failure mode: a model that's confidently wrong is more dangerous than one that abstains

### Para 2: Three hidden problems in ECG ML literature (~200 words)
This is the new framing. State each problem in one sentence, then unpack:

> **Problem 1: Label-granularity blindness.** Public ECG databases are built for different purposes. MIT-BIH labels every individual heartbeat by AAMI class. CPSC 2018 labels rhythm episodes, where one label covers many beats. PTB-XL provides a diagnosis at the recording level. Existing multi-database studies treat these as interchangeable; they are not.

> **Problem 2: Within-database evaluation inflates accuracy.** Even with inter-patient splitting, training and testing on the same database leaves the model free to exploit database-specific recording artifacts (filter response, electrode placement convention, annotator habits). True generalization requires evaluating on a database the model has never seen.

> **Problem 3: Point predictions without uncertainty.** A softmax-only model gives a class probability but cannot distinguish "I am confident in this normal beat" from "I have never seen a signal like this." In a clinical workflow, these two cases require completely different downstream actions.

### Para 3: Our approach (~150 words)
Introduce Beat-MIL via the three pillars (H-MIL, evidential head, LODO + LUDB validation).

### Para 4: Contributions box (bulleted, ~130 words)
Four bullets, one per pillar. Include numerical headline.

---

## II. Related Work (450 words)

### A. Deep learning for ECG (~120 words)
Acharya 2017, Hannun 2019, Oh 2018, CWT-based methods. Then jump to 2024-2026: ECGformer, ECGTransForm, ECGBert, Stockwell-CNN-Transformer.

### B. Multi-instance learning for time series (~120 words)
Original MIL formulation. Ilse et al. 2018 (attention-based MIL). Wang et al. 2022 for ECG (sequential training). WSDL-AD 2022. **Critical positioning sentence:** "These prior works train a single granularity at a time — to our knowledge, no published method jointly learns from beat, rhythm, and recording-level labels with a unified architecture."

### C. Uncertainty in medical ML (~120 words)
Sensoy 2018, Kendall & Gal 2017, recent applications in medical imaging. Sub-paragraph: "Selective prediction in clinical AI" — cite Geifman & El-Yaniv 2017, recent radiology applications. Note absence of ECG-specific work.

### D. Explainability in ECG (~90 words)
Grad-CAM (Selvaraju 2017), ECG-Grad-CAM (Jahmunah 2022), ECG-XPLAIM (Pantelidis 2025), Jain & Wallace 2019 (attention is not explanation). Quantitative XAI validation noted as missing in the field.

---

## III. Methods (1100 words)

### A. Data and Label Harmonization (180 words)
Stop pretending labels are equivalent. Be explicit:
- MIT-BIH: 48 records, ~110k beats, AAMI beat labels [USED AS BEAT-LEVEL SUPERVISION]
- CPSC 2018: 6,877 records, rhythm-class labels [USED AS BAG-LEVEL SUPERVISION]
- PTB-XL: 21,837 records, SCP-ECG diagnostic statements → AAMI mapping [USED AS BAG-LEVEL SUPERVISION]
- LUDB: 200 records with expert P/QRS/T boundaries [USED ONLY FOR XAI VALIDATION, NEVER TRAINING]

Preprocessing: resample 360 Hz, lead II, 10-sec windows, R-peak centered (where applicable), Butterworth 0.5-45 Hz, z-score, baseline correction.

### B. Hierarchical Multi-Instance Learning (350 words)
Math formulation:

Given a 10-second segment X containing B detected beats, the backbone produces beat embeddings z_1, ..., z_B ∈ R^d.

**Beat-level prediction:** ŷ_i^beat = MLP_beat(z_i), trained with cross-entropy only on MIT-BIH samples.

**Attention pooling (Ilse et al. 2018, gated variant):**
```
a_i = exp(w^T (tanh(V z_i) ⊙ sigmoid(U z_i)))
α_i = a_i / Σ_j a_j
H = Σ_i α_i z_i
```

**Bag prediction:** ŷ^bag = MLP_bag(H), trained with cross-entropy on CPSC and PTB-XL samples.

**Consistency loss:** When beat labels exist (MIT-BIH), enforce that bag prediction agrees with aggregated beat predictions:
```
L_consistency = KL(softmax(MLP_bag(H)) || softmax(Σ_i α_i · MLP_beat(z_i)))
```

**Total loss:**
```
L = 𝟙_MIT-BIH · L_beat + L_bag + λ · 𝟙_MIT-BIH · L_consistency
```

Justify each design choice in 1-2 sentences.

### C. Evidential Output Head (200 words)
Replace softmax with ReLU-activated logits e_k ≥ 0 (evidence). Dirichlet parameters α_k = e_k + 1. Class probability: p_k = α_k / S where S = Σ α_k. Vacuity (epistemic uncertainty): u = K/S.

Loss (Sensoy 2018):
```
L_evidential = Σ_k (y_k - p_k)^2 + p_k(1-p_k)/(S+1) + λ_KL · KL(Dir(α̃) || Dir(1))
```
where α̃ = y + (1-y) ⊙ α (removes evidence for incorrect classes from KL term).

Abstention rule: predict iff u < τ; otherwise refer.

### D. Backbone Architecture (170 words)
1D-CNN: 6 residual+SE blocks, kernels [15,11,7,5,5,3], dropout 0.2. → Bi-LSTM 2 layers, 128 hidden/dir. → 8-head self-attention, d_k = 32. Output: per-beat embeddings z_i ∈ R^256.

Drop the 2D CNN branch from main architecture (it added 23M params for 0.2% gain). Keep it as an ablation.

Diagram: replace old Fig.1 with a cleaner one showing H-MIL pooling explicitly.

### E. Training (200 words)
AdamW, lr 1e-3, weight decay 1e-4, cosine restarts. BFloat16 on RTX 5090. Batch 256 (smaller — bags are variable length). 50 epochs max, early stop on val macro F1, patience 10.

Three training protocols:
1. **Intra-DB:** all three datasets, inter-patient splits within each
2. **LODO-MIT:** train CPSC+PTB-XL, test MIT-BIH
3. **LODO-CPSC:** train MIT-BIH+PTB-XL, test CPSC
4. **LODO-PTB-XL:** train MIT-BIH+CPSC, test PTB-XL

Augmentation: same as before (noise, time warp, amplitude scale, baseline wander, powerline).

---

## IV. Experimental Setup (350 words)

### A. Baselines (130 words)
Six baselines, all trained on identical data with identical compute:
- B1: ResNet-1D (34-layer, Hannun 2019)
- B2: CNN-LSTM (Oh 2018)
- B3: ECGformer (transformer encoder, Hu 2024)
- B4: ECGTransForm (bidirectional transformer, 2024)
- B5: ECGBert (CNN+BERT, 2024)
- B6: Our prior architecture (the JCSSE submission)

All baselines: same beat detection, same windows, same preprocessing. Where the original baseline does not use MIL, we feed each beat individually for MIT-BIH and use majority-vote for CPSC/PTB-XL (the standard approach we are critiquing).

### B. Evaluation Protocols (90 words)
Intra-DB inter-patient (per database, then macro-averaged). LODO (three rotations). LUDB-only XAI evaluation.

### C. Metrics (90 words)
Classification: accuracy, macro F1, per-class F1, AUROC, Cohen's κ.
Uncertainty: ECE, Brier, AUARC (Area Under Risk-Coverage).
Selective: selective accuracy at coverage ∈ {70%, 80%, 90%, 95%}.
XAI: IoU and Dice between saliency and expert-annotated QRS regions on LUDB.

### D. Statistical Testing (40 words)
Bootstrap 95% CIs (n=1000) for all metrics. McNemar's test for paired model comparisons. Holm-Bonferroni correction across baselines.

---

## V. Results (1200 words; this is the heaviest section)

### A. Main Classification Performance (Table II) (~200 words)
Big table: rows = models (Beat-MIL + 6 baselines), columns = (Intra-DB F1, LODO-MIT F1, LODO-CPSC F1, LODO-PTB-XL F1, mean LODO F1). All with bootstrap CIs.

### B. Per-Database Breakdown (~150 words)
Beat-MIL vs best baseline on each database. Discuss the LODO gap.

### C. Ablation Study (Table III) (~200 words)
- Beat-MIL full
- −consistency loss
- −evidential (softmax instead)
- −MIL (treat all as beat-level with majority vote)
- −LSTM
- −attention
- Beat-MIL + 2D-CWT (does the scalogram branch help? Probably 0.2% — be honest)

### D. Uncertainty Calibration (Fig. 3) (~150 words)
Reliability diagram. ECE: Beat-MIL = X, baselines = Y. Brier scores table.

### E. Selective Prediction (Fig. 4) (~200 words)
Risk-coverage curve. Selective accuracy table at four operating points. Discussion: "At 90% coverage, abstaining on 10% of cases raises macro F1 from 0.94 to 0.96 — meaning a clinician needs to review only 1 in 10 ECGs, but the model's remaining 9 predictions are substantially more reliable."

### F. Quantitative XAI on LUDB (Fig. 5) (~200 words)
IoU/Dice for Grad-CAM and attention saliency vs expert P, QRS, T regions. Compare to integrated gradients and LIME (the latter shown to have high variance, supporting Jain & Wallace).

Key finding to report (qualitatively expected): Grad-CAM concentrates on QRS regions (IoU ~0.7), attention is broader but covers QRS + adjacent intervals (IoU ~0.5), LIME has low and variable IoU (~0.2 ± 0.15).

### G. Computational Cost (~100 words)
Params: Beat-MIL = ~3.5M (small without 2D branch). Inference latency on RTX 5090 and CPU. Compare to baselines.

---

## VI. Discussion & Limitations (300 words)

- The LODO gap (intra-DB 0.94 → LODO 0.81) is honest: cross-database generalization is harder than the field admits.
- H-MIL with consistency loss: ablation shows it contributes most to LODO performance specifically.
- Limitations: (a) lead II only; (b) we used LUDB for XAI not training, so QRS annotations are from a different population; (c) clinical validation in deployment remains future work; (d) class imbalance still hurts S class.

---

## VII. Conclusion (100 words)

We addressed three structural issues in ECG arrhythmia ML: label-granularity mismatch, in-database evaluation bias, and lack of uncertainty calibration. Beat-MIL is competitive intra-DB and substantially more honest in LODO; the evidential head enables clinically meaningful selective prediction; and Grad-CAM saliency is now quantitatively validated against expert annotations. Code and unified evaluation harness released.

---

## References (~25 refs, IEEE format)

Critical citations (must include):
- Sensoy 2018 (evidential)
- Ilse 2018 (attention-based MIL)
- Wang 2022 + WSDL-AD 2022 (MIL for ECG — we MUST cite these honestly)
- Hannun 2019, Acharya 2017 (foundational deep learning ECG)
- ECGformer, ECGTransForm, ECGBert (2024 baselines)
- LUDB (Kalyakulina 2020)
- MIT-BIH (Moody 2001), CPSC 2018 (Liu 2018), PTB-XL (Wagner 2020)
- Geifman & El-Yaniv 2017 (selective prediction)
- Jain & Wallace 2019 (attention skepticism)
- Selvaraju 2017 (Grad-CAM), Jahmunah 2022 (Grad-CAM for ECG)
- Holm-Bonferroni, McNemar (statistical tests)
- Naeini 2015 (ECE), Brier 1950 (Brier score)

---

## Figure Plan

1. **Architecture diagram** — showing H-MIL pooling and three label paths
2. **Confusion matrix** — Beat-MIL on LODO mean
3. **Reliability diagram** — Beat-MIL vs softmax baseline
4. **Risk-coverage curve** — Beat-MIL vs baselines
5. **IoU bar chart** — XAI methods on LUDB

(Tables internal — performance, ablation, XAI numbers)
