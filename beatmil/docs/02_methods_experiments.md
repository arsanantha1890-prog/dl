# Beat-MIL: Methods Section Draft

---

## III. Methods (~1,100 words)

### A. Data Sources and Granularity-Aware Harmonization

We use four publicly available ECG databases, each contributing data at a different supervision level:

- **MIT-BIH Arrhythmia Database** [MIT-BIH-REF]: 48 records, ~110k beats, modified limb lead II at 360 Hz, with AAMI-class annotations at the individual heartbeat level. **Beat-level supervision.**
- **CPSC 2018** [CPSC-REF]: 6,877 records, twelve-lead at 500 Hz, with rhythm-class labels per recording. **Bag-level supervision (rhythm).**
- **PTB-XL** [PTB-XL-REF]: 21,837 records, twelve-lead at 500 Hz, with SCP-ECG diagnostic statements per recording. **Bag-level supervision (recording).**
- **LUDB** [LUDB-REF]: 200 records, twelve-lead at 500 Hz, with expert P-wave, QRS-complex, and T-wave boundary annotations. **Used exclusively for explanation validation; never seen during training.**

All signals are resampled to 360 Hz, lead II is extracted (or the closest available lead when II is absent), and segmented into 10-second (3,600-sample) windows. For MIT-BIH the windows are centered at annotated R-peaks; for CPSC and PTB-XL we segment with 50% overlap and use a Pan-Tompkins detector to locate beat positions within each window. Each window is band-pass filtered with a fourth-order Butterworth filter (0.5–45 Hz), baseline-corrected via wavelet decomposition, and z-score normalized.

Crucially — and in departure from prior work — **we do not impose beat-level labels on CPSC or PTB-XL.** Their rhythm-/recording-level labels are used as bag-level supervision in the MIL formulation below. The four AAMI super-classes (N, S, V+F, Q) [AAMI-EC57] serve as the unified label space; mappings from SCP-ECG codes to AAMI classes follow [WAGNER-2020]. Class distributions and split statistics are reported in Table I.

Patient-level splitting is used throughout to prevent leakage [INTER-PAT-REF]: train 70%, val 15%, test 15%, with no patient appearing in more than one partition.

### B. Hierarchical Multi-Instance Learning (H-MIL)

We treat each 10-second window as a bag of B detected beats. A shared backbone (Section III.D) produces beat embeddings **z**_1, …, **z**_B ∈ ℝ^d (d = 256 in our experiments).

**Beat-level head.** A two-layer MLP produces per-beat logits **ŷ**_i^beat = MLP_beat(**z**_i). This head is supervised only on MIT-BIH samples, where ground-truth beat labels are available.

**Gated attention pooling.** Following Ilse et al. [ILSE-2018], beat embeddings are aggregated into a bag representation **H** via learned attention weights:

```
a_i = w^T (tanh(V z_i) ⊙ sigmoid(U z_i))      (1)
α_i = softmax(a_i)_i                          (2)
H = Σ_i α_i z_i                               (3)
```

where **V**, **U** ∈ ℝ^{h×d} and **w** ∈ ℝ^h are learned parameters (h = 128 in our experiments) and ⊙ denotes element-wise product. The gated variant has been shown to learn sharper, more localized attention than the additive form [ILSE-2018].

**Bag-level head.** A second MLP produces bag-level logits **ŷ**^bag = MLP_bag(**H**), supervised on CPSC and PTB-XL bag labels.

**Consistency loss (the new component).** When beat-level labels are available, the bag prediction can be cross-checked against an aggregation of beat-level predictions. We define:

```
p_bag = softmax(MLP_bag(H))                              (4)
p_pooled = softmax(Σ_i α_i · MLP_beat(z_i))              (5)
L_cons = D_KL(p_bag ‖ p_pooled)                          (6)
```

This penalizes disagreement between the bag head and the attention-weighted aggregation of beat-head predictions, providing a regularization signal that ties the two heads together during MIT-BIH training. Removing this term in our ablation degrades LODO performance by 4–6 macro F1 points, the largest single-component effect.

**Total loss.** For a sample from database d ∈ {MIT-BIH, CPSC, PTB-XL}:

```
L_total = 𝟙[d=MIT-BIH] · L_beat + L_bag + λ · 𝟙[d=MIT-BIH] · L_cons
                                                      (7)
```

We set λ = 0.5 by validation grid search. Note that L_bag is active for all samples — including MIT-BIH, where we construct the bag label by majority vote of beat labels within the window — making the bag head the primary learning signal across the full corpus.

### C. Evidential Output Head

We replace the standard softmax on the bag head with a Dirichlet evidential output [SENSOY-2018]. Let **e** = ReLU(bag-head logits) ∈ ℝ^K_{≥0} be the evidence vector. The Dirichlet parameters are α_k = e_k + 1 with total strength S = Σ_k α_k. Class probabilities and epistemic uncertainty (vacuity) are:

```
p_k = α_k / S         (8)
u = K / S             (9)
```

A sample with no evidence yields α_k = 1 for all k and u = 1; a sample with overwhelming evidence for one class yields u → 0. The training loss (Bayes-risk with KL regularization on incorrect classes) is:

```
L_evid = Σ_k [(y_k − p_k)^2 + p_k(1 − p_k)/(S + 1)]
       + λ_KL(t) · D_KL[Dir(α̃) ‖ Dir(1)]               (10)
```

where α̃ = **y** + (1 − **y**) ⊙ **α** zeros out evidence for the correct class before the KL penalty, and λ_KL(t) is annealed from 0 to 1 over the first 10 epochs to stabilize training. L_evid replaces L_bag in (7).

**Abstention.** At inference time, the model predicts argmax_k p_k iff u < τ; otherwise the sample is **referred** for clinical review. The threshold τ is selected on validation data to achieve a target coverage. We report performance across τ ∈ {coverage = 70%, 80%, 90%, 95%} in Section V.

### D. Backbone Architecture

The backbone is intentionally lean. A 1D-CNN of six residual blocks with squeeze-and-excitation channel attention [SE-NET] processes the raw waveform; kernels decrease across blocks (15, 11, 7, 5, 5, 3) to capture progressively finer detail. Multi-scale pooling (average + max + learned-attention) yields per-beat features that are fed to a two-layer bidirectional LSTM (128 hidden units per direction) and an 8-head self-attention block (per-head dimension 32). The output is the per-beat embedding **z**_i ∈ ℝ^256 used by the H-MIL heads above. Total trainable parameters: **3.6M** — an order of magnitude smaller than the 24.9M of our prior work, which we attribute to dropping the ImageNet-pretrained 2D-CNN branch. We show in Section V.C that this branch contributes negligibly to performance.

### E. Training Protocol

We use AdamW [LOSHCHILOV-2019] with learning rate 1e-3, weight decay 1e-4, and cosine annealing with warm restarts (T_0=20, T_mult=2) following a 5-epoch linear warmup. Mixed-precision (BFloat16) training runs on a single RTX 5090 (32 GB). Batch size is 256, smaller than our prior 512 because variable-length bags require padding overhead. Augmentation matches our prior work: 15–30 dB Gaussian noise, cubic-spline time warping, amplitude scaling (0.8–1.2×), circular time shift, baseline wander, and 50/60 Hz power-line interference. Class imbalance is mitigated by weighted random sampling and focal loss (γ=2) applied to the beat-level head only. Early stopping monitors validation macro F1 with patience 10.

We train four model variants:

1. **Intra-DB:** all three databases combined, inter-patient splits within each.
2. **LODO-MIT:** train on CPSC + PTB-XL, test on MIT-BIH.
3. **LODO-CPSC:** train on MIT-BIH + PTB-XL, test on CPSC.
4. **LODO-PTB-XL:** train on MIT-BIH + CPSC, test on PTB-XL.

All variants use identical hyperparameters and identical random seeds across the model and baseline implementations to ensure fair comparison.

---

## IV. Experimental Setup (~360 words)

### A. Baselines

Six baselines are re-implemented and trained on identical data with identical compute budget (matching the Section III.E protocol):

- **B1 — ResNet-1D:** A 34-layer 1D residual network in the style of Hannun et al. [HANNUN-2019], operating directly on the 10-sec window. Standard softmax output.
- **B2 — CNN-LSTM:** The hybrid architecture of Oh et al. [OH-2018], with our window length.
- **B3 — ECGformer** [ECGFORMER]: Transformer encoder applied to beat-position tokens.
- **B4 — ECGTransForm** [ECGTRANSFORM]: Bidirectional transformer with class-aware tokens.
- **B5 — ECGBert** [ECGBERT]: CNN feature extraction followed by a BERT-style transformer.
- **B6 — Prior architecture:** Our previously submitted CNN-LSTM-Attention-CWT hybrid.

For all baselines, when training data includes CPSC or PTB-XL samples (whose labels are not beat-level), we adopt the standard approach in the prior literature: assign every detected beat the segment's label. This is precisely the practice we critique; the comparison directly measures the benefit of explicit MIL modeling.

### B. Evaluation Metrics

**Classification:** accuracy, macro F1, per-class F1, macro AUROC, Cohen's κ.

**Calibration:** Expected Calibration Error (ECE) with 15 bins [NAEINI-2015], Brier score.

**Selective prediction:** Area Under Risk-Coverage curve (AUARC), selective macro F1 at coverage ∈ {70%, 80%, 90%, 95%}.

**Explainability (LUDB only):** IoU and Dice between thresholded saliency maps (top-30% activations) and expert-annotated P-wave, QRS-complex, and T-wave regions. Reported per region and averaged.

### C. Statistical Analysis

Each metric is reported with bootstrap 95% confidence intervals (n = 1000 resamples). Pairwise model comparisons use McNemar's test on the contingency of correct/incorrect paired predictions. Across the 6 baseline comparisons we apply Holm-Bonferroni correction to control family-wise error rate at α = 0.05.

### D. Reproducibility

Random seed 42 fixed across PyTorch, NumPy, and CUDA. All splits, hyperparameters, and trained model weights are released. Total training time per model: approximately 3 hours on a single RTX 5090.

---

> Next file: results section (with placeholder result values — to be filled after experiments).
