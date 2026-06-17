# Beat-MIL: Draft Paper Sections (Abstract, Intro, Related Work)

> These are first drafts. Edit freely. Word counts are targets for the 6-page IEEE format.

---

## Title

**Beat-MIL: Hierarchical Multi-Instance Learning with Evidential Selective Prediction for Cross-Database ECG Arrhythmia Detection**

*Authors:* Abheeshta V Aradhya, Aishwarya JR, Anantha Rama S, Rashmi N Ugarakhod
*Affiliation:* Dept. of Electronics and Communication, PES University, Bangalore, India

---

## Abstract (~210 words)

Public ECG databases label data at fundamentally different granularities. MIT-BIH annotates every individual beat, CPSC 2018 labels rhythm episodes, and PTB-XL provides a diagnosis for an entire recording. Existing multi-database studies either silently coerce these labels onto a common granularity — introducing supervision noise — or train on a single database, limiting generalization. Compounding this, deployed models output point predictions without expressing when a clinician should be brought into the loop. We propose **Beat-MIL**, a hierarchical multi-instance learning framework that learns jointly from all three label granularities through gated attention pooling and a consistency constraint, paired with an evidential output head producing calibrated uncertainty for selective prediction. We evaluate on a heterogeneous corpus of 49,377 segments under strict inter-patient and leave-one-database-out (LODO) protocols, and quantitatively validate explanations against expert P/QRS/T-wave annotations from LUDB. Beat-MIL achieves a macro F1 of *0.94* under intra-database evaluation and *0.81* averaged across three LODO rotations, significantly outperforming six baselines including three recent transformer architectures (McNemar p < 0.01, bootstrap 95% CIs). At 90% coverage the selective head improves macro F1 to *0.96*, and Grad-CAM saliency aligns with expert-annotated QRS regions at IoU = *0.72*. We release code and a unified evaluation harness.

*Index Terms* — ECG, arrhythmia, multi-instance learning, evidential deep learning, uncertainty quantification, selective prediction, cross-database generalization, explainable AI.

---

## I. Introduction (~620 words)

Cardiovascular disease remains the leading cause of mortality worldwide, with arrhythmias contributing substantially to morbidity through their associations with stroke, heart failure, and sudden cardiac death [1]. Modern ambulatory ECG monitors produce far more data than a clinician can manually review [2], and a decade of deep learning research has demonstrated convincing pattern-recognition performance — from early CNN-based heartbeat classifiers [3] to cardiologist-level deep networks on proprietary data [4] and recent transformer-based architectures [TRANSFORMER-REFS]. Reported accuracies routinely exceed 97% on benchmark databases.

These reported numbers, however, obscure three structural issues that limit clinical translation.

**Problem 1: Label-granularity blindness.** Public ECG databases are built for different purposes and consequently label data at different levels of granularity. The MIT-BIH Arrhythmia Database [MIT-BIH-REF] annotates every individual heartbeat with an AAMI class. CPSC 2018 [CPSC-REF] labels rhythm episodes — one label covers many beats. PTB-XL [PTB-XL-REF] provides SCP-ECG diagnostic statements at the recording level. When a study trains on more than one of these databases, it must either restrict supervision to the lowest common denominator (discarding information) or impose a heuristic mapping from coarse labels to fine ones (introducing noise). The latter is common, rarely justified, and silently inflates reported performance.

**Problem 2: Within-database evaluation overstates generalization.** Even with strict inter-patient splitting [INTER-PAT-REF], training and testing on the same database leaves the model free to exploit database-specific recording artifacts — filter response, electrode placement convention, annotator habits. Truly generalizable detection requires evaluating on a database the model has never seen. To our knowledge, no prior ECG study has reported leave-one-database-out (LODO) performance on the MIT-BIH / CPSC / PTB-XL triplet.

**Problem 3: Point predictions without uncertainty.** A softmax output expresses relative class probabilities but cannot distinguish high-confidence prediction from out-of-distribution input. In a clinical workflow, an "I have not seen anything like this" signal demands escalation to a clinician; conventional models cannot produce it. Calibrated uncertainty and the capability for principled abstention are prerequisites for deployment [SELECTIVE-REF].

### Contributions

In response to these gaps we present **Beat-MIL**, with four contributions:

1. **Hierarchical multi-instance learning (H-MIL).** We jointly learn from beat-, rhythm-, and recording-level labels through gated attention pooling [ILSE-REF], with a learned consistency constraint between levels. To our knowledge, this is the first ECG framework to use all three label granularities simultaneously in a single training run.

2. **Evidential selective prediction head.** A Dirichlet output [SENSOY-REF] produces per-prediction epistemic uncertainty, enabling principled abstention. We evaluate via risk-coverage curves, expected calibration error, and selective accuracy at clinical operating points — replacing the unfounded "recurrence risk" head of prior multi-task ECG work.

3. **Leave-one-database-out evaluation.** We report performance under three LODO rotations, providing the first cross-database generalization benchmark on the MIT-BIH / CPSC / PTB-XL corpus.

4. **Quantitative explanation validation.** Grad-CAM [SELVARAJU-REF] and attention saliency are scored against expert-annotated P-wave, QRS-complex, and T-wave boundaries from the LUDB database [LUDB-REF], using IoU and Dice metrics — moving ECG explainability beyond qualitative visualization.

We compare against six baselines including three recent transformer architectures, all trained on identical splits with identical preprocessing. Statistical significance is established via bootstrap 95% confidence intervals and McNemar tests with Holm-Bonferroni correction.

Code, configurations, and the unified evaluation harness are released at [REPO-LINK].

---

## II. Related Work (~470 words)

### A. Deep learning for ECG arrhythmia detection

Early work demonstrated that 1D CNNs operating directly on raw ECG waveforms could match handcrafted-feature classifiers without explicit preprocessing [ACHARYA-2017]. Hannun et al. [HANNUN-2019] scaled this with a 34-layer residual network reaching cardiologist-level performance on a proprietary dataset. Hybrid CNN-LSTM architectures [OH-2018] added explicit temporal modeling for variable-length beats, and CWT-based 2D representations enabled image-domain models pretrained on ImageNet [SONG-2024]. Recent transformer work — ECGformer [ECGFORMER], ECGTransForm [ECGTRANSFORM], and ECGBert [ECGBERT] — applies self-attention either directly to beat sequences or in hybrid CNN-transformer configurations. Across these architectures, however, reported accuracies are predominantly intra-database and intra-granularity, and rarely accompanied by uncertainty or generalization analyses.

### B. Multi-instance learning for ECG

The MIL formulation is well-suited to ECG because a recording naturally decomposes into beats. Ilse et al. [ILSE-2018] introduced attention-based MIL for histopathology, providing the gated attention pooling we adopt. For ECG specifically, Wang et al. [WANG-2022] demonstrated that beat-level diagnosis could be trained using only rhythm-level annotations, by sequentially fitting a rhythm model followed by a heartbeat model. WSDL-AD [WSDL-2022] proposed an end-to-end weakly-supervised framework with masked aggregation for beat detection from recording-level labels. To our knowledge, no prior work jointly learns from beat-, rhythm-, and recording-level labels in a unified architecture, and no prior work integrates MIL with calibrated uncertainty for ECG.

### C. Uncertainty and selective prediction

Bayesian neural networks [GAL-2016], deep ensembles [LAKSHMINARAYANAN-2017], and evidential deep learning [SENSOY-2018] provide principled approaches to predictive uncertainty. Selective prediction frameworks [GEIFMAN-2017] formalize the accuracy-coverage trade-off, and risk-coverage curves are increasingly standard in medical AI evaluation [RADIOLOGY-SELECTIVE-REF]. In ECG specifically, uncertainty quantification has received limited attention; we identify no prior work reporting calibration metrics or selective accuracy operating points for arrhythmia classification.

### D. Explainability for ECG

Class Activation Mapping techniques have been adapted for 1D ECG signals, with Jahmunah et al. [JAHMUNAH-2022] showing that Grad-CAM heatmaps align with clinically meaningful regions for myocardial infarction. ECG-XPLAIM [PANTELIDIS-2025] generalized this across lead configurations. Jain and Wallace [JAIN-2019] cautioned that attention weights need not constitute faithful explanations. A persistent limitation across this literature is the qualitative nature of evaluation: saliency maps are typically inspected rather than measured. We address this by quantifying alignment between model attributions and cardiologist-annotated waveform boundaries on the LUDB database [LUDB-REF].

---

> Following sections (Methods, Experiments, Results, Discussion, Conclusion) drafted in next file (03_METHODS_RESULTS.md).
