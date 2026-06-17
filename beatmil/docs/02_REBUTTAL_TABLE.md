# Reviewer Concerns → Beat-MIL Resolutions

## Reviewer 1 (Solid work, novelty/SOTA/stats concerns)

| R1 Concern | Old Paper Status | Beat-MIL Resolution | Paper Location |
|---|---|---|---|
| "Novelty is relatively limited" | Hybrid arch only — fair criticism | **Hierarchical MIL (joint beat+rhythm+recording) + evidential selective prediction + LODO eval + quantitative XAI** — four distinct contributions | Sec I.4 (contribution box), Sec III.B (H-MIL), Sec III.C (evidential) |
| "No comparison with transformer baselines" | Cited but not compared | **Three transformer baselines re-implemented:** ECGformer, ECGTransForm, ECGBert — all trained on identical data | Sec IV.A, Sec V.A (Table II) |
| "Absence of statistical significance testing" | None | **Bootstrap 95% CIs on every metric (n=1000) + McNemar's test for paired model comparison + Holm-Bonferroni correction** | Sec IV.D, all tables |
| "Prediction head not properly evaluated" | Fair — no real ground truth | **Prediction head removed.** Replaced with evidential uncertainty head, evaluated via risk-coverage, ECE, Brier, selective accuracy at multiple coverage operating points | Sec III.C, Sec V.D, Sec V.E |
| "Explainability limited to qualitative visualization" | Just heatmaps | **Quantitative XAI:** IoU and Dice between Grad-CAM/attention saliency and expert-annotated P/QRS/T regions from LUDB (independent test set, never seen during training) | Sec V.F (Fig. 5) |

## Reviewer 2 (Label-granularity correctness concerns)

| R2 Concern | Old Paper Status | Beat-MIL Resolution | Paper Location |
|---|---|---|---|
| "How were beat-level labels derived from CPSC 2018?" | Heuristic rhythm→beat mapping (problematic) | **We don't derive beat labels from CPSC.** CPSC samples are treated as bags with rhythm-level supervision via MIL attention pooling. | Sec III.A, Sec III.B |
| "How was target beat label determined when segment-level rhythm and center beat may not match?" | Forced the center beat's label = segment label (this is what R2 was rightly suspicious of) | **No forcing.** MIL bag prediction learns from rhythm label; beat-level head supervised only on MIT-BIH where beat labels exist. Consistency loss aligns the two. | Sec III.B |
| "At what level were PTB-XL diagnostic statements mapped?" | Recording → forced to beat | **Recording-level only.** PTB-XL samples treated as recording-level bags. We never claim per-beat labels for PTB-XL. | Sec III.A |

## Reviewer 3 (Good work, wants stronger baselines)

| R3 Concern | Old Paper Status | Beat-MIL Resolution | Paper Location |
|---|---|---|---|
| "Lack of comprehensive comparative analysis" | Only literature comparison | **Six baselines on harmonized data:** ResNet-1D, CNN-LSTM, ECGformer, ECGTransForm, ECGBert, our prior arch. Same preprocessing, same splits, same compute. | Sec IV.A, Sec V.A (Table II) |
| "Evaluate established baseline models on the same harmonized dataset" | Not done | **Done.** Single evaluation harness applies all baselines to all three databases with both intra-DB and LODO protocols. | Sec IV, Sec V |

---

## Concerns Reviewers Didn't Raise But We Pre-empt

| Potential reviewer attack | Pre-emption |
|---|---|
| "MIL for ECG is not new" (Wang 2022, WSDL-AD 2022) | We **cite these explicitly** in Related Work and clearly state our contribution is **joint multi-granularity training**, which is new |
| "Evidential learning is from 2018" | We acknowledge Sensoy 2018 fully; our contribution is **applying it for ECG selective prediction with clinical operating points and validating calibration on cross-database data** — a combination not previously studied |
| "LODO will give lower numbers" | We frame this as a feature, not a bug. "First honest LODO evaluation on this corpus" |
| "LUDB for XAI is unusual" | Yes — that's the point. It is the only public ECG corpus with expert P/QRS/T boundaries. Anyone wanting to refute our XAI numbers must use the same database. |
| "Why only AAMI 4-class?" | AAMI EC57 is the standard. We add a supplementary analysis on PTB-XL's super-classes as future work. |
