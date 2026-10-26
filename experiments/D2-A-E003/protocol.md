# D2-A-E003 preregistered protocol

**Protocol version:** 1.1
**Status:** Frozen before R0/R1/R2 execution and before creating training/evaluation data.
**Date:** 2026-10-26
**Research charter:** v1.2; unchanged.
**Historical results:** E001, R3, E002, T001 and R4 remain immutable.
**Data:** synthetic only; no final blind is authorized or generated.

**Pre-execution amendment:** Version 1.1 makes the physical-diagnostic sample size and source splits explicit because version 1.0 described fixed quotas without numeric values. This clarification adds no model, condition, endpoint, acceptance gate, or training budget. It supersedes version 1.0 before any R0/R1/R2 run or generated E003 data.

## 1. Question and scope

Test whether one fixed physical coherence representation helps the existing three-channel GMTI `TaskStudent` directly learn detection, range and continuous Doppler in the frozen stationary-clutter model. The work supports charter Q1/Q4 and the task-level student goal. It does not claim that invariance alone improves the final task, that a teacher is reliable, or that results transfer to measured data.

Preserve the E001 model: 3 channels, PRF 1300 Hz, 130 pulses, 32 range cells, stationary point-scatterer clutter, circular complex AWGN, one target, and paired H0/H1 sharing the same clutter/noise. Retain E001 C0–C3 difficulty definitions; exact-overlap D is descriptive only. No multi-target, calibration mismatch, moving clutter, raw ADC, or tracking is introduced.

## 2. Frozen candidate and mathematical definition

Input `X ∈ C[B,3,130,32]`. Let `q(X)=max(mean_{c,n,r}|X[c,n,r]|², eps)` with `eps=1e-12` in power units. The front end emits, in this fixed order:

1. Real and imaginary parts of `x_i[n+τ,r] conj(x_j[n,r])/q(X)` for same-channel pairs `i=j∈{0,1,2}`, `τ∈{1,4,16}` (9 complex/18 real channels).
2. Real and imaginary parts of `x_i[n,r] conj(x_j[n,r])/q(X)` for `(i,j)∈{(0,1),(0,2),(1,2)}` (3 complex/6 real channels).
3. Three channels `|x_i[n,r]|²/q(X)`.
4. One scene-constant channel `log(q(X))` broadcast over slow time and range.

Output is 28 real channels. For every positive lag, values at `n+τ≥130` are zero padded; no truth or validity mask is appended. A learned pointwise 1×1 projection maps 28→6 before the otherwise unchanged `TaskStudent`; no raw-IQ bypass is allowed. S0 is the unchanged raw-IQ `TaskStudent`. S2 adds 174 projection parameters; total S2 trainable parameter increase must remain ≤5% of S0.

Public phase rotation multiplies every complex coherence by `exp(jφ)exp(-jφ)=1`; normalized powers are also unchanged; `log(q)` is unchanged. The representation is not declared globally amplitude invariant: the first 27 channels are normalized, while `log(q)` intentionally retains observable scene scale. Lag 1 resolves the principal Doppler interval; lags 4 and 16 are aliased individually and may add discrimination but cannot remove PRF ambiguity. Cross-channel phase differences remain observable. The products are quadratic, so target/clutter cross terms remain and the front end is not an additive target/background separator.

## 3. R0 stable sampling and physical diagnostics

Use stable unique `base_scene_id` strings and predeclared stratum labels. For each stratum, sort IDs, apply a fixed-seed generator to that sorted ID list, select the fixed quota, and save selected IDs (not only transient array indices). The selection ID set must be identical after input-row permutation. Report H0 and H1 separately, condition, sample count, detected/correctly detected status, and valid parameter estimates. All Doppler differences use the PRF-periodic wrap.

For public-phase diagnostics apply 32 fixed rotations from 0 through `2π·31/32`. Report score absolute changes, decision agreement, range changes and circular Doppler changes both overall and by H0/H1, C0–C3, detected/missed and parameter-localization strata. Reusing a calibration threshold is mandatory.

Select exactly 64 scene IDs per C0–C3 from the independent H0 verification split and exactly 64 per C0–C3 from H1 development, using `physical_diagnostic_seed` and the stable-ID sampler. Save the selected IDs. Use these same 512 scenes for all 32 public-phase rotations and both models; report resulting H0/H1 sample counts and correctness strata. The separate frequency interventions use the full 64-scene per-geometry R2 control sets.

Frequency diagnostics remain three distinct interventions: (A) whole-observation slow-time modulation; (B) target-component-only modulation `H0 + S·exp(j2πδn/PRF)` as a synthetic counterfactual; (C) regenerated target geometry with changed radial velocity and the same random scene seed so background/noise are paired. Report truth movement, alias/wrap status, target/background Doppler separation and identifiability for each. Never label A/B as a regenerated physical target trajectory.

## 4. R1 correctness and numerical gates

Before training, test:

- fixed output shape/order and finite output for valid `[B,3,130,32]` inputs;
- common-phase invariance against complex128 reference and between complex64 transformed inputs;
- same-channel lag phase `2πfτ/PRF`, lag-1 unambiguous mapping on `[-PRF/2,PRF/2)`, and expected aliases at lags 4/16;
- preservation of all three pairwise cross-channel phase differences;
- no range-bin mixing;
- finite zero, tiny and high-dynamic-range behavior;
- normalized coherence/power behavior under complex scaling and the expected `2 log|a|` shift in the scale channel;
- trainable projection and finite nonzero task gradients;
- parameter count and front-end compute/memory measurements.

For complex64 comparisons, pre-register relative max error `max|y64−y128| / max(1,max|y128|) ≤ 1e-4`, plus finite outputs. This is a relative numerical tolerance for a float32 path, not a claim of `1e-4` absolute physical accuracy. Report the unrounded error and test that the tolerance does not mask a lag/phase decision change. Complex128 reference tests use `atol=1e-10, rtol=1e-8`. Any failure stops training until a code defect is fixed or a revised protocol is frozen.

## 5. R2 controlled observable-information diagnostics

Generate 64 paired scenes for each identifiable geometry A/B/C at fixed 10 dB SNR and 10 dB CNR for each control, with independent scene seeds and common H0/H1 background/noise within each pair.

- Target-level control: multiply only the target component by `{0.25,0.5,1,2}`; leave clutter and noise byte-identical. Record physical component powers and paired feature increments.
- Clutter-level control: multiply only clutter by `{0.5,1,2}`; leave target and noise unchanged. Record input scale and feature increments.
- Doppler control: compare whole-IQ shifts and target-only shifts `δ∈{-25,-10,-5,5,10,25} Hz`; separately regenerate geometry at radial velocities `{-4,-1.2,-0.4,0.4,1.2,4} m/s` using matched scene seeds where the named scenario permits. Report D exact-overlap only as a boundary reference.

Use no learned model to select a feature. Report per-scene `F(H0)`, `F(H1)`, target-only `F(S)`, paired `F(H1)-F(H0)` norms, feature-domain standardized separation, Doppler/overlap group, and B3/T1 descriptive score association where computationally available. Oracle components are analysis-only. R2 is a pretraining information diagnostic, not evidence that a student can use the information. The gate is finite, nondegenerate observable increments above the measured numerical floor in identifiable A/B/C; specifically, the median per-scene RMS feature increment in each A/B/C base cell must exceed 10× the maximum absolute complex64-vs-complex128 feature error measured on the pre-registered R1 random fixture. D is exempt. Because normalized second-order features include target/background cross terms, monotonicity of `F(H1)-F(H0)` under target scaling is not assumed; its empirical shape is reported. Report all measurements without post-hoc thresholds.

## 6. Splits, scene sizes and seeds

All generated scene IDs are unique across model splits; paired H0/H1 stay in the same split. Chunk size is 16; scene seed stride is 1009. Seed blocks are fixed below and checked for collisions by enumerating every used chunk seed before generation. The R2 paired interventions intentionally re-use the same seed within a matched intervention family only.

| Split | Count and composition | Seed bases |
|---|---:|---|
| Train | 51,200 paired scenes; 12,800 per C0–C3 using the E002 cell proportions | 51M, 56M, 61M, 66M by course |
| Validation | 5,000 pairs; 1,250 per course, E002 cell proportions | 71M, 76M, 81M, 86M |
| H0 calibration | 8,192; equal course mixture (2,048 each), with E002 within-course nuisance mixture | 91M |
| H0 verification | 8,192; same mixture, independent IDs | 96M |
| Dedicated C3 H0 verification | 4,096; equal A/B/C × 0/10 dB mixture | 101M |
| H1 development | 4,096 pairs; 1,024 each C0–C3, E002 cell proportions | 106M, 111M, 116M, 121M |
| D exact-overlap boundary | 512 pairs, 10/10 dB, descriptive only | 126M |
| R2 controlled scenes | 64 per A/B/C per fixed-factor condition; paired by scene seed | 131M onward, one disjoint 1M block per intervention family |

Training uses seed 101, matched S0/S2 initialization for all shared `TaskStudent` tensors, and one shared fixed pair permutation/batch plan. Bootstrap seed 2026102605; phase/shift diagnostic seed 2026102604. The final blind reserve `[9M,10M)` is untouched. No E002/R4 development scene is reused. Dataset/seed manifests and IDs are saved.

## 7. S0/S2 training and checkpoint selection

One E1 physical-truth-only pilot seed 101 per arm, strictly sequential on RTX 3050 Laptop `cuda:0`, FP32, AdamW, 1,600 optimizer updates, effective batch 64 (32 paired scenes/update), cosine learning rate `1e-3→1e-4`, weight decay `1e-4`, gradient clip 1.0, validation every 100 updates. No teacher, K1/K2, curriculum or auxiliary output is used. S0 and S2 consume the same train pair IDs once each, same pair plan/order, same data, same task loss and validation rule. Shared raw TaskStudent weights begin identical; only S2 has the fixed front end and 28→6 projection. Select minimum common validation truth-task loss; save initial/best/last weights, every-update history, optimizer/resource records and actual microbatch.

Start at microbatch 16. On OOM restart only the affected arm from the same initial weights and fixed batch plan at 8, then 4, preserving effective batch and updates by accumulation. If 4 OOMs, stop; do not use CPU, AMP or reduce the scientific budget. Never load both models on GPU simultaneously. Evaluation H0 thresholds are model-specific, using only the independent H0 calibration split and `quantile(0.95, method="higher")`; decisions use `score >= threshold`.

## 8. Endpoints and statistical analysis

Primary pilot endpoint: paired S2−S0 joint-success difference in pooled identifiable C2/C3 development scenes. Joint success is detection plus absolute range error ≤1 cell and circular Doppler error ≤20 Hz; misses fail. The pilot signal requires point gain ≥0.02 and a two-sided paired scene-bootstrap 95% interval lower bound >0. This is a screening result for one seed only.

Task protection gates (S2 vs S0 on the same IDs):

- pooled H1 Pd point difference ≥−0.01 and paired 95% interval lower bound >−0.05;
- pooled H1 joint-success point difference ≥−0.01 and interval lower bound >−0.05;
- C2/C3 joint-success difference interval lower bound >−0.05;
- C2/C3 range and circular Doppler RMSE ratio point estimate ≤1.05 and paired 95% upper bound ≤1.10;
- C3 H0 Pfa S2−S0 paired 95% upper bound ≤+0.02, and each model's C3 point Pfa ≤0.10.

These are frozen pilot-screening margins, not formal population non-inferiority claims. The joint endpoint's bootstrap is 2,000 resamples by `base_scene_id`; RMSE is recalculated inside every resample. H0 Pfa includes a one-sided exact 95% Clopper–Pearson upper bound, paired bootstrap difference, actual threshold, empirical ties and both strict/non-strict comparator sensitivity. Report AUC/ROC, overall and per-course/per-scenario Pd/Pfa, range MAE/RMSE in cells/metres, Doppler MAE/RMSE/bias, predicted-range mismatch, true-range-gated Doppler, joint success, all-H1 and detected-only results, and 95% scene-bootstrap intervals. Bootstrap quantifies scene uncertainty only; the single seed does not estimate training-seed uncertainty.

G4 is passed only by the preselected C2/C3 joint-success endpoint above. Doppler RMSE improvement ≥10% is secondary descriptive evidence, not an alternative success endpoint. Public-phase and frequency consistency gates are reported independently and cannot substitute for task gates. Compute timing compares complete input→student outputs for both arms on the same GPU, separately reports fixed front end, projection/network, transfer, peak GPU memory, parameter count, and repeated batch sizes; it is not full-chain realtime evidence.

## 9. Stage gates and stop rules

- **G0 Engineering:** split/seed audit, stable-ID sampler, label/input separation, finite gradients, output decoder, matched cohort/order/initialization and CUDA preflight pass.
- **G1 Front end:** all R1 tests pass, including finite precision and nondegenerate frequency, relative-phase and range mapping. Any failed subtest stops training.
- **G2 Controlled information:** R2 increments remain finite/nondegenerate and respond to target scaling in identifiable A/B/C. Any information collapse stops training; D is descriptive.
- **G3 Task learning:** both S0 and S2 produce meaningful development discrimination (AUC lower 95% scene-bootstrap bound >0.5) and beat train-truth constant prediction for range and Doppler RMSE. If either arm fails, do not claim a fair successful comparison; diagnose the failure.
- **G4 Task protection and benefit:** apply the frozen protection margins and primary C2/C3 joint-success gain exactly as above. If protection fails, S2 stops as a route candidate regardless of physical invariance. If protection passes but the primary gain does not, classify as no demonstrated task gain and do not enter formal multi-seed/KD.
- **G5 Physical consistency:** report score, decision, range and wrapped Doppler changes under public phase and the three separate frequency interventions. No invariant claim from decision agreement alone.
- **G6 Compute:** report costs; any speed claim is limited to measured boundaries.

If S2 fails, redirect only to the diagnosed bottleneck. If S2 passes the pilot, recommend a separately preregistered three-seed confirmation; do not start it, distillation or final blind under this protocol. Do not modify `docs/RESEARCH_CHARTER.md`.

## 10. Reproducibility, artifacts and cleanup

Before generation, require committed and pushed protocol/config/code, `main`, CUDA visibility, disk-space record, source/config commit, environment record, and seed collision audit. Keep generated IQ and temporary memmaps under `experiments/D2-A-E003/data/cache/`; do not track them. Preserve model initial/best/last checkpoints, logs, IDs, seed manifest, metrics, per-scene result tables, plots, controlled diagnostics, environment/resource records, analysis and review. After evaluation, remove only cache files listed in the E003 cache manifest and verify the project is below 20 GiB. No pre-existing file or user data is eligible for cleanup.

