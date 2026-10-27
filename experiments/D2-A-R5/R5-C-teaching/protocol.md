# D2-A-R5-C Teaching Content Pilot

**Status: DRAFT — not frozen and not authorized to run.** The independent
ChatGPT review of R5-C is still required. This document must remain a draft
until that review has returned and the stage decision supports this pilot.

**Research charter:** `docs/RESEARCH_CHARTER.md` v1.3. This draft does not
modify the charter or any frozen R5-A/R5-B/R5-C result.

## 1. Research question and scope

R5-C found task-related information in the K1, K3, and K2-local response forms,
but did not test whether a student can use that information. This one-seed,
development-only pilot asks whether K3's full response or K2-local response
improves the final detection/range/Doppler task beyond matched physical-truth
supervision. A truth-assisted response arm controls for the extra auxiliary
head and response objective.

The work serves the charter's task-level GMTI student objective; it is not a
standalone detector or CFAR study. It does not test teacher timing, reliability
weighting, formal multi-seed reproducibility, deployment latency, or a final
blind set. Results are a screening pilot and cannot establish seed-level
uncertainty or detector qualification.

The signal remains the D2-A-E001 three-channel complex IQ model: PRF 1300 Hz,
130 pulses, 32 range cells, stationary clutter, and the existing A/B/C
identifiability conditions. D (exact overlap) remains excluded from source
attribution claims.

## 2. Matched student and four arms

All arms use `TaskStudent(width=96)` with the same task trunk and
presence/range/range-offset/circular-Doppler heads. A training-only response
head has 520 Doppler bins (2.5 Hz grid spacing); inference export removes this
head. All arms start from one seed-303 state whose parameters must be
bit-identical across arms, receive the same paired scenes in the same order,
and use the same 12,800-update FP32 schedule. The current implementation has
120,717 training parameters and 70,277 inference parameters after export; G0
must record and verify these counts at runtime.

The physical-truth task terms use the frozen R5-A/E003 weights: presence BCE,
range-cell cross-entropy, range-offset Smooth L1 (`beta=0.1`), and circular
Doppler cosine loss each have weight 1.0. Only the separate response KL is
weighted by `lambda=0.05` on T1–T3.

| Arm | Physical-truth task loss | Auxiliary response loss |
|---|---|---|
| T0 | Presence, range and Doppler truth | None; response head is instantiated but unused |
| T1 | Same | Full K3 `[520,32]` normalized relative-power support KL |
| T2 | Same | K3 response conditioned on the IQ-only global-peak range cell |
| T3 | Same | Analytic truth-assisted local rectangular-DFT response KL |

K3/K2 inputs use only H1 IQ and the frozen physical configuration. T2 selects
its range cell from the K3 map's global maximum; it is not an independent range
estimator. T3 selects `floor(truth_range_cell)` and uses the true Doppler to
form the length-130 rectangular coherent-integration response on the centered
520-bin frequency axis. The finite coherent sum is evaluated directly and
normalized; the response is PRF-periodic at the Nyquist boundary. T3 is an
oracle auxiliary-supervision control, not a teacher output.

Auxiliary response loss is applied only to H1 rows. H0 rows still contribute to
the physical-truth presence loss. Truth labels are not passed into T1/T2 target
selection. All auxiliary arms use fixed `lambda=0.05`; no temperature,
weight, timing, or reliability search is allowed.

## 3. Training data, updates, and compute

- Reuse the fixed R5-A 51,200 paired training scenes, the frozen pair plan
  (`training_pair_plan.npz`, seed base `2026102711`), and the shared R5-A
  5,000-scene validation set. No arm resumes an R5-A checkpoint. At 32 paired
  scenes per update, 12,800 updates are eight full traversals of the fixed
  51,200-pair cohort.
- Train four arms sequentially, one model resident on the GPU at a time, for
  12,800 optimizer updates. Each update uses 32 paired base scenes (64 CPI
  examples), with effective batch size 64 and gradient accumulation.
- Use AdamW, weight decay `1e-4`, gradient clipping at `1.0`, and the R5 two
  segment cosine schedule: `1e-3` to `1e-4` through update 6,400, then `1e-4`
  to `1e-5` through update 12,800. Validate every 400 updates; select the
  minimum common validation truth-task loss checkpoint.
- Use FP32 without AMP on RTX 3050 Laptop `cuda:0`; CPU fallback is forbidden.
  Start at microbatch 16, then only on OOM reduce to 8 or 4 while preserving
  effective batch and updates. If 4 still OOMs, stop.
- Generate K3 teacher maps online from the current H1 batch using the existing
  complex128 CUDA implementation. Do not persist full IQ or teacher maps.
- Before full training, run G0/G1 checks, including a 16-pair, 200-update
  overfit sanity check per arm, excluded from scientific metrics. It passes
  only if truth-task loss drops by at least 50%, presence accuracy is 100%,
  H1 range-cell accuracy is at least 95%, and task-head gradients are finite
  and nonzero. Keep checkpoints, resumable state, RNG state, update history,
  CUDA/resource records, and failure logs.

## 4. New evaluation splits and seed allocation

The new calibration and development scenes are independent from R5-C's audit
scenes and from the R5-A training/validation sets. H0/H1 are generated as
paired scenes with shared background; only the requested presence side is
retained for each split. The protocol's equal-course H0 mixture is a benchmark
mixture, not an operational prevalence estimate.

| Split | Scenes | Course allocation |
|---|---:|---|
| H0 calibration | 8,192 | 2,048 each for C0, C1, C2, C3 |
| H0 overall verification | 8,192 | Same fixed equal-course allocation |
| H0 C3 verification | 4,096 | Six C3 cells: 683, 683, 683, 683, 682, 682 |
| H1 development | 4,352 | R5-C cell allocation: C0 256; two C1 cells 128 each; three C2 cells 256 each; six C3 cells 512 each |

For overall calibration/verification, the fixed 12-cell counts are
`[2048,1024,1024,683,683,682,342,342,341,341,341,341]` in the configuration's
condition order. The H1 development counts are
`[256,128,128,256,256,256,512,512,512,512,512,512]`. C3-only verification
uses only the six C3 conditions, indexed 0–5 within that split.

Each split uses its own seed base: H0 calibration `728,000,000`, overall H0
verification `748,000,000`, C3 H0 verification `768,000,000`, and H1
development `788,000,000`. Within a split, condition seed blocks are spaced by
1,000,000; generation chunks contain at most 16 scenes and use
`split_base + condition_index*1,000,000 + chunk_index*1009`. Any collision with
known manifests or another registered split stops preflight; seeds are never
silently renumbered. Training and validation IDs are reused only as specified
above. These evaluation scenes are development data, never final blind data.

## 5. Metrics, comparisons, and pilot gates

For each arm, calibrate its presence threshold on its own H0 calibration split
using the raw `presence_logit` score, the `higher` 95th empirical quantile and
decision `score >= threshold`.
For calibration, report the empirical false-alarm rates above and at/above the
threshold, along with tie count; do not report a nominal exact confidence bound
because the threshold is selected on those same samples. For the independent
overall H0 verification and C3 H0 verification splits, report exact one-sided
95% false-alarm upper bounds and ties.
For each H1 development condition, calculate AUC against independent H0
verification scores from the same condition, then weight condition-level AUCs
by the H1 development counts. Use the dedicated C3 H0 verification split for
C3 condition AUCs. Do not infer or retain the paired H0 side in the H1
development split. Evaluate H1 with Pd/AUC, range and circular-Doppler
MAE/RMSE/bias, predicted range mismatch, true-range-gated Doppler, and joint
success by course and each C3 cell. Joint success requires detection, absolute
range error at most one cell, and PRF-wrapped Doppler error at most 20 Hz; a
miss is a joint failure.

The primary endpoint is paired C2+C3 joint success. The only primary contrasts
are T1−T0 and T2−T0. A candidate needs a point gain of at least `+0.02` and a
Bonferroni simultaneous one-sided 97.5% paired-scene bootstrap lower bound
above zero (2,000 resamples, resampling `base_scene_id`). Also report nominal
95% intervals. These intervals describe scene sampling for one trained seed;
they are not training-seed uncertainty. T3 contrasts are secondary and
descriptive.

Task-protection screening is relative to the newly trained T0: overall Pd and
joint point differences at least `-0.01`, with nominal 95% lower bounds above
`-0.05`; C2+C3 joint lower bound above `-0.05`; C2+C3 range and Doppler RMSE
ratios at most `1.05` by point estimate and at most `1.10` at the nominal 95%
upper bound; C3 H0 Pfa difference upper bound at most `+0.02`. Each arm seeking
advancement must also have C3 H0 Pfa at most `0.10`, with its exact upper bound
reported. These are pilot-screening limits, not formal non-inferiority claims.

- **G0:** Exact config, split counts, seeds, no-collision check, pushed code,
  disk/VRAM, and visible RTX 3050 `cuda:0` all pass; no CPU fallback.
- **G1:** Four arms have finite forward/backward and gradients; teacher inputs
  exclude truth; T2 selection is IQ-only; T3 normalization and PRF periodicity
  pass; the fixed-batch overfit sanity check passes.
- **G2/G5:** A teacher arm passes only if its primary contrast and all task and
  C3 false-alarm protections pass. Loss reduction alone cannot pass the gate.

If the independent R5-C review does not support the student pilot, do not
freeze this protocol or train. If all arms fail G1, stop and diagnose the
implementation/optimization issue. If T0 learns but neither T1 nor T2 passes
G2/G5, stop this teacher-content branch; do not proceed to teacher timing,
reliability weighting, formal multi-seed training, or final blind. Only a
single teacher arm passing the pilot can motivate a separately reviewed and
frozen multi-seed confirmation protocol.

## 6. Required records and execution boundary

The run should retain protocol/config, split manifests, training history,
selected checkpoints, per-scene predictions, metrics, analysis, phase review,
environment/resource records, and failure logs. It should not retain complete
IQ or response-map caches. The 20 GiB project-size limit applies after the
stage's cleanup; intermediate size may exceed it. Preserve all user data and
historical negative results.

Current execution state is **not authorized**: external R5-C ChatGPT review is
pending. Do not run `preflight`, `overfit`, or `train` until the returned review
has been considered and a preregistered config/protocol is committed and
pushed. A later update must record the review decision, update the protocol
version/status before any data generation or training, and retain this draft
history.
