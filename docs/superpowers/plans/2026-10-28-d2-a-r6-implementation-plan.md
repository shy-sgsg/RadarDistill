# D2-A-R6 Research Plan

> **For agentic workers:** Execute the approved stages inline on `main`. Keep each data/model family in its own protocol and commit; do not use a missing interim ChatGPT review to block authorized work. Steps use checkbox syntax for tracking.

**Goal:** Determine how to build a low-false-alarm, accurate, resource-aware GMTI system by jointly evaluating physical baselines, student scale/training, detection and parameter tasks, representative architectures, and hybrid processing.

**Architecture:** R5-C T0–T3 teaching pilot is complete and its current teacher-content branch stopped by its frozen gate. Next establish a shared-input R6-A benchmark and a statistically supported R6-B false-alarm study, followed by staged capacity/training, joint-model, architecture, hybrid-system, and only evidence-supported teacher-transfer experiments. One GPU model resides at a time; all tests and formal comparisons use independent, recorded scene IDs.

**Tech Stack:** Existing Python/PyTorch/AISAR simulation and evaluation code, CUDA on RTX 3050 Laptop GPU, NumPy memmaps/streaming, pytest, exact binomial intervals and paired scene bootstrap.

**Spec:** `docs/superpowers/specs/2026-10-28-d2-a-r6-research-design.md`

## Global Constraints

- Preserve R5-C T0–T3 v1.0 settings and results as a separate experiment.
- Keep `docs/RESEARCH_CHARTER.md` v1.3 unchanged unless later evidence demonstrates a real scope gap and the user approves a scoped edit.
- Use identical IQ scene IDs for all candidate methods within each R6 comparison; never reuse R5-C evaluation IDs as R6 data.
- Calibrate thresholds on H0 calibration only; report independent H0 verification, condition Pfa, Pd, parameter error, and joint success.
- Treat 1e-6 as unverified until the independent CPI sample size supports it; do not infer system-level false-alarm control from CUT-level results.
- Separate oracle/correct-candidate estimates from actual end-to-end estimates.
- Record parameters, updates, unique scene exposure, fixed-set traversals, schedule, validation curves, seeds, GPU/disk use, and full processing cost.
- Keep frozen R5-C and preregistered R6-A/B batch settings unchanged. A later independent training protocol may screen up to 2x the current effective batch if VRAM allows; all compared arms must preserve matched fairness, otherwise keep the batch fixed or make batch an explicit matched factor.
- R6 model/training jobs must wait until the R5-C GPU run is finished; no two training models reside on GPU at once.
- Stream IQ and teacher outputs; preserve manifests/results and user files; final project size below 20 GiB after authorized cache cleanup.
- Do not launch broad model × teacher × loss searches; each experiment has one stated scientific contrast and a stop rule.
- Write ChatGPT review prompts only after the corresponding authorized work package is complete; continue other authorized phases without waiting for a new review.

## Review Focus

- Scene reuse across R5-C/R6-A/R6-B — tests compare scene IDs and reserved seed blocks.
- H0 CPI maximum and score ties — evaluation tests exact `>=` decisions, tie counts, and one-sided Clopper–Pearson intervals.
- CFAR/NP threshold transfer — tests prove thresholds use calibration only and verification cannot alter them.
- Oracle parameter leakage — tests keep true target/range candidate out of runtime decisions and label oracle metrics separately.
- Joint soft-candidate feedback — tests check circular Doppler, null/H0 state, normalization, and gradients in both directions.
- Low-Pfa sample claims — tests validate the zero-event sample-size calculation and prevent unsupported 1e-6 claims.

---

### Task 0: Finish the frozen R5-C teaching pilot

**Files:** Existing R5-C teaching runner, runtime records under `experiments/D2-A-R5/R5-C-teaching/`, and the R5-C status/review files.

- [x] Finish fresh train/validation data generation without changing the frozen protocol.
- [x] Verify exact counts, pair plan, scene-ID uniqueness and disjointness; all four CUDA G1 overfit checks pass.
- [x] Train T0/T1/T2/T3 sequentially on CUDA for the frozen 12,800 updates; all completed at microbatch 16 without OOM or CPU fallback.
- [x] Evaluate frozen H0 calibration/verification and H1 development, including C3 partial-overlap/0 dB and teacher-correct/incorrect strata.
- [x] Analyze the frozen primary contrasts and protections, and write the stage report.
- [x] Remove only manifest-authorized IQ files; resulting project size is below 20 GiB.
- [x] Write the final ChatGPT review request against the result commit, then commit and push the result package.

### Task 1: R6-A same-scene detector/estimator baseline

**Files:**
- Create: `src/radardistill/evaluation/d2a_r6a.py`
- Create: `scripts/run_d2a_r6a.py`
- Create: `experiments/D2-A-R6/R6-A/{protocol.md,config.yaml}`
- Test: `tests/test_d2a_r6a.py`

- [x] Freeze the independent R6-A protocol/config, fresh seed blocks, 12 condition cells, and exact frozen Medium checkpoint path after R5-C completion. Implementation, G0/G1, data generation and evaluation are complete; protocol-order deviation is recorded.

**Interfaces:** `evaluate_r6a_scene(iq, detector, estimator, config) -> dict[str, float]`; stream generator yields `base_scene_id`, condition metadata, IQ and physical truth. Detector output is one CPI-level score plus candidate location. Estimator output includes actual-candidate and separately labeled oracle-candidate estimates.

- [x] Add tests for same-IQ method fan-out, threshold isolation, Doppler wrap, H0/H1 outputs, and oracle/actual separation.
- [x] Implement B0–B3, R3 T1/T2 (as non-qualified development candidates), frozen Medium, FFT/zero-pad/interpolation/phase-difference/profile-ML baselines using existing validated functions where possible.
- [x] Generate new R6-A H0 calibration (12,000), H0 verification (12,000), H1 development (6,144) scenes on fresh seed blocks; compare all methods on identical split IDs.
- [x] Report workpoints 0.05 and 0.01, ROC/AUC, global and condition Pfa, Pd, all-H1 and hit/correct-candidate RMSE, joint success and diagnostic costs.
- [x] Keep R6-A development data labeled as development; archive and push the baseline package.

### Task 2: R6-B low-false-alarm diagnosis

**Files:**
- Create: `src/radardistill/evaluation/d2a_r6b_low_pfa.py`
- Create: `scripts/run_d2a_r6b_low_pfa.py`
- Create: `experiments/D2-A-R6/R6-B/{protocol.md,config.yaml}`
- Test: `tests/test_d2a_r6b_low_pfa.py`

- [x] Freeze the R6-B low-Pfa protocol/config, sample-size limits, branch-selection firewall, and conditional resource budget. R6-A and the preregistered resource gate are complete; hard-negative branch R is selected.

- [ ] Test exact Clopper–Pearson intervals and required zero-event sample counts for 0.05/0.01/0.001/0.0001/1e-6.
- [ ] On R6-B-only scenes, run H0 calibration and independent verification of 120,000 CPI each (10,000 per registered condition), plus matched H1 development; no R5-C/R6-A IDs are reused.
- [ ] At each method's fixed global threshold, compare pooled and six-cell C3 Pfa with family-aware intervals, condition ROC, H0 score scale, and Pd/joint protection.
- [x] Use the result to select exactly one next diagnostic: observable background normalization/score calibration if conditional ROC is good but global transfer fails; H0 coverage/representation if conditional ROC is poor. R6-A hard-cell AUC selected the preregistered hard-negative-resampling R branch.
- [ ] Evaluate the selected intervention against matched Medium baseline on fresh train/validation IDs, using hard-H0 mining only from training data and no oracle nuisance inputs.
- [ ] Stop an intervention that lowers Pfa while violating the frozen Pd/parameter/joint protection; record untested 1e-4/1e-6 work honestly.

### Task 3: R6-C model scale, training budget, and data scale

**Files:**
- Create: `src/radardistill/students/r6_scaled_cnn.py`
- Create: `src/radardistill/training/d2a_r6c_scaling.py`
- Create: `experiments/D2-A-R6/R6-C/{protocol.md,config.yaml}`
- Test: `tests/test_d2a_r6c_scaling.py`

- [ ] Build 0.3M and 1M CNNs from the same declared topology as the width-96 baseline; report inference and trainable parameter counts separately.
- [ ] Add parameter-shape, forward/backward, CUDA memory, and 16-pair/200-update learning checks.
- [ ] Screen at 12,800 updates on matched R6 training/validation scenes; record unique samples, exposure count, and all validation metrics.
- [ ] Extend to 25,600 then at most 51,200 only if the predeclared validation trend remains improving without a widening overfit gap; defer 102,400 to a separate gate.
- [ ] Compare 51,200 vs 102,400 unique training pairs under fixed updates, then add an exposure-matched comparison only if the first result motivates it.
- [ ] Use one screening seed; take no more than two finalists to at least three independent seeds. Keep H0 verification independent from model selection.

### Task 4: R6-D joint D/P/J models and correctness

**Files:**
- Create: `src/radardistill/students/r6_joint.py`
- Create: `src/radardistill/training/d2a_r6d_joint.py`
- Create: `experiments/D2-A-R6/R6-D/joint_architecture_design.md`
- Test: `tests/test_d2a_r6_joint.py`

- [x] Write the J1/J2 architecture and correctness-gate design; implementation and CUDA tests remain pending.

- [ ] Implement J1 shared-backbone multi-head baseline with D-only, P-only and J losses under matched trunk/optimizer controls.
- [ ] Implement one J2 soft-candidate bidirectional interaction model with explicit null class and continuous range/Doppler refinement.
- [ ] Test output probability normalization, no-target/H0 behavior, PRF-periodic boundaries, and finite bidirectional gradient flow; test detached-branch controls.
- [ ] Measure parameter count, forward/backward and RTX3050 peak VRAM before any scientific training.
- [ ] Run small-tensor CPU/CUDA correctness and 16-pair/200-update G1; only then propose a matched development pilot after R6-A/R6-B.

### Task 5: R6-E representative architecture screening

**Files:** R6-E protocol/config, compact candidate models, candidate-specific tests and a single ordered screen runner.

- [ ] Compare only mechanism-motivated candidates: multiscale/TCN or time-range-separable CNN; axis-factorized periodic Transformer; parameter-matched complex CNN; bidirectional slow-time/range SSM; one CNN+sequence hybrid.
- [ ] Audit the existing TFAS input/position/attention/pooling path and document structural differences in the new Transformer.
- [ ] Screen parameter count, memory, finite forward/backward, overfit behavior and short-budget diagnostics before long training.
- [ ] Do not reject a model family on one short run; send at most the strongest representative candidates to a fresh matched budget/seed protocol.

### Task 6: R6-F hybrid systems

**Files:** R6-F protocol/config, end-to-end pipeline evaluators and tests.

- [ ] Build matched A–F systems: traditional/traditional, NN/NN, NN/profile-ML, traditional/NN, joint NN, NN/profile-ML/statistical validation.
- [ ] Test candidate propagation, null/abstain behavior, and final-chain H0 calibration; never inherit a submodule's Pfa.
- [ ] Evaluate all chains on the same new IQ scenes at the same quality gates and count complete preprocessing/postprocessing/transfer cost.

### Task 7: R6-G targeted teacher collaboration

**Files:** R6-G protocol/config and matched student/teacher training/evaluation modules.

- [ ] Use R5-C T0–T3 and R6 task/baseline outcomes to select at most one teacher knowledge form and one or two student types.
- [ ] Compare against matched truth-only and required truth-assisted controls; keep true labels, teacher hard outputs, soft responses, and reliability features separately logged.
- [ ] Add reliability or teaching-time interventions only as a later one-factor protocol if the knowledge-content contrast first shows task gain.

### Task 8: Formal confirmation and final stage review

**Files:** per-stage `metrics.json`, `analysis.md`, figures, `docs/stage_reviews/`, progress/decisions/assumptions and one consolidated ChatGPT review request after the authorized work package is complete.

- [ ] Select few candidates based on frozen development gates; generate independent formal multi-seed and final evaluation data only after model and analysis choices freeze.
- [ ] Report Pfa sample size and exact bounds, Pd, conditional metrics, parameter/joint outcomes, architecture/training evidence, uncertainty and complete cost.
- [ ] Preserve negative results and route changes; make no unsupported 1e-6 or deployment claim.
- [ ] Update the research status and report one overall CONTINUE / REFINE / REDIRECT / STOP decision with one next priority.
- [ ] Write and push the review request once, after the relevant study package is complete; do not block intermediate authorized stages on it.
