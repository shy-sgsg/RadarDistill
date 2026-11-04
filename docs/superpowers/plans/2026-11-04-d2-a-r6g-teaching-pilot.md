# D2-A-R6-G Teaching Pilot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Implement and run one matched K3 teacher-content pilot on R6G CE-TCN, with truth-only and truth-assisted controls, then report task and convergence evidence.

**Architecture:** Keep R6-E/F models and R5-C frozen code unchanged. Add a new CE-TCN training subclass with a training-only 520×32 response head, a standalone R6-G runner/data owner, and reuse the proven CUDA K3 response and truth-task loss utilities. Run all three arms sequentially at equal effective batch and update budget.

**Tech Stack:** Python, PyTorch CUDA, NumPy, PyYAML, existing RadarDistill simulation/evaluation helpers.

**Spec:** `docs/superpowers/specs/2026-11-04-d2-a-r6g-teaching-pilot-design.md` and `experiments/D2-A-R6/R6-G/protocol.md`.

## Global Constraints

- Use three-channel, PRF=1300 Hz, N=130 and the frozen 12 R6-A conditions.
- Use fresh disjoint scene IDs; each training arm uses seed707 bit-identical initialization and sample order.
- Train T0/T1/T3 with effective batch64, FP32, 51,200 updates, microbatch4, and sequential CUDA residency.
- T1 teacher maps use only H1 IQ and frozen physics on CUDA; no truth, SNR/CNR labels, or teacher correctness may enter teacher/student inference.
- Keep R5-C and R6-E/F source, protocols, and results unchanged.
- Do not log each update; write history every400 updates and checkpoints every800 updates.
- No teacher-weight/temperature/reliability/timing search, no multi-seed confirmation, no final blind.
- Do not remove unrelated or user untracked files. Clean only owner-registered R6-G caches after preserving audit outputs.

## Review Focus

- Teacher/truth target leakage: test T0/T1/T3 source separation and assert T1 receives no truth-derived target.
- Wrong IQ layout or frequency/range axis: test response shape `[B,520,32]`, normalization, finite values and Nyquist wrapping.
- Mismatched arm starts/data exposure: assert initial state equality, identical sampled scene IDs and equal optimizer updates.
- CUDA fallback or OOM changing the experiment: fail if CUDA unavailable; permit microbatch reduction only while effective batch64 remains fixed.
- H0/H1 calibration leakage: test unique scene IDs and each arm's independent threshold calibration.

### Task 1: Freeze R6-G protocol and configuration

**Files:**
- Create: `experiments/D2-A-R6/R6-G/protocol.md`
- Create: `experiments/D2-A-R6/R6-G/config.yaml`
- Create: `docs/task_alignments/D2-A-R6G_alignment.md`
- Modify: `docs/superpowers/specs/2026-11-04-d2-a-r6g-teaching-pilot-design.md`

- [ ] Freeze T0/T1/T3, K3 target, fixed 0.05 weight, from-scratch seed707 cloned initialization, common validation truth-task-loss checkpoint selection, 51,200 updates, fresh split seed bases, evaluation gates and stops.
- [ ] Add protocol validation tests for the fixed arms, budgets, teacher-only inputs and no blind authorization.
- [ ] Run the config validation tests and commit this bounded protocol package.

### Task 2: Add R6-G CE-TCN student and behavior tests

**Files:**
- Create: `src/radardistill/students/r6g_student.py`
- Create: `tests/test_d2a_r6g_student.py`

- [ ] Add `R6GCEStudent(width=64, spectrum_frequency_bins=520)` with the R6-E CE-TCN task outputs plus optional training-only `spectrum_logits` of shape `[B,520,32]`.
- [ ] Test equality of task-output names/shapes with R6-E CE-TCN when loading shared task weights, finite forward/backward, nonzero response-head gradients and public phase invariance of task outputs.
- [ ] Test that `compute_spectrum=False` omits the auxiliary map while retaining identical task-head behavior.
- [ ] Run the focused student tests and commit the model package.

### Task 3: Implement isolated data and CUDA teacher-target preparation

**Files:**
- Create: `src/radardistill/training/d2a_r6g.py`
- Create: `tests/test_d2a_r6g_data_and_targets.py`
- Modify: `scripts/run_d2a_r6g.py`

- [ ] Implement fresh split-ID planning, prior-ID collision audit, owner-bound temporary paired-IQ caches, and disk-budget preflight.
- [ ] Precompute K3 maps for H1 training rows through the existing CUDA teacher path; store only compact CPU targets. Build T3 analytic soft responses from truth only in the control-target path.
- [ ] Test H0 rows never receive teacher maps, T1 maps are finite/normalized, T3 wraps at the Nyquist boundary, splits are disjoint, and cache owner mismatch stops resume.
- [ ] Run the targeted data/teacher tests and fresh G0/G1; require RTX 3050 `cuda:0` and no CPU fallback before formal training.
- [ ] Commit code, tests and preflight records before training.

### Task 4: Train the matched T0/T1/T3 pilot

**Files:**
- Modify: `src/radardistill/training/d2a_r6g.py`
- Create: `experiments/D2-A-R6/R6-G/runs/`

- [ ] Initialize one CE-TCN state from scratch with seed707 and clone it exactly for all three arms; do not warm-start from R6-E seed606.
- [ ] Train T0, T1 and T3 sequentially for 51,200 updates, effective batch64, microbatch4, FP32, identical pair plan, validation every400 and checkpoint every800.
- [ ] Select all three arms by the shared validation truth-task loss only, excluding auxiliary response loss.
- [ ] Save resume/best checkpoints, optimizer/sampler RNG state, bounded training history and CUDA memory/time; stop on nonfinite loss or failed fairness assertion.
- [ ] Verify all arms completed identical update/exposure counts before evaluation.

### Task 5: Evaluate and analyze the teaching pilot

**Files:**
- Create: `src/radardistill/evaluation/d2a_r6g.py`
- Create: `scripts/run_d2a_r6g_evaluation.py`
- Create: `tests/test_d2a_r6g_evaluation.py`
- Create: `experiments/D2-A-R6/R6-G/metrics.json`
- Create: `experiments/D2-A-R6/R6-G/analysis.md`
- Create: `experiments/D2-A-R6/R6-G/figures/`
- Create: `docs/stage_reviews/D2-A-R6G_review.md`

- [ ] Independently calibrate each arm on H0 calibration and evaluate H0 verification/H1 development with paired scene IDs.
- [ ] Report Pd/Pfa, AUC, range/Doppler RMSE, joint success, C3 per-cell familywise Pfa, T1−T0 primary contrast, T1−T3/T3−T0 controls, and validation trajectory summaries.
- [ ] Compute K3 correctness only after training, then report teacher-correct/incorrect and predeclared C3 hard-group student results.
- [ ] Run evaluation tests, generate figures, and verify all report values against metrics.
- [ ] Clean only owner-listed IQ/soft-map caches; record final project size and cleanup audit.

### Task 6: Update project state and complete the R6-G work package

**Files:**
- Modify: `docs/CURRENT_STATUS.md`
- Modify: `docs/RESEARCH_ROADMAP.md`
- Modify: `docs/research/ASSUMPTION_REGISTER.md`
- Modify: `docs/progress.md`
- Modify: `docs/decisions.md`

- [ ] Record the R6-G result, limits, and next single priority without editing the research charter or prior frozen results.
- [ ] Create one internal stage report; do not create a ChatGPT review request until all authorized R6 P0–P7 goals are complete.
- [ ] Verify scoped diff, commit exact R6-G artifacts, and push when GitHub remote operations are available.
