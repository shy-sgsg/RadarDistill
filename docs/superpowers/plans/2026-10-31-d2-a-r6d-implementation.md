# D2-A-R6-D Implementation Plan

> **For agentic workers:** Execute inline on `main`, as authorized by the R6 route. Steps use checkbox syntax; do not stop for architecture re-selection.

**Goal:** Implement and evaluate matched detection-only, parameter-only, shared joint, and soft-candidate interaction students on fresh D2-A scenes.

**Architecture:** J1 reuses the TaskStudent residual TCN backbone and heads with D-only, P-only, or joint truth objectives. J2 reuses the same trunk, adds a learned 130-to-520 temporal candidate projection, a normalized null-plus-grid posterior, local continuous range/Doppler refinement, and one bounded parameter-to-detection feedback step. No FFT frontend, teacher map, truth label, or scenario metadata enters forward inference.

**Tech Stack:** Existing RadarDistill PyTorch/AISAR simulation, CUDA on RTX 3050 Laptop, NumPy streaming/memmaps, pytest, exact binomial intervals and paired scene bootstrap.

**Spec:** `experiments/D2-A-R6/R6-D/joint_architecture_design.md`.

## Global Constraints

- Preserve `docs/RESEARCH_CHARTER.md` v1.3 and all R6-A/B/C frozen protocols and results.
- R6-D uses a new training seed and fresh train/validation/calibration/verification/development scene IDs disjoint from R6-A/B/C.
- Pre-register the scientific pilot before any R6-D scene generation; no final blind in this stage.
- Train one model at a time on RTX 3050 Laptop `cuda:0`, FP32, no CPU fallback; effective batch64 for all arms, OOM handling only via microbatch/gradient accumulation.
- Use 25,600 optimizer updates for each arm; identical H0/H1 slots, data exposure, optimizer schedule, validation/checkpoint cadence and seed across matched arms. Checkpoint every800 updates; history every400; never append per-update logs.
- Keep D-only detection evidence and P-only conditional-H1 parameter evidence distinct; do not assign P-only a detection/Pfa qualification.
- C3 H0 Pfa remains an independent protection gate. Do not select checkpoints, models or thresholds from H0 verification or H1 development.
- Preserve manifests, checkpoints, histories, metrics and high-value predictions; remove only R6-D owner-manifest caches after reports are complete; final project <20 GiB.
- Push completed R6-D work packages to `main`; defer the single consolidated ChatGPT review request until the entire authorized R6 route is complete.

## Review Focus

- Null and candidate probabilities must normalize together, including H0 and uniform-logit edge cases.
- The continuous Doppler head must wrap at ±PRF/2 and use physical frequency centers from the frozen 520-bin grid.
- Detection loss must reach J2 parameter predictions through feedback; parameter losses must reach detection logits through soft candidate pooling.
- P-only has no detection objective: it is evaluated for conditional H1 parameter error, never as an end-to-end detector.
- Four arms must share fresh scene IDs, training order, seed, updates, effective batch, optimizer schedule and checkpoint budget.

---

### Task 1: Finalize the implementation contract and TDD model tests

**Files:**
- Modify: `experiments/D2-A-R6/R6-D/joint_architecture_design.md`
- Test: `tests/test_d2a_r6_joint.py`

- [ ] Record the R6-C evidence boundary and the R6-D fixed 25,600-update/effective-batch64 matched budget.
- [x] Record the R6-C evidence boundary and the R6-D fixed 25,600-update/effective-batch64 matched budget; clarify that the budget is not a convergence claim.
- [x] Record the one-round candidate implementation, output/loss contract, H0/null behavior, and why the feedback output layer uses small nonzero initialization (strict zero initialization blocks first-step cross-branch gradients).
- [x] Write tests first for J1/J2 output shapes, finite values, null-plus-candidate probability normalization, PRF wrap, H0 loss masking, batch permutation, and no truth/teacher inputs in `forward`.
- [x] Add tests that demonstrate both cross-task gradient paths, plus detached controls that remove each expected cross-gradient.
- [x] Run the new tests, first confirming RED for the missing implementation, then GREEN after implementation.

### Task 2: Implement J1/J2 students and truth losses

**Files:**
- Create: `src/radardistill/students/r6_joint.py`
- Create: `src/radardistill/training/d2a_r6d_joint.py`
- Modify: `tests/test_d2a_r6_joint.py`

- [x] Implement the shared TaskStudent trunk, J1 task heads and D/P/J objective modes.
- [x] Implement J2's learned temporal candidate projection, `[null, 520×32]` posterior, per-candidate continuous residual outputs, bounded feedback and soft-posterior parameter pooling.
- [x] Implement R6-D loss accounting: D-only BCE; P-only H1-masked range/Doppler losses; J1 joint task losses; J2 presence, candidate/range, range-offset and circular-Doppler losses.
- [x] Preserve zero parameter loss on H0 and ensure all reported component losses are finite.
- [x] Run targeted CPU tests; require GREEN for normalization, periodic boundaries, H0 masking and both gradient directions before CUDA use.

### Task 3: CUDA correctness, resource and G1 gate

**Files:**
- Modify: `tests/test_d2a_r6_joint.py`
- Create/update: `experiments/D2-A-R6/R6-D/g1.json`

- [ ] Measure actual parameters and CUDA forward/backward peak memory per arm on the registered GPU; keep one model resident.
- [ ] Run J1/J2 loss gradient and 16-pair/200-update micro-overfit checks on CUDA; require the preregistered task-loss reduction and finite gradients.
- [ ] Verify effective batch64 under available microbatch/accumulation and record throughput; no training fallback to CPU.
- [ ] If a candidate fails, fix only a diagnosed implementation error or stop that candidate; do not weaken the scientific question.

### Task 4: Freeze the R6-D experiment protocol before data generation

**Files:**
- Create: `experiments/D2-A-R6/R6-D/protocol.md`
- Create: `experiments/D2-A-R6/R6-D/config.yaml`
- Update: `docs/task_alignments/D2-A-R6_alignment.md`

- [x] Register four arms: J1 D-only, J1 P-only, J1 joint, and J2 joint; state which metrics are undefined for specialized D/P arms.
- [x] Register fresh non-overlapping scene blocks, 51,200 paired training scenes, 5,000 validation pairs, H0 calibration/verification120,000 each and H1 development6,144; use identical scenes for all arms.
- [x] Register training seed505, E1 physical truth supervision, FP32 CUDA, effective batch64, 25,600 fixed updates, identical AdamW/learning-rate plan, H0/H1-balanced batches, validation every400 and resume checkpoints every800.
- [x] Register per-arm checkpoint selection based only on the corresponding validation objective; H0 verification and H1 development are evaluation-only.
- [x] Register primary endpoint/protections, C3 Pfa handling, paired scene bootstrap, stopping rules, resource/disk budget, cleanup manifest and no-final-blind boundary.
- [x] Freeze protocol and config at version1.2 before generating any R6-D scenes; G0 will bind their committed hashes before generation.

### Task 5: Implement and audit the R6-D runner

**Files:**
- Create: `scripts/run_d2a_r6d_joint.py`
- Modify: `src/radardistill/training/d2a_r6d_joint.py`
- Modify: `tests/test_d2a_r6_joint.py`

- [x] Implement G0, deterministic split manifests, owner-scoped streaming/cache, matched sampler, checkpoint/resume and 400-update history.
- [x] Implement per-arm validation checkpoint selection and H0 calibration-only threshold selection.
- [x] Implement independent H0 verification, H1 development, scene-paired comparisons and compact per-scene prediction archives.
- [x] Test seed/split disjointness, common arm IDs/order, no verification/development leakage, exact `>=` threshold behavior, null outputs and checkpoint identity.

### Task 6: Run the matched single-seed development pilot

- [ ] Pass G0/G1 and confirm at least4 GiB disk headroom beyond any stage-owned data cache before generation.
- [ ] Generate new R6-D train/validation/calibration/verification/development manifests and pass independence audits.
- [ ] Train D-only, P-only, J1-joint and J2-joint sequentially on CUDA for exactly25,600 updates each; keep effective batch64 and prescribed checkpoint/log intervals.
- [ ] Evaluate only after all training arms and checkpoint choices are frozen; keep H0 verification independent and all H1 outputs labeled development.

### Task 7: Analyze, report, clean and push

- [ ] Produce `metrics.json`, `analysis.md`, training/metric curves, and `docs/stage_reviews/D2-A-R6D_review.md`.
- [ ] Report paired task gains, conditional Pfa/Pd, range/Doppler, C3, uncertainty, budget/exposure, parameter count, VRAM and wall time; preserve negative results.
- [ ] Update status, roadmap, assumptions, decisions, progress and the R6 implementation-plan checkboxes; do not edit the charter.
- [ ] Remove only R6-D manifest-owned reconstructible caches after all reports exist; verify project size remains <20 GiB and user-untracked files remain untouched.
- [ ] Review exact staged paths, run the relevant test suite, commit and push the completed package to `main`.
