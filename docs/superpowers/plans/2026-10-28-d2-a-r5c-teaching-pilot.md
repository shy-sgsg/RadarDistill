# D2-A-R5-C Teaching Content Pilot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Implement and run one matched T0–T3 pilot to test whether R5-C teacher responses improve Medium TaskStudent's final detection/range/Doppler task beyond physical-truth supervision.

**Architecture:** Keep the width-96 TaskStudent task trunk and task heads unchanged; expose a configurable training-only response head with 520 Doppler bins. Add pure response-target/KL helpers, a stage-specific CUDA runner, and independent streamed evaluation; reuse the existing R5-A pair plan, TaskStudent losses, schedule, K3 CUDA teacher, and D2-A evaluation primitives without modifying their frozen protocols.

**Tech Stack:** Python, PyTorch FP32 training, CUDA complex128 K3 teacher, NumPy scene/cache data, pytest.

**Spec:** `docs/superpowers/specs/2026-10-28-d2-a-r5c-teaching-pilot-design.md`

## Global Constraints

- Preserve `docs/RESEARCH_CHARTER.md` v1.3 and every frozen R5-A/R5-B/R5-C result.
- R5-C independent ChatGPT review is pending; keep the new protocol as draft and do not freeze or start training until that review is returned and the stage decision supports the pilot.
- Use four arms T0/T1/T2/T3, identical initial state (seed 303), 51,200 fixed training pairs, frozen pair plan seed base 2026102711, 12,800 updates, 32 pairs per update, effective batch 64, FP32, two-segment R5 cosine schedule, and one model resident at a time.
- T1/T2 teachers receive only H1 IQ and frozen config; generate K3 responses online on `cuda:0` with complex128; never cache complete IQ or teacher maps; apply auxiliary loss only on H1 rows.
- Use `lambda=0.05` fixed for T1/T2/T3; T3 is analytic truth-assisted rectangular-DFT response, not teacher output; no reliability weighting or timing search.
- Use separate new seed ranges 728M/748M/768M/788M for H0 calibration/overall verification/C3 verification/H1 development, exact split counts from the design, chunk size 16, condition stride 1M, chunk seed stride 1009; any collision stops without auto-renumbering.
- Calibrate each arm on 8,192 H0 using an equal-course mixture, target Pfa 0.05, `higher` quantile and `score >= threshold`; verify on new 8,192 H0 plus independent C3 H0 n=4,096; evaluate H1 on new 4,352 scenes. These are development data, never final blind.
- C2+C3 joint success is the primary endpoint. T1−T0/T2−T0 gains must be ≥0.02 and each Bonferroni simultaneous one-sided 97.5% paired-scene bootstrap lower bound >0; report nominal 95% intervals as well. Task-protection deltas compare with T0; Holm is not used.
- RTX 3050 Laptop `cuda:0` is mandatory, CPU fallback forbidden, FP32 only; on OOM reduce microbatch 16→8→4 while maintaining effective batch, otherwise stop. Preserve all untracked user files and keep the post-stage repository below 20 GiB.

## Review Focus

- Nyquist wrap in T3's analytic Dirichlet target: test equivalent Doppler values across ±PRF/2 produce the same normalized target.
- K2-local leakage: test its selected range comes only from the K3 teacher map, not range/Doppler truth.
- H0 teacher supervision: test all teacher/auxiliary losses exclude H0 rows while H0 still contributes to truth presence loss.
- Split reproducibility: test fixed condition counts and seed formulas are unique, and any manifest collision is rejected.
- Legacy compatibility: test default `TaskStudent` still emits `[B,130,32]`, while the pilot option emits `[B,520,32]` and inference export omits the training head.

---

### Task 1: Add the configurable 520-bin training-only response head

**Files:**
- Modify: `src/radardistill/students/task_student.py`
- Test: `tests/test_d2a_r5c_teaching.py`

**Interfaces:**
- `TaskStudent(width=96, spectrum_frequency_bins=520)` returns `spectrum_logits` with shape `[B,520,32]` when `compute_spectrum=True`.
- Default `spectrum_frequency_bins=130` preserves existing R5-A/E002 behavior.

- [x] **Step 1: Add failing tests** for the legacy default, 520-bin shape, and export without the auxiliary head.
- [x] **Step 2: Run the tests and confirm the new constructor option is missing.**
- [x] **Step 3: Implement the optional frequency-bin count** with positive-integer validation; leave task heads and existing default unchanged.
- [x] **Step 4: Run `pytest tests/test_d2a_r5c_teaching.py -q`.**
- [x] **Step 5: Commit** only the model and test files.

### Task 2: Implement full, local, and truth-assisted response targets/losses

**Files:**
- Create: `src/radardistill/distillation/r5c_teaching.py`
- Test: `tests/test_d2a_r5c_teaching.py`

**Interfaces:**
- `k2_local_targets(teacher_map: Tensor) -> tuple[Tensor, Tensor]` returns IQ-only teacher range indices and normalized local frequency distributions from `[B,520,32]` maps.
- `truth_dirichlet_targets(range_cell: Tensor, doppler_hz: Tensor, *, prf_hz: float=1300.0, pulse_count: int=130) -> tuple[Tensor, Tensor]` returns `floor(range_cell)` and normalized 520-bin rectangular coherent-integration power responses.
- `response_kl(student_logits: Tensor, arm: str, *, teacher_map: Tensor | None, truth_range_cell: Tensor | None, truth_doppler_hz: Tensor | None) -> Tensor` implements T1 full-map KL, T2 teacher-selected local KL, T3 truth-selected local KL, and T0 zero loss.

- [x] **Step 1: Write failing tests** for KL zero on identical categorical distributions, T1 normalization, T2 IQ-only argmax selection, and T3 Dirichlet normalization/periodicity.
- [x] **Step 2: Run the focused tests and verify expected missing-function failures.**
- [x] **Step 3: Implement the pure target and KL functions** with strict shapes, finite/nonnegative checks, and no truth arguments on the teacher-target path.
- [x] **Step 4: Run the focused tests and ensure all pass.**
- [x] **Step 5: Commit** the loss module and tests.

### Task 3: Draft pilot protocol/config and validate seeds/splits

**Files:**
- Create: `experiments/D2-A-R5/R5-C-teaching/protocol.md`
- Create: `experiments/D2-A-R5/R5-C-teaching/config.yaml`
- Modify: `src/radardistill/training/d2a_r5c_teaching_pilot.py`
- Test: `tests/test_d2a_r5c_teaching_runner.py`

**Interfaces:**
- `validate_r5c_teaching_config(config) -> None` rejects changes to arms, data, schedules, losses, seed ranges, CUDA, or gates.
- `build_split_seed_specs(config) -> list[dict]` enumerates the exact split/condition/chunk seeds in the design.
- `assert_split_seed_disjoint(specs, existing_manifest_seeds) -> None` raises on any overlap; it never picks replacement seeds.

- [x] **Step 1: Write failing tests** for exact per-course counts, per-condition seed formula, split disjointness, collision rejection, and frozen config values.
- [x] **Step 2: Run the focused tests and verify they fail for missing validator/seed helpers.**
- [x] **Step 3: Implement config and split helpers** and write a protocol/config explicitly marked `draft` until independent R5-C review is returned.
- [x] **Step 4: Run the focused tests.**
- [x] **Step 5: Commit** the draft protocol/config, helper/tests, and matching task-alignment/design records.

### Task 4: Implement matched CUDA training with online teacher responses

**Files:**
- Modify: `src/radardistill/training/d2a_r5c_teaching_pilot.py`
- Create: `scripts/run_d2a_r5c_teaching_pilot.py`
- Test: `tests/test_d2a_r5c_teaching_runner.py`

**Interfaces:**
- `train_arm(arm, initial_state, train_arrays, pair_plan, config, device, microbatch) -> dict` runs one arm for 12,800 updates, uses the common validation truth-task loss for checkpoint selection, and saves resumable model/optimizer/scheduler/RNG/data-cursor state.
- `compute_h1_teacher_maps(iq_h1, config, device) -> Tensor` calls the existing strict-CUDA K3 response (factor 4), returns only the current update's temporary targets, and does not persist them.
- Runner phases: `preflight`, `overfit`, `train`, `evaluate`, `analyze`.

- [x] **Step 1: Write failing CPU unit tests** for matched initialization, paired H0/H1 ordering, H1-only K3 calls/loss, accumulation scale, sequential arm selection, and review-gated preflight.
- [x] **Step 2: Run the focused tests and confirm missing pilot behavior.**
- [x] **Step 3: Implement a stage-specific runner** that reuses `make_paired_examples`, `scale_d2a_e002_microbatch_loss`, `R5CosineSchedule`, `TaskStudent`, `task_loss`, and `k2_response_cuda`; do not modify R5-A runner behavior.
- [x] **Step 4: Run focused tests plus the existing R5 student/teacher regression tests.**
- [x] **Step 5: Commit and push this implementation while the protocol/config remain draft.** Do not run preflight, overfit, or training before the independent R5-C review supports proceeding.

### Task 5: Implement independent streamed evaluation and pilot gates

**Files:**
- Create: `src/radardistill/evaluation/d2a_r5c_teaching_pilot.py`
- Modify: `scripts/run_d2a_r5c_teaching_pilot.py`
- Test: `tests/test_d2a_r5c_teaching_evaluation.py`

**Interfaces:**
- `summarize_pilot_arm(predictions, calibration, verification, config) -> dict` computes per-arm threshold/Pfa/Pd/AUC/range/Doppler/joint metrics.
- `paired_teaching_contrasts(arm_predictions, config) -> dict` computes T1−T0, T2−T0, T3−T0 and T2−T3 scene bootstrap intervals and Bonferroni simultaneous primary lower bounds.
- Evaluation streams generated scenarios and saves only IDs and per-scene predictions; no IQ/map cache or final blind.

- [x] **Step 1: Write failing tests** for the higher quantile and `>=` comparator, exact one-sided Pfa interval, circular Doppler error, joint-failure-on-miss, paired bootstrap unit and 2.5th-percentile Bonferroni lower bounds, plus the descriptive T2−T3 contrast.
- [x] **Step 2: Run tests and verify missing or incorrect evaluation behavior.**
- [x] **Step 3: Implement evaluation using existing D2-A calibration/metrics helpers where compatible.**
- [x] **Step 4: Run focused evaluation tests and R5 regression tests.**
- [x] **Step 5: Commit** the evaluation module, runner integration and tests.

### Task 6: Execute the one-seed pilot and complete the research report

**Files:**
- Generate under: `experiments/D2-A-R5/R5-C-teaching/`
- Update: `docs/CURRENT_STATUS.md`, `docs/RESEARCH_ROADMAP.md`, `docs/research/ASSUMPTION_REGISTER.md`, `docs/progress.md`, `docs/decisions.md`, `README.md` only where evidence requires it.
- Create: `docs/stage_reviews/D2-A-R5C-teaching_review.md`
- Create: `experiments/D2-A-R5/R5-C-teaching/review_request.md`

- [ ] **Step 1: After the independent R5-C review supports proceeding, record its decision and freeze protocol/config in a separate commit; verify the frozen code is pushed, then run preflight and G0/G1 and record environment, CUDA, disk and model parameter counts.**
- [ ] **Step 2: Train T0, T1, T2, T3 sequentially** using one seed and the frozen 12,800-update schedule; preserve checkpoints, per-update history, resource data and failure logs.
- [ ] **Step 3: Run separate H0 calibration, H0 verification, C3 H0 verification and new H1 development evaluation.**
- [ ] **Step 4: Analyze T1/T2 versus T0 and T2 versus truth-assisted T3; evaluate C3 Pfa and all task protections; distinguish scenario bootstrap from seed uncertainty.**
- [ ] **Step 5: Decide whether any teacher content passes G5.** If none passes, stop timing/reliability/mechanism expansion. If exactly one passes, propose a separate ≥3-seed confirmation; do not auto-run blind.
- [ ] **Step 6: Update only evidence-supported dynamic status/assumptions, preserve R5-C/R5-A/R5-B historical results, run the requested tests and diff/size review, commit to `main`, and push.**

## Self-Review

- **Spec coverage:** All four arms, 520-bin representation, H1-only teachers, fixed training budget, seed isolation, CUDA-only execution, false-alarm protection, paired statistics, report, and stop gates have an owning task.
- **Scope boundary:** External review of the completed R5-C teacher audit remains a required decision point before protocol freeze/training. The design/code work can be prepared, but the pilot cannot be described as preregistered or executed before that review.
- **Compatibility:** 130-bin behavior stays the default; R5-A/E002 frozen configs and outputs are not changed.
- **Data/compute:** The selected new seed ranges and exact split allocations are frozen in the design; target maps are online and temporary; models train sequentially on `cuda:0`.
- **Multiple comparisons:** Only the two teacher-vs-T0 primary contrasts use Bonferroni simultaneous lower bounds; all protection intervals are nominal screening intervals.
- **Limitations:** One training seed gives no cross-seed claim; pilot development is not blind; C3 Pfa gate remains absolute.
