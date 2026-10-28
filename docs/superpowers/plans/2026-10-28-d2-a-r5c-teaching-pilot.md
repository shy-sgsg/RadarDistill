# D2-A-R5-C Teaching Pilot Implementation Plan

> **For agentic workers:** Execute inline on the user-authorized `main` branch. Each completed work package is committed and pushed to `origin/main` before moving to the next gate.

**Goal:** Freeze and execute one matched, single-seed T0–T3 student teaching pilot on fresh R5-C-independent scenes, then report whether teacher soft knowledge adds task-level GMTI value beyond truth supervision.

**Architecture:** Keep the width-96 TaskStudent, frozen truth task loss, 12,800 updates, effective batch 64, and sequential CUDA execution. Generate new train/validation pairs into stage-owned CPU memmaps, create one deterministic shared eight-pass pair plan, and compute K3 teacher responses online on `cuda:0`. Use a shared `[520,32]` response head and categorical KL over the same joint map for T1, T2, and T3; embed the IQ-selected local response or truth-assisted local response at its selected range gate.

**Tech Stack:** Python, NumPy memmaps, PyTorch/AISAR, CUDA, pytest.

**Spec:** `experiments/D2-A-R5/R5-C-teaching/protocol.md`

## Global Constraints

- Preserve `docs/RESEARCH_CHARTER.md` v1.3 and all historical R5-A/B/C results and gates.
- Bind independent review to result commit `7563c822e70afe761897ddd5e91bb64207720f89`, decision `REFINE`, pilot support `true`.
- New training, validation, calibration, verification, and development scene IDs must be disjoint from all registered scene IDs and each other.
- T0–T3 use identical initialization, scene order, task loss, optimizer, 12,800 updates, and effective CPI batch 64.
- Preserve the C3 partial-overlap / 0 dB hard subgroup, all physical truth, and teacher outputs separately.
- CUDA is mandatory; no CPU fallback. One student model at a time. Reduce microbatch on OOM while preserving effective batch and updates.
- Keep generated IQ and teacher maps out of Git; remove only this stage's verified IQ cache after evaluation, preserving manifests, pair plan, checkpoints, logs, metrics, and audit outputs. Final project size must remain below 20 GiB.
- No teacher-timing, reliability-weight, multiseed, final-blind, or full R5-D mechanism experiment in this pilot.

## Review Focus

- Fresh training and validation scenes — tests must reject reuse of R5-A cache IDs and verify seed-block disjointness.
- T2/T3 range-gate supervision — tests must show the common joint-map KL penalizes mass at an incorrect range gate.
- H0 teacher leakage — tests must show auxiliary response loss uses H1 rows only and T1/T2 inputs exclude truth.
- Stage authorization — tests must keep data generation and training blocked until the exact reviewed commit is recorded and the protocol/config is pushed.
- CUDA and resumability — tests must reject CPU fallback and preserve matched updates/effective batch after microbatch fallback.

---

### Task 1: Freeze the reviewed protocol and fresh data plan

**Files:**
- Modify: `experiments/D2-A-R5/R5-C-teaching/protocol.md`
- Modify: `experiments/D2-A-R5/R5-C-teaching/config.yaml`
- Modify: `docs/task_alignments/2026-10-28-D2-A-R5C-teaching-pilot.md`
- Modify: `tests/test_d2a_r5c_teaching_runner.py`
- Modify: `src/radardistill/training/d2a_r5c_teaching_pilot.py`

- [x] Record the returned independent review and its exact target commit; freeze version 1.0 before generation.
- [x] Specify fresh train/validation seed blocks, cell allocations, evaluation blocks, and one fixed pair plan.
- [x] Add tests that reject reused R5-A scene IDs, seed collisions, and changes to the review-linked design.
- [x] Update the exact config validator and verify the tests in AISAR.
- [x] Commit and push the reviewed preregistration and implementation package before preflight.

### Task 2: Match T1–T3 auxiliary response support

**Files:**
- Modify: `src/radardistill/distillation/r5c_teaching.py`
- Modify: `tests/test_d2a_r5c_teaching.py`
- Modify: `experiments/D2-A-R5/R5-C-teaching/protocol.md`

- [x] Add a test proving T2/T3 loss penalizes a response placed at an incorrect range gate.
- [x] Embed T2 teacher-local and T3 truth-assisted Doppler distributions in the shared normalized `[520,32]` map.
- [x] Keep truth task losses active for all arms and teacher map generation IQ-only.
- [x] Run the focused response tests, then the pilot unit tests; commit and push.

### Task 3: Implement fresh train/validation cache generation

**Files:**
- Modify: `scripts/run_d2a_r5c_teaching_pilot.py`
- Modify: `src/radardistill/training/d2a_r5c_teaching_pilot.py`
- Modify: `tests/test_d2a_r5c_teaching_runner.py`

- [x] Add a committed `prepare-data` phase gated by a passing preflight.
- [x] Use the existing paired scene materializer to write only stage-owned CPU memmaps and a deterministic shared pair plan.
- [x] Add tests for seed-derived IDs, exact quotas, cache manifests, and preserving unrelated files.
- [x] Estimate required disk before generation; record generated files and seed metadata without committing IQ arrays.
- [x] Run focused tests and commit/push before invoking preflight.

### Task 4: Preflight, G1, and one-seed matched training

**Files:**
- Modify as required: pilot runner/trainer and focused tests.
- Create runtime records under `experiments/D2-A-R5/R5-C-teaching/`.

- [ ] Run preflight from pushed `main`; verify RTX 3050 `cuda:0`, seed isolation, exact config, storage, and model parameter counts.
- [ ] Generate the fresh train/validation data, then run the frozen 16-pair/200-update G1 overfit check for all arms.
- [ ] If G1 passes, train T0, T1, T2, T3 sequentially for 12,800 updates each using the same pair plan and validation selection rule.
- [ ] Save best and resumable checkpoints, per-update histories, environment, and CUDA resources.
- [ ] Stop without modifying scientific settings on implementation errors or failure gates; commit/push runtime records only after the run evidence is reviewed.

### Task 5: Independent evaluation, hard-subgroup analysis, and stage decision

**Files:**
- Modify as required: evaluation/analysis and focused tests.
- Create: `metrics.json`, prediction summaries, figures, `analysis.md`, stage review, review request.
- Modify: `docs/CURRENT_STATUS.md`, `docs/RESEARCH_ROADMAP.md`, `docs/research/ASSUMPTION_REGISTER.md`, `docs/progress.md`, `docs/decisions.md`.

- [ ] Evaluate independent H0 calibration/verification and H1 development scenes; no final blind.
- [ ] Report T1−T0/T2−T0 primary C2+C3 joint contrasts, task and C3-Pfa protection, T3 truth-assisted controls, and the C3 partial-overlap/0 dB subgroup.
- [ ] Report teacher-correct/incorrect strata as offline analysis only and keep truth separate from teacher inputs.
- [ ] Remove only cache files listed by this stage's completed data manifest; keep audit manifests and final project under 20 GiB.
- [ ] Write a four-layer stage review, one overall decision, and a ChatGPT review request bound to the new commit; commit and push the completed package.

---

**Research ruling:** The previously suggested 808M–920M seed region produces false collision reports because the existing manifest scanner interprets unrelated `size_bytes` integers as seeds. Use the scanned-clean 1.008B–1.120B region instead; this preserves the scanner and avoids a nonessential infrastructure change. Cost if wrong: a real collision outside the scanned manifests would compromise nominal data independence, so the preflight must repeat the full seed/ID scan immediately before generation.
