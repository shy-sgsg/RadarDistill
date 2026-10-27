# D2-A-R5-C Teacher Quality Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run a bounded, new-scene, CUDA-backed quality audit of the frozen B3 soft RD response and its fixed 4× Doppler-support/local-response forms, without student training or KD.

**Architecture:** Keep the historic NumPy K2 implementation frozen and add a separate strict-CUDA implementation that uses the same DPCA alignment/difference equation. A stream runner generates paired D2-A scenes in CPU batches, transfers only each batch to CUDA, and writes compact per-scene response-quality metrics without caching IQ or maps.

**Tech Stack:** Python 3.13, NumPy, PyTorch 2.8 CUDA 12.8, PyYAML, pytest, RTX 3050 Laptop GPU.

**Spec:** `docs/superpowers/specs/2026-10-28-d2a-r5c-teacher-quality-design.md`

## Global Constraints

- Use three channels, PRF 1300 Hz, N=130, 32 range cells, the frozen E001 static-clutter simulator, and paired H0/H1 scenes.
- New data seeds start at 726,000,000, are disjoint across condition blocks, and must not overlap any tracked/untracked R5 data manifests.
- Teacher IQ processing must use `cuda:0`; CPU fallback is forbidden. Scene generation may remain CPU and batchwise.
- Evaluate 4,352 paired base scenes from the preregistered C0/C1/C2/C3 condition allocation; use batch size 16 and save no raw IQ cache.
- K1 is the 130×32 B3 DPCA power response; K3 is its fixed 4× zero-padded 520×32 support response and teacher-selected-range 520-bin Doppler response.
- Normalize response power only as relative support, never call it a posterior or calibrated likelihood.
- Do not train students, run KD, tune a reliability threshold, alter the simulator, or create final blind data.
- Keep final repository usage below 20 GiB and preserve all untracked user data.

## Review Focus

- Complex FFT frequency-axis ordering and fractional DPCA phase sign: test exact agreement against the frozen NumPy K2 implementation.
- CUDA failure must be explicit with no CPU fallback: test CPU device rejection and require visible CUDA for the run.
- Zero-padding must refine a known continuous tone without frequency-axis wrap/sign error: test a deterministic off-bin sinusoid to within 2.5 Hz.
- H0/H1 identity and leakage: test matched scene IDs, H1−H0 target generation through the frozen simulator, and ensure the teacher function accepts IQ/config only.
- Batch streaming must not materialize or persist whole-split IQ/maps: test fixed batch bounds and output row count/IDs.

---

### Task 1: Freeze the R5-C protocol

**Files:**
- Create: `docs/task_alignments/D2-A-R5C_alignment.md`
- Create: `experiments/D2-A-R5/R5-C/protocol.md`
- Create: `experiments/D2-A-R5/R5-C/config.yaml`
- Existing design: `docs/superpowers/specs/2026-10-28-d2a-r5c-teacher-quality-design.md`

**Interfaces:**
- Consumes the frozen E001 simulation config, existing B3/K2 definition, and R5-C design.
- Produces the fixed conditions, sample counts, seed blocks, metrics, gates, and stopping rules consumed by Tasks 2–4.

- [x] Add task alignment mapping R5-C to charter Q1/Q2/Q4/Q9/Q10, H-TEACH-002/H-TEACH-004, prior K2/T1/T001 evidence, route reason, alternatives, and explicit no-student/no-KD boundaries.
- [x] Freeze C3 6 strata ×512, C2 3×256, C1 2×128, C0 1×256; total 4,352 paired scenes, batch 16, 2,000 paired-scene bootstrap replicates, and seeds from 726,000,000 in disjoint 100,000 blocks.
- [x] Freeze K1/K3 response definitions, ±1 range cell/±20 Hz truth window, observability sharpness score, scale factors, numerical tolerances, and G0/G4 eligibility rules from the spec.
- [x] Validate YAML parsing, count totals, and `git diff --check`; review exact protocol/spec alignment.
- [x] Commit and push only these reviewed protocol/alignment/spec/plan paths to `main` before running data or CUDA teacher evaluation.

### Task 2: Add and test the strict-CUDA B3 response backend

**Files:**
- Create: `src/radardistill/teachers/d2a_k2_cuda.py`
- Create: `tests/test_d2a_k2_cuda.py`
- Reference only: `src/radardistill/teachers/d2a_k2.py`, `src/radardistill/d2a_baselines.py`

**Interfaces:**
- Consumes observable IQ shaped `[B,3,130,32]`, frozen physical config, explicit CUDA device, and oversampling factor.
- Produces a normalized NumPy support map `[B,130×factor,32]`; rejects non-CUDA devices or unavailable CUDA.

- [x] Write failing tests for CPU-device rejection, malformed/nonfinite input rejection, K1 parity at absolute tolerance `1e-6`, normalized/nonnegative finite maps, deterministic output, global complex-scale invariance, and a known off-bin tone peak within one 2.5 Hz bin at factor 4.
- [x] Run the focused test file to confirm the expected missing-module/API failures.
- [x] Implement the minimal Torch complex128 DPCA shift, channel difference, centered FFT/zero-padding, power normalization, explicit synchronization, and CPU-array result export.
- [x] Run the focused tests under `/home/shy/anaconda3/envs/AISAR/bin/python -m pytest` on the visible CUDA device.
- [x] Compare CUDA peak allocation and actual runtime on batch 16; reject execution without `cuda:0`.
- [x] Commit and push the module/tests after reviewing only their exact paths.

### Task 3: Implement the registered streaming audit runner

**Files:**
- Create: `scripts/run_d2a_r5c_teacher_audit.py`
- Create: `tests/test_d2a_r5c_teacher_audit.py`
- Create during execution: `experiments/D2-A-R5/R5-C/preflight.json`, `environment.json`, `teacher_scene_metrics.csv`, `metrics.json`

**Interfaces:**
- Consumes committed protocol/config, E001 simulator, K1/K3 CUDA functions, and new condition-specific seed blocks.
- Produces preflight/environment records, one metric row per paired scene and candidate, condition-stratified bootstrap summaries, and GPU/storage usage.

- [x] Write failing tests for exact condition sample allocation, seed-block non-overlap, paired IDs, fixed chunk size 16, and per-scene K1/K3 metric correctness on a small synthetic map.
- [x] Run focused tests to observe missing runner/helper behavior.
- [x] Implement preflight checking `main == origin/main`, committed protocol/code, PyTorch CUDA availability, RTX 3050 identity, free VRAM, disk headroom, data manifest seed disjointness, and `git status` preservation rules.
- [x] Implement ordered CPU scene batches with explicit per-batch seeds; transfer only IQ batches to GPU; compute K1/K3 for H0/H1; emit metrics/IDs and discard IQ/maps before the next batch.
- [x] Implement paired target-window mass lift, top-cell localization, range/circular Doppler errors, fixed observable peak-mass confidence, condition-stratified 2,000-replicate base-scene bootstrap, scale-invariance checks, and runtime/peak-memory records.
- [x] Run focused tests in AISAR; push runner/tests before actual scene generation.

### Task 4: Run R5-C teacher audit on new scenes

**Files:**
- Generate only under: `experiments/D2-A-R5/R5-C/`
- Do not write IQ/map caches.

**Interfaces:**
- Consumes the committed frozen protocol/config and runner on `main`.
- Produces fresh R5-C metrics and compact per-scene rows from new independent synthetic scenes.

- [ ] Run `preflight`; inspect that it records CUDA available on RTX 3050 Laptop `cuda:0`, seed disjointness, and sufficient disk.
- [ ] Run the registered audit in fixed condition/batch order; abort on CPU fallback, OOM, seed collision, numeric/parity/invariance error, or teacher leakage.
- [ ] Verify all 4,352 paired base-scene IDs exactly once, candidate outputs finite/normalized, expected condition counts, confidence intervals, and GPU peak allocation.
- [ ] Keep temporary outputs and project total below 20 GiB; retain no reconstructible IQ, full support maps, or checkpoints.

### Task 5: Analyze teacher eligibility and report the stage

**Files:**
- Create: `experiments/D2-A-R5/R5-C/analysis.md`
- Create: `experiments/D2-A-R5/R5-C/review_request.md`
- Create: `docs/stage_reviews/D2-A-R5C_review.md`
- Modify: `docs/CURRENT_STATUS.md`, `docs/RESEARCH_ROADMAP.md`, `docs/research/ASSUMPTION_REGISTER.md`, `docs/progress.md`, `docs/decisions.md`

**Interfaces:**
- Consumes frozen protocol, verified runner outputs, existing K2/T1/T001 history, and R5-A/B student evidence.
- Produces a bounded G4 judgment, explicit limits, independent ChatGPT review package, and one evidence-based next action.

- [ ] Compute all registered results without altering gates or omitting failed strata; distinguish soft-response eligibility from fixed-threshold teacher qualification.
- [ ] Update H-TEACH-002 and H-TEACH-004 only if the registered evidence changes their status; do not modify the research charter.
- [ ] Draft analysis and stage review answering whether any knowledge form is eligible for one new student pilot, whether reliability gating is justified, and what must stop.
- [ ] Update dynamic state and decision records; preserve the current R5-B negative result and ChatGPT review request.
- [ ] Create the review request beginning with the required charter-first instruction and include current commit SHA, protocol, code, metrics, analysis, historical K2/T1/T001 context, and route question.
- [ ] Re-read the spec/protocol and inspect exact git diff; run focused/full required verification and `git diff --check`; commit/push exact paths to `main`.
- [ ] Verify pushed SHA equals `origin/main` and final repository usage remains below 20 GiB.
