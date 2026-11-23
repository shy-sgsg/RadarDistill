# R6-I Observable Background Score Normalization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Test whether one H0-fitted observable background-scale correction stabilizes the frozen CNN-SSM detector's C3 false-alarm behavior while protecting task performance.

**Architecture:** Build a bounded evaluation module that generates fresh paired scenes in chunks, evaluates frozen CE-TCN/CNN-SSM checkpoints on CUDA and B3 on the same CPU IQ, fits one right-tail logit correction using H0 fit scenes only, and applies separately calibrated global thresholds. Preserve compact prediction arrays and compute exact Pfa bounds plus paired task contrasts.

**Tech Stack:** Python, NumPy, PyTorch CUDA, SciPy, PyYAML; existing RadarDistill simulation, detector, model, and metric helpers.

**Spec:** `docs/superpowers/specs/2026-11-23-r6-i-observable-score-normalization-design.md`

## Global Constraints

- Keep three channels, PRF 1300 Hz, 130 pulses, 32 range cells, stationary clutter, and the R6-A 12-condition set.
- Use only fresh scene IDs and pass the complete prior-manifest/registered-plan overlap audit before generating scenes.
- Fit the correction with H0-only observed logits and IQ-derived scale; do not use H1 truth, SNR, CNR, or condition labels in the correction.
- Freeze the single correction before threshold calibration; do not search further transforms after development evaluation.
- Student inference must run FP32 on AISAR `cuda:0`; CPU fallback is forbidden; keep one model resident at a time.
- Do not create final blind data; persist no full IQ cache; keep the project under 20 GiB.
- Do not modify the research charter, R6-H protocol, checkpoints, or historical results.

## Review Focus

- Scene plans or generated IDs overlap prior experiments: G0 must build and reject overlaps before evaluation.
- Condition labels or truth leak into score correction: unit tests must show the correction uses only the score and observable log-power vectors.
- Threshold calibration leaks into verification: keep fit, threshold-calibration, verification, and H1 scene IDs disjoint and assert it.
- The correction hides false alarms by sacrificing target detections: paired Pd/joint and parameter protections are mandatory.
- GPU unavailability or OOM causes an unnoticed CPU path: require CUDA device checks and stop or reduce inference microbatch without changing sample counts.

---

### Task 1: Freeze R6-I protocol, config, and alignment

**Files:**
- Create: `experiments/D2-A-R6/R6-I/protocol.md`
- Create: `experiments/D2-A-R6/R6-I/config.yaml`
- Create: `docs/task_alignments/D2-A-R6-I_alignment.md`
- Modify: `docs/RESEARCH_ROADMAP.md`, `docs/CURRENT_STATUS.md`, `docs/research/ASSUMPTION_REGISTER.md`, `docs/decisions.md`, `docs/progress.md`

**Interfaces:**
- Consumes the R6-H fixed checkpoints and R6-A/R6-H condition, physics, and metric definitions.
- Produces an authorized, preregistered evaluation contract and disjoint scene plan; no data is generated in this task.

- [x] Record the primary hypothesis, one score correction, sample counts, gates, exclusions, CUDA requirement, and no-blind boundary in protocol/config.
- [x] Register `H-R6I-001` as TESTING and record why R6-H results motivate this finite follow-up.
- [x] Run YAML loading and plan-construction checks without generating IQ; confirm the registered split IDs are internally disjoint.

### Task 2: Implement pure score and robust-scale functions with tests

**Files:**
- Create: `src/radardistill/evaluation/d2a_r6i.py`
- Create: `tests/test_d2a_r6i.py`

**Interfaces:**
- Produces `robust_iq_power(iq) -> np.ndarray`, `fit_tail_scale_slope(logits, log_power, *, bin_count, quantile) -> dict`, and `correct_presence_logit(logits, log_power, *, beta, center) -> np.ndarray`.
- Produces exact binomial and paired bootstrap summaries by reusing existing R6-E/R6-F helpers where their definitions match.

- [x] Test robust power shape, positivity, finite validation, and exact `|a|^2` scaling for `aX`.
- [x] Test tail-slope fitting on synthetic linear and no-variance inputs; reject nonfinite or insufficient samples.
- [x] Test corrected score values against the registered equation, and prove no labels/condition variables are accepted by the function interface.
- [x] Implement the three pure functions and run the focused tests.

### Task 3: Implement scene audit and streamed CUDA evaluation

**Files:**
- Modify: `src/radardistill/evaluation/d2a_r6i.py`
- Create: `scripts/run_d2a_r6i.py`

**Interfaces:**
- CLI commands: `preflight`, `g1`, `evaluate`, `analyze`.
- The evaluator streams IQ batches to the fixed CE-TCN and CNN-SSM checkpoints on `cuda:0`, obtains B3 scores from the same observable IQ, and writes compact per-split predictions.

- [x] Add deterministic plans for `h0_scale_fit`, `h0_threshold_calibration`, `h0_verification`, and `h1_development` using the frozen 12-condition configuration.
- [x] Add G0 checks for HEAD/protocol/config/source/checkpoint hashes, all prior CSV manifests and registered plans, CUDA device, available disk, and split-ID uniqueness.
- [x] Add G1 checks for checkpoint ownership, model output contracts, finite predictions, robust power behavior, and correction arithmetic; do not train models.
- [x] Add sequential model loading/inference, fixed threshold calibration, exact H0 verification, H1 task metrics, and deterministic paired contrasts.
- [x] Add the `a=0.5/2.0` matched scale intervention for the registered C3 partial-overlap subset.
- [ ] Run focused tests, then run G0/G1 on AISAR; commit/push protocol, code, and tests before evaluation data generation.

### Task 4: Run the frozen evaluation and produce the result package

**Files:**
- Create: `experiments/D2-A-R6/R6-I/preflight.json`, `g1.json`, manifests, prediction arrays, `metrics.json`, and `analysis.md`
- Create: `docs/stage_reviews/D2-A-R6I_review.md`

**Interfaces:**
- Consumes only the four registered independent evaluation splits and the fixed R6-H checkpoints.
- Produces a single development result, a P7 re-admission decision, and a cleanup/resource record if any generated temporary files require removal.

- [ ] Run only after the frozen implementation commit is pushed and G0/G1 pass.
- [ ] Evaluate global Pfa workpoints 0.05/0.01/0.001; treat 0.05 as the sole primary gate.
- [ ] Report C3 six-cell Bonferroni exact upper bounds, score-scale relationship, paired task protections, hard-cell results, and scale intervention.
- [ ] Apply the frozen stop rule; do not tune the correction on these results.
- [ ] Update current status, roadmap, assumption register, progress, and decisions with the actual result; do not draft a ChatGPT prompt while the authorized route remains open.
- [ ] Verify disk usage, exact staged paths, pushed `main` SHA, and preserve all unrelated untracked user files.
