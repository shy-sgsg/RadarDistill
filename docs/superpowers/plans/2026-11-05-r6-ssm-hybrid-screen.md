# R6 Complex SSM and CNN-SSM Screen Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add and fairly evaluate a phase-equivariant complex diagonal SSM student and a matched CNN-SSM student against the existing CE-TCN for D2-A-R6 architecture coverage.

**Architecture:** Keep the R6-E CE-TCN task interface and raw complex IQ input. Implement stable complex diagonal state kernels as zero-padded linear convolutions, with three bidirectional SSM blocks; the hybrid adds one local complex temporal residual block. Use magnitude-only real gates and invariant task readouts so common-phase behavior follows from the construction and remains testable.

**Tech Stack:** Existing AISAR Python 3 / PyTorch 2.8.0+cu128, torch complex tensors and differentiable `torch.fft`; no new runtime dependency.

**Spec:** `docs/task_alignments/D2-A-R6_SSM_hybrid_followup_draft.md` (non-frozen; exact protocol is frozen only after R6-G completes).

## Global Constraints

- Do not modify or interrupt the active R6-G run, its code/config/protocol, or tracked files before its safe completion boundary.
- Preserve `docs/RESEARCH_CHARTER.md` v1.3 and all frozen R5-C/R6-A–G results.
- Use three-channel raw complex IQ with shape `[B,6,130,32]`; keep pulse and range axes distinct.
- Compare E1 truth-only students with matching data, sample order, effective batch, updates, optimizer, validation, and checkpoint selection.
- Train sequentially on explicit RTX 3050 Laptop `cuda:0`; no CPU fallback. Do not use final blind data for model selection.
- Do not create or submit a ChatGPT review prompt until all authorized R6 goals are complete.
- Keep the stage budget, seed allocation, stopping gates, and development split provisional until a standalone protocol passes preflight and is committed before data generation.

## Review Focus

1. **Circular-convolution leakage:** an FFT implementation must equal zero-padded linear convolution and must not wrap the end of the CPI to its beginning; test against a direct reference convolution.
2. **Common phase handling:** all complex transformations must be complex-linear and every gate/normalization must use phase-invariant magnitudes; test both feature equivariance and task-output invariance.
3. **Slow-time axis integrity:** recurrence/convolution runs only over 130 pulses per range cell; tests must detect accidental flattening across range cells.
4. **Frequency and boundary behavior:** test exact-bin/off-bin tones, positive/negative shifts, Nyquist wrap, and both sequence ends; report frequency-shift performance without presuming equivariance.
5. **Fair candidate size and interface:** verify both student output dictionaries match CE-TCN and actual inference parameter counts remain in the frozen matched range.

---

### Task 1: Specify the diagonal SSM kernel contract with failing tests

**Files:**
- Create: `tests/test_d2a_r6h_ssm.py`
- Create later: `src/radardistill/students/r6_ssm.py`

**Interfaces:**
- Consumes: CE-TCN complex feature tensors shaped `[B,D,130,32]`.
- Produces: a testable helper `_complex_linear_conv_fft(values, kernel, time_dim=2)` whose output has the same shape and complex dtype as `values`; a direct small-tensor reference is used only in tests.

- [x] **Step 1: Write tests** `test_fft_linear_convolution_matches_direct_reference`, `test_fft_linear_convolution_does_not_wrap_cpi_boundary`, and `test_fft_convolution_supports_complex128_and_complex64`.
- [x] **Step 2: Run the targeted tests and confirm the expected import/API failure.**

Run: `PYTHONPATH=src /home/shy/anaconda3/envs/AISAR/bin/python -m pytest -q tests/test_d2a_r6h_ssm.py`

Expected: FAIL because the SSM helper has not yet been implemented.

- [x] **Step 3: Implement `_complex_linear_conv_fft`** using `torch.fft.fft/ifft`, pad to at least `T + K - 1` (next power of two in the stage configuration), and crop only the causal linear-convolution prefix.
- [x] **Step 4: Re-run the targeted tests and confirm PASS.**

### Task 2: Implement a phase-equivariant complex diagonal SSM block

**Files:**
- Modify: `src/radardistill/students/r6_ssm.py`
- Modify: `tests/test_d2a_r6h_ssm.py`

**Interfaces:**
- `R6ComplexDiagonalSSMBlock(width: int, modes: int, sequence_length: int = 130)` consumes a complex tensor `[B,width,130,32]` and returns the same shape.
- The learned per-channel kernel is `K_d[l] = sum_m C[d,m] * lambda[d,m]**l * B[d,m]`, with complex poles constrained to `abs(lambda) < 1`.
- The two directions use separate poles/coefficient sets; reversed-input causal convolution supplies the reverse scan. Channel mixers are bias-free complex linear transforms. Gates and normalization use magnitudes only.

- [x] **Step 1: Add failing tests** for shape/dtype, common phase equivariance, finite nonzero gradients, range isolation, and exact/off-bin/Nyquist-neighbor tones.
- [x] **Step 2: Run these tests and confirm the expected missing-module/API failure.**
- [x] **Step 3: Implement the minimum SSM block** with stable poles, magnitude-only input gate, separate forward/reverse kernels, phase-preserving residual path, and magnitude RMS normalization.
- [x] **Step 4: Run the targeted tests and confirm PASS.**

### Task 3: Add matched SSM and CNN-SSM students

**Files:**
- Modify: `src/radardistill/students/r6_ssm.py`
- Modify: `src/radardistill/students/__init__.py`
- Modify: `tests/test_d2a_r6h_ssm.py`

**Interfaces:**
- `R6ComplexSSMStudent(width: int = 56, modes: int = 8)` reuses the CE-TCN stem and task heads and replaces its temporal stack with three SSM blocks.
- `R6ComplexCNNSSMStudent(width: int = 56, modes: int = 8)` uses the same stem/task heads and three SSM blocks, with one complex temporal residual block before the SSM stack.
- Both return the CE-TCN keys and shapes: `presence_logit [B]`, `range_logits [B,32]`, `range_offset [B,32]`, `doppler_vector [B,32,2]`.

- [x] **Step 1: Add failing tests** for task output contract, parameter matching, and common-phase invariance.
- [x] **Step 2: Run the tests and confirm the expected missing-module/API failure.**
- [x] **Step 3: Implement both candidates** by composing the existing CE-TCN complex stem/readout with the tested blocks; do not modify R6-E/R6-G implementations.
- [x] **Step 4: Run targeted model tests and confirm PASS.**
- [x] Add a fixed three-arm factory (`ce_tcn`, `complex_ssm`, `cnn_ssm`) with seed-scoped initialization; reject unregistered names.

### Task 4: Add architecture-screen training/evaluation integration after R6-G

**Files:**
- Create: `src/radardistill/training/d2a_r6h.py`
- Create: `src/radardistill/evaluation/d2a_r6h.py`
- Create: `scripts/run_d2a_r6h.py`
- Create: `experiments/D2-A-R6/R6-H/protocol.md`, `config.yaml`, and result files
- Create: `tests/test_d2a_r6h_training.py`, `tests/test_d2a_r6h_evaluation.py`, and `tests/test_run_d2a_r6h.py`

**Interfaces:**
- Training runner accepts a frozen config, output directory, and explicit `cuda:0`; outputs checkpoint, fixed-interval history, and owner-scoped cache manifest.
- Evaluation consumes a selected checkpoint plus disjoint H0 calibration, H0 verification, and H1 development splits; outputs per-scene rows and frozen metrics.
- Budget and seeds are not encoded until the R6-G curve and full-repository scene-ID audit are reviewed and the standalone protocol is frozen.

- [ ] **Step 1: Write failing config, CUDA-device, split-disjointness, checkpoint-selection, and metric-contract tests.**
- [ ] **Step 2: Run targeted tests and verify they fail for the missing runner/evaluator contracts.**
- [ ] **Step 3: Implement config validation and deterministic scene/sampler manifests.**
- [ ] **Step 4: Implement matched CUDA training, fixed-interval logging, validation-only checkpoint selection, and owner-scoped cache cleanup.**
- [ ] **Step 5: Implement independent H0 calibration/verification and H1 development metrics, paired scene bootstrap, common-phase diagnostics, and frequency-shift diagnostics with Nyquist wrapping.**
- [ ] **Step 6: Run targeted tests and the full project test suite; run G0/G1 preflight; freeze and commit protocol/code before generating formal data; then execute the pre-registered screen.**

### Task 5: Report the bounded screen and preserve the route boundary

**Files:**
- Create: `experiments/D2-A-R6/R6-H/analysis.md`, `metrics.json`, figures, and `docs/stage_reviews/D2-A-R6H_review.md`
- Modify after R6-G completes: `docs/CURRENT_STATUS.md`, `docs/RESEARCH_ROADMAP.md`, `docs/research/ASSUMPTION_REGISTER.md`, `docs/progress.md`, and `docs/decisions.md`

- [ ] **Step 1: Report train/validation curves, parameter counts, SNR/condition metrics, six-cell C3 false alarm, physical diagnostics, and resource use.**
- [ ] **Step 2: Keep single-seed development conclusions distinct from formal multi-seed confirmation and final blind.**
- [ ] **Step 3: If an architecture advances, write a new matched knowledge-transfer protocol only after this truth-only screen.**
