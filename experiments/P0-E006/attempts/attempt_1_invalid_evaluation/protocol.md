# P0-E006 preregistered protocol

**Status:** frozen before E006 model training and before generation of final blind-set predictions. The earlier E001–E005 test set (seed 7070) is a development benchmark because it has informed prior modeling decisions.

## Question and hypotheses

**Question:** Can explicit coherent-signal features improve lightweight neural Doppler estimation, and does teacher information improve the same students beyond truth supervision?

- H1: the original local CNN lacks a useful inductive bias for coherent phase accumulation.
- H2: multi-lag complex correlations or an explicit frequency response improve learning efficiency and/or accuracy.
- H3: a local profile-likelihood response transfers information beyond the teacher point estimate or its circular mean.

These are hypotheses. Results may reject any or all of them.

## Signal model and data roles

Use `x[n] = A exp(j(2π f n / PRF + φ)) + w[n]`, PRF=1300 Hz, N=130, continuous uniform `f` in `[-PRF/2, PRF/2)`, uniform phase in `[-π,π)`, amplitude in `[0.5,1.5]`, and circular complex Gaussian noise. Training/validation SNR levels are `[-15,-5,0,10,20]` dB; final per-SNR results use `[-15,-10,-5,0,5,10,15,20]` dB. No clutter, channel mismatch, acceleration, multiple targets, or motion compensation errors are included.

| Role | Seeds | Size |
|---|---|---:|
| Train | 1101, 1202, 1303 | 6000 per seed; SNR sampled from training levels |
| Validation | 3303, 3404, 3505 | 1200 per seed; independent of train |
| Development benchmark | existing seed 7070 | existing E001–E005 test set; never treated as final |
| Final blind test | 909090 plus the fixed per-SNR offsets in config | 1024 per SNR |
| Noiseless correctness | seed 777771 | 2048 continuous-frequency examples |

The final seed, seeds, model definitions, training limits, metrics, and decision threshold are fixed here before final evaluation. If a post-test design change is made, a new independent final-test seed will be required.

## Models and physical baselines

- **S0:** load each frozen E005 E1 checkpoint with the same phase-increment input; do not retrain or tune it.
- **S1:** normalized complex autocorrelation features for lags 1, 2, 4, 8, represented as real/imaginary pairs, followed by a small MLP with a circular two-value frequency output. The feature uses average lag product divided by average signal power.
- **S1 analytic baseline:** maximize the weighted multi-lag phase-coherence score on the same four complex correlations; grid is dense over one PRF and locally refined.
- **S2:** fixed 4N zero-padded FFT; input is log power centered by its per-example maximum. A small MLP predicts circular frequency. This is a physics-guided hybrid.
- **Classical references:** N-point FFT peak, log-parabolic interpolation, one-lag phase difference, local FFT-initialized profile ML, high-resolution global-grid profile ML, and 4N FFT peak.

## Teacher information and training protocols

- **T1 hard point:** continuous profile-ML teacher frequency as a circular target.
- **T2 circular soft target:** existing 65-candidate local profile likelihood over ±2 native FFT bins (grid spacing 4/64 native bins), converted to its circular mean. This is a compressed local target, not a global posterior.
- **T3 full local response:** normalized likelihood over those same 65 candidates, retained as a vector target and matched with KL divergence from a student response head.
- **E1:** physical truth target; model selection on truth validation.
- **E2-A:** teacher-only training and model selection on the corresponding teacher target. No true validation label participates in checkpoint selection.
- **E2-B:** same teacher-only training with truth-based checkpoint selection; explicitly labeled as using true validation labels.
- **E3:** truth loss plus the fixed teacher loss. Validation selection uses the truth objective because truth remains part of training. Teacher point/soft/full-response variants are recorded separately.

Train S1 and S2 with E1, E2-A/T1, E2-A/T2, and E3/T2. For S1 additionally compare E2-A/T3, E2-B/T2, and E3/T3. This is the complete bounded comparison; do not add architectures or hyperparameter searches.

## Optimization and reproducibility

PyTorch in Conda environment `AISAR`; CUDA is mandatory. Fail before training if CUDA is unavailable. Adam, learning rate `0.001`, batch size `128`, at most `160` epochs, early-stopping patience `40`, fixed torch and data seeds. Use the same train/validation splits and training budget within each seed. Save all checkpoints and per-epoch train/validation objective histories.

## Evaluation

- Circular frequency error wraps with period PRF. Report RMSE, MAE, bias, outlier rate (`|error| > PRF/(2N)`), CRLB, per-seed values, and grouped results by frequency, phase, amplitude, and FFT-bin offset.
- Primary comparison: E1 RMSE minus N-point FFT-peak RMSE on the final set, paired by sample and training seed. Use a two-level bootstrap with 2000 replicates.
- Primary gate passes only if the paired 95% interval for `E1 RMSE - FFT peak RMSE` is entirely negative at two adjacent points among 0, 5, 10, 15, 20 dB.
- **G1 learning feasibility passes** if at least one student, for at least two of three training seeds, has noiseless full-range MAE below 1 Hz and 95th-percentile absolute wrapped error below 5 Hz, and its 20 dB E1 RMSE is at least 50% below the zero-frequency predictor. Report phase/amplitude/boundary sensitivities separately.
- **G2 classical competitiveness passes** only if the primary FFT-peak gate passes and at least one student also beats its same-input physical-feature reference at two adjacent primary SNR points with wholly negative paired 95% intervals. Report interpolation and global ML gaps regardless of the decision.
- **G3 distillation value passes** only if a preregistered E3 variant beats its matched E1 model at two adjacent primary SNR points with wholly negative paired 95% intervals, and the seed-level direction is consistent in at least two of three seeds. Otherwise report no demonstrated gain.
- Include comparison to the explicit S1/S2 physical frontend baselines; do not attribute frontend gains to the MLP. G1, G2, and G3 do not substitute for one another. P1-A is permitted only if implementation is sound and G1 passes; failure of G2 does not claim classical competitiveness, while failure of G1 means STOP/REDIRECT.
- Evaluate common phase rotation and full-observation amplitude scaling as exact feature/model sensitivity diagnostics. Evaluate frequency shifts, conjugation, and Nyquist wrapping on fixed development/noiseless cases; report measured student equivariance rather than assuming it.
- Test no-noise continuous frequencies, including off-grid and near-boundary cases. Keep same-distribution SNR results separate from these designed correctness checks.

## Stopping and reporting

Run this one frozen round. Do not tune after opening the final set. Report failed runs, seed dispersion, paired intervals, teacher errors, uncertainty and limitations. No P0 result is evidence of real GMTI detection, clutter robustness, angle/velocity estimation, or deployment speed. The P0 phase review will choose GO, CONDITIONAL GO, or STOP based on the independent gates. P1-A may proceed only as the simplest relative-phase feasibility study, without claiming angle or velocity.
