# P0 experiment analysis

**Experiment:** `P0-E002`  
**Scope:** single-target complex slow-time Doppler estimation under the stated AWGN model. All data are simulated.

## Conditions

- PRF: 1300.0 Hz; samples per dwell: 130.
- Training SNR levels: -15, -5, 0, 10, 20 dB.
- Test SNR levels: -15, -10, -5, 0, 5, 10, 15, 20 dB.
- Independent training replicates: 1; held-out samples per SNR: 1024.
- Student: NumPy Conv1D with 6 filters of length 9, ReLU, global average pooling, and a two-value circular output.
- E1 trains on physical truth; E2 trains only on local teacher likelihood soft targets; E3 combines physical truth loss with teacher loss of weight 0.5.
- Errors wrap modulo PRF. An outlier is an absolute wrapped error above half one native FFT bin (5 Hz).

## Highest-SNR RMSE comparison

At 20 dB, mean student RMSE across training seeds and reference-method RMSE are:

| Method | RMSE (Hz) |
|---|---:|
| B0_fft_peak | 2.96366 |
| B1_fft_interpolated | 1.23289 |
| B2_phase_difference | 0.206079 |
| B3_continuous_ML | 0.0331511 |
| E1_supervised | 8.55482 |
| E2_teacher_KD | 9.16842 |
| E3_supervised_plus_KD | 8.30495 |

Best measured method at this SNR: **B3_continuous_ML** (0.0331511 Hz). This comparison is specific to the simulated signal model and test distribution.

## Did the student learn Doppler?

E1 is at least 20% better than the zero-frequency predictor at 7/8 test SNR points (-10 dB, -5 dB, 0 dB, 5 dB, 10 dB, 15 dB, 20 dB). This is evidence of basic frequency learning, but it is not sufficient for the P0 continuation gate. E1 beats FFT peak at: -15 dB.

Read `result.csv` for all SNRs and grouped diagnostics by frequency, initial phase, amplitude, and FFT-bin offset. The grouped estimates use the same independent test set.

## Distillation comparison

The paired bootstrap compares E3 minus E1 RMSE on shared held-out samples and resamples training replicates when multiple are available. A negative delta favors joint training; an interval spanning zero is inconclusive. Single-seed diagnostics do not estimate between-training-seed variability.

| SNR (dB) | E3 − E1 RMSE (Hz) | 95% paired bootstrap interval (Hz) |
|---:|---:|---:|
| -15 | 0.312679 | [-2.88351, 4.60072] |
| -10 | -1.09572 | [-5.35083, 3.66874] |
| -5 | -0.314935 | [-1.72182, 1.129] |
| 0 | -0.349994 | [-0.826636, 0.139174] |
| 5 | -0.357395 | [-0.664943, -0.0313781] |
| 10 | -0.116945 | [-0.383652, 0.193685] |
| 15 | -0.968401 | [-1.55617, -0.435591] |
| 20 | -0.249873 | [-0.691943, 0.124662] |

## Bound and interpretation

`crlb_sd_hz` in `result.csv` is the standard-deviation Cramér–Rao bound for an unbiased estimator with unknown complex amplitude/phase and known per-sample SNR. It is a theoretical reference, not a guaranteed finite-sample RMSE. The continuous ML estimator is the profile-likelihood reference for this correctly specified single-tone model.

## P0 stage assessment

- **Implementation and pipeline:** the experiment completed with independent train/validation/test seeds and saved inputs, labels, weights, and predictions; repository test results are recorded separately.
- **Basic signal learning:** supported, with limited scope. E1 improves on a constant predictor at several SNRs, but large errors remain and the frequency-subset results show variation.
- **Continue to P1:** **not passed.** E1 does not beat FFT peak at the tested SNRs in the primary run. At the highest SNR, the best classical baseline is B3_continuous_ML at 0.0331511 Hz while E1 is 8.55482 Hz. The strong classical-reference gap remains the main result.

Do not interpret these single-target AWGN results as GMTI detection, clutter rejection, real-data generalization, or deployment evidence. Keep the held-out test data fixed. The next action is one targeted P0 experiment testing a constrained position-aware readout or an explicit phase-increment feature, motivated by weak mean-pooled performance and overfitting in the larger flatten heads. Predefine its architecture and criterion before training; do not start P1 until a student beats FFT peak over at least part of the normal-SNR range with stable held-out performance.
