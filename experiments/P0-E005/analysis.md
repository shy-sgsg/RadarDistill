# P0 experiment analysis

**Experiment:** `P0-E005`  
**Scope:** single-target complex slow-time Doppler estimation under the stated AWGN model. All data are simulated.

## Conditions

- PRF: 1300.0 Hz; samples per dwell: 130.
- Training SNR levels: -15, -5, 0, 10, 20 dB.
- Test SNR levels: -15, -10, -5, 0, 5, 10, 15, 20 dB.
- Independent training replicates: 3; held-out samples per SNR: 1024.
- Student input: 单位相位增量 `u[n] = x[n+1]·conj(x[n]) / |x[n+1]·conj(x[n])|`；输入长度为 129，零幅度增量设为 0。
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
| E1_supervised | 15.4199 |
| E2_teacher_KD | 15.7686 |
| E3_supervised_plus_KD | 14.9178 |

Best measured method at this SNR: **B3_continuous_ML** (0.0331511 Hz). This comparison is specific to the simulated signal model and test distribution.

## Did the student learn Doppler?

E1 is at least 20% better than the zero-frequency predictor at 7/8 test SNR points (-10 dB, -5 dB, 0 dB, 5 dB, 10 dB, 15 dB, 20 dB). This is evidence of basic frequency learning, but it is not sufficient for the P0 continuation gate. E1 beats FFT peak at: none of the tested SNR points.

Read `result.csv` for all SNRs and grouped diagnostics by frequency, initial phase, amplitude, and FFT-bin offset. The grouped estimates use the same independent test set.

## Distillation comparison

The paired bootstrap compares E3 minus E1 RMSE on shared held-out samples and resamples training replicates when multiple are available. A negative delta favors joint training; an interval spanning zero is inconclusive. Single-seed diagnostics do not estimate between-training-seed variability.

| SNR (dB) | E3 − E1 RMSE (Hz) | 95% paired bootstrap interval (Hz) |
|---:|---:|---:|
| -15 | 1.07312 | [-0.834849, 4.14928] |
| -10 | 0.186534 | [-3.10435, 2.7896] |
| -5 | -0.515226 | [-2.84984, 1.02869] |
| 0 | 1.44413 | [0.584389, 2.69268] |
| 5 | 0.602822 | [-0.00855132, 1.1553] |
| 10 | -0.159225 | [-0.53791, 0.194757] |
| 15 | -0.218418 | [-0.954392, 0.390149] |
| 20 | -0.573755 | [-1.90272, 0.372213] |

## Preregistered E1 vs. FFT peak comparison

Each delta is E1 minus B0 FFT-peak RMSE; negative values favor E1. The paired bootstrap resamples the three training seeds and shared held-out samples.

| SNR (dB) | E1 − FFT peak RMSE (Hz) | 95% paired bootstrap interval (Hz) |
|---:|---:|---:|
| -15 | 10.1546 | [-3.919, 24.7757] |
| -10 | 104.675 | [88.1153, 121.545] |
| -5 | 74.8388 | [69.6213, 80.1981] |
| 0 | 29.3069 | [22.9719, 35.7952] |
| 5 | 11.6419 | [8.14065, 13.8882] |
| 10 | 9.14765 | [4.50526, 11.6965] |
| 15 | 11.4758 | [9.11982, 13.2999] |
| 20 | 12.586 | [10.2455, 14.9066] |

Primary gate: **not passed**. It requires wholly negative intervals at 2 consecutive primary SNR points (0, 5, 10, 15, 20 dB). Passing windows: 无.

## Bound and interpretation

`crlb_sd_hz` in `result.csv` is the standard-deviation Cramér–Rao bound for an unbiased estimator with unknown complex amplitude/phase and known per-sample SNR. It is a theoretical reference, not a guaranteed finite-sample RMSE. The continuous ML estimator is the profile-likelihood reference for this correctly specified single-tone model.

## P0 stage assessment

- **Implementation and pipeline:** the experiment completed with independent train/validation/test seeds and saved inputs, labels, weights, and predictions; repository test results are recorded separately.
- **Basic signal learning:** E1 is at least 20% better than the zero-frequency predictor at 7/8 test SNR points (-10 dB, -5 dB, 0 dB, 5 dB, 10 dB, 15 dB, 20 dB). This is evidence of basic frequency learning, but it is not sufficient for the P0 continuation gate. E1 beats FFT peak at: none of the tested SNR points.
- **P0-E005 preregistered gate:** **not passed**. Passing windows: 无.
- **P1:** this diagnostic alone does not start P1. Record the result, then follow the P0 error-source analysis and route decision.

Do not interpret these single-target AWGN results as GMTI detection, clutter rejection, real-data generalization, or deployment evidence. Keep the held-out test data fixed.
