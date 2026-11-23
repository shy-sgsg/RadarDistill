# R6-I Observable Background Score Normalization Design

**Status:** Authorized within the user's R6 low-false-alarm research scope; design decisions below are the implementation choice for this bounded experiment.

## Intent

R6-H found that CNN-SSM improved C2+C3 joint success over CE-TCN but failed the Range RMSE protection and all tested students failed the C3 six-cell familywise false-alarm gate. R6-I tests whether a scene-level background scale that can be computed from observed IQ can explain and reduce the CNN-SSM score drift without retraining the model or using scenario labels at inference.

## Approaches

1. **Condition-specific threshold tables.** These could lower false alarms in known cells, but they require condition labels that are not available to the detector and would not test a transferable observable correction. Rejected.
2. **Observable robust-scale correction on a frozen CNN-SSM score.** Estimate one scene-level power statistic from IQ, fit one right-tail score-versus-scale slope on H0 calibration-fit scenes, and subtract that scale contribution from the CNN-SSM presence logit. This directly tests the R6-H score-scale hypothesis without changing model weights. Selected.
3. **Retrain with scale augmentation or hard-negative sampling.** This could improve robustness but would mix representation and training effects with score-scale correction and would require another matched training study. Defer until R6-I identifies a remaining train-time problem.

## Selected design

- Keep the R6-H seed808 CE-TCN and CNN-SSM best checkpoints fixed; compare raw CE-TCN, raw CNN-SSM, corrected CNN-SSM, and B3 on the same new simulated scenes.
- For each CPI, compute observable robust background power as the median of `|IQ|^2` over all channel, pulse, and range samples. A point target occupies only a small fraction of these samples, so the median resists target contamination while retaining exact common-amplitude scaling. Convert it to natural log power. No SNR, CNR, target truth, or condition ID enters the correction.
- Fit the one correction slope only on an H0 fit split: sort scenes into 20 equal-count log-power bins, compute each bin's empirical 95th percentile of raw CNN-SSM presence logits using the `higher` rule, and fit a single least-squares slope to those 20 tail summaries. The corrected score is `logit_corrected = logit_raw - beta * (log_power - median_fit_log_power)`. Freeze beta before threshold calibration.
- Use a separate H0 threshold-calibration split to set global 0.05, 0.01, and 0.001 thresholds for every detector. Use independent H0 verification and H1 development scenes for evaluation. The 0.05 workpoint is primary; 0.01 and 0.001 are descriptive secondary workpoints.
- Primary correction admission requires C3 six-cell Bonferroni one-sided exact Pfa upper bounds at or below 0.10, global Pfa upper bound at or below 0.055, at least a one-point absolute reduction in pooled C3 Pfa versus raw CNN-SSM with paired 95% interval below zero, and paired task noninferiority: Pd and C2+C3 joint-success lower bounds no worse than −0.02, Range and Doppler RMSE ratio upper bounds no greater than 1.05.
- Add a controlled `X` versus `aX` diagnostic at `a=0.5` and `a=2` on the registered C3 partial-overlap development scenes. This is a scale intervention, not a claim about hardware AGC behavior.

## Scope and constraints

- This is a bounded inference/score study, not a new architecture search, training run, knowledge-distillation experiment, final blind, or system-latency evaluation.
- The fixed simulation remains three-channel, PRF 1300 Hz, 130 pulses, 32 range cells, stationary clutter, and the R6-A 12-condition set.
- Planned fresh scenes: 60,000 H0 fit, 60,000 H0 threshold calibration, 120,000 H0 verification, and 6,144 H1 development, equally allocated across the 12 conditions. All IDs must pass a repository-wide disjointness audit before generation.
- Student inference must use AISAR `cuda:0` on the RTX 3050 Laptop GPU with FP32 and one model resident at a time. IQ is generated in CPU chunks and is not cached; only compact predictions, manifests, metrics, and audit records are retained.
- No final blind is authorized. If the correction fails its frozen gates, stop this correction and report the result; do not search additional transforms on the same development set.
