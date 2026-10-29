# D2-A-R6-B analysis

- Result generation commit: `293f1733b417d7fdc65fd9e294c8511cef43613c`
- Preflight: `G0_PASS`; technical G1: `G1_PASS`; selected branch: `R`.
- Independent scenes: calibration 120,000, verification 120,000, H1 development 6,144.
- H0 thresholds come from calibration only; all reported Pfa values use independent verification and exact one-sided Clopper–Pearson upper limits.
- Training used effective batch 64, 12,800 updates per arm, seed 303, FP32 CUDA, and the shared two-segment R5-A Medium cosine schedule. History is recorded at validation/checkpoint intervals only.
- The hard-negative pool and its top-quartile ordering use only the R6-B training H0 split. No evaluation IQ or teacher response maps were retained.

## Training record

| Arm | Selected update | Microbatch | Seconds | Peak allocated GiB | Validation task loss | Validation AUC | Doppler RMSE (Hz) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Truth-only | 12000 | 32 | 3176.7 | 1.532 | 0.228539 | 0.99568 | 17.414 |
| Hard-negative | 12000 | 32 | 3278.4 | 1.533 | 0.231157 | 0.99560 | 17.261 |

## Detection and task results

| Method | Pfa upper @ .05 | C3 Pfa upper @ .05 | Hard-cell Pfa upper @ .05 | Pd | C3 Pd | Hard-cell Pd | Joint success | C3 joint | Hard-cell joint |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B0 | 0.05087 | 0.10168 | 0.20841 | 0.05013 | 0.10026 | 0.21875 | 0.00033 | 0.00065 | 0.00195 |
| B1 | 0.05102 | 0.10199 | 0.20922 | 0.05013 | 0.10026 | 0.21875 | 0.00033 | 0.00065 | 0.00195 |
| B2 | 0.05102 | 0.10199 | 0.20922 | 0.05013 | 0.10026 | 0.21875 | 0.00033 | 0.00065 | 0.00195 |
| B3 | 0.05029 | 0.10052 | 0.20143 | 0.35449 | 0.32975 | 0.21875 | 0.32747 | 0.27572 | 0.00781 |
| R3_T1 | 0.05074 | 0.10140 | 0.09740 | 0.80273 | 0.64714 | 0.08789 | 0.78369 | 0.60905 | 0.00000 |
| R3_T2 | 0.05119 | 0.10187 | 0.10211 | 0.77979 | 0.60872 | 0.08398 | 0.75798 | 0.56510 | 0.00000 |
| Medium_Frozen | 0.05177 | 0.10276 | 0.10762 | 0.93359 | 0.86751 | 0.22070 | 0.83496 | 0.69661 | 0.03125 |
| Medium_TruthOnly | 0.05123 | 0.10192 | 0.10895 | 0.93620 | 0.87240 | 0.24023 | 0.85042 | 0.71712 | 0.03125 |
| Medium_HardNegative | 0.05108 | 0.10180 | 0.10722 | 0.93473 | 0.86947 | 0.22266 | 0.84831 | 0.71452 | 0.02734 |

## Matched intervention contrast at global target Pfa 0.05

- Global Pd difference (hard-negative minus truth-only): -0.00146, paired 95% bootstrap CI [-0.0035807291666666665, 0.0006510416666666666].
- Global joint-success difference: -0.00212, paired 95% bootstrap CI [-0.006510416666666667, 0.001953125].
- C3 joint-success difference: -0.00260, paired 95% bootstrap CI [-0.011067708333333334, 0.005533854166666667].
- Hard-cell joint-success difference: -0.00391, paired 95% bootstrap CI [-0.015625, 0.005859375].
- For a conservative no-loss statement, the paired 95% interval must have a nonnegative lower bound; intervals crossing zero are inconclusive, not proof of equivalence.

## Low-Pfa qualification

- target 0.05: Medium_TruthOnly: upper=0.0512322 (not qualified); Medium_HardNegative: upper=0.0510808 (not qualified).
- target 0.01: Medium_TruthOnly: upper=0.0106987 (not qualified); Medium_HardNegative: upper=0.0110226 (not qualified).
- target 0.001: Medium_TruthOnly: upper=0.000947832 (PASS); Medium_HardNegative: upper=0.00090264 (PASS).

## Interpretation and limits

- This is one matched training seed. It estimates this pilot's effect; it does not estimate training-seed uncertainty or authorize final blind evaluation.
- R6-B compares train-only H0 resampling against a matched truth-only retrain. It does not compare a new architecture, normalization, teacher loss, or larger effective batch.
- The frozen original Medium is a contextual baseline. The causal comparison for the R intervention is the paired TruthOnly versus HardNegative retraining.
- A low-Pfa point is qualified only when its independent verification exact upper bound meets the target. Pfa point estimates, ROC curves, and calibration results do not replace this gate.
- Review the condition-level nuisance correlations, RMSE curves, hard-cell protection and the protocol-order deviation documented in the task alignment before a stage decision.

## Training-budget interpretation and execution provenance

- Both matched arms selected update 12,000. Validation task loss at update 12,800 was slightly higher: 0.229891 versus 0.228539 for Truth-only, and 0.232303 versus 0.231157 for Hard-negative. The interval training loss continued to fall, but validation does not show continued improvement through the final checkpoint. This single-seed pilot cannot establish convergence or the benefit of extending the budget. R5-A's longer-budget trend motivates a separately preregistered matched budget study; it does not justify extending only one R6-B arm. The user's permission to explore longer schedules and up to 2x batch applies to future experiments; the frozen R6-B effective batch 64 and 12,800 updates remain unchanged.
- The first formal launch stopped before any optimizer update because the initialization guard compared against a fresh random state; after correction, both arms completed the frozen plan. An initial post-training evaluation stopped on a config-loader error before producing valid aggregate results. These attempts are retained under `attempts/`; the reported evaluation was regenerated from the same registered predictions/splits after the loader fix. Training code SHA: `091e96b9d570bdaa16444c1e696a633145c6655d`; result generation/evaluation SHA: `293f1733b417d7fdc65fd9e294c8511cef43613c`.
