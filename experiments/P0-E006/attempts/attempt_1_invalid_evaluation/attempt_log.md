# Attempt 1: invalid evaluation output

Date: 2026-10-23  
Training device: NVIDIA GeForce RTX 3050 Laptop GPU (PyTorch CUDA)  
Final test seed opened: `909090` (consumed; do not reuse)

All 33 S1/S2 model instances trained to the frozen protocol. The checkpoints, training history, train/validation datasets, and final test inputs are preserved in this directory. Do not use the archived metrics, analysis, prediction archive, or result table as evidence.

## Evaluation defect

Classical estimators returned one NumPy vector of N predictions. The evaluator iterated that vector as if each scalar were a distinct training seed, broadcasting each scalar against the full truth vector. This produced an invalid per-seed error matrix and invalid baseline metrics and paired comparisons. The prediction archive also iterated baseline vectors as scalars. The archived CSV had 3,278,901 lines and 474,335,833 uncompressed bytes.

The CSV was losslessly compressed to `result.csv.gz`; its decompressed SHA-256 is `40faa36f8e9f629d1798d0236b359cfe33bc6f778bc0f7d73c1e4c7e281ee2d7`. The compressed file passed `gzip -t` and a decompressed SHA-256 comparison. Other artifacts are preserved byte-for-byte; the archived manifest records their original experiment-relative paths.

## Disposition

- The evaluation and serialization code now normalizes a one-dimensional prediction vector as one estimator replicate, validates every prediction shape, and serializes one full vector per replicate.
- Regression coverage verifies baseline vector shape, row count, sample count, and RMSE. The full suite passes with 35 tests.
- Because seed `909090` was opened, the corrected full run uses the new independent final seed `919191`. The frozen models, training data, and training rules are unchanged. No model selection was performed from the invalid results.
