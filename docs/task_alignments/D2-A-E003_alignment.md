# D2-A-E003 task alignment

**Date:** 2026-10-26  
**Status:** User-authorized, preregistration and staged execution  
**Research charter:** `docs/RESEARCH_CHARTER.md` v1.2, read at HEAD `b2174df409bc5d13c15ce93209c41b28d26bdbe8`; charter edits are out of scope.  
**Prior state:** D2-A-R4 review at `b2174df409bc5d13c15ce93209c41b28d26bdbe8`.

## Research alignment

The project goal is to train a task-level GMTI student from physical truth and useful teacher knowledge, then assess detection, parameter estimation, and only later full processing cost. E003 addresses Q1 and Q4 directly, and Q3/Q8 indirectly: it tests whether fixed, public-phase-invariant slow-time and cross-channel coherence features preserve identifiable target information and improve direct truth-supervised task learning. It does not claim a new standalone CFAR or generic feature-learning project.

The question is whether a bounded physical representation helps the existing `TaskStudent` use target information in the frozen three-channel stationary-clutter model while protecting detection, conditional false alarm, range, continuous Doppler, and joint success. The representation is one fixed candidate; no architecture search, teacher loss, curriculum change, final blind, or charter revision is authorized here.

## Evidence and reason for priority

- D2-A-E002 established one-seed task learning on identifiable synthetic A/B/C scenes (A development AUC 0.9846, Pd 0.9158), but conditional false alarm, public-phase consistency, and frequency-shift equivariance failed; curriculum and K2 showed no pilot benefit.
- D2-A-R4's public-phase S1 pilot reduced Doppler RMSE by 6.29 Hz, while Pd and joint success both declined; the physical diagnostic's original first-256 sampling selected only C0. Its balanced retrospective subset still showed only 0.909 phase agreement for S1.
- D2-A-R3 and D2-A-T001 do not establish a generally reliable teacher. Their negative findings remain unchanged.
- D1-E002's lag-one representation improved one high-SNR metric but failed detection protection and physical-consistency gates. E003 therefore fixes a multi-lag candidate before evaluation and keeps task protection primary.

The new user-provided E003 task explicitly authorizes a bounded test of this route after the R4 review. It does not erase R4's REFINE decision or authorize formal multi-seed training, distillation, or final blind data.

## Falsifiable hypotheses

- **H-ARCH-D2-002:** fixed multi-delay and cross-channel complex coherence features can improve task learning over raw-IQ `TaskStudent` without weakening public-phase invariance. Falsified for this candidate if its mathematical feature tests pass but the paired pilot shows no pre-registered C2/C3 joint-success gain or fails task-protection gates.
- **H-FREQ-D2-001:** lags `{1,4,16}` retain useful Doppler information beyond lag-one alone. Falsified for this candidate if controlled frequency changes are aliased or indistinguishable in the full feature vector, or if S2 fails the frequency sensitivity gate.
- **H-DET-D2-002:** coherence features preserve detection and parameter estimation in difficult identifiable clutter. Falsified for this candidate if C2/C3 false alarm or task-protection gates fail despite valid implementation.
- **H-IDENT-D2-001:** some C2/C3 degradation is due to overlap/SCNR information limits rather than only student architecture. Supported if controlled R2 results show target increments collapse with overlap or target-to-clutter ratio before model learning; inconclusive if oracle and observable-feature diagnostics disagree.

The hypotheses apply only to the frozen synthetic stationary-clutter model. They do not establish performance on measured data, hardware mismatch, multiple targets, or operational GMTI.

## Decision impact and stop conditions

R0/R1/R2 are mandatory before training. If stable ID sampling, feature correctness, finite-precision behavior, or controlled target-information checks fail, stop before CUDA training and repair only a demonstrable implementation defect or submit a revised preregistration. If those gates pass, run one matched E1 truth-only CUDA pilot for S0 and S2 using seed 101, 1,600 updates and effective batch 64. No E2/E3 distillation is part of this task.

The pilot can support a later formal multi-seed proposal only if S2 passes task learning, task protection, and the pre-registered C2/C3 joint-success primary gain. Public-phase invariance alone is insufficient. A failed pilot redirects the next question to the observed bottleneck; it does not stop the RadarDistill research program.

## Alternatives considered

The bounded alternatives were to continue tuning the R4 public-phase augmentation, resume teacher search/KD, or make the fixed coherence representation comparison. E002/R4 provide direct evidence against the first two as the next immediate step. The user-authorized single-candidate representation comparison best isolates whether the student's observable representation is a bottleneck while preserving the charter's task-level objective.

