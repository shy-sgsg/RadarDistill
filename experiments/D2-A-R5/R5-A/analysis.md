# D2-A-R5-A 分析：训练充分性与固定相干候选

**协议：** `protocol.md` v1.1
**训练源提交：** `d996b66418221e027c29927789ff84f3c1df3c89`
**评估源提交：** `d1d46617deac234f9456121b2adf0752b44c6f5d`
**总纲：** `docs/RESEARCH_CHARTER.md` v1.3
**范围：** R5-A seed 101、E1 truth-only 训练充分性 pilot；不含容量对照、KD 或 final blind。

## 研究问题与设计

本阶段检验 D2-A-E003 的 1,600 updates（一遍固定训练集）是否足以评价 S0 和固定相干前端 S2。R5-A 从头训练相同两臂，保持 E003 冻结的 51,200 对 train、5,000 对 validation scene IDs、任务损失、配对样本顺序、初始化规则和优化器相同。S0 为原始三通道复 IQ `TaskStudent(width=32)`；S2 为 28→6 固定相干前端及同宽主干。两臂分别有 13,383 和 13,557 个参数。

每次更新抽取 32 对 H0/H1 scene，1,600 updates 对应一遍训练集。6,400 updates 为预注册首阶段上限；两臂在 6,400 时都通过 v1.1 的延长条件，故不中断 optimizer、scheduler、RNG 和 pair plan，连续扩至 12,800 updates（8 遍固定数据）。验证每 400 updates 一次，按共同 truth-task validation loss 选 checkpoint：S0 选 update 12,400，S2 选 update 12,000。

训练在 AISAR、Python 3.13.7、PyTorch 2.8.0+cu128、CUDA 12.8、RTX 3050 Laptop `cuda:0` 完成。FP32、microbatch 16、梯度累积 4、有效 batch 64；每次仅一个模型驻留，无 CPU 回退、无 OOM。G1 小样本过拟合检查两臂均通过。峰值 allocated/reserved 显存：S0 274,465,280/337,641,472 bytes；S2 285,836,800/371,195,904 bytes。

### 数据隔离边界

训练/验证沿用 E003 冻结 ID。R5 缓存清单覆盖 81,288 个跨 split 不重叠的 base-scene IDs（train 51,200；validation 5,000；H0 calibration 8,192；H0 verification 8,192；C3 H0 4,096；development 4,096；exact-overlap 描述边界 512）。其中校准、普通 verification、C3 H0、development 及物理诊断 ID 都是已使用的 E003 开发材料，不能视作 R5 独立 blind。R5-A 未生成 final blind，未使用教师硬标签或软响应。

## 训练曲线与预算证据

在 R5 同一长程 cosine schedule 内，1,600 至 12,800 updates 的验证表现持续改善：

| Arm | Update | Task loss | AUC | Range RMSE (cell) | Doppler RMSE (Hz) | Range-cell mismatch |
|---|---:|---:|---:|---:|---:|---:|
| S0 | 1,600 | 0.5190 | 0.9885 | 2.844 | 35.49 | 0.0830 |
| S0 | 6,400 | 0.3344 | 0.9938 | 2.243 | 24.85 | 0.0582 |
| S0 | 12,800 | 0.3208 | 0.9942 | 2.203 | 24.16 | 0.0562 |
| S2 | 1,600 | 1.0235 | 0.9811 | 3.750 | 72.51 | 0.2480 |
| S2 | 6,400 | 0.7412 | 0.9892 | 3.196 | 55.13 | 0.1896 |
| S2 | 12,800 | 0.6935 | 0.9904 | 2.907 | 52.13 | 0.1792 |

The within-run comparison uses the same long schedule at each checkpoint. The old E003 1,600-update run used its own short cosine schedule, so its absolute metrics are historical context only, not a matched schedule contrast. On the reused E003 development data, R5 selected S0 improves joint success from 0.6865 to 0.8389; S2 from 0.4763 to 0.6055. Because the train/validation scenes and development materials were already used in E003 and only one training seed was run, these are fixed-seed development results, not independent generalization evidence.

## Development task results

All values below use each arm's own student-H0 threshold calibrated to 5% using the frozen higher empirical quantile and `score >= threshold`. Intervals for paired task comparisons use 2,000 base-scene paired bootstrap replicates. They quantify scene uncertainty for this seed; they do not estimate training-seed variability.

| Metric | S0 raw-IQ TaskStudent | S2 fixed coherence front end |
|---|---:|---:|
| Development Pd | 0.9619 [0.9563, 0.9680] | 0.9336 [0.9258, 0.9414] |
| AUC | 0.99374 [0.99262, 0.99481] | 0.98999 [0.98852, 0.99148] |
| All-H1 range RMSE | 2.239 cells [1.994, 2.476] | 2.824 cells [2.573, 3.073] |
| All-H1 circular Doppler RMSE | 25.23 Hz [23.47, 26.81] | 54.25 Hz [51.20, 57.44] |
| Joint success | 0.8389 [0.8276, 0.8503] | 0.6055 [0.5901, 0.6204] |

`S2−S0` paired differences are Pd −0.02832 [−0.03564, −0.02075] and joint success −0.23340 [−0.24878, −0.21875]. In the preregistered C2/C3 difficulty groups, joint success is 0.6919 for S0 (C2 0.8516; C3 0.5322) and 0.2974 for S2 (C2 0.4473; C3 0.1475); paired difference −0.39453 [−0.41895, −0.36962]. S2/S0 C2/C3 RMSE ratios are 1.261 [1.116, 1.444] for range and 2.160 [2.004, 2.335] for Doppler.

### H0 calibration and verification

| Split | S0 Pfa | S2 Pfa | Protocol interpretation |
|---|---:|---:|---|
| Ordinary H0 verification, n=8,192 per arm | 0.05017; exact upper 0.05432 | 0.04834; exact upper 0.05242 | Overall fixed-threshold rate is near 0.05 on this reused split |
| C3 H0 verification, n=4,096 per arm | 0.20410; exact upper 0.21474 | 0.18384; exact upper 0.19408 | Both exceed the frozen 0.10 C3 limit |
| Overall development H0, n=4,096 | 0.05322; exact upper 0.05935 | 0.05469; exact upper 0.06089 | Development diagnostic only |

The pooled H0 result does not establish difficult-condition threshold stability. Both candidates fail the C3 H0 point limit. Calibration and verification were not used to retune thresholds.

## Physical consistency

- Across 32 common public phase rotations, S2 predictions are numerically stable: decision agreement is 1.0; score p95 changes are about `1.1–1.4e-4`, range p95 changes about `8e-6` cell. S0 is phase-sensitive; decision agreement reaches about 0.939 at some rotations, and Doppler p95 change reaches about 49 Hz.
- The S2 phase invariance did not translate into task benefit. In the separated-scene whole-observation −25 Hz intervention, S2's estimated mean shift was +118.94 Hz, p95 absolute shift error 511.89 Hz, and joint success fell from 0.516 to 0.141. Its target-only counterfactual shift mean was −12.27 Hz. S0's separated-scene whole-observation estimate moved −21.75 Hz for a requested −25 Hz, but joint success still fell from 0.969 to 0.859.
- These interventions reuse E003 diagnostic IDs and are mechanism diagnostics, not independent tests. A phase-invariant front end alone is insufficient evidence that a network learns frequency translation or task-relevant coherence.

## Gate and hypothesis interpretation

The basic-learning screen passes for both arms: development AUC lower bounds exceed 0.5, range and Doppler RMSE beat the train-truth constant baseline, and small-set overfit checks pass. The extension decision is consistent with the frozen continuation rule. Validation metrics improve through 12,800 updates, so this run does not demonstrate convergence or a width-32 capacity ceiling.

The current S2 candidate fails task protection and the C2/C3 primary gain criterion; it should not be retained as the current fixed-coherence architecture candidate. This is limited to the tested E003 front end, projection, initialization, and training procedure. S0 is the stronger of these two task baselines, but it fails the C3 fixed-threshold Pfa limit and is not thereby established as a deployment-ready detector or teacher.

`metrics.json` is written using R5-A gate names: `G1_basic_learning`, `S2_task_protection_screen`, `S2_primary_C2_C3_joint_gain_screen`, and `G3_architecture_value`. The inherited E003 evaluator used the label `G3_task_learning` for a different basic-learning check; this naming mismatch was corrected in the R5 adapter without changing metric calculations or frozen thresholds.

- **H-CAP-001:** supported within this one seed, fixed training cohort, schedule, and reused development distribution. The in-run extension improves validation task and parameter metrics; no cross-seed or final generalization claim follows.
- **H-CAP-002:** untested; the width-only Small/Medium comparison remains the next pre-registered capacity study.
- **H-ARCH-003:** untested. The negative S2 result concerns the fixed coherence front-end candidate, not the raw-IQ non-CNN hypothesis.
- **H-LEARN-002:** inconclusive. S2 has common-phase stability but poor frequency intervention and task results; S0 learns the task but is phase-sensitive. Neither observation isolates the causal value of learned physical coherence.

## Resource accounting and reproducibility limits

The final training records contain `training_seconds_this_invocation` from the extension invocation only: S0 878.3 s and S2 1,005.4 s. The initial 6,400-update segment duration was overwritten and is not recoverable from saved records. These numbers are not total 12,800-update wall times. The training runner now retains separate segment times and cumulative elapsed time for future runs; this correction does not alter the historical R5-A record.

Evaluation command:

```bash
PYTHONPATH=src conda run --no-capture-output -n AISAR python scripts/run_d2a_r5.py --phase evaluate
```

Training command and preflight are recorded in `training.json` and `preflight.json`; the result came from the pushed training SHA above, with the current evaluation SHA recorded in `metrics.json`. R5 cache: 14,232,309,192 bytes across 90 files; it is a local rebuildable cache and is not committed. No final blind, teacher training, or KD run occurred.

## Stage recommendation

**REFINE.** R5-A supports the claim that one epoch did not settle the measured S0/S2 task curves, rejects the tested S2 fixed-coherence candidate under the frozen task-protection/gain screen, and leaves capacity unresolved. Proceed only to the already frozen width=32/96 Small/Medium comparison after this training-sufficiency review; keep the same train/validation IDs, seed, truth loss, optimizer, sample plan, effective batch and total update schedule. Do not start Large, KD, formal multi-seed confirmation, or final blind from this result alone.

The evidence does not change RadarDistill's long-term objective. It refines the immediate student study: retain raw-IQ S0 as the comparison baseline, test capacity as the next isolated factor, and require C3 Pfa and task metrics to remain explicit protection outcomes.
