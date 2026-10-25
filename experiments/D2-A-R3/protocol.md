# D2-A-R3 预注册协议：阈值迁移与 H0 整体尺度 nuisance

**冻结日期：** 2026-10-25
**状态：** 预注册；任何数据生成前提交并推送。
**数据性质：** 合成数据；属于开发诊断，不是 final blind。
**基线总纲：** `docs/RESEARCH_CHARTER.md` v1.1。
**冻结历史：** 不修改 D2-A-E001 的协议、配置、代码行为、数据或结论。

## 1. 目标、假设与范围

本实验服务于物理仿真真值与传统教师支持的任务级 GMTI 学生主线。唯一当前阻塞问题是 D2-A-E001 固定 H0 阈值的 Pfa 迁移。实验先在未经修改的 B0–B3 分数上估计抽样与校准不确定性，再比较两个预先定义的 B3 归一化统计量，不搜索额外窗、变换或超参数。

主要问题：

1. E001 B3 校准经验 `Pfa_CPI=0.05078125`、独立 A/10/10 开发 `Pfa_CPI=0.08984375` 的差异，能否由 n=512 校准和 n=256 验证的有限样本波动解释？
2. H0 的场景整体幅度是否随潜在目标参考幅度 nuisance 变化，并使原始绝对功率分数的固定阈值不稳？
3. 两个只使用 IQ 的归一化候选能否稳定 H0 虚警，同时保护可辨识 H1 的 Pd 与联合参数任务？
4. 合格传统检测统计量是否仍有足够的目标检测/参数估计能力，支持一个后续 E1 truth-only 学生 pilot？

检验 H-CLUTTER-002（原始 B3 主工作点阈值迁移未支持，原因未决）和 H-CLUTTER-003（可观测整体尺度归一化可改善 H0 稳定性且不造成超过冻结限值的 H1 退化）。不改变长期研究总纲，不把 CFAR 发展为独立科研项目。

## 2. 固定信号与代码审计边界

保持 D2-A-E001 原有 3 通道、`PRF=1300 Hz`、`N=130`、32 距离单元、静止地杂波、平地/匀速平台、波形、几何、range PSF、散射和 AWGN 假设。H0=`clutter+noise`；H1 与同 base scene 的 H0 共享背景和噪声，且 `H1−H0=target_signal_iq`。

运行前代码审计记录：

- 生成器把潜在目标参考幅度 `A` 抽自 `[-6,6] dB`，按 `noise_power=A²·10^(-SNR/10)` 生成噪声；再令目标 CNR 下的期望杂波功率为 `noise_power·10^(CNR/10)`，按场景完整 CPI 杂波功率归一化。因此固定 SNR/CNR 时，A 同时缩放 H0 杂波与噪声。这是当前名义 SNR/场景尺度定义的结果，需要测量，不能预先称为实现错误。
- A、背景散射、噪声使用 `SeedSequence.spawn(3)` 的独立随机流；H0/H1 共用散射/噪声。场景布局为 `[B,C,N,R]`，基线按通道/脉冲/距离轴处理；R3 用新 seed 流并按 batch=16 流式运行。
- 场景 A/B/C/D 只改变目标构造；背景和噪声代码路径不读取场景标签。R3 用相同 seed 的 A/B/C/D H0 回归检查该点，但不能把相同 seed 的场景当作独立重复。
- 冻结基线阈值比较符是 `score >= threshold`；校准采用经验 95th percentile、`numpy.quantile(method="higher")`。R3 同时报告严格 `>`、`>=` 及恰等阈值计数，主报告遵循冻结的 `>=`。
- 现有代码不改。若 R3 测试发现生成器或冻结基线错误，先保存复现并停止本轮解释；只有有证据且影响任务的必要修复才可另行实施，不能覆盖 E001。

## 3. 第一阶段：冻结原始分数 T0

T0 完全沿用 `src/radardistill/d2a_baselines.py` 的 B0、B1、B2、B3。第一阶段仅运行原始统计量，不计算或使用 T1/T2，不用新开发结果调阈值。

E001 已有 `10 dB SNR / CNR=0,10,20 dB` 的 512-scene 校准阈值必须原样重放。主 `10/10 dB` 的 B0–B3 阈值分别为 `180598.23002948496`、`536397.2435161747`、`178629.93962331084`、`961.397913450906`；各自由 E001 metrics.json 的冻结记录复核。对 R3 新样本先报告这些历史固定阈值的 Pfa，不允许重拟合后替代。

### R3 H0 样本组与场景数

每个列出的组均为 4096 个独立 base scenes。生成器按 batch=16 调用；每组单独报告，不把不同校准重复当作一个校准样本池。

| 用途 | SNR/CNR | 场景/组 | seed start | n |
|---|---|---|---:|---:|
| Primary calibration | 10/10 dB | cal-1 | 10,000,000 | 4096 |
| Primary calibration | 10/10 dB | cal-2 | 10,010,000 | 4096 |
| Primary calibration | 10/10 dB | cal-3 | 10,020,000 | 4096 |
| Primary validation | 10/10 dB | A separated | 11,000,000 | 4096 |
| Primary validation | 10/10 dB | B near edge | 11,010,000 | 4096 |
| Primary validation | 10/10 dB | C partial overlap | 11,020,000 | 4096 |
| Primary validation | 10/10 dB | D exact overlap | 11,030,000 | 4096 |
| Secondary calibration | 10/0 dB | cal | 12,000,000 | 4096 |
| Secondary validation/H1 | 10/0 dB | A separated | 13,000,000 | 4096 |
| Secondary validation/H1 | 10/0 dB | C partial overlap | 13,010,000 | 4096 |
| Secondary calibration | 10/20 dB | cal | 14,000,000 | 4096 |
| Secondary validation/H1 | 10/20 dB | A separated | 15,000,000 | 4096 |
| Secondary validation/H1 | 10/20 dB | C partial overlap | 15,010,000 | 4096 |
| Controlled `X,aX` check | 10/10 dB | scale diagnostic | 16,000,000 | 128 |

R3 seed groups 不与 E001 校准 `[3,000,000,3,099,999]`、E001 开发集 `[4,000,000,4,099,999]`、R2 规划的学生数据划分或预留 blind `[9,000,000,9,999,999]` 重叠。primary 阈值变化通过 3 个互不重叠的校准组和 4 个独立验证组测量。每个 primary 校准阈值都分别报告在每个验证组上的结果及合并独立场景估计；3×4 的阈值—验证矩阵各项并非独立重复。

每个 secondary CNR 条件包含一个 4096-scene 独立校准组和两个各 4096-scene 的独立配对验证组。它们是有界灵敏度检查，不作为重复校准证据。将匹配 CNR 的新阈值和冻结的 primary CNR 阈值都应用于验证组，以量化跨 CNR 迁移。

## 4. H0 diagnostics and finite-sample analysis

For every H0 scene, save only compact diagnostics: split/group/seed/base ID, latent potential-target amplitude (linear and dB), requested SNR/CNR, nominal noise power, measured noise/clutter/H0 total powers, and each frozen T0 raw score. Component powers and latent amplitude are analysis-only and never detector inputs. Do not save/cache full IQ arrays.

Report score quantiles (0.50/0.90/0.95/0.99/max), thresholds, calibration `Pfa(>)`, `Pfa(>=)`, tie count/fraction, validation `Pfa_CPI` and 95% Wilson interval for each group, plus the 3×4 primary threshold/validation matrix. Record per-scene noise, clutter, and total power by fixed amplitude bins `[-6,-3),[-3,0),[0,3),[3,6] dB` and quantify score versus scale; include scene-level data so the analysis can be reproduced.

To assess whether E001's `23/256=0.08984375` could arise from finite samples:

1. Apply E001's frozen B0–B3 threshold to the new 10/10 H0 validation scenes; estimate its marginal Pfa from all four independent primary validation groups and show group-specific intervals.
2. At the observed R3 marginal rate under the legacy threshold, calculate the exact binomial tail `Pr(K≥23 | n=256)` for validation-sample fluctuation alone.
3. Run a preregistered 10,000-replicate hierarchical nonparametric bootstrap: sample one of the three primary calibration groups and one of the four validation groups, resample 512 calibration scores and 256 independent validation scores with replacement, recompute the `higher` 95th-percentile threshold, and record validation Pfa. Report threshold and Pfa distributions and `Pr(Pfa≥23/256)`. This estimates combined finite calibration/validation sampling under the observed common distribution; group labels are retained in sampling and reporting.

For controlled scale diagnostics, use the same H0 IQ `X` and factors `a=0.5`, `a=2.0`, and `a=0.6+1.2j`; report raw score ratios and normalized-score differences. The expected T0 law is `score(aX)=|a|² score(X)`.

## 5. 第二阶段：两个固定统计量候选

只有在 T0 raw phase 已运行并形成其分析后，才运行下列候选；seed 和 H0/H1 scenes 不改变，不根据第一阶段结果搜索参数。

- **T1：** 对 B3 的原始 RD 功率图 `P`，找到全图最大原始功率 cell；对该 cell 的固定 guard 区（Doppler ±2、range ±2；Doppler 轴循环、range 轴裁剪）之外的所有 cells 取中位数 `b1`。每个 cell 的分数图为 `P/b1`，CPI 分数为该分数图的最大值。H1 的 background estimate contamination 用配对 H0/H1 的 `b1` 相对变化报告；truth 不进入检测器。
- **T2：** 对每个 RD cell 作为 CUT，取 9-Doppler×7-range 对称参考窗，排除 3×3 guard cells；Doppler 方向循环，range 方向只取有效 cells；参考数少于 10 时该 CUT 不可用。分数为 `CUT power / local reference median`；CPI 分数取所有可用 CUT 的最大值。它是经验校准的局部 CFAR-like 统计量，不假定 RD 参考 cells 独立同分布，也不宣称理论恒虚警。

两个候选都只读 IQ/由 IQ 计算的 RD map；检测阈值在各自独立 H0 calibrator 上取 `higher` 95th percentile。报告每个配对 H1 场景的 raw-power map 与检测-statistic map 分离，防止阈值量纲混淆。T1/T2 不替换 B3 的物理前端，不构成纯学习结果。

H1 使用同一 4096 scene 组下的 H0/H1 配对观测；条件为 CNR10 的 A/B/C/D 以及 CNR0/20 的 A/C。报告每组的 `Pd`、Pfa、ROC/AUC、range RMSE、周期 Doppler RMSE、joint success（检测命中且 range error≤1 cell、Doppler error≤20 Hz），各自 95% 区间。Pfa/Pd 使用 Wilson 区间；配对候选−B3 差异、AUC、RMSE 与 joint success 使用 2000 次 scene-cluster bootstrap。D 的来源归因不可辨识，只报告描述性结果，不作为 H1 性能门。

为验证参考背景未被目标峰严重污染，T1 对 H0/H1 配对的 scene-wide robust background estimate 比值；T2 对物理真值对应附近 CUT 的局部背景估计比值只作诊断。后者的真值不得传入 T2 本身。

## 6. 冻结阶段门

- **G1 工程与数据正确性：** R3 行为/物理测试通过；种子组不重叠；新生成 H0/H1 继续满足配对恒等式；A/B/C/D 不造成非目标场景 H0 分布差异；B0–B3 的 R3 T0 与现有实现逐样本一致；数据和标签从未输入 detector。失败则停止统计结论，修复需新增证据和记录，不能改 E001。
- **G2 匹配条件固定阈值：** 对每个方法和每个 primary 4096-scene 校准组，经验校准 `Pfa_CPI(>=)` 应在 `[0.04,0.06]`。每个校准阈值应用到四个独立 primary validation 组后，合并验证点估计需在 `[0.04,0.06]` 且 95% Wilson CI 覆盖 0.05；每个单独 validation 组的 95% Wilson CI 必须与 `[0.04,0.06]` 有交集。secondary CNR matched-threshold 合并验证点估计同样要求在 `[0.04,0.06]` 且 CI 覆盖 0.05。
- **G3 整体幅度 nuisance：** 在每个 primary 校准阈值下，四个预定潜在幅度箱各约 4096 个验证 H0。至少 2/3 校准阈值要求最高与最低幅度箱 Pfa 差的 95% bootstrap CI 完全落在 `[-0.02,0.02]`，且四箱 Pfa 点估计 max−min≤0.03。 controlled `X,aX` 检查中 T1/T2 normalized map 最大绝对差≤`1e-6`；T0 raw score 比率相对 `|a|²` 最大相对误差≤`1e-5`。
- **G4 H1 性能保护：** 对候选和 raw B3 在同一条件分别使用独立校准阈值；对每个可辨识 A/B/C 条件，candidate−B3 的 paired 95% CI 下界对 Pd 与 joint success 均须高于 `−0.05`。D 只作为不可辨识边界，不设来源恢复门。range/Doppler RMSE 均需报告，joint success 防止只报告命中条件误差。

### 结果处置与 E1 授权

- 若原始 T0/B3 通过 G2 和 G3，且 G4 对可辨识条件没有任务退化，记录历史偏差是否与新重复结果相容；不要求为引入 CFAR 而替换 T0。
- 若 T0 未过 G2/G3，而 T1 或 T2 通过 G2/G3/G4，则选择满足全部门的最简单统计量作为 R2 的传统教师检测前端候选，并清楚保留其 B3 物理前端身份。
- 只有存在一种前置统计量通过 G1–G4，且 R2 冻结教师的既有 A/20dB/CNR10 检测门仍符合原报告，才允许一个 E1 truth-only 单 seed pilot。严格沿用 `experiments/D2-A-E001/protocol.md` 的 D1 `TaskStudent`、三通道 complex IQ、K3 物理真值、1600 updates、有效 batch 64；通过实际 CUDA 设备检查后只用 RTX 3050 Laptop GPU。不得启用 K1/K2 loss、E2/E3、正式三种子训练或 final blind。
- 若所有方法失败 G2/G3/G4，不启动 E1/KD；记录负结果与是否提出独立 truth-only 路线建议，等待单独预注册，不绕过现行前门。

## 7. 资源、输出与复现

总量为 20,480 H0 calibration scenes、32,768 paired validation scenes（包含 H0 与指定 H1）、128 个受控缩放诊断 scenes；仿真按 batch 16 流式运行，不缓存全量 IQ。依照 E001 173.3 s 完成约 5.6k 生成 scenes 的记录，本轮含两阶段候选的预计 CPU 生成/评估时间约 1–2 小时；首次运行先记录实际吞吐和内存，若超过 2 小时预算则停止在完整的已预注册边界或按组续跑，不缩小已开始的关键组、不改门槛。R3 不进行实时性测试。

保存 `config_used.yaml`、`environment.json`、`metrics.json`、`result.csv`、`h0_scene_diagnostics.csv`、`analysis.md`、静态图、复现命令、运行时长与 CPU/GPU 环境。只有通过前门时额外保存 E1 pilot 的单 seed checkpoints、训练历史、GPU/CUDA、峰值显存和指标。不要提交缓存 IQ、临时 profiler、全量训练缓存或无关用户数据。

复现命令：

```bash
conda run --no-capture-output -n AISAR python scripts/run_d2a_r3.py --config experiments/D2-A-R3/config.yaml --phase raw
conda run --no-capture-output -n AISAR python scripts/run_d2a_r3.py --config experiments/D2-A-R3/config.yaml --phase candidates
```

阶段审查必须回答用户列出的十个问题，并对整体研究方向作 CONTINUE / REFINE / REDIRECT / STOP 决策。无论正负结果都保留；不声称真实硬件、生产系统或 `Pfa=1e-6` 已验证。
