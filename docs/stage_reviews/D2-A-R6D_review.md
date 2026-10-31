# D2-A-R6-D 阶段审查

**阶段结论：REFINE。** J2-J 的 C2+C3 joint-success 主筛选通过，但 Doppler RMSE 保护和 C3 六格 familywise 虚警门失败。J2-J 不进入正式多 seed 或 final blind；保留 J1-J 作下一阶段联合任务 baseline，按 R6 P4 进入有界 R6-E 架构比较。

**结果绑定：** 训练/评价代码 SHA `9a631f4fae445d9fd3793faeba94cdcbc09cc062`；protocol/config v1.2 SHA 分别为 `c5b30ff2f395617558484ad48821b9239da09278922331c326745f84dc8dddb5` / `e01cb65244338fc3957b39b0db233c022a9befa467f130bda37ecc667abe3ee9`；研究总纲 `docs/RESEARCH_CHARTER.md` v1.3。最终结果文档在后续交付提交中，所有 H1 仍为 development，未生成 final blind。

## 1. 对总体研究目标的贡献

本阶段比较任务级学生在物理真值监督下如何共同学习检测与参数估计，直接对应研究总纲关于学生学习、架构表示和最终任务收益的问题。它没有发展独立传统检测项目，也没有训练新教师或开展知识蒸馏。

R6-D说明：候选交互有可测的 range/joint 收益，但会伴随连续 Doppler 估计显著退化。它支持将任务质量拆成 detection、range、Doppler 和 joint 保护门，不支持按单一 joint 指标筛选学生。

## 2. 工程与实验科学审查

- G0/G1 通过。四臂均在 NVIDIA RTX 3050 Laptop GPU `cuda:0`、PyTorch 2.8.0+cu128、CUDA 12.8、FP32 上顺序训练，effective batch 64、microbatch 16；训练期间未回退 CPU、未发生 OOM。
- J1-D/J1-P/J1-J 各 70,277 参数，J2-J 143,294 参数。峰值 allocated 显存依次为约 0.821/0.821/0.821/1.336 GB；训练用时约 6,758/6,390/6,216/9,492 秒。
- 四臂同 seed 505、25,600 updates、同一 51,200 对训练场景和 5,000 对 validation，使用相同 H0/H1 exposure、优化计划及 checkpoint 规则。checkpoint 仅由对应 validation 目标选取。
- 独立评价为 H0 calibration 120,000、H0 verification 120,000、H1 development 6,144 场景。检测阈值只由 calibration 分数设定；verification 与 development 未用于调阈值或 checkpoint。scene ID 与已登记历史不重合。
- 阶段运行的测试证据：R6-D 定向测试与代码修复后的项目测试共 72 项通过；G1 中每个臂在相同 16-pair/200-update 数据上的主 loss 下降均超过预注册 10% 要求。最终提交前的定向回归 `tests/test_d2a_r6_joint.py` 为 19 passed（1.52 s）。
- 仿真 Train/Validation cache 生成器 SHA 为 `76815bb2c2068b3c43643340e640fc008dfebff4`，训练/评价代码 SHA 为 `9a631f4fae445d9fd3793faeba94cdcbc09cc062`；来源已记录于 owner manifest 与每臂 training record。结果包保留 manifests、history、checkpoint、指标和压缩逐场景预测。

## 3. 主要证据

J2-J−J1-J 的 C2+C3 joint-success 差为 `+0.04470`，配对 scene bootstrap 95% CI `[+0.03602,+0.05360]`，超过预注册 +0.02 筛选门。Overall Pd 差为 `+0.00130`（95% CI `[-0.00212,+0.00456]`），满足非劣保护；range RMSE ratio 为 `0.8966`（95% CI `[0.8248,0.9720]`），满足保护。Doppler RMSE ratio 为 `1.3066`（95% CI `[1.2015,1.4148]`），明显未达 1.05 上限。

各臂 C3 pooled Pfa 为 J1-D/J1-J/J2-J `0.10193/0.09865/0.09578`。J2-J 的 pooled exact 上界为 `0.09778`，但六个 Bonferroni familywise exact 上界中有四个超过 0.10，所以不能宣称通过 C3 虚警保护。

C3 partial-overlap、0 dB、20 dB CNR 条件下，J2-J 相对 J1-J 的 joint success 为 `0.1445` 对 `0.1035`，range RMSE 为 `8.706` 对 `9.655` cells；Doppler RMSE 为 `76.75` 对 `61.81` Hz。J2 的 Doppler 退化在预测候选正确子集中仍存在，故错误距离候选不足以单独解释它；因果机制未在本阶段检验。

训练曲线还显示训练预算依模型而异：J1-D 和 J2-J 的验证目标在 12,800 后继续改善，最佳点分别在 20,400 和 22,800；J1-P 与 J1-J 则在 14,400 达到最佳，之后出现验证回升。25,600 是 R6-D 的冻结比较预算，不是通用学习极限。用户允许后续在公平比较下适当延长；R6-D 不追溯改动，R6-E 可按本轮曲线预注册更长的共同预算，并匹配各臂更新数、有效 batch、训练 exposure、schedule 和 checkpoint 规则。

完整指标、分层结果和训练曲线见 [`R6-D 分析`](../../experiments/D2-A-R6/R6-D/analysis.md)、[`metrics.json`](../../experiments/D2-A-R6/R6-D/metrics.json)、[`training_curves.png`](../../experiments/D2-A-R6/R6-D/figures/training_curves.png) 与 [`development_metrics.png`](../../experiments/D2-A-R6/R6-D/figures/development_metrics.png)。

R6-D owner-scoped Train/Validation cache 已清理，共释放 `11,233,315,805` bytes；项目树（包括 `.git` 和项目目录内本地文件）清理后测得约 `1.26 GB`，低于 20 GiB 限制。完整审计见 [`cleanup.json`](../../experiments/D2-A-R6/R6-D/cleanup.json)。

## 4. 研究假设状态

- **H-CAP-001 训练预算影响学习：有限支持。** 单 seed 中 J1-D/J2-J 的验证目标在 12,800 后继续下降；J1-P/J1-J 较早达到验证最佳并回升，说明延长收益不能跨模型泛化。
- **H-ARCH-R6D-001 软候选交互能在保护检测、range、Doppler 与 C3 Pfa 下提高最终 joint：未获支持（该冻结候选/协议）。** 主 joint 与 range 通过，但 Doppler 与 C3 familywise 保护失败；不能外推否定其它交互结构。
- **复数相位归纳偏置/频移等变：未评价。** J2 使用可学习时间投影，本阶段没有公共相位或 Doppler 频移干预。
- **教师知识能否改善学生：未评价。** 没有蒸馏或教师分支。

没有证据要求修改研究总纲 v1.3，也没有改变 R6-A/B/C 的冻结结论。

## 5. 门控与下一步

| 决策项 | 结果 |
|---|---|
| R6-D G0/G1/G2 | PASS / PASS / pilot evaluation complete |
| J2 C2+C3 joint screening | PASS |
| Overall Pd / range protection | PASS / PASS |
| Doppler RMSE protection | FAIL |
| C3 familywise false-alarm protection | FAIL |
| 正式多 seed / final blind | 不启动 |
| 阶段裁决 | **REFINE** |

下一唯一优先任务是按 R6 P4 冻结 R6-E 的有界架构比较，使用 J1-J 作当前联合 baseline；候选范围限于少量有物理依据的复数/序列结构。R6-E 应在预注册中选择共同训练预算，允许高于 25,600，但不得依据本轮 H1 development 调模型或门槛。J2-J 不直接晋级，只保留为解释 range/joint 与 Doppler 权衡的开发候选。R6-D 没有证据支持传统基线优越/劣化、CRLB 达成、知识蒸馏增益或三通道相位机制结论。
