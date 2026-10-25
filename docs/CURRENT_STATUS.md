# 当前研究状态

**更新日期：** 2026-10-25
**工作分支：** `main`
**最新完成工作：** D2-A-E002 独立 truth-only `TaskStudent` pilot 协议提案；仅完成设计，未运行训练。
**阶段判断：** D2-A-R3 **REFINE** 保持不变；D2-A-E002 为 **CONDITIONAL GO（协议提案）**，不构成训练授权。学生与蒸馏训练仍未启动。

## 当前结论

RadarDistill 的长期目标仍是通过物理仿真真值和传统算法知识蒸馏，训练直接完成 GMTI 动目标检测与参数估计的学生，并在后期评估精度和完整处理效率。当前优先问题仍是学生能否从复数多通道观测学习可辨识物理量。

D2-A-R3 使用新种子完成了三个独立 H0 主校准组、四个独立主验证场景组和预先规定的 CNR=0/20 dB 辅助组。H0 总体尺度随潜在目标参考幅度 `A` 变化，因为生成器用 `noise_power=A²×10^(-SNR/10)` 控制噪声，并按 CNR 设定杂波功率。该幅度变量不是 H0/H1 标签分支泄漏，但使绝对功率分数的跨尺度固定阈值不稳定。

冻结 E001 的 B3 阈值在 R3 新主条件的 16,384 个 H0 上得到 `Pfa_CPI=0.06464`（95% Wilson CI `[0.06097,0.06850]`）。E001 的 `23/256=0.08984` 可受有限样本影响，但联合校准/验证 bootstrap 产生该值的比例为 0.0216，不能据此把问题完全归结为抽样波动。

归一化候选 T1/T2 都通过幅度/整体尺度 G3 与可辨识 H1 性能保护 G4；T1 在部分重叠 C 的检测、参数精度和背景稳健性方面整体更有前景。两者都未通过 G2 固定阈值迁移门，故目前没有获准的 D2-A 传统教师统计量。E1 truth-only pilot 依据现行 R2/R3 规则未运行。T0、T1、T2 结果和限制见 [阶段分析](../experiments/D2-A-R3/analysis.md) 与 [阶段审查](stage_reviews/D2-A-R3_review.md)。

D1-E002 的 AWGN Doppler 结构搜索 **STOP / REDIRECT**；D1-E001 原决策 GO → D2 仅适用于其冻结实验范围，不覆盖 D2-A-R3 的失败门。D2-A-E001 冻结结果保持原样。

E002 提出独立假设 H-LEARN-D2-001：冻结的 D1-E001 `TaskStudent` 能否只凭 IQ 和物理真值，在 A/B/C 静止杂波场景学习检测、距离及连续 Doppler。协议使用新 scene seed、学生自己的全局 H0 校准/验证和任务/物理门；不使用教师损失。它不改变 R3 未通过 G2 的历史结论、不批准 T1 为教师，也不重写原 R2 的 A-only E1 记录。当前没有 E002 代码入口、数据、指标或训练结果。

## 下一项唯一优先任务

先将 E002 协议提交 ChatGPT 阶段审查，重点审查 IID 混合下的 Pfa 置信界、G2/G4/G5 门槛及其与总纲的关系；之后须取得用户对单 seed truth-only pilot 执行的明确批准。批准前不实现/运行训练入口、不生成 E002 数据、不训练、不开展蒸馏或 final blind。

## 关键材料

- 总纲/路线：[研究总纲](RESEARCH_CHARTER.md)、[动态路线图](RESEARCH_ROADMAP.md)、[假设登记](research/ASSUMPTION_REGISTER.md)
- 当前审查：[D2-A-R3](stage_reviews/D2-A-R3_review.md)、[D2-A-E001](stage_reviews/D2-A-E001_review.md)、[D1-E001](stage_reviews/D1-E001_review.md)、[D1-E002](stage_reviews/D1-E002_review.md)
- 当前提案：[D2-A-E002 对齐](task_alignments/D2-A-E002_alignment.md)、[E002 协议](../experiments/D2-A-E002/protocol.md)、[E002 协议审查](stage_reviews/D2-A-E002_protocol_review.md)
- R3 协议和结果：[`experiments/D2-A-R3/`](../experiments/D2-A-R3/)
- D2-A 物理模型、可辨识性与传统基线审查：[物理模型](research/D2-A_physical_model.md)、[可辨识性](research/D2-A_identifiability.md)、[基线审查](research/D2-A_baseline_review.md)
- 环境：AISAR / PyTorch 2.8.0+cu128 / CUDA 12.8 / RTX 3050 Laptop GPU，4 GiB；R3 是 CPU 统计诊断，后续学生训练如获批准须使用可见 CUDA。
