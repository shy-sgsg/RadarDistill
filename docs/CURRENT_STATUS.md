# 当前研究状态

**更新日期：** 2026-10-25
**工作分支：** `main`
**最新完成工作：** D2-A-E002 v2 课程学习 + K2 A/B/C pilot 已完成预注册并获用户执行授权；代码、数据生成、审计、测试及训练尚未完成。
**阶段判断：** D2-A-R3 **REFINE** 保持不变；E002 当前处于已授权的训练前实现/验证阶段，尚无 E002 运行结论。总纲 v1.1 未修改。

## 当前结论

RadarDistill 的长期目标仍是通过物理仿真真值和传统算法知识蒸馏，训练直接完成 GMTI 动目标检测与参数估计的学生，并在后期评估精度和完整处理效率。当前优先问题仍是学生能否从复数多通道观测学习可辨识物理量。

D2-A-R3 使用新种子完成了三个独立 H0 主校准组、四个独立主验证场景组和预先规定的 CNR=0/20 dB 辅助组。H0 总体尺度随潜在目标参考幅度 `A` 变化，因为生成器用 `noise_power=A²×10^(-SNR/10)` 控制噪声，并按 CNR 设定杂波功率。该幅度变量不是 H0/H1 标签分支泄漏，但使绝对功率分数的跨尺度固定阈值不稳定。

冻结 E001 的 B3 阈值在 R3 新主条件的 16,384 个 H0 上得到 `Pfa_CPI=0.06464`（95% Wilson CI `[0.06097,0.06850]`）。E001 的 `23/256=0.08984` 可受有限样本影响，但联合校准/验证 bootstrap 产生该值的比例为 0.0216，不能据此把问题完全归结为抽样波动。

归一化候选 T1/T2 都通过幅度/整体尺度 G3 与可辨识 H1 性能保护 G4；T1 在部分重叠 C 的检测、参数精度和背景稳健性方面整体更有前景。两者都未通过 G2 固定阈值迁移门，故目前没有获准的 D2-A 传统教师统计量。E1 truth-only pilot 依据现行 R2/R3 规则未运行。T0、T1、T2 结果和限制见 [阶段分析](../experiments/D2-A-R3/analysis.md) 与 [阶段审查](stage_reviews/D2-A-R3_review.md)。

D1-E002 的 AWGN Doppler 结构搜索 **STOP / REDIRECT**；D1-E001 原决策 GO → D2 仅适用于其冻结实验范围，不覆盖 D2-A-R3 的失败门。D2-A-E001 冻结结果保持原样。

E002 v1 truth-only 提案从未执行，现由 v2 matched A/B/C 路线取代。v2 比较直接真值学习、同样本与预算下的课程真值学习，以及课程 + IQ-only K2 软响应；物理真值任务损失在所有臂保留。新路线针对总纲 Q1–Q4、Q6，区分课程顺序和教师响应增量，不改变 R3 未通过 G2 的历史结论、不批准 T1 为固定阈值教师，也不重写 D1/D2-A 冻结结果。当前只有协议、配置与设计审查材料；尚无 runner、E002 数据、审计指标或训练结果。

## 下一项唯一优先任务

按已获用户批准的 E002 v2 协议完成 TDD 实现、数据/标签/split/CUDA preflight、K2 质量审计和微型过拟合；门通过后顺序运行 A/B/C 单 seed pilot。暂不启动正式三 seed、final blind 或真实数据实验。

## 关键材料

- 总纲/路线：[研究总纲](RESEARCH_CHARTER.md)、[动态路线图](RESEARCH_ROADMAP.md)、[假设登记](research/ASSUMPTION_REGISTER.md)
- 当前审查：[D2-A-R3](stage_reviews/D2-A-R3_review.md)、[D2-A-E001](stage_reviews/D2-A-E001_review.md)、[D1-E001](stage_reviews/D1-E001_review.md)、[D1-E002](stage_reviews/D1-E002_review.md)
- 当前提案：[D2-A-E002 v2 对齐](task_alignments/D2-A-E002_curriculum_k2_alignment.md)、[模型与损失审查](../experiments/D2-A-E002/architecture_and_loss_review.md)、[E002 v2 协议](../experiments/D2-A-E002/protocol.md)；[v1 对齐与审查](task_alignments/D2-A-E002_alignment.md)保留为历史记录
- R3 协议和结果：[`experiments/D2-A-R3/`](../experiments/D2-A-R3/)
- D2-A 物理模型、可辨识性与传统基线审查：[物理模型](research/D2-A_physical_model.md)、[可辨识性](research/D2-A_identifiability.md)、[基线审查](research/D2-A_baseline_review.md)
- 环境：AISAR / PyTorch 2.8.0+cu128 / CUDA 12.8 / RTX 3050 Laptop GPU，4 GiB；R3 是 CPU 统计诊断，后续学生训练如获批准须使用可见 CUDA。
