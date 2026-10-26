# 当前研究状态

**更新日期：** 2026-10-26
**工作分支：** `main`
**最新完成工作：** D2-A-E002 v2.1 的 A/B/C 单 seed CUDA pilot、评价、阶段审查及缓存归档清理均已完成；D2-A-T001 v1.2 CUDA 教师实现与测试现已完成本地验证，待提交推送和重新 preflight。
**阶段判断：** D2-A-R3 **REFINE**；D2-A-E002 **REFINE**；D2-A-T001 **执行中，尚无评分结果**。E002 的 G4 任务学习门通过，G3 分层 H0 虚警与 G6 公共相位/频移等变性门未通过，G5 未发现课程或 K2 pilot 增益。总纲 v1.2 未修改。

## 当前结论

RadarDistill 的长期目标仍是通过物理仿真真值和传统算法知识蒸馏，训练直接完成 GMTI 动目标检测与参数估计的学生，并在后期评估精度和完整处理效率。当前优先问题仍是学生能否从复数多通道观测学习可辨识物理量。

D2-A-R3 使用新种子完成了三个独立 H0 主校准组、四个独立主验证场景组和预先规定的 CNR=0/20 dB 辅助组。H0 总体尺度随潜在目标参考幅度 `A` 变化，因为生成器用 `noise_power=A²×10^(-SNR/10)` 控制噪声，并按 CNR 设定杂波功率。该幅度变量不是 H0/H1 标签分支泄漏，但使绝对功率分数的跨尺度固定阈值不稳定。

冻结 E001 的 B3 阈值在 R3 新主条件的 16,384 个 H0 上得到 `Pfa_CPI=0.06464`（95% Wilson CI `[0.06097,0.06850]`）。E001 的 `23/256=0.08984` 可受有限样本影响，但联合校准/验证 bootstrap 产生该值的比例为 0.0216，不能据此把问题完全归结为抽样波动。

归一化候选 T1/T2 都通过幅度/整体尺度 G3 与可辨识 H1 性能保护 G4；T1 在部分重叠 C 的检测、参数精度和背景稳健性方面整体更有前景。两者都未通过 G2 固定阈值迁移门，故目前没有获准的 D2-A 传统教师统计量。E1 truth-only pilot 依据现行 R2/R3 规则未运行。T0、T1、T2 结果和限制见 [阶段分析](../experiments/D2-A-R3/analysis.md) 与 [阶段审查](stage_reviews/D2-A-R3_review.md)。

D1-E002 的 AWGN Doppler 结构搜索 **STOP / REDIRECT**；D1-E001 原决策 GO → D2 仅适用于其冻结实验范围，不覆盖 D2-A-R3 的失败门。D2-A-E001 冻结结果保持原样。

E002 v1 truth-only 提案从未执行，已由 v2.1 matched A/B/C 路线取代。A/B/C 分别为直接真值、课程真值、课程 + IQ-only K2；每臂 seed 101、1600 updates，RTX 3050 Laptop GPU 顺序训练。K2 质量审计和 G1 通过。G4 三臂均通过：A/B/C AUC 为 0.9846/0.9787/0.9815，Pd 为 0.9158/0.8996/0.9000；A 在主 A/10/10 条件 joint success 为 0.8489（单侧下限 0.8201）。学生整体 H0 Pfa 为 0.0457–0.0495，但最差分层精确上限为 0.195–0.204，故 G3 分层门失败。G5 的 B−A joint-success 差为 −0.1254（95% CI `[−0.1386,−0.1122]`），C−B 为 +0.0036（`[−0.0088,+0.0162]`），且无收敛加速信号。G6 公共相位决策一致率仅 0.943–0.950，频移等变残差 p95 最大 73.4 Hz，等变门失败。补充的 range 门诊断显示 validation 错门率为 10.30%/11.14%/10.96%，真值门下 Doppler RMSE 仍为 40.95/56.03/59.55 Hz；错门会增加部分误差，但不足以解释主要 Doppler 残差。阶段决定 **REFINE**：结果支持学生在当前可辨识仿真开发分布学习任务，但不支持课程/K2 增益、跨分组虚警稳定或相位/频率等变性；单 seed 开发证据不代表跨 seed 泛化。无正式三 seed 或 final blind。D2-A-R3 的阈值迁移失败与 T1 未获固定阈值教师资格保持不变。

### D2-A-T001 当前执行状态

用户批准的有界 ACE×AMF 教师比较已在协议 v1.2 中改为强制 CUDA covariance 拟合和候选推理；候选公式、场景分组、种子、统计门和学生前置条件保持冻结。CUDA/CPU同批数值对照、GPU路由和最终体积约束测试均通过，`AISAR` 整仓测试 270 项通过。一次初始 CPU covariance试跑仅生成并拟合了4,096个H0场景，尚未评分 calibration、verification 或 development，已在用户要求 GPU 后中止；该临时 bank 不作结果并将被 CUDA 重拟合覆盖。当前没有 T001 性能统计、学生训练或 final blind。

## 下一项唯一优先任务

提交推送 T001 v1.2 的 CUDA 执行修订，运行新 preflight，然后从冻结 covariance seed block 在 GPU 重拟合并开始预注册教师评分。若 G1/G2/G3 任一失败，不运行学生分支；仍不生成 final blind。阶段最终清理后项目总占用必须小于20 GiB，运行中间体积不受该限制。

## 关键材料

- 总纲/路线：[研究总纲](RESEARCH_CHARTER.md)、[动态路线图](RESEARCH_ROADMAP.md)、[假设登记](research/ASSUMPTION_REGISTER.md)
- 当前审查：[D2-A-R3](stage_reviews/D2-A-R3_review.md)、[D2-A-E001](stage_reviews/D2-A-E001_review.md)、[D1-E001](stage_reviews/D1-E001_review.md)、[D1-E002](stage_reviews/D1-E002_review.md)
- 最新学生阶段：[D2-A-E002 阶段审查](stage_reviews/D2-A-E002_review.md)、[结果分析](../experiments/D2-A-E002/analysis.md)、[指标 JSON](../experiments/D2-A-E002/metrics.json)；缓存清理记录：[容量审计](maintenance/CACHE_CLEANUP_2026-10-25.md)
- 当前提案：[D2-A-E002 v2 对齐](task_alignments/D2-A-E002_curriculum_k2_alignment.md)、[模型与损失审查](../experiments/D2-A-E002/architecture_and_loss_review.md)、[E002 v2 协议](../experiments/D2-A-E002/protocol.md)；[v1 对齐与审查](task_alignments/D2-A-E002_alignment.md)保留为历史记录
- R3 协议和结果：[`experiments/D2-A-R3/`](../experiments/D2-A-R3/)
- D2-A-T001 当前协议及实现状态：[协议 v1.2](../experiments/D2-A-T001/protocol.md)、[任务对齐](task_alignments/D2-A-T001_teacher_alignment.md)
- D2-A 物理模型、可辨识性与传统基线审查：[物理模型](research/D2-A_physical_model.md)、[可辨识性](research/D2-A_identifiability.md)、[基线审查](research/D2-A_baseline_review.md)
- 环境：AISAR / PyTorch 2.8.0+cu128 / CUDA 12.8 / RTX 3050 Laptop GPU，4 GiB；R3 是 CPU 统计诊断，E002 学生 pilot 已在可见 `cuda:0` 上完成。
