# 当前研究状态

**更新日期：** 2026-10-28
**工作分支：** `main`
**最新完成工作：** D2-A-R5-A Small/Medium width-only capacity pilot 已在 CUDA 完成；代码提交 `1678a468d5c150ff734d80ad40cf75407a49651f`，结果与状态归档提交 `760cb30c7ce9dafdbcb5a328d9228b0f2eb9f585`。结果、训练曲线、分层指标和 Codex 阶段自审见 [capacity analysis](../experiments/D2-A-R5/R5-A/capacity/analysis.md) 与 [stage review](stage_reviews/D2-A-R5A-capacity_review.md)；ChatGPT 审查请求见 [review request](../experiments/D2-A-R5/R5-A/capacity/review_request.md)。
**阶段判断：** R5-A **REFINE**：延长训练在固定 seed/data 下持续改善；Medium width=96 相对 Small width=32 的 C2/C3 joint-success 提升 `+0.1309`（95% CI `[+0.1123,+0.1509]`），range/Doppler RMSE 亦改善；但两者 C3 H0 Pfa 分别 `0.2041/0.1890`，均超 0.10 门，因此 G3 任务保护失败，Medium 不晋级、不运行 Large。E003、R3、E002、T001、R4 和旧 R5-A 结论保持不变。

**当前授权工作：** R5-A Phase 1/2 已完成：S0/S2 及 Small/Medium 均按12,800-update预算使用RTX 3050 Laptop `cuda:0` FP32训练；未运行KD、正式多 seed 或 final blind。Medium 未通过C3 Pfa绝对保护。R5-B 架构调查与单一轻量原始IQ时频注意力候选 protocol/config v1.0 已冻结；代码、G1与CUDA pilot 尚未运行。大型可重建缓存不提交，整个项目最终须低于20 GiB。

## 当前结论

RadarDistill 的长期目标仍是通过物理仿真真值和传统算法知识蒸馏，训练直接完成 GMTI 动目标检测与参数估计的学生，并在后期评估精度和完整处理效率。当前优先问题仍是学生能否从复数多通道观测学习可辨识物理量。

### D2-A-R5-A 阶段结果

在同一 R5 长程学习率计划内，seed101 的验证任务指标从 update1600 到 12,800 持续改善：S0 task loss `0.519→0.321`、range RMSE `2.844→2.203 cells`、Doppler RMSE `35.49→24.16 Hz`；S2 分别为 `1.023→0.694`、`3.750→2.907 cells`、`72.51→52.13 Hz`。两臂均触发预注册连续延长条件；S0/S2最佳checkpoint分别来自 update12,400/12,000。该证据支持训练预算影响，但不证明达到收敛或跨 seed 泛化。

复用 E003 development 的 Pd 为 S0/S2 `0.9619/0.9336`，AUC `0.99374/0.98999`，全 H1 range RMSE `2.239/2.824 cells`，Doppler RMSE `25.23/54.25 Hz`，joint success `0.8389/0.6055`。S2−S0 joint差为 `−0.2334`（95% paired scene bootstrap CI `[-0.2488,-0.2188]`），C2/C3 joint差 `−0.3945`（`[-0.4189,-0.3696]`）。S0/S2普通 H0 verification Pfa 为 `0.05017/0.04834`，但 C3 H0 Pfa 为 `0.20410/0.18384`，均超过0.10门。S2公共相位旋转下基本不变，但该性质没有转化为任务收益；频移干预也暴露估计不稳定。所有校准、verification、development 与物理诊断均为 E003 已使用材料，不是盲测。

固定相干 S2 候选按 R5-A任务保护与C2/C3收益筛选停止；不外推否定所有物理结构。H-CAP-001限定支持；H-CAP-002在width-only试验中显示有限的C2/C3容量收益，但因C3绝对Pfa保护失败而未通过G3。R5-A训练计时文件只保留了扩展段用时，未保存前6,400 updates墙钟时间；此限制已写入分析，后续训练记录器已修正累计计时。完整门控、统计区间和物理干预见 [R5-A分析](../experiments/D2-A-R5/R5-A/analysis.md)。

### D2-A-R5-A Small/Medium 容量结果

在 seed101、相同 E003 train/validation IDs、pair plan、E1 truth-only、有效 batch64、FP32和12,800 updates下，Medium width=96（82,887参数）相对复用的Small width=32（13,383参数）取得 C2/C3 joint-success `+0.13086`（95% paired-scene bootstrap CI `[+0.11230,+0.15088]`），C2/C3 range/Doppler RMSE比为 `0.839/0.713`。G1和primary gain筛选通过。常规H0 verification Pfa为`0.05017/0.04797`，但C3 H0 Pfa为`0.20410/0.18896`；两者的单侧 exact 95%上限均远高于冻结0.10限制。因此 G3 任务保护未通过，Medium不得晋级，Large不运行。结果限于一个训练seed与已使用开发材料，不是blind或跨seed确认。详细数据见 [capacity analysis](../experiments/D2-A-R5/R5-A/capacity/analysis.md)、[metrics](../experiments/D2-A-R5/R5-A/capacity/metrics.json) 和 [capacity review](stage_reviews/D2-A-R5A-capacity_review.md)。

### D2-A-E003 阶段结果

E003 固定相干前端 R1 正确性与 R2 可观测信息门通过。seed 101 的 S0/S2 均使用 RTX 3050 `cuda:0`、FP32、1,600 updates、有效 batch64；本轮无教师损失、蒸馏、多 seed 或 final blind。独立 H0 verification 总体 Pfa 为 S0/S2 `0.0483/0.0475`，但 C3 H0 Pfa 为 `0.1814/0.1528`，两者均超过冻结 0.10 限值。S2 相对 S0 的总体 Pd 差 `−0.0474`（95% CI `[−0.0571,−0.0376]`）、overall joint 差 `−0.2102`、C2/C3 joint 差 `−0.2593`（`[−0.2822,−0.2368]`）；C2/C3 Doppler RMSE 比为 `1.835`（`[1.720,1.959]`）。所以停止当前 S2 表示候选，不做正式三 seed 或 KD。

S2 公共相位 32 旋转下的 H0/H1 决策一致率为 1.0，分数变化约 `1e−5`，但 C1–C3 joint success 与参数估计恶化；该结果说明物理不变性不能替代任务质量。固定前端、归一化、28→6 投影的独立贡献尚未分离。训练后严格按 `cache_manifest.json` 删除了 90 个 E003 可重建缓存文件（14,232,309,192 bytes）；保留检查点、history、metrics、scene IDs、受控诊断、环境与审查材料。清理后项目总占用为 801,766,373 bytes（0.747 GiB），低于最终 20 GiB 限制；审计记录见 `experiments/D2-A-E003/cleanup.json`。清理未触及清单外文件或用户未跟踪文件。

D2-A-R3 使用新种子完成了三个独立 H0 主校准组、四个独立主验证场景组和预先规定的 CNR=0/20 dB 辅助组。H0 总体尺度随潜在目标参考幅度 `A` 变化，因为生成器用 `noise_power=A²×10^(-SNR/10)` 控制噪声，并按 CNR 设定杂波功率。该幅度变量不是 H0/H1 标签分支泄漏，但使绝对功率分数的跨尺度固定阈值不稳定。

冻结 E001 的 B3 阈值在 R3 新主条件的 16,384 个 H0 上得到 `Pfa_CPI=0.06464`（95% Wilson CI `[0.06097,0.06850]`）。E001 的 `23/256=0.08984` 可受有限样本影响，但联合校准/验证 bootstrap 产生该值的比例为 0.0216，不能据此把问题完全归结为抽样波动。

归一化候选 T1/T2 都通过幅度/整体尺度 G3 与可辨识 H1 性能保护 G4；T1 在部分重叠 C 的检测、参数精度和背景稳健性方面整体更有前景。两者都未通过 G2 固定阈值迁移门，故目前没有获准的 D2-A 传统教师统计量。E1 truth-only pilot 依据现行 R2/R3 规则未运行。T0、T1、T2 结果和限制见 [阶段分析](../experiments/D2-A-R3/analysis.md) 与 [阶段审查](stage_reviews/D2-A-R3_review.md)。

D1-E002 的 AWGN Doppler 结构搜索 **STOP / REDIRECT**；D1-E001 原决策 GO → D2 仅适用于其冻结实验范围，不覆盖 D2-A-R3 的失败门。D2-A-E001 冻结结果保持原样。

E002 v1 truth-only 提案从未执行，已由 v2.1 matched A/B/C 路线取代。A/B/C 分别为直接真值、课程真值、课程 + IQ-only K2；每臂 seed 101、1600 updates，RTX 3050 Laptop GPU 顺序训练。K2 质量审计和 G1 通过。G4 三臂均通过：A/B/C AUC 为 0.9846/0.9787/0.9815，Pd 为 0.9158/0.8996/0.9000；A 在主 A/10/10 条件 joint success 为 0.8489（单侧下限 0.8201）。学生整体 H0 Pfa 为 0.0457–0.0495，但最差分层精确上限为 0.195–0.204，故 G3 分层门失败。G5 的 B−A joint-success 差为 −0.1254（95% CI `[−0.1386,−0.1122]`），C−B 为 +0.0036（`[−0.0088,+0.0162]`），且无收敛加速信号。G6 公共相位决策一致率仅 0.943–0.950，频移等变残差 p95 最大 73.4 Hz，等变门失败。补充的 range 门诊断显示 validation 错门率为 10.30%/11.14%/10.96%，真值门下 Doppler RMSE 仍为 40.95/56.03/59.55 Hz；错门会增加部分误差，但不足以解释主要 Doppler 残差。阶段决定 **REFINE**：结果支持学生在当前可辨识仿真开发分布学习任务，但不支持课程/K2 增益、跨分组虚警稳定或相位/频率等变性；单 seed 开发证据不代表跨 seed 泛化。无正式三 seed 或 final blind。D2-A-R3 的阈值迁移失败与 T1 未获固定阈值教师资格保持不变。

### D2-A-T001 阶段结果

协议 v1.2 的 H0 covariance 拟合和 ACE×AMF 候选评分强制使用 RTX 3050 Laptop `cuda:0`；候选、种子、样本和门槛保持冻结。4096 covariance H0、15,000 calibration H0、15,000 verification H0 和12,000 paired development scenes 均已完成。运行代码提交为 `91dfb81673e45cad3f3330d90adc707a0cdf859d`；结果归档后的 `AISAR` 整仓测试270项通过。总体 ACE×AMF verification `Pfa_CPI=0.05007`，单侧 exact 95%上限0.05309，但 C3、SNR=0 dB、CNR=20 dB 三个大分组的上限为0.125–0.127，未过冻结分层门。主困难条件 joint success 为 ACE×AMF 0、B3 0.01、T1 0.004；可辨识 A/B/C 上候选相对 T1 的 Pd 差为−0.2267（95% CI `[−0.2345,−0.2188]`）、joint-success 差为−0.2441（`[−0.2522,−0.2361]`）。公共复尺度 G1 误差为4.58e−5，高于冻结1e−6；目标单独频移残差对四个预设频移均约等于其相反数，未跟随注入目标频率移动。G1/G2失败，按协议 G3 未评估且 S2 pilot 不符合前置条件；没有学生训练或 final blind。完整数据、图表和解释见 [T001 分析](../experiments/D2-A-T001/analysis.md) 与 [T001 阶段审查](stage_reviews/D2-A-T001_review.md)。

CUDA covariance 拟合用时47.95秒（85.42场景/秒）；候选内核复用16个CPI时 batch=1 CUDA约为CPU的7.40倍、batch=16约15.63倍。整个教师阶段用时3539.7秒（约59分钟），该时间包括CPU场景仿真、旧基线与CPU bootstrap，不代表端到端实时性能。阶段后项目占用9,379,091,889 bytes（8.735 GiB），低于最终20 GiB上限；未生成原始IQ或大型模型文件，GMTI用户目录未改动。

### D2-A-R4 阶段结果

R4 新复尺度不变性诊断在正确的 complex128 变换构造下通过 `1e-5` 门；T001 历史 `4.58e-5 > 1e-6` G1 失败不变。legacy complex64 先乘路径最大差为`2.44e-4`，CPU/CUDA ACE×AMF 最大绝对差`3.09e-4`，相对差`9.30e-7`，64例无峰值或阈值判决变化。`1e-4` 不是适用于所有指标的通用容差。

H0 场景标签独立性检查逐元素通过；噪声/杂波尺度按 A、SNR、CNR 变化。B3 的 H0 幅度四分位 Pfa 从0到0.119，T1 对应 Pfa 为0.046–0.052；但 C3 的 T1 Pfa 仍约0.178–0.217，冻结学生 A/B/C 约0.152–0.195。T1改善了 C2 检测和联合成功，但峰图/参数估计与 B3 相同，属于分数归一化收益；T1 仍未取得跨条件固定阈值教师资格。

首轮phase/频移诊断的256例全部是C0。原结果保留，另按每条件64例做后验分层复核：冻结 A/B/C phase agreement 为0.916/0.925/0.916；pilot S0/S1 为0.905/0.909，不能支持 phase-augmented S1 跨条件不变。相同2048个pilot H1的配对 bootstrap 中，S1−S0 Doppler RMSE为`−6.29 Hz`（95% CI `[-9.16,-2.84]`），Pd为`−0.0117`、joint success为`−0.0313`且区间低于0，range RMSE区间跨0。单seed的频率改善以检测/联合任务退化为代价，不启动新seed、KD或blind。阶段建议 **REFINE**，先进行独立 ChatGPT 方向审查。

## 当前唯一优先任务

按冻结的 R5-B protocol/config v1.0 实现 TFAS 轻量时频注意力学生、CUDA-only runner 和定向测试。先核对物理/频率表示、模型规模、显存及 G1 可学习性，再完成与 R5-A Small/Medium 的单 seed、同数据同暴露比较；维持独立校准与开发评估，报告 C3 H0 Pfa 和 C2/C3 任务保护。Medium 的正向参数估计增益与失败的绝对虚警门均需进入比较；不运行 Large、KD 或 final blind。当前仅完成架构调查/预注册，没有 R5-B 运行结果。

## 关键材料

- 总纲/路线：[研究总纲](RESEARCH_CHARTER.md)、[动态路线图](RESEARCH_ROADMAP.md)、[假设登记](research/ASSUMPTION_REGISTER.md)
- 当前审查：[D2-A-R3](stage_reviews/D2-A-R3_review.md)、[D2-A-E001](stage_reviews/D2-A-E001_review.md)、[D1-E001](stage_reviews/D1-E001_review.md)、[D1-E002](stage_reviews/D1-E002_review.md)
- 最新学生阶段：[D2-A-E002 阶段审查](stage_reviews/D2-A-E002_review.md)、[结果分析](../experiments/D2-A-E002/analysis.md)、[指标 JSON](../experiments/D2-A-E002/metrics.json)；缓存清理记录：[容量审计](maintenance/CACHE_CLEANUP_2026-10-25.md)
- 当前提案：[D2-A-E002 v2 对齐](task_alignments/D2-A-E002_curriculum_k2_alignment.md)、[模型与损失审查](../experiments/D2-A-E002/architecture_and_loss_review.md)、[E002 v2 协议](../experiments/D2-A-E002/protocol.md)；[v1 对齐与审查](task_alignments/D2-A-E002_alignment.md)保留为历史记录
- R3 协议和结果：[`experiments/D2-A-R3/`](../experiments/D2-A-R3/)
- D2-A-T001 协议、[分析与指标](../experiments/D2-A-T001/analysis.md)、[阶段审查](stage_reviews/D2-A-T001_review.md)、[任务对齐](task_alignments/D2-A-T001_teacher_alignment.md)
- D2-A-R4 [实验协议](../experiments/D2-A-R4/protocol.md)、[分析](../experiments/D2-A-R4/analysis.md)、[Codex 自审](stage_reviews/D2-A-R4_review.md)、[指标表](../experiments/D2-A-R4/result.csv)、[分层复核](../experiments/D2-A-R4/sampling_sensitivity.json)
- D2-A-R5-A [容量对照分析](../experiments/D2-A-R5/R5-A/capacity/analysis.md)、[指标](../experiments/D2-A-R5/R5-A/capacity/metrics.json)、[阶段自审](stage_reviews/D2-A-R5A-capacity_review.md)、[容量研究汇总](research/D2-A_student_capacity_study.md)
- D2-A-R5-A [ChatGPT容量阶段审查请求](../experiments/D2-A-R5/R5-A/capacity/review_request.md)，审查目标结果SHA `760cb30c7ce9dafdbcb5a328d9228b0f2eb9f585`
- D2-A-E003 [实验协议](../experiments/D2-A-E003/protocol.md)、[固定特征设计](../experiments/D2-A-E003/feature_design.md)、[训练与指标](../experiments/D2-A-E003/training.json)、[分析](../experiments/D2-A-E003/analysis.md)、[审查请求](../experiments/D2-A-E003/review_request.md)、[阶段自审](stage_reviews/D2-A-E003_review.md)
- D2-A 物理模型、可辨识性与传统基线审查：[物理模型](research/D2-A_physical_model.md)、[可辨识性](research/D2-A_identifiability.md)、[基线审查](research/D2-A_baseline_review.md)
- 环境：AISAR / PyTorch 2.8.0+cu128 / CUDA 12.8 / RTX 3050 Laptop GPU，4 GiB；R3 是 CPU 统计诊断，E002 学生 pilot 已在可见 `cuda:0` 上完成。
