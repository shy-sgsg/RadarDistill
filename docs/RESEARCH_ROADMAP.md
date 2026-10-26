# RadarDistill 动态研究路线图

**更新时间：** 2026-10-26
**权威关系：** 长期使命见 [`RESEARCH_CHARTER.md`](RESEARCH_CHARTER.md)；简要状态见 [`CURRENT_STATUS.md`](CURRENT_STATUS.md)；历史指标和冻结门以对应 protocol、metrics 与阶段审查为准。

## 当前阶段

**D2-A-E001、R3 与 E002 v2.1 pilot 已完成；R3 和 E002 阶段决定均为 REFINE。** R3 的 T1/T2 没有通过独立固定阈值 G2。E002 v2.1 的 A/B/C CUDA pilot 表明三臂能在当前可辨识开发分布学习任务，但 G3 分层 H0 门和 G6 相位/频移等变门失败；G5 没有课程或 K2 增益。此结果不覆盖 R3 结论、不授予 T1 固定阈值教师资格、不改变 Charter 或 D1/E001 历史门。D1-E001 的 GO → D2 与 D1-E002 的 STOP/REDIRECT 适用范围不同，均保留原结论。

**当前执行：D2-A-T001 v1.2。** 唯一 post-Doppler ACE×AMF 教师候选、样本种子/分组、指标和 G1–G3 门保持冻结；按用户最新要求，协方差拟合和候选推理使用 RTX 3050 CUDA，旧 B3/T1/K2 CPU 基线不改。一次 CPU covariance试跑仅生成和拟合了4,096个H0场景，未评分校准/验证/开发集，已中止且临时 bank 将由同种子 CUDA拟合覆盖。CUDA 实现整仓 270 项测试通过；新的 preflight/教师阶段尚待推送代码后重新启动，不能据此预判候选通过。最终项目体积需在阶段收尾清理后小于20 GiB，中间体积不设此限制。

## 已完成实验与证据

- **D0-E001：** 12/12 简化仿真与 APX15 慢时间 FFT 子链正确性检查通过；不覆盖完整生产链。
- **D1-E001：** 三种子 formal、独立校准和 blind 完成；G1/G2 PASS，G3 FAIL，原阶段决定 GO → D2。C/D 的联合指标满足冻结 G2 规则，但没有收敛加速证据；D 的平均改善量受 seed 1 影响明显。
- **D1-R1：** 仅使用开发材料。高 SNR 预测距离门与 oracle 的 Doppler RMSE 接近；池化后线性 probe 比池化前好，但维数/样本数差异使池化因果未决；最终 checkpoint 梯度均值没有持续强冲突证据；teacher-KL 梯度因训练期 spectrum-head checkpoint 不存在而不可用。
- **D1-E002：** S1/S2 各三 seed、1600 更新，独立校准和 40,960 条 blind 完成。S2 高 SNR 等权 RMSE 23.352 Hz、较 S0 改善 10.86%，3/3 seed 同向；但低 SNR Pd 非劣区间未达门槛。S1 未改善。S3 未通过 pilot screen。总决策 **STOP / REDIRECT**。
- **D2-A-E001：** 合成静止杂波、3 通道、N=130、32 距离单元；R0 T1–T12 全通过。B3 几何 DPCA 静止杂波抑制中位数 15.406 dB，目标差分功率比中位数 3.614；分离 A 和边缘 B 有强检测，部分重叠 C 退化，同流形 D 近随机。校准 H0 每条件 512 个，开发每条件 256 个配对场景；主条件 A 的开发 `Pfa_CPI=0.0898` 高于标定经验值 0.0508。条件 Range/Doppler RMSE 区间已修正为 RMSE 统计量并重新生成。未训练学生、未生成 final blind。决定 **CONDITIONAL GO**，先复核阈值迁移，再考虑 E1。
- **D2-A-R3：** 新种子、3 个 4096 场景主校准组、4 个 4096 场景主 H0/H1 配对验证组，并含 CNR=0/20 dB 有界检查。原始 E001 B3 阈值在 16,384 个新主 H0 上得到 `Pfa=0.06464`（95% Wilson CI `[0.06097,0.06850]`）；有限样本可能解释 E001 `23/256` 的一部分，但层级 bootstrap 达到该点估计的比例为 0.0216。潜在目标参考幅度通过噪声/杂波生成尺度影响 H0，导致 B3 固定阈值的幅度箱 Pfa 极差达 0.169–0.216。T1 scene-wide B3 峰值/中位背景统计量与 T2 local median CFAR-like 分数均通过 G3/G4，但 T1 三次主合并验证 Pfa 为 0.0444–0.0462，T2 一次主验证 CI 不覆盖 0.05；因此都未过 G2，无方法完成全部门。T1 是未来候选，未被选为教师。无 E1、KD 或 final blind。阶段决定 **REFINE**。
- **D2-A-E002：** v2.1 A/B/C 已按相同 1,600 updates、有效 batch64、单 seed 101 使用 RTX 3050 CUDA 顺序训练。K2 audit 通过，G4 三臂通过；development AUC 为 A/B/C `0.9846/0.9787/0.9815`，Pd `0.9158/0.8996/0.9000`。学生整体 Pfa 与精确上限门通过，但 C3/SNR0 最差分层上限达 `0.195–0.204`，G3 分层门失败。G5 B−A joint-success gain `−0.1254`（95% CI `[−0.1386,−0.1122]`），C−B `+0.0036`（`[−0.0088,+0.0162]`）；课程和 K2 均无预注册 pilot signal。G6 phase agreement `0.943–0.950`、频移 residual p95 最大 `73.4 Hz`，等变门失败。阶段为 **REFINE**，没有 formal three-seed 或 final blind。
- **P0/P1 历史先导：** 按各自冻结阶段审查保留；不将旧开发测试重新称为盲测，也不把其结论扩大到 D0/D1。

## 当前支持、否定与未决判断

| 判断 | 状态 | 证据范围 |
|---|---|---|
| D0 简化仿真和 FFT 子链适配正确 | SUPPORTED | D0-E001 冻结 12 项检查 |
| D1 TCN 能从距离压缩 IQ 学习受测任务 | SUPPORTED | D1-E001 简化单目标复 AWGN、三训练种子 |
| D1 蒸馏改善收敛速度 | NOT_SUPPORTED | D1-E001 B/C/D 均未过冻结收敛门 |
| D1 C/D 蒸馏改善最终联合指标 | 有范围支持 | 原 G2 PASS；多策略比较未作家族错误率调整，D 的效果量受 seed 1 影响 |
| D1-E002 lag-one 特征能改善频率估计并保护检测/距离 | NOT_SUPPORTED（完整门） | S2 Doppler、range 门通过，低 SNR Pd 门失败；只限该协议 |
| 时间均值池化是 Doppler 误差主因 | INCONCLUSIVE | R1 probe 不支持简单因果解释；S1 也未改善 |
| 多任务持续梯度冲突是主要瓶颈 | INCONCLUSIVE | 只有最终 checkpoint、小批量诊断 |
| 教师低 SNR 全图响应可靠 | NOT_SUPPORTED（受测区间） | R1 teacher 的低 SNR 检测和 Doppler 质量弱，且 E001 蒸馏未改善低 SNR Pd |
| 学生有完整链速度优势 | INCONCLUSIVE | 输入边界、硬件不一致且无完整链计时 |
| D2-A 固定 5% H0 阈值可稳定跨独立主场景迁移 | NOT_SUPPORTED（当前仿真协议） | R3 冻结 B3 阈值 Pfa=0.06464；三次校准及新验证仍未使 T0/T1/T2 全部通过 G2 |
| 归一化可消除 H0 整体幅度尺度敏感性 | SUPPORTED（当前仿真内） | T1/T2 通过 G3 幅度分层与受控 `X,aX` 尺度检查；G2 未通过，因此不等于可靠固定阈值 |
| T1 可作为后续稳定传统教师 | INCONCLUSIVE | G3/G4 通过且 H1 能力强，但 G2 失败；保留候选，不进入教师选择 |
| D2-A-E002 学生能在可辨识开发场景学习任务 | SUPPORTED（当前单 seed 仿真 pilot） | 三臂 G4 通过，A AUC=0.9846、Pd=0.9158；未验证跨 seed 或盲测 |
| 课程顺序提高 joint success | NOT_SUPPORTED（当前 pilot） | B−A 差 −0.1254，配对 CI `[−0.1386,−0.1122]`；未见收敛提前 |
| K2 软响应在课程之外带来增益 | NOT_SUPPORTED（当前 pilot） | C−B 差 +0.0036，配对 CI `[−0.0088,+0.0162]`；未见收敛提前 |
| 当前 TaskStudent 满足所需公共相位/频移等变性 | NOT_SUPPORTED（当前 pilot） | G6 phase agreement 0.943–0.950；最大 frequency-shift residual p95 73.4 Hz |

## 当前主要问题

1. 当前 D2-A 教师候选的独立 H0 固定阈值门仍未通过；不能用 T1/T2 的 G3/G4 通过替代 G2。
2. 单 seed 学生可学习当前开发任务，但 C3 分层 H0 虚警不稳定，公共相位/频移等变性未通过；D1 与 E002 单 seed 结果均不能替代独立盲测和跨 seed 确认。
3. 真实 GMTI、系统级虚警率 `1e-6` 和完整链成本均未验证。

## 当前唯一优先任务

推送 T001 v1.2 CUDA 执行修订并重跑 preflight，再依冻结 seed block运行教师阶段。只在 G1/G2/G3 全通过时评审和执行一个预先冻结的 S2 student pilot；否则保留负结果并停止该分支。旧 E002 分层虚警与相位/频移问题仍登记为后续研究问题，不在 T001 中改动其冻结结论。

详细阶段门见 [`D2-A-E001_review.md`](stage_reviews/D2-A-E001_review.md) 和 [`D2-A-R3_review.md`](stage_reviews/D2-A-R3_review.md)。本结果不触发多目标、通道误差或加速度扩展，也不构成网络优于传统方法的证据。

## 后续依赖和阶段决策

- **D2：** 以单因素协议逐步引入可辨识的杂波/弱目标条件；每次扩展另设 gate。
- **D3：** 依赖可靠下游任务，再比较距离压缩前端与原始三通道回波。
- **D4：** 只有任务质量可靠后才测量结构、算子和计算优化。
- **D5：** 同硬件、同输入边界、完整前后处理链；未满足样本量时不声称 `Pfa=1e-6`。

任何新盲测都在模型、数据生成器、阈值、统计门、预算和候选集冻结后生成。新路线不能覆盖旧结果；旧阶段门也不妨碍提出单独登记的新假设。
