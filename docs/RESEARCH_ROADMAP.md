# RadarDistill 动态研究路线图

**更新时间：** 2026-10-25
**权威关系：** 长期使命见 [`RESEARCH_CHARTER.md`](RESEARCH_CHARTER.md)；简要状态见 [`CURRENT_STATUS.md`](CURRENT_STATUS.md)；历史指标和冻结门以对应 protocol、metrics 与阶段审查为准。

## 当前阶段

**D2-A-E001 的 R0/R1 开发验证已完成，R2 协议已冻结；阶段判断为 CONDITIONAL GO。** R2 沿用 D1-E001 `TaskStudent`，并匹配 1,600 update/有效 batch 64 预算；学生训练仍需等待阈值迁移前门。D1-E001 的 GO → D2 与 D1-E002 的 STOP/REDIRECT 适用范围不同：前者允许进入 D2，后者停止当前 AWGN Doppler 结构搜索。

## 已完成实验与证据

- **D0-E001：** 12/12 简化仿真与 APX15 慢时间 FFT 子链正确性检查通过；不覆盖完整生产链。
- **D1-E001：** 三种子 formal、独立校准和 blind 完成；G1/G2 PASS，G3 FAIL，原阶段决定 GO → D2。C/D 的联合指标满足冻结 G2 规则，但没有收敛加速证据；D 的平均改善量受 seed 1 影响明显。
- **D1-R1：** 仅使用开发材料。高 SNR 预测距离门与 oracle 的 Doppler RMSE 接近；池化后线性 probe 比池化前好，但维数/样本数差异使池化因果未决；最终 checkpoint 梯度均值没有持续强冲突证据；teacher-KL 梯度因训练期 spectrum-head checkpoint 不存在而不可用。
- **D1-E002：** S1/S2 各三 seed、1600 更新，独立校准和 40,960 条 blind 完成。S2 高 SNR 等权 RMSE 23.352 Hz、较 S0 改善 10.86%，3/3 seed 同向；但低 SNR Pd 非劣区间未达门槛。S1 未改善。S3 未通过 pilot screen。总决策 **STOP / REDIRECT**。
- **D2-A-E001：** 合成静止杂波、3 通道、N=130、32 距离单元；R0 T1–T12 全通过。B3 几何 DPCA 静止杂波抑制中位数 15.406 dB，目标差分功率比中位数 3.614；分离 A 和边缘 B 有强检测，部分重叠 C 退化，同流形 D 近随机。校准 H0 每条件 512 个，开发每条件 256 个配对场景；主条件 A 的开发 `Pfa_CPI=0.0898` 高于标定经验值 0.0508。条件 Range/Doppler RMSE 区间已修正为 RMSE 统计量并重新生成。未训练学生、未生成 final blind。决定 **CONDITIONAL GO**，先复核阈值迁移，再考虑 E1。
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

## 当前主要问题

1. 最佳纯 IQ 学生 S2 的 Doppler RMSE 仍约为四倍 FFT 抛物插值的 238 倍，且公共相位旋转/频移一致性不足。
2. E001 蒸馏的最终联合指标收益不等于低 SNR 检测收益、收敛收益或传统算法竞争力。
3. 简化 AWGN 测试还没有回答静止杂波中学生能否利用三通道相干结构；真实 GMTI、系统级虚警率 `1e-6` 和完整链成本均未验证。

## 当前唯一优先任务

**D2-A 阈值迁移诊断：** 以新开发 H0 seeds 验证固定 H0 calibration 阈值在独立场景和 nuisance 幅度变化下的 `Pfa_CPI`；若分数需修订，先冻结归一化/CFAR规则和新的校准、开发数据，再运行诊断。不得用已看过的 D2-A 开发 H0 调阈值。只有该前门达到预注册标准后，才运行 R2 的 E1 truth-only 小规模学生 pilot；基础学习通过后才考虑 E2/E3。此处不生成 final blind。

详细阶段门见 [`D2-A-E001_review.md`](stage_reviews/D2-A-E001_review.md)。本结果不触发多目标、通道误差或加速度扩展，也不构成网络优于传统方法的证据。

## 后续依赖和阶段决策

- **D2：** 以单因素协议逐步引入可辨识的杂波/弱目标条件；每次扩展另设 gate。
- **D3：** 依赖可靠下游任务，再比较距离压缩前端与原始三通道回波。
- **D4：** 只有任务质量可靠后才测量结构、算子和计算优化。
- **D5：** 同硬件、同输入边界、完整前后处理链；未满足样本量时不声称 `Pfa=1e-6`。

任何新盲测都在模型、数据生成器、阈值、统计门、预算和候选集冻结后生成。新路线不能覆盖旧结果；旧阶段门也不妨碍提出单独登记的新假设。
