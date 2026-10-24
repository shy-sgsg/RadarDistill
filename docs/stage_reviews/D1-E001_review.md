# D1-E001 阶段审查

**审查日期：** 2026-10-24
**决策：GO → D2，限于逐项增加一项复杂度。**
**证据范围：** 冻结的 D1-E001 仿真协议、三个正式训练 seed、独立 H0 阈值校准和独立盲测。完整结果见 [`experiments/D1-E001/analysis.md`](../../experiments/D1-E001/analysis.md) 与 [`experiments/D1-E001/blind/metrics.json`](../../experiments/D1-E001/blind/metrics.json)。

## 决策摘要

按预注册规则，G1 通过，G2 通过，G3 未通过。G1 与 G2 足以进入 D2；G3 失败意味着学生未达到冻结的传统竞争触发标准，阶段决定不表示可替代 APX15/profile-ML，也不允许声称满足 `Pfa=1e-6`。

## 证据

- **G1：PASS，3/3 seed。** A 真值监督组在 20 dB 的三 seed AUC 与校准 `Pfa=1e-2` 下 Pd 均为 1.0。全 H1 range RMSE 为 0.0592–0.0682 cell，Doppler RMSE 为 25.95–35.13 Hz；相对常值预测的配对 95% 区间均低于 0。低 SNR 仍有显著退化：-20 dB AUC 约 0.48–0.50，-10 dB 约 0.61–0.64。
- **G2：PASS。** C 真值/教师联合训练的 ΔJ=-0.00013472，配对层级 bootstrap 95% CI `[-0.00025944,-0.00000810]`，3/3 seed 改善；D 渐进蒸馏的 ΔJ=-0.00086207，95% CI `[-0.00214046,-0.00004611]`，3/3 seed 改善。B 预训练后微调未通过。B/C/D 全部未达到单独的收敛加速规则；G2 仅由 C/D 的最终盲测指标规则通过。
- **G3：FAIL。** A/B/C/D 在 0/5、5/10、10/20 dB 预注册相邻对均未同时满足 Pd 与 profile-ML/APX15 参数误差门槛。20 dB profile 子集上，D range/Doppler RMSE 为 0.0689 cell/25.42 Hz，APX15 为 0.0415 cell/0.0360 Hz，profile-ML 为 0.01994 cell/0.01906 Hz。
- **虚警范围：** 盲测每方法 32,768 个 H0，阈值来自独立 50,000 个 H0。结果支持报告冻结的 `Pfa=1e-2/1e-3` 工作点，不足以验证 `Pfa=1e-6`。
- **资源与范围：** 正式训练使用 RTX 3050 Laptop GPU `cuda:0`、FP32、batch 64。结论只适用于单目标三通道距离压缩复 IQ 与复 AWGN 仿真；没有真实 GMTI、多目标、杂波、通道误差或完整 CoreX 链证据。

## 盲测分析更正记录

全部学生/基线预测已按冻结配置生成并保存后，评估器因读取预注册 G2 AUC 非劣界限的配置路径错误未能完成指标写盘。补丁修正了查找位置，并支持从已保存预测重放分析。盲测样本、阈值、模型、指标定义及阶段门均未改变；没有生成新预测来选择或调整模型。错误、修正提交和重放标记记录在 `blind/metrics.json.analysis_replay` 及 `blind/metrics_partial.json`。该处理只修复分析阶段，不改变盲测数据或冻结评估定义。

## 阶段限制和下一步授权范围

D1 证实该轻量学生能在受测仿真中学习最终检测/估计任务，并显示联合与渐进软频谱蒸馏在预注册联合指标上有收益。它没有证明蒸馏提升收敛速度，且低 SNR 检测与高 SNR 连续参数估计仍落后于传统教师或 profile-ML。

初步成本测量见 [`experiments/D1-E001/benchmark/inference_cost.json`](../../experiments/D1-E001/benchmark/inference_cost.json)：输入边界为已距离压缩 IQ。D 学生 CUDA 单样本平均/P95 为 `1.53/1.91 ms`；batch 128 平均/P95 为 `59.75/62.11 ms`，峰值分配显存约 `359 MB`。APX15 FFT 子链在 CPU 上单样本平均为 `0.83 ms`。由于设备不同且 APX15 计时只含 FFT 子链，不能据此声称学生速度更快或更慢；完整部署时延仍未验证。

允许进入 D2 的最低复杂度、可证伪验证。按 `docs/research_plan.md` 一次只引入一个因素，先冻结 D2 的可辨识条件、对照、预算、阈值和独立测试；不得把多目标、杂波、角度和通道失配同时堆入，也不得把本次 GO 当作完整场景验收。

当前建议优先处理精度。唯一的后续任务建议是冻结一个 Doppler 表示对照，比较直接 IQ 学生与 FFT 辅助连续修正，并保留匹配非学习插值、APX15 和 profile-ML 对照；不同时引入新场景因素。细节见 [`D0-D1_overall_report.md`](D0-D1_overall_report.md)，该建议尚未预注册或运行。
