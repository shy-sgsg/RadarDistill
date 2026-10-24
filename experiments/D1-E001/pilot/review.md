# D1-E001 Pilot 筛选审查

## 运行范围

这是一个独立开发 seed 的预注册 pilot，不是正式三种子结果或盲测。A/B/C/D 在同一训练/验证数据、初始化、batch 顺序和 1600 optimizer updates 下顺序运行。训练设备为 RTX 3050 Laptop GPU `cuda:0`、FP32；可见总显存 3942 MiB。四组均完成预算，没有 OOM 回退。训练用时 133.0–140.0 秒/组，峰值分配显存 1046–1049 MiB，峰值保留显存 1150 MiB。

训练权重、初始/最佳/最终 checkpoint、训练历史、环境和完整逐 SNR 指标保存在本目录。训练数据使用本机忽略的 CPU memmap 缓存，未纳入提交。

## 筛选结果

| 方法 | 验证主指标 J（0/5/10/20 dB） | 20 dB AUC | 20 dB range RMSE (cell) | 20 dB Doppler RMSE (Hz) |
|---|---:|---:|---:|---:|
| A 真值监督 | 0.01203 | 1.000 | 0.0624 | 32.12 |
| B 教师预训练后真值微调 | 0.01095 | 1.000 | 0.0703 | 24.49 |
| C 联合真值与教师 | 0.01358 | 1.000 | 0.0595 | 38.48 |
| D 渐进蒸馏 | 0.01076 | 1.000 | 0.0582 | 23.76 |
| APX15 教师 | 0.00353 | 1.000 | 0.0380 | 0.0384 |

A 组验证任务损失从初始化的 `5.238` 降至 `2.118`。四个学生在 0、5、10、20 dB 的验证 AUC 均为 1.0；A 组对应 Doppler RMSE 为 31.38、25.18、22.19、32.12 Hz。-5 dB 时 AUC 为 0.914、Doppler RMSE 为 59.60 Hz；-10 dB 及以下检测与参数估计明显退化，例如 -10 dB AUC 为 0.593、range RMSE 为 9.02 cells、Doppler RMSE 为 326.27 Hz。

B、D 的单 seed 主指标比 A 低约 9% 和 11%，20 dB Doppler RMSE 也较低；C 没有显示一致收益。该差异只有一个训练 seed，不能据此断言蒸馏有效。APX15 教师在 20 dB 的频率估计远强于学生；pilot 尚未运行 FFT 插值、相位差和 profile-ML 盲测比较，不能据此声称传统方法竞争力。

## 物理一致性

对 A 组最佳 checkpoint，在相同的 256 条验证样本上做共同相位旋转和频率平移。参数变化只在 H1 样本统计，并按 SNR 分组；presence 变化使用全部 H0/H1 样本。20 dB H1 的 Doppler 变化 p95 在三种相位旋转下为 6.94、10.93、13.87 Hz；频移等变残差 p95 对 0.625 Hz 和 5 Hz 分别为 2.87、9.34 Hz。0 dB 相应最大 p95 为 60.65 Hz 和 37.82 Hz。-10 dB 及以下偏差很大，与该区间的参数估计失败一致。每个 SNR 的 H1 样本数为 13–19 条，因此这里只作描述性诊断，不作统计门判定。

最初跨全部 SNR 的汇总包含了低 SNR 样本，参数输出在 H0 上也没有定义；该原始汇总仍保留在 `metrics.json` 的 `physical_consistency_validation_original_aggregate`。按 H1/SNR 条件重算的结果同时保存在 `physical_consistency_validation` 与 `analysis.md`，用于正确解释高 SNR 学习行为。

## 决策

**通过 pilot 筛选，进入 D1-E001 正式三种子 A/B/C/D 对比。** A 组在 20 dB 达到冻结 G1 的 AUC、range 和 Doppler 数值参考，损失有明显下降；高 SNR 的相位旋转与频移诊断较稳定。低 SNR 的失败和 B/C/D 的单 seed 差异继续作为正式实验必须回答的问题。正式阶段冻结代码与 checkpoint 后使用新的独立校准流和盲测流；pilot 不进入显著性检验。

这不是 D1 阶段 GO：正式 G1/G2/G3 尚未审查，亦未达到或声称达到 `Pfa=1e-6`。如果三 seed/盲测结果未通过冻结门槛，按 `protocol.md` STOP 或 CONDITIONAL GO，不增加未经预注册的模型搜索。

## 执行异常记录

首次两次直接 CLI 启动在训练前因 PyTorch CUDA device 尚未选择而于 `reset_peak_memory_stats(cuda:0)` 报 `Invalid device argument`，没有发生 optimizer update。诊断确认 availability/count 检查没有初始化 device；成功的 pilot 在调用同一原始显存统计 API 前显式查询当前 device。根因修正为 `require_cuda` 显式选择 `cuda:0`，见提交 `651ece3`；此后该 API 的直接回归检查通过。训练算法、种子和损失未因该环境修正而变更。

formal 顺序训练入口现已实现并通过测试，但尚未运行正式三 seed。入口会按当前冻结配置从 pilot 原始策略摘要重新计算筛选门，且要求配置恰有三组互不重叠的训练/验证/初始化 seed；不信任早期仅表示技术完整性的旧 `pilot_gate.passed` 字段。正式训练、基线比较、校准和盲测完成后，会在相应报告中记录命令与结果。本文件不把尚未运行的正式训练写作结果。
