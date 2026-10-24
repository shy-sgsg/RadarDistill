# D1-R1 Doppler 误差诊断审查

**审查日期：** 2026-10-25
**判断：** 诊断证据有效；对 D1-E002 的 S1/S2 结构搜索作 **REDIRECT**。
**范围：** D1 开发/验证材料及其后续冻结的 E002 结果。D1-E001 的原阶段门和盲测结论保持不变。

## 研究问题和材料

D1-R1 检查高 SNR Doppler 误差是否主要来自预测距离门、时间均值池化、任务梯度冲突或教师响应。诊断沿用 PRF=1300 Hz、N=130、三通道、32 个距离门、单目标复 AWGN 与距离压缩 IQ。共同 `pilot_validation` 有每个 SNR、每类各 128 条；线性 probe 使用正式训练/验证缓存。没有训练新模型，没有读取 E001 blind predictions 或校准数据。执行记录为 AISAR、PyTorch 2.8.0+cu128、CUDA 12.8、RTX 3050 Laptop GPU、FP32，峰值分配显存约 548 MiB。

## 一、工程正确性

- 正常 Doppler 解码按学生预测距离门取值；oracle 路径只在诊断时用 `floor(true_range_cell)` 取同一个 Doppler 头的输出。真值距离没有进入模型 forward 或正常解码。
- 频率误差按 PRF 周期回绕。分组、训练/验证缓存、checkpoint 和代码 SHA 记在 [`experiments/D1-R1/metrics.json`](../../experiments/D1-R1/metrics.json) 与 [`protocol.md`](../../experiments/D1-R1/protocol.md)。`result.csv` 是 metrics 的展开，不是额外独立样本。
- D1 学生把 IQ 实虚部送入普通实值 TCN；时间维均值池化后，32 维特征送到逐距离门圆周 Doppler 头。训练期全图频谱头在推理导出时移除。该实现满足本轮冻结输入/标签边界，但没有显式复数相位等变结构。
- R1 是只读诊断，未重跑原始数据生成或训练。审查未发现材料显示 oracle 标签泄漏或把教师输出覆盖物理真值；未对训练作独立复现。

## 二、误差来源和证据

### 预测距离门与 oracle 距离门

| SNR | 预测门 RMSE (Hz) | 真值门 oracle RMSE (Hz) | 说明 |
|---:|---:|---:|---|
| -10 dB | 310.95 ± 5.92 | 186.16 ± 10.81 | 距离选择有明显额外影响，但 oracle 仍有大残差 |
| 0 dB | 32.86 ± 3.42 | 32.83 ± 2.29 | 几乎无差别 |
| 5 dB | 22.13 ± 1.28 | 22.14 ± 1.29 | 几乎无差别 |
| 10 dB | 19.68 ± 1.80 | 19.43 ± 2.21 | 差异很小 |
| 20 dB | 32.36 ± 6.45 | 31.87 ± 7.07 | 差异很小 |

表内为三个训练 seed 在共同开发集上的均值和标准差，每个 seed/SNR 128 条 H1。高 SNR 下距离门选择不能解释主要 Doppler 残差；低 SNR（尤其 -10 dB）则存在显著误差传播。

### 时间池化与任务梯度

固定 ridge α=1、1024 个 H1 probe 样本下，池化前 4160 维特征的 RMSE 为 258.3、272.4、267.0 Hz；池化后 32 维为 180.7、187.7、191.4 Hz。高 SNR 的池化后 probe 在 0/5/10/20 dB 为：seed 1 `35.8/23.5/28.6/40.7` Hz，seed 2 `36.5/23.8/26.2/38.0` Hz，seed 3 `28.9/24.2/24.6/46.7` Hz。它不支持“均值池化单独丢失频率信息”的解释；但 4160 维对 32 维且训练样本仅 1024，不能作池化因果结论。

冻结 checkpoint 上共享主干任务梯度的跨 SNR/seed 平均 cosine 接近零：presence-Doppler 在 A/D 分别约 -0.003/+0.014；range-cell-Doppler 约 +0.001/-0.006。单组有正负波动，且只计算最终 checkpoint、小 batch，故不能排除训练早期冲突，但无证据支持持续强冲突是主要原因。inference checkpoint 不含训练期 spectrum head，teacher-KL 主干梯度不可复算；KL 标量不能代替该证据。

### 教师可靠度

教师在 -20/-15 dB 的共同开发集检测 AUC 为 0.511/0.666，Doppler RMSE 为 369.59/332.42 Hz；-10 dB 检测 AUC 已为 0.982，但 Doppler RMSE 仍为 286.50 Hz。检测知识与连续频率估计知识的可靠区间不同。由此可指出低 SNR 全图软响应可能不可靠，不能据此断言它造成了学生误差。

## 三、对 E002 候选的复核

R1 后最小的物理候选是 S2：给原 IQ TCN 的 Doppler readout 增加逐通道、逐距离门归一化 lag-one 复相干统计；它的相位直接编码相邻脉冲相位增量，无完整 Doppler FFT。S1 保留四段粗时间信息作为反证，S3 是明确的 FFT 混合候选。候选与匹配非学习基线均写入冻结的 [`D1-E002 protocol`](../../experiments/D1-E002/protocol.md)。

E002 已完成三个训练 seed 和独立盲测。S2 在 0/5/10/20 dB 等权 RMSE 为 23.352 Hz，相对 S0 改善 10.86%，配对 95% 区间 `[-6.338,-0.173]` Hz，3/3 seed 同向；高 SNR range 门通过，但低 SNR Pd 区间下界 -0.0939 未通过 -0.05 非劣界。S1 的主 Doppler 指标恶化 8.02%。S2 仍远落后四倍 FFT 抛物插值 0.104 Hz、APX15 0.122 Hz 和有限网格 profile-ML 0.100 Hz；也未通过公共相位旋转和频移一致性诊断。完整结果见 [`D1-E002_review.md`](D1-E002_review.md)。

因此，S2 是受测纯 IQ 学生中最好的频率候选，但“接入一个物理特征即可同时提高估计并保护检测”的合取假设未通过。此证据不支持继续增加损失项或变体来绕过基础性能门。

## 四、三层结论与下一步

1. **工程：** 诊断路径按协议把 oracle 用作只读对照，物理真值、教师响应和学生输出分开。E002 评估器首次运行曾因配置键读取错误中断；修正后对同一缓存/样本 ID 重放，无重生成或重校准，失败记录保留。
2. **实验科学性：** R1 仅为开发诊断；E002 的三 seed 盲测按冻结门判定，但种子数有限。频率物理一致性、传统基线、检测保护和 range 保护均需合并解读。profile-ML 每 SNR 仅使用预定 128 条 H1，属于有限网格参考。
3. **研究主线：** 高频率残差不是距离门的主要问题；池化与最终任务梯度因果解释未获支持。E002 提供了相干特征有帮助的有限证据，却没有形成可用估计器或检测安全的学生。

**D1-R1 的最小结构建议：** 当时优先验证 S2；该验证已经完成并未通过合取门。现在没有证据支持新的 D1 结构改动。停止当前 AWGN Doppler 结构搜索，保留 S2 的部分收益与完整失败门结果。

**唯一优先任务：** 准备 D2-A“静止杂波下单目标检测与估计”的单因素预注册协议和传统基线/可辨识性检查。冻结原 D1-E001 学生结构作为一个参考，不开展新结构搜索；只引入静止杂波这一场景因素，先验证 H0/H1 生成、三通道观测可辨识性、传统杂波抑制/检测基线与校准方案，再决定是否训练。D1-E001 原 GO → D2 仍有效；D2-A 的协议尚未冻结，不能把本建议写成已运行结果。

### 复核入口

- [`protocol.md`](../../experiments/D1-R1/protocol.md)、[`analysis.md`](../../experiments/D1-R1/analysis.md)、[`metrics.json`](../../experiments/D1-R1/metrics.json)、[`result.csv`](../../experiments/D1-R1/result.csv)
- [`D1-E001 review`](D1-E001_review.md)、[`D1-E002 review`](D1-E002_review.md)
- 实现：[`task_student.py`](../../src/radardistill/students/task_student.py)、[`task_losses.py`](../../src/radardistill/distillation/task_losses.py)、[`d1_evaluation.py`](../../src/radardistill/evaluation/d1_evaluation.py)、[`d1_runner.py`](../../src/radardistill/training/d1_runner.py)
