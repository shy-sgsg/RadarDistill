# D1-E002 阶段审查：受控 Doppler 读出比较

**日期：** 2026-10-25
**决策：** **STOP / REDIRECT**
**冻结协议：** [`experiments/D1-E002/protocol.md`](../../experiments/D1-E002/protocol.md)
**完整结果：** [`experiments/D1-E002/blind/metrics.json`](../../experiments/D1-E002/blind/metrics.json)、[`result.csv`](../../experiments/D1-E002/blind/result.csv)、[`analysis.md`](../../experiments/D1-E002/blind/analysis.md)

## 问题、假设与边界

D1-R1 的验证诊断显示，高 SNR Doppler 误差在真值距离门下仍然存在。E002 在冻结 D1-A TCN 上比较四段时间读出 S1 与逐通道归一化 lag-one 复相干读出 S2。假设是保留时间位置或显式相干量可改善连续 Doppler 读出，同时满足低 SNR 检测与距离非劣门。

信号仍为 PRF=1300 Hz、N=130、三通道单目标复 AWGN；目标频率在 `[-650,650) Hz`，距离在 `[4,28)`。未加入杂波、多目标、通道失配或加速度。S1/S2 formal 候选仅用 E1 仿真真值监督；未训练 E2/E3。S3 FFT 辅助混合模型未通过 pilot 屏选，未进入 formal。

## 数据与执行

- S1/S2 各完成三个独立训练种子、每个 1600 次更新，RTX 3050 Laptop GPU、FP32、有效 batch 64、按种子顺序训练；12 个正式/历史 checkpoint 在生成盲测前冻结。
- 使用独立 50,000-H0 calibration；14 个分数的经验 Pfa 均为 0.00998。盲测每 SNR 4096 H0 与 1024 H1，8 个 SNR 共 32,768 H0 和 8,192 H1。
- 盲测只作最终评价。profile-ML 每 SNR 采用协议预定的前 128 个 H1；它是 D1 冻结实现的有限网格数值参考，不宣称连续全局 ML。
- 初次盲测已保存样本及多数方法预测，但 evaluator 在 profile-ML 距离区间配置路径处以 `KeyError: 'waveform'` 中断。提交 `e7404f6` 修正 D0 波形区间读取后，用同一缓存与 IDs 补算 profile-ML 并完成汇总。保留初次失败清单与 partial metrics；盲测没有重生成，模型、阈值、指标和门槛没有变化。

## 冻结阶段门

主终点为 0/5/10/20 dB 等权平均 Doppler RMSE；候选需改善至少 10%、配对层级 bootstrap 95% 区间上界小于零、至少 2/3 seed 同向，并同时通过低 SNR Pd 与高 SNR range 非劣门。

| 候选 | 高 SNR Doppler RMSE（Hz） | 相对 S0 改善 | Doppler 差值 95% CI（Hz） | seed 方向 | 低 SNR 检测门 | range 门 | 总门 |
|---|---:|---:|---:|---|---|---|---|
| S1 | 28.298 | −8.02% | [0.378, 4.622] | 0/3 | FAIL，CI 下界 −0.0554 | PASS，Δ=+0.0104 cell，CI [−0.0069,+0.0227] | FAIL |
| S2 | 23.352 | +10.86% | [−6.338, −0.173] | 3/3 | FAIL，CI 下界 −0.0939 | PASS，Δ=+0.0155 cell，CI [−0.0128,+0.0321] | FAIL |

S2 的低 SNR Pd 点估计相对 S0 在四个预设 SNR 均非负，但 paired 95% CI 下界未达 −0.05。主要 seed 波动来自 −5 dB：S2 seed_2 的 Pd 为 0.471，而 seed_1/3 为 0.837/0.864。按冻结的区间门，不能以点估计代替不确定性门槛。

## 估计精度与传统基线

下表为 0/5/10/20 dB 等权 Doppler RMSE。学生项先在每个 SNR 对三个 seed 求均值，再对四个 SNR 求均值；profile-ML 的每个 SNR 只有 128 个 H1。

| 方法 | Doppler RMSE（Hz） | 评价范围 |
|---|---:|---|
| S0 真值监督锚点 | 26.197 | 每 SNR 1024 H1，3 seed |
| S1 四段时间读出 | 28.298 | 每 SNR 1024 H1，3 seed |
| S2 lag-one 相干读出 | 23.352 | 每 SNR 1024 H1，3 seed |
| 非学习 lag-one | 3.743 | 每 SNR 1024 H1 |
| 原生 130 点 FFT peak | 2.860 | 每 SNR 1024 H1 |
| 原生 FFT 三点插值 | 1.494 | 每 SNR 1024 H1 |
| 四倍 FFT peak | 0.738 | 每 SNR 1024 H1 |
| 四倍 FFT 三点抛物插值 | 0.104 | 每 SNR 1024 H1 |
| APX15 冻结教师 | 0.122 | 每 SNR 1024 H1 |
| profile-ML 有限网格参考 | 0.100 | 每 SNR 固定 128 H1 |

S2 比 S0 有统计支持的相对改善，但离传统精细估计仍很远。0/5/10/20 dB 的条件 CRLB 平均约 0.098 Hz；S2 平均误差约为其 238 倍。S2 20 dB RMSE 为 22.65–28.58 Hz，反而高于 10 dB 的 16.67–22.75 Hz。精度增益不能解释为网络接近 CRLB 或胜过传统频谱方法。

## 物理一致性与误差诊断

- S2 三个 seed 对公共相位旋转 `2.83 rad` 的平均 Doppler 输出变化为 11.17–14.62 Hz，存在明显公共相位敏感性。
- 对 `±0.625 Hz` 的频率平移，等变残差均值为 2.94–4.18 Hz；对 `±5 Hz` 为 9.79–14.73 Hz。加入 lag-one 复相干特征并未自动赋予网络频移等变性。
- `metrics.json` 保存了 Doppler/距离四分位、正负频率、Nyquist 25 Hz 邻域、初相八分箱、幅度四分位和逐 SNR 结果。Nyquist 邻域样本较少，分组结论应结合其 `n` 阅读。
- 校准目标为 `Pfa=1e-2`。盲测 H0 每 SNR 为 4096；不据此声称达到 `Pfa=1e-6`。经典 FFT/lag-one 频率估计器共用协议定义的非相干能量检测分数，其 Pd 不归因于频率估计器。

## 阶段结论

S1 未通过 Doppler 主门；S2 通过 Doppler 与 range 门，但低 SNR 检测区间门失败。由于冻结验收是合取条件，没有纯 IQ 候选完整通过，故本阶段为 **STOP / REDIRECT**，按协议停止当前结构搜索。S2 是本轮学生中最好的 Doppler 候选，但不构成可替代 FFT 插值、APX15 或 profile-ML 的证据。

E002 没有新增 E2/E3 蒸馏训练。历史 S0_D checkpoint 仅是次要参考，不能用于推断本实验的蒸馏因果收益；D1-E001 的 G1/G2/G3 结论保持不变。任何后续模型或蒸馏研究都应另行提出可证伪假设、冻结预算和门槛，并使用新独立盲测。

## 复核入口

协议与配置：[`protocol.md`](../../experiments/D1-E002/protocol.md)、[`config.yaml`](../../experiments/D1-E002/config.yaml)。逐 SNR 数值见 `blind/result.csv`，完整统计和分组见 `blind/metrics.json`，曲线见 `blind/rmse_snr.png` 与 `blind/detection_roc.png`。修正后的盲测重放记录位于 `blind/run_manifest.json`，初次失败位于 `blind/run_manifest_initial_failure.json` 与 `blind/metrics_partial.json`。

本阶段实现回归验证命令：

```bash
conda run --no-capture-output -n AISAR python -m pytest -q tests/test_d1_e002_evaluation.py
```

结果：**10 passed**。冻结盲测的重放入口为 `python scripts/replay_d1_e002_blind.py --batch-size 64`；由于结果已存在，该入口会拒绝覆盖完成的盲测产物。
