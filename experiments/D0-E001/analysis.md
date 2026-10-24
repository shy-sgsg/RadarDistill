# D0-E001 仿真与 APX15 教师正确性结果

**结果：PASS**。本实验仅验证距离压缩简化模型及慢时间频谱子链，不训练学生，不代表完整 CoreX 生产链或实测数据。

## 正确性门

| 检查 | 结果 |
|---|---|
| complex_awgn_power | PASS |
| counterfactual_h0_h1_background | PASS |
| lfm_matched_filter_psf | PASS |
| native_apx15_fft_axis | PASS |
| noiseless_teacher_doppler_max | PASS |
| noiseless_teacher_doppler_rmse | PASS |
| noiseless_teacher_range_max | PASS |
| noiseless_teacher_range_rmse | PASS |
| nyquist_and_range_boundaries | PASS |
| seed_reproducibility_and_shape | PASS |
| teacher_auc_at_minus_10_db | PASS |
| three_channel_steering | PASS |

## 物理模型与教师核对

- 离散 LFM 匹配滤波 PSF 最大归一化幅度误差：`0.00012662415`。
- 复 AWGN 经验功率：`0.99881278`，相对误差 `0.1187%`。
- 无噪声教师：range RMSE `7.1699882e-10` cell，Doppler RMSE `0.029039694` Hz，最大 Doppler 误差 `0.040522559` Hz。
- Nyquist 邻域/范围端点检查：range 最大误差 `9.9856692e-09` cell，Doppler 最大环绕误差 `0.00081711999` Hz。

## 含噪教师诊断

教师检测分数为细网格峰值/全图中位功率。D0 不从这些样本选择工作阈值；D1 将使用独立 H0 校准集。

| SNR (dB) | H0/H1 数量 | AUC (95% bootstrap CI) | H1 range RMSE (cell) | H1 Doppler RMSE (Hz) |
|---:|---:|---:|---:|---:|
| -20 | 1024/1024 | 0.4867 [0.4622, 0.5109] | 11.294 | 372.26 |
| -15 | 1024/1024 | 0.6443 [0.6197, 0.6674] | 10.735 | 357.42 |
| -10 | 1024/1024 | 0.9886 [0.9841, 0.9926] | 9.1261 | 292.88 |
| -5 | 1024/1024 | 1.0000 [1.0000, 1.0000] | 2.2438 | 78.387 |
| 0 | 1024/1024 | 1.0000 [1.0000, 1.0000] | 0.2589 | 0.2344 |
| 5 | 1024/1024 | 1.0000 [1.0000, 1.0000] | 0.21827 | 0.13264 |
| 10 | 1024/1024 | 1.0000 [1.0000, 1.0000] | 0.14917 | 0.077841 |
| 20 | 1024/1024 | 1.0000 [1.0000, 1.0000] | 0.04028 | 0.036963 |

## 解释边界

通过只说明冻结的理想距离 PSF、噪声采样和 APX15 FFT 布局适配符合本协议。距离轴是理论 `c/(2B)` 分辨单元；本阶段没有验证实际 5 m 距离分辨率字段、完整快时间脉压链、杂波、多目标、通道误差或 CoreX 执行。教师 AUC 是 AWGN 单目标开发诊断，不是运行检测阈值或系统虚警结论。
