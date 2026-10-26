# D2-A-T001 分析

**运行代码提交：** `91dfb81673e45cad3f3330d90adc707a0cdf859d`<br>
**研究总纲：** `docs/RESEARCH_CHARTER.md` v1.2（T001 未修改总纲）<br>
**数据：** 新生成的合成 H0/H1 开发数据；不是实测，也不是 final blind。<br>
**候选：** 唯一模型辅助三通道 post-Doppler ACE×AMF。<br>
**候选设备：** `cuda:0`；B3/T1/K2 对照：CPU NumPy (B3, T1, and K2)。<br>

## 冻结阶段门

- G1：FAIL。
- G2：FAIL。
- G3：NOT EVALUATED。
- 是否允许 S2 pilot：禁止。

## 总体开发指标

| 方法 | 验证 Pfa | 单侧 exact 95% 上限 | Pd | AUC | Range RMSE | Doppler RMSE (Hz) | Joint success |
|---|---:|---:|---:|---:|---:|---:|---:|
| B3 | 0.04913 | 0.05213 | 0.5635 | 0.9400 | 3.0956 | 45.271 | 0.5407 |
| T1 | 0.05160 | 0.05467 | 0.9189 | 0.9862 | 3.0956 | 45.271 | 0.8970 |
| ACE_AMF | 0.05007 | 0.05309 | 0.6923 | 0.9128 | 6.4485 | 139.093 | 0.6529 |

## 主要困难条件与知识质量

主要困难条件：`C3_partial_overlap_snr_0_cnr_20`。ACE×AMF 相对最佳旧教师 `B3` 的 joint-success 差及 bootstrap CI：`{'candidate_minus_reference': -0.01, 'ci95': [-0.018, -0.002]}`。
G2 未通过；按预注册协议不评价 soft response 质量，G3 记为 NOT EVALUATED。

## 物理一致性与运行资源

公共相位旋转的 score/map 最大绝对差为 `0.000e+00`。公共复尺度的 score/map 最大绝对差为 `4.578e-05`；冻结 G1 上限为 `1.000e-06`，尺度不变门未通过。
目标单独 Doppler 频移残差（估计频移减预设频移）：
- 频移 -25.0 Hz：残差均值 `+25.0005 Hz`，5 Hz 内比例 `0.0000`；原阈值下检出比例 `0.1328`。
- 频移 -10.0 Hz：残差均值 `+10.0004 Hz`，5 Hz 内比例 `0.0000`；原阈值下检出比例 `0.1328`。
- 频移 +10.0 Hz：残差均值 `-10.0001 Hz`，5 Hz 内比例 `0.0000`；原阈值下检出比例 `0.1328`。
- 频移 +25.0 Hz：残差均值 `-25.0001 Hz`，5 Hz 内比例 `0.0000`；原阈值下检出比例 `0.1328`。

独立 H0 总体 Pfa 点估计/单侧精确上限为 `0.05007/0.05309`；大分组门要求 `12` 个分组的上限均不超过 0.10，门未通过。超过 0.10 的分组：`course_level=C3 (0.125), snr_db=0 (0.127), cnr_db=20 (0.125)`。
可辨识 A/B/C 相对 T1：Pd 差 `-0.2267`（95% CI `[-0.2345020833333333, -0.21883333333333332]`），joint-success 差 `-0.2441`（95% CI `[-0.25216666666666665, -0.23608333333333334]`），Range RMSE 差 `+3.353 cells`，Doppler RMSE 差 `+93.822 Hz`。
CUDA covariance 拟合 `47.95 s`（`85.42` scenes/s）；教师全阶段 `3539.7 s`。候选内核 batch=1 为 CPU `0.016932 s/CPI`、CUDA `0.002288 s/CPI`（`7.40×`）；batch=16 为 CPU `0.019008`、CUDA `0.001217 s/CPI`（`15.63×`）。
CUDA 峰值 allocated/reserved `1233.3/1582.0 MiB`。候选内核计时仅比较复用的 16 CPI 输入，不代表完整链路实时性能；全阶段时间包含 CPU 场景生成、B3/T1/K2 基线和 CPU bootstrap。

## 解释边界

检测阈值由独立 H0 calibration 以 `higher` 经验 95% 分位冻结，并只在独立 H0 verification 评价。开发集仅用于预注册阶段门和分层误差；没有生成 final blind。若任一教师门失败，不运行新蒸馏学生；负结果保留，不追加候选或调参。

精确数值、分组结果、bootstrap 区间、运行资源和物理诊断见 `metrics.json`、`result.csv`、`verification_subgroups.csv`、`physical_diagnostics.json` 与 `environment.json`。G2 未通过，因此本协议未评价 soft response，`soft_teacher_audit.csv` 未产生有效记录。
