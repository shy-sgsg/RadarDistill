# S2：L0/L1模型与传统方法完整能力结果

## 数据覆盖与表格定义

冻结R7-D共同场景复评覆盖47个历史checkpoint和7种传统参考，共形成 **485条实际能力记录**：L0为53条，L1为432条。L0是无噪声、无杂波的目标单独映射诊断；profile-ML按原协议没有生成L0输出，因此不补造。L1是无杂波复AWGN，每方法、每SNR分别有2,048个H0 calibration、2,048个独立H0 verification和2,048个H1；profile-ML每SNR固定使用首256个H1。检测阈值按方法和SNR独立校准到目标Pfa_CPI 0.01。

**用户所要求的完整模型×数据集表在此：** [`all_models_l0_l1_capability_matrix.csv`](./all_models_l0_l1_capability_matrix.csv)。一行对应实际方法、L0/L1数据集、split和SNR；字段同时列出检测(Pfa_CPI/Pd/AUC)、距离与周期Doppler估计、联合成功、全部H1和检测命中子集误差、样本数、置信区间、checkpoint/训练身份与来源指针。无检测输出的参数专用方法会明确标为不适用。可读索引 [`model_capabilities.md`](./model_capabilities.md) 提供方法实例、表格导航与图表。

## 固定同场景的结果观察

| SNR | 受测学生最高Pd | 传统最高Pd | 学生最低Range RMSE | 传统最低Range RMSE | 学生最低Doppler RMSE | 传统最低Doppler RMSE | 最高学生/传统Joint |
|---:|---|---|---|---|---|---|---|
| −20 dB | R6-D J1 detection 0.0190 | B0 FFT 0.0166 | R6-E RAX 10.01 cell | profile-ML 10.14 cell (n=256) | R5-C TaskStudent 346.85 Hz | lag-one 366.75 Hz | 0.0005 / 0.0000 |
| −15 dB | D1-E001 TCN 0.0151 | B1三通道非相干FFT 0.0264 | R6-J4 9.81 cell | profile-ML 8.40 cell (n=256) | R5-C TaskStudent 348.24 Hz | profile-ML 326.18 Hz (n=256) | 0.0039 / 0.0107 |
| −10 dB | R6-J4 0.2021 | B1三通道非相干FFT 0.1621 | R6-J4 8.15 cell | profile-ML 1.00 cell (n=256) | R6-J4 297.96 Hz | profile-ML 48.34 Hz (n=256) | 0.1919 / 0.1597 |
| −5 dB | R6-J4 0.6509 | B1三通道非相干FFT 0.5049 | R6-J4 5.22 cell | profile-ML 0.152 cell (n=256) | R6-J4 194.89 Hz | profile-ML 0.432 Hz (n=256) | 0.6460 / 0.5049 |
| 0 dB | D1-E002 S1 0.8213 | B1三通道非相干FFT 0.8882 | D1-E001 TCN 1.11 cell | profile-ML 0.145 cell (n=256) | D1-E002 S2 113.85 Hz | profile-ML 0.247 Hz (n=256) | 0.7852 / 0.8882 |
| 5 dB | D1-E001 TCN 1.0000 | B1三通道非相干FFT 0.9990 | D1-E002 S2 0.209 cell | profile-ML 0.140 cell (n=256) | D1-E002 S2 54.04 Hz | profile-ML 0.131 Hz (n=256) | 0.8770 / 0.9990 |
| 10 dB | D1-E001 TCN 1.0000 | B0 FFT 1.0000 | D1-E002 S2 0.196 cell | profile-ML 0.150 cell (n=256) | D1-E002 S2 38.83 Hz | profile-ML 0.072 Hz (n=256) | 0.9448 / 1.0000 |
| 20 dB | D1-E001 TCN 1.0000 | B0 FFT 1.0000 | D1-E002 S2 0.176 cell | profile-ML 0.149 cell (n=256) | D1-E002 S2 49.31 Hz | profile-ML 0.022 Hz (n=256) | 0.9746 / 1.0000 |

表中每列的最优方法可能不同，profile-ML只有预定的256个H1；它是逐指标描述性摘要，不是多重比较校正后的正式胜者选择。Pd需与同一方法/同一SNR的Pfa_CPI和区间一起读取；完整行在主能力表中。

## 结果解释

1. **检测：** 在−10/−5 dB，R6-J4固定DPCA/FFT前端加学习读出取得最高受测学生Pd与Joint点值；它是混合方法。0 dB以上，传统FFT检测总体更强。低SNR各法Pd整体较低。
2. **距离与Doppler：** 高SNR下受测学生Range RMSE最佳约0.18–0.21 cell；profile-ML为0.14–0.15 cell（n=256）。Doppler方面，学生最佳点仍约39–54 Hz（5–20 dB），而FFT细化约0.08–3 Hz，profile-ML约0.131、0.072、0.022 Hz（n=256）。受测网络对频率精估仍有很大差距。
3. **检测与估计不能合成单一排序：** R6-J4可在部分SNR提供较高检测/Joint，但其Doppler误差、最难C3格表现和虚警条件保护不佳；传统参数专用方法的精准估计也不包含检测能力。
4. **误差分母：** 表7同时给全部H1和检测命中H1误差。全部H1是完整任务能力主口径；命中子集可帮助诊断估计头，但会受到不同检测器筛选出的样本不同影响。二者不可互相替代。
5. **L0限制：** L0用于目标单独映射/数值诊断，不校准阈值，不报告部署Pfa或随机Pd工作点。

## 结果边界

这是历史checkpoint的新共同场景复评，不是重新训练、跨seed确认或最终blind。L1无杂波，不代表静止/复杂杂波。每SNR独立校准不代表未知SNR下一个全局阈值。所有R7-D检测器只有CPI标量输出，`Pfa_cell`不适用/未验证。本表内参数散点跨阶段预算和数据分布，不应作模型大小因果解释。矩阵记录485条，但独立场景数按protocol计数，不能由行数推断。

## 随附图表

能力总表及表格链接集中在 [`model_capabilities.md`](./model_capabilities.md)。八幅图分别展示Pd、Pfa_CPI、Range/Doppler RMSE、Joint、L0/L1对照、参数量与性能、全方法能力热图；图7为描述性跨阶段图，热图按指标列归一颜色但标注原值。
