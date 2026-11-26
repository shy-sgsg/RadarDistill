# D2-A-R7-S S2：无杂波基础能力总表与图表

主表记录 **485 条实际结果**：L0 53 条，L1 432 条；覆盖47个历史checkpoint和7种传统方法。R7-D原 protocol 没有为 profile-ML 生成L0结果，故不补造该行。

这张CSV是用户要求的逐方法、逐数据集能力主表；检测、参数估计、全H1/检测命中子集、样本数、区间、训练/checkpoint身份和来源均保留。跨阶段原协议结果请同时参阅[完整统一性能矩阵](../../D2-A-R7/unified_performance_matrix.csv)。

L0为无噪声、无杂波目标单独场景（每方法2,048 H1），仅作映射诊断，不报告Pfa/Pd工作点。L1为无杂波复AWGN；每SNR每方法有2,048 H0校准、2,048独立H0验证和2,048 H1，Pfa目标按每SNR校准为0.01。profile-ML固定使用每SNR前256个H1；检测专用或参数专用方法的非适用指标留空/写明状态。

## 固定方法实例快览

以下列出同一方法在0和10 dB的行，避免把不同模型的单点最佳值拼成一个虚构方法。完整模型/教师结果以主CSV和跨阶段矩阵为准。

| 方法 | SNR | n H1 | Pfa_CPI | Pd | AUC | Range RMSE (cell) | Doppler RMSE (Hz) | Joint success |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| R6-J4 J1-J raw-IQ student | 0.0 | 2048 | 0.009766 | 0.4673 | 0.774 | 6.782 | 257.2 | 0.1006 |
| R6-J4 J1-J raw-IQ student | 10.0 | 2048 | 0.005859 | 0.9355 | 0.9844 | 1.066 | 262.1 | 0.373 |
| R6-J4 J4 fixed-DPCA/FFT hybrid student | 0.0 | 2048 | 0.007324 | 0.7881 | 0.9078 | 3.809 | 150.8 | 0.7852 |
| R6-J4 J4 fixed-DPCA/FFT hybrid student | 10.0 | 2048 | 0.01025 | 0.9453 | 0.9771 | 2.318 | 78.84 | 0.9448 |
| B1 FFT detector + 4× zero-pad estimator | 0.0 | 2048 | 0.01074 | 0.8882 | 0.9881 | 0.2391 | 0.2846 | 0.8882 |
| B1 FFT detector + 4× zero-pad estimator | 10.0 | 2048 | 0.01758 | 1 | 1 | 0.2085 | 0.07613 | 1 |
| lag-one phase estimator (parameter-only) | 0.0 | 2048 | — | — | — | 0.2391 | 60.64 | — |
| lag-one phase estimator (parameter-only) | 10.0 | 2048 | — | — | — | 0.2085 | 5.481 | — |
| profile-ML (parameter-only; 256 H1) | 0.0 | 256 | — | — | — | 0.1451 | 0.2466 | — |
| profile-ML (parameter-only; 256 H1) | 10.0 | 256 | — | — | — | 0.1502 | 0.07194 | — |

参数RMSE针对全部H1；检测命中H1子集误差另见表7。检测专用/参数专用方法无关指标不参与排名。

## 九个表格

1. [table_1_model_identity_training.csv](./table_1_model_identity_training.csv)

2. [table_2_l0_all_methods.csv](./table_2_l0_all_methods.csv)

3. [table_3_l1_detection_by_snr.csv](./table_3_l1_detection_by_snr.csv)

4. [table_4_l1_range_estimation_by_snr.csv](./table_4_l1_range_estimation_by_snr.csv)

5. [table_5_l1_doppler_estimation_by_snr.csv](./table_5_l1_doppler_estimation_by_snr.csv)

6. [table_6_l1_joint_by_snr.csv](./table_6_l1_joint_by_snr.csv)

7. [table_7_l1_all_vs_detected_errors.csv](./table_7_l1_all_vs_detected_errors.csv)

8. [table_8_l1_traditional_references.csv](./table_8_l1_traditional_references.csv)

9. [table_9_method_comparability.csv](./table_9_method_comparability.csv)

## 八幅图

- [Pd–SNR](./figure_1_pd.png)
- [Pfa_CPI–SNR](./figure_2_pfa_cpi.png)
- [Range RMSE–SNR](./figure_3_range_rmse_cells_all_h1.png)
- [Doppler RMSE–SNR](./figure_4_doppler_rmse_hz_all_h1.png)
- [Joint success–SNR](./figure_5_joint_success.png)
- [L0与L1对照](./figure_6_l0_vs_l1.png)
- [参数量与性能](./figure_7_parameter_count_vs_performance.png)
- [全方法能力热图](./figure_8_model_family_capability_heatmap.png)

代表性曲线按阶段/方法类别预先固定，不按本图的最高分挑选；完整逐行结果见主CSV。图7是跨训练预算、数据分布和任务臂的描述性散点，不能作参数量因果结论。热图颜色只在各列内部归一化，格中数字为原始指标。
