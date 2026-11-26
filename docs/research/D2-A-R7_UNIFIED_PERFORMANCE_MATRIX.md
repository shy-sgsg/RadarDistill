# D2-A-R7：模型、教师与传统方法统一性能总表

**用途：** 回答“各模型在不同数据集上的检测和参数估计能力如何，教师又如何”，并保留每个数值的实验协议、原始指标路径和适用范围。

本阶段报告已将此矩阵作为正式结果部分：见[D2-A-R7综合阶段报告](../stage_reviews/D2-A-R7_review.md)，其中提供R7-D逐SNR检测、Range/Doppler估计与Joint摘要，以及教师正确性/教学结果。

## 完整机器可读表

完整表为单张长格式CSV：[下载/打开完整矩阵](../../experiments/D2-A-R7/unified_performance_matrix.csv)。它纳入D0/D1可追溯结果（包括D1-R1、D1-E001和D1-E002）、D2-A-E001/R3/R4/E002/E003/T001、R5/R6历史结果、R7-B物理教师诊断、R7-D共同L0/L1复评，以及R7-E明确尚未运行/不适用的方法。R5/R6原有60行综合记录以原文JSON保留，同时增加可追溯的详细指标记录。

CSV的基本粒度是“一个指标对象 × 方法 × 数据划分/条件 × checkpoint”，并不是去重后的独立样本表。相同方法/条件可能有不同seed、initial/best checkpoint、阈值校准/verification、总体/子组或历史汇总行；这些行通过`source_pointer`和`metric_payload_json`区分，不能相加成独立场景数。不同阶段测试集、SNR定义和阈值协议不同，不能将整张表视为一个总排行榜。

## 字段和指标解释

- 身份与来源：`study`、`experiment`、`dataset`、`split`、`record_grain`、`metric_group`、`condition_key`、`source_metrics`、`source_protocol`、`source_analysis`、`source_git_sha`、`source_pointer`。
- 方法与训练：`method_id/name/role`、`task_scope`、`seed`、`checkpoint_role/path/SHA256`、参数量、训练更新数、可追溯训练场景暴露量与训练seed数量。缺失训练元数据明示`NOT_RECORDED...`。
- 条件：`scenario_level`仅在映射到R7 L0–L6时使用；历史C0–C3困难度不强行映射。`snr_db`/`cnr_db`与对应定义并列，未嵌入历史指标时指回来源协议。
- 检测：`false_alarm_unit`严格保留CPI/单元/其他统计单位；`pfa_cpi`、虚警CPI计数、H0分母与置信区间分开。检测器若无dense CUT输出，`pfa_cell`是`NOT_APPLICABLE...`或`NOT_REPORTED...`，不从Pfa_CPI换算。
- 参数估计/教师：range/Doppler RMSE区分全H1和检测命中样本；joint success保持来源协议定义。教师定位率、响应lift与DPCA相干/残余/抑制量是独立指标，不能当作学生Pd或蒸馏收益。
- 空值：`NOT_RUN`表示没有运行；`NOT_APPLICABLE`表示方法/接口不产出该指标；`MISSING_CHECKPOINT`表示所需checkpoint缺失；`NOT_VERIFIED`表示缺少该资格对应的独立统计证据。只有实际数字填数字；不会以零代替缺项。

## 数据集和可比性

R7-D新数据包括L0（无噪声目标）以及L1（无杂波复AWGN），每SNR独立H0校准、H0验证、H1均为2,048 CPI。L1的SNR定义为目标参考振幅平方除以连续距离PSF峰值处复噪声功率；每方法、每SNR单独校准Pfa_CPI=0.01。R7-D的稀疏存在分数不能报告Pfa_cell；profile-ML使用每SNR预先固定的首256个H1。R7-B是单独的静止杂波物理分量诊断，不是检测排行榜。历史各阶段保持原数据切分、工作点和false-alarm单位。

## 本轮R7-D复评摘要

R7-D已完成47/47个历史checkpoint和7种传统参考的同场景复评，模型失败0个。L1覆盖−20至20 dB的8个SNR；每SNR分别有2,048个H0校准、2,048个独立H0验证和2,048个H1样本。R7-D评估使用AISAR环境的RTX 3050 Laptop GPU，且没有重新训练历史模型。

逐方法、逐SNR的Pd/Pfa_CPI/AUC、距离与Doppler RMSE、Joint、样本数、置信区间、checkpoint元信息和原始来源指针均在[完整CSV](../../experiments/D2-A-R7/unified_performance_matrix.csv)中。该表现在共28,058行，其中R7-D有485行，包含47个模型、7种传统参考及L0/L1适用记录；还包括历史D0/D1与D2-A阶段数据、R7-B物理教师诊断和R7-E未运行/不适用方法。R7-D的`Pfa_cell`均为不适用/未验证，因为这些方法未输出逐CUT分数图。

每SNR的最佳学生与传统参考摘要、估计误差和教师证据见[R7-D结果分析](../../experiments/D2-A-R7/R7-D/analysis.md)。摘要中的最佳值是逐指标对47个checkpoint进行的描述性筛选，不能代替配对统计检验或最终盲测。
