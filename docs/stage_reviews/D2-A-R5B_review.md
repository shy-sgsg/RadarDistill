# D2-A-R5-B 阶段自审：TFAS 原始 IQ 时频注意力 pilot

**审查范围：** 冻结的 R5-B protocol/config v1.0、TFAS 实现与 seed 101 E1 truth-only pilot。
**结论：** 阶段总体 **REFINE**；TFAS 候选按预注册 G3 **STOP**。
**研究总纲：** `docs/RESEARCH_CHARTER.md` v1.3，正文未修改。
**独立审查：** 待 ChatGPT 审查；本文件为执行方阶段自审，不替代独立审查。

## 1. 与研究总纲的关系

总纲的核心目标是通过物理仿真真值和传统算法知识训练任务级学生，直接完成 GMTI 检测与参数估计，并在后期评估精度和完整处理效率。R5-B 对应 Q3/Q4/Q6/Q8/Q10，直接检验一个保留原始三通道复 IQ 的时频注意力学生是否能以接近 Medium CNN 的参数量改进任务表现；本实验没有测试教师知识或蒸馏。

实验为一个有界非 CNN 架构对照，不是独立 Transformer 研究，也不证明其他非 CNN 结构无效。候选未通过 G3，当前没有理由继续扩充该结构或借由 KD 掩盖其基本任务保护失败。

## 2. 工程正确性

- TFAS 保留 6 个 I/Q 实虚平面；采用实值分平面表示，没有固定 FFT/相干前端，也没有形式化公共相位不变或频移等变保证。参数量为 79,638，比 Medium 的 82,887 少 3.92%，在预设匹配范围内。
- CUDA preflight 记录 NVIDIA GeForce RTX 3050 Laptop GPU、`cuda:0`、FP32、PyTorch 2.8.0+cu128/CUDA 12.8；训练无 CPU 回退、无 OOM。数据缓存 manifest 为90个文件、14,232,309,192 bytes；冻结 split IDs 与训练 pair plan 审计通过。
- G1 使用16对真实训练场景、200 updates；truth-task loss从5.57683降至0.000867（下降99.984%），presence与range-cell accuracy均为1.0，任务头梯度有限且非零。
- 模型/runner定向及 R5 训练测试合计26项通过；compileall、配置校验与差异空白检查通过。完整运行和资源记录见 R5-B `preflight.json`、`g1_overfit_sanity.json`、`training.json` 与训练history。

## 3. 实验科学性

严格按预注册执行：seed101、E1物理真值监督、12,800 optimizer updates、有效batch64、固定训练集8遍、FP32；未用教师硬标签/软响应、KD或truth-assisted输入。checkpoint按共同validation truth-task loss选择于update 12,400，验证task loss从初始5.46615降至1.71301，但末段已接近平台且未达到可证明收敛的依据。单seed不估计训练seed方差。

同一冻结development上，TFAS AUC=`0.96139`（paired-scene bootstrap 95% CI `[0.95760,0.96512]`）；全H1 range RMSE=`4.595 cells`、Doppler RMSE=`82.00 Hz`，都优于训练真值常值baseline，因此 G2 通过。但这仅证明学到部分基本映射。

G3未通过：

- 总体 Pd=`0.8020`，joint success=`0.5054`；TFAS相对Medium的总体Pd差`−0.17261`（95% CI `[-0.18433,-0.16088]`）、总体joint差`−0.40527`（`[-0.42042,-0.39037]`）。
- 主要端点 C2/C3 joint-success差`−0.60449`（paired 95% CI `[-0.62598,-0.58350]`），而冻结门要求至少`+0.02`且CI下界大于0。
- TFAS C3 H0 verification为`622/4096=0.15186`，单侧exact 95%上限`0.16138`，超过绝对`0.10`门。虽然相对Small/Medium的C3 Pfa分别低`0.05225/0.03711`，但TFAS和两个CNN都没有满足绝对门；不能把相对改善描述为合格检测器。
- C2/C3 range与Doppler RMSE分别为Medium的2.385倍和4.599倍；比值95% CI分别`[2.085,2.814]`与`[4.224,5.032]`。候选的较低C3 Pfa伴随明显的漏检和定位损失。

所有主要对比是同一已使用development ID上的单seed配对场景bootstrap，区间反映场景抽样，不代表seed方差或blind泛化。R5-B未重训 Small/Medium，沿用R5-A capacity metrics；没有新传统基线或教师资格检验。

## 4. 物理一致性与结论边界

诊断复用了 E003 已见 development/verification IDs，仅作描述。公共相位旋转下，H0误警率跨32种旋转为0.02734–0.07813；H1判决一致率中位数0.9258，score变化p95最高2.572，Doppler变化p95最高122.18 Hz，range变化p95最高14.98 cells。整体Doppler频移诊断也未显示稳定等变；partial-overlap组的检测一致率平均0.695。结果与模型缺少严格群等变结构相符，但不能据此确定唯一失败机制。

结论适用范围限于当前三通道、130脉冲、32距离单元、静止杂波合成模型及复用的开发材料。没有真实数据、盲测、跨seed或系统时延证据。不能宣称 TFAS 超过传统方法、CNN路线整体通过、或教师知识将改善它。

## 5. 假设和总体研究意义

- **H-ARCH-003：** 对本轮79.6K TFAS候选 **NOT_SUPPORTED**。在matched训练暴露下，其主困难条件联合任务显著低于Medium CNN且绝对C3 Pfa失败。不得外推至全部非CNN/attention模型。
- **H-LEARN-002：** **INCONCLUSIVE**。TFAS能优于真值常值baseline，但复用开发材料的相位/频移响应不稳，且任务保护失败；没有教师知识吸收或物理机制因果证据。
- **H-CAP-002：** 保留R5-A限定范围内的有限支持。R5-B不覆盖其历史结果。
- 本阶段向总体目标提供有用的负架构证据：单纯把原始I/Q分平面并添加时域/距离注意力，尚未优于matched CNN，也未解决绝对虚警保护。它没有回答可用教师知识是否能提升学生，也没有找到目前可直接晋级的学生。

## 6. 决策与下一步

**阶段建议：REFINE。** 停止当前TFAS候选后续训练；不做额外seed、KD、Large、架构网格或final blind。此决定是冻结G3的直接结果，不是对整个学生研究主题的否定。

**唯一下一步优先任务：** 将本阶段材料送独立 ChatGPT 审查，审查总纲一致性、是否仍值得研究现有学生/教师路线，以及最小高价值后续任务。审查前暂停R5-C/D和新架构训练。若审查认为继续有价值，再明确新的可证伪问题、独立数据/前置门和停止条件并冻结新协议；总纲不变。

## 7. 复核材料

- 预注册：`experiments/D2-A-R5/R5-B/protocol.md`、`config.yaml`、`architecture_survey.md`
- 实现：`src/radardistill/students/temporal_frequency_attention.py`、`scripts/run_d2a_r5_attention.py`
- 测试：`tests/test_temporal_frequency_attention_student.py`、`tests/test_d2a_r5_attention_runner.py`、`tests/test_d2a_r5_training.py`
- 结果：`experiments/D2-A-R5/R5-B/analysis.md`、`metrics.json`、`training.json`、`result.csv`、`roc.csv`、`scene_predictions.csv`、`physical_diagnostics.json`、`figures/rmse_by_snr.png`
- 冻结总纲：`docs/RESEARCH_CHARTER.md` v1.3
