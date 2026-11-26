# ChatGPT 独立阶段审查请求：D2-A-R7-S

请首先读取本次Git提交对应版本的 `docs/RESEARCH_CHARTER.md`，理解RadarDistill的整体研究主题、研究目标、研究内容、科学问题和技术路线，然后结合这些内容审查本阶段的代码、实验、数据和研究结论。

## 审查基准

- **结果/报告审查SHA：** `415fa10020ad601866d72b34ca6c3f4ad859da3f`
- **该SHA中的研究总纲：** [`docs/RESEARCH_CHARTER.md`](../../docs/RESEARCH_CHARTER.md)，v1.4；文件SHA-256 `903ba06b97bf02c1fa5b15aa012f049e5830ebd7607342338ffd2e6a0290bea9`。该文件在R7-S中保持不变。
- **结果基线分支：** `main`
- **工作区状态：** 不干净。工作区有与R7-S无关的其他暂存和未跟踪用户文件；它们不在审查SHA的R7-S提交中，也未被此次结果提交收录。请严格按上述SHA及其父提交审查R7-S变更，不以当前工作树覆盖提交内容。
- **研究任务：** D2-A-R7-S跨阶段证据综合和统一性能矩阵审查；本轮最终阶段结论为REFINE。

## 研究总纲对应关系

本阶段服务于总纲的GMTI任务级学生目标：分别审查检测、距离/Doppler估计、虚警统计、传统教师物理与任务能力、知识质量和学生教学收益。重点关联Q1–Q10，尤其Q1/Q2/Q9教师知识与迁移、Q3/Q4学生表示及物理先验、Q6检测与连续参数精度、Q8传统计算取舍、Q10容量/预算/知识吸收。

R7-S不训练、不蒸馏、不执行新的阈值搜索、不生成final blind；它重建和审查现有证据，判断哪些结论可以成立、哪些仍未验证，以及下一项最有价值的研究任务。没有修改研究总纲、历史实验协议、冻结结果或checkpoint。

## 最重要的结果与审查焦点

1. 28,058行/76列跨阶段统一矩阵独立审计：重复全行0、重复记录身份组0；27,661个JSON、328个CSV、69个Markdown来源指针全部解析；47/47个R7-D checkpoint SHA匹配；最终未解决finding=0。
2. R7-D共同新场景能力表有485条实际L0/L1记录，覆盖47个历史checkpoint和7种传统方法。L1无杂波AWGN每方法/SNR为2,048 H0 calibration、2,048独立H0 verification、2,048 H1；profile-ML预先固定每SNR 256个H1。该表含Pd、Pfa_CPI、AUC、Range/Doppler RMSE、Joint、全H1/检测命中误差、样本量、区间和来源。
3. 当前矩阵没有数值型Pfa_cell结果；R7-D检测方法仅输出CPI标量。所有历史Pfa_CPI不可换算为cell级低虚警资格，`Pfa_cell≈10^-6`仍NOT_VERIFIED。
4. R7-D中学生优势依SNR和任务变化：R6-J4固定DPCA/FFT+学习读出在−10/−5 dB受测学生检测/Joint点值较高，但Doppler保护和最难C3条件虚警/joint保护失败；中高SNR FFT/profile-ML参数估计显著更准确。profile-ML仅n=256，其他大部分方法n=2,048。
5. R3识别了H0分数随潜在目标参考幅度改变尺度的nuisance；归一化T1/T2通过部分尺度与H1保护，但未过固定阈值G2。R6-I单斜率校正降低部分pooled C3 Pfa却损害整体Pd且C3分层仍失败。
6. B3 DPCA仅在声明的合成几何下具物理组件正确性支持；K1/K3/K2-local有部分教师响应质量证据，但困难C3重叠条件正确定位仅约2–4%。R5-C与R6-G matched教学均未证明K3/K2-local相对truth-only带来预设学生收益；T3是真值辅助控制，不是蒸馏。
7. 7kg工程参考未获得版本、许可、输入适配和正确性所需证据，未进行数值复评。R7-E列出的其它高级方法为未运行/不适用状态。
8. R5-A训练预算/容量结果有有限开发集收益，R6-C/R6-G/R6-H及R7-D证据都没有证明普遍模型容量上限、最优参数量或普遍最优架构；不得把跨阶段参数量散点作因果结论。

请重点判断：R7-S的证据整合与范围限定是否可靠；“没有cell级资格、现有K3教学未获益、暂不增加学生训练”的结论是否被证据支持；当前唯一优先建议——先建立L1 dense CUT输出和独立H0统计路径，再重审G5——是否比继续架构或教师搜索更有信息价值。请主动指出可能改变整体研究方向的证据或替代路线。

## 用户要求的性能总表

- **跨阶段主表（学生、教师、传统方法及各数据集/阶段）：** [`experiments/D2-A-R7/unified_performance_matrix.csv`](../../experiments/D2-A-R7/unified_performance_matrix.csv)，28,058行；同一文件内的来源字段保留跨协议边界，不可将所有行拼成一个排名。
- **共同条件逐模型能力表：** [`experiments/D2-A-R7-S/S2/all_models_l0_l1_capability_matrix.csv`](../../experiments/D2-A-R7-S/S2/all_models_l0_l1_capability_matrix.csv)，485行；逐个L0/L1、split、SNR列出检测、距离和Doppler估计、Joint、H1分母及来源。
- **模型身份/参数/训练记录：** [`table_1_model_identity_training.csv`](../../experiments/D2-A-R7-S/S2/table_1_model_identity_training.csv)。训练暴露、有效batch或架构未能从原记录确认的行明示`NOT_RECORDED`，不得推断填补。
- **完整可读索引、九个指标表与八幅图：** [`model_capabilities.md`](../../experiments/D2-A-R7-S/S2/model_capabilities.md)。
- **教师资格矩阵：** [`teacher_qualification_matrix.csv`](../../experiments/D2-A-R7-S/S5/teacher_qualification_matrix.csv)。分别记录实现/物理、任务能力、知识质量、学生迁移和虚警资格。

## 阶段综合报告与审计材料

- **综合报告：** [`docs/stage_reviews/D2-A-R7-S_review.md`](D2-A-R7-S_review.md)
- **研究问题Q1–Q10证据矩阵：** [`charter_question_evidence_matrix.csv`](../../experiments/D2-A-R7-S/S6/charter_question_evidence_matrix.csv)
- **主矩阵审计：** [`S1 analysis`](../../experiments/D2-A-R7-S/S1/analysis.md)、[`matrix_audit.json`](../../experiments/D2-A-R7-S/S1/matrix_audit.json)、[`source_validation.csv`](../../experiments/D2-A-R7-S/S1/source_validation.csv)
- **参数与训练证据：** [`D2-A-R7-S_PARAMETER_EVIDENCE_REVIEW.md`](../research/D2-A-R7-S_PARAMETER_EVIDENCE_REVIEW.md)
- **虚警资格：** [`S4 analysis`](../../experiments/D2-A-R7-S/S4/analysis.md)、[`false_alarm_evidence_audit.csv`](../../experiments/D2-A-R7-S/S4/false_alarm_evidence_audit.csv)
- **教师资格：** [`D2-A-R7-S_TEACHER_QUALIFICATION.md`](../research/D2-A-R7-S_TEACHER_QUALIFICATION.md)、[`teacher_qualification_matrix.csv`](../../experiments/D2-A-R7-S/S5/teacher_qualification_matrix.csv)
- **动态状态与路线：** [`CURRENT_STATUS.md`](../CURRENT_STATUS.md)、[`RESEARCH_ROADMAP.md`](../RESEARCH_ROADMAP.md)、[`ASSUMPTION_REGISTER.md`](../research/ASSUMPTION_REGISTER.md)、[`decisions.md`](../decisions.md)、[`progress.md`](../progress.md)

## 代码和复核

- 矩阵构建：`python scripts/build_d2a_r7_unified_performance_matrix.py`
- 独立来源/数据审计：`python scripts/audit_d2a_r7s_matrix.py`
- L0/L1能力表、训练身份表和图表：`python scripts/build_d2a_r7s_l0_l1_report.py`
- 定向测试：`pytest -q tests/test_d2a_r7_performance_matrix.py tests/test_d2a_r7s_matrix_audit.py tests/test_d2a_r7s_l0_l1_report.py`；本次 **22 passed**。
- 已运行`py_compile`及`git diff --check`；没有重新训练、蒸馏、数据生成或低虚警实验。

## 请求审查

请按 `docs/governance/CHATGPT_REVIEW_PROTOCOL.md` 的四层流程检查工程正确性、实验科学性、对研究总纲的推进和战略替代方向。请区分已测、未运行、不适用、推断和开发结果；核对用户所需总表中的检测/估计/教师口径、全H1与检测子集误差、profile-ML样本量及历史Pfa单位。

最终只给出一个总判断：**CONTINUE / REFINE / REDIRECT / STOP**。请说明是否支持当前REFINE结论、唯一下一项优先任务、最小可证伪方案、会改变判断的证据、停止条件、研究总纲若需修订时应提交用户讨论的具体建议。阶段门通过不自动代表长期研究方向正确。
