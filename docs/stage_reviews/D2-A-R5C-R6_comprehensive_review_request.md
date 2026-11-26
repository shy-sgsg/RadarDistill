# ChatGPT 综合阶段审查请求：D2-A-R5-C 与 R6

请首先读取本次Git提交对应版本的 `docs/RESEARCH_CHARTER.md`，理解RadarDistill的整体研究主题、研究目标、研究内容、科学问题和技术路线，然后结合这些内容审查本阶段的代码、实验、数据和研究结论。

## 审查基准

- **审查结果基准提交SHA：** `bf99c3bf268a1c136bd082b4be8f481918bf1d58`（`main`；含本报告、R6-J4指标和阶段报告）。本审查请求文件在后续单独的文档提交中；涉及R6-J4的冻结训练/评价实现与指标记录绑定先前冻结提交`e6b17e1e8d01b66457d9889db6d9672efb02ae7f`，阶段报告说明了之后仅绘图映射的修正。
- **工作区状态：** 审查基准提交的跟踪文件完整；本地另有未跟踪的用户源码包及历史实验进度/数据，均未纳入审查提交。可重建训练IQ和验证IQ缓存未提交，已按R6-J4 owner manifest清理；R6-J4训练记录、best checkpoint、metrics、分析、G0/G1与清理审计已提交。
- **研究总纲路径及版本：** `docs/RESEARCH_CHARTER.md`，v1.3（审查基准提交中的版本）。执行中未修改总纲。
- **阶段：** D2-A-R5-C教学pilot与D2-A-R6整合路线，含R6-A至R6-J4已完成候选筛选。
- **路线结论：** 当前综合判断为 **REFINE**。这表示已证明若干学生可以从合成物理真值学习任务信息，但没有候选同时通过C3逐条件虚警、连续Doppler和joint任务保护；现有K3/K2-local也未被证明带来truth-only之外的学生收益。

## 与总体研究目标的关系

本阶段服务于总纲中“利用物理仿真真值与传统算法知识，训练直接完成GMTI目标检测与参数估计的学生，并评价其精度和完整处理能力”的目标。重点对应Q1/Q2（教师知识是否改善学生及哪些知识有效）、Q3/Q4（学生表示和物理归纳偏置）、Q6（困难目标/背景能力）以及Q9/Q10（教师能力、学生能力与教学机制）。

请独立判断：本阶段实际推动了总体目标多少；当前低虚警和困难C3问题是否仍值得研究；仅依赖当前开发集扩展架构是否还有研究价值；是否有更高价值的可证伪替代方向；阶段门通过但没有P7候选晋级，是否支持调整近期研究思路。

## 实验假设、协议和主要结果

R5-C对照T0 truth-only、T1 truth+K3完整RD响应、T2 truth+K2-local响应及T3匹配truth-assisted监督。R5-C教师审计中K3在可辨识C2+C3的top-cell定位率为81.22%，但在C3 partial-overlap / SNR=0 dB / CNR=20 dB困难格仅4.10%。R5-C T1/T2没有优于T0的最终joint任务表现。R6-G再次比较T0 truth-only、T1 truth+K3及T3真值辅助控制：T1−T0的C2+C3 joint差为−0.00130（95% CI `[-0.00781,+0.00521]`），未证明教师增益；T3−T0为+0.01172，低于预设+0.02收益门，且T3不是教师蒸馏证据。

R6路线覆盖同场景教师/学生低虚警基线、困难H0诊断、模型规模、检测/参数/联合任务、代表性CNN/Transformer/复数SSM/CNN-SSM架构、完整传统/混合处理链、K3教学及可观测尺度分数校正。不同阶段数据划分、预算和模型不同，不能把跨阶段点估计拼成单一排行榜。完整60行R5/R6结果见[`D2-A_model_performance_matrix.md`](../research/D2-A_model_performance_matrix.md)。该表标出每行split、样本量、参数量、H0 Pfa、H1 Pd/AUC、Range/Doppler RMSE、joint success和指标来源。

最后的R6-J4以新split比较固定几何DPCA/520点零填充频谱候选上的学习式联合读出，与matched raw-IQ J1-J及固定DPCA/B3基线。两臂使用seed909、51,200对训练场景、5,000对验证场景、有效batch64、FP32、25,600 updates；模型分别70,277和10,494参数。AISAR RTX 3050 Laptop `cuda:0`顺序训练；独立H0校准/验证各120,000，H1 development 6,144。

J4−J1-J的C2+C3 joint差为`+0.03255`，配对95% bootstrap CI `[+0.02408,+0.04080]`；总体Pd与Range保护通过。Doppler RMSE比CI为`[1.723,2.085]`，C3六格familywise Pfa门失败。最难condition10中J4/J1-J Pfa分别18.31%/9.77%、Pd 23.05%/27.73%、joint 2.15%/10.35%。J4相对固定DPCA+520峰值显示学习读出带来明显joint收益，但完整系统没有胜过J1-J的所有核心任务门，因此当前J4参数化不晋级多seed、KD或blind。

容量筛选中，R6-C约70K→292K→1.01M参数的truth-only模型在同一split上C2+C3 joint有所提升；R6-H约72K CNN-SSM的joint收益伴随Range保护和C3虚警失败。训练历史尚未证明这些模型达到容量或优化上限。R6-F显示profile-ML在预测候选上的连续Doppler误差远高于CE-TCN原生参数读出，说明传统精估模块并非可直接替换。所有已评价H1均为development，不是最终盲测；多数架构筛选只有单训练seed。

## 审查材料

- **研究总纲：** `docs/RESEARCH_CHARTER.md`
- **动态状态与路线：** `docs/CURRENT_STATUS.md`、`docs/RESEARCH_ROADMAP.md`
- **假设、决定与进度：** `docs/research/ASSUMPTION_REGISTER.md`、`docs/decisions.md`、`docs/progress.md`
- **执行和审查治理：** `docs/governance/CODEX_TASK_PROTOCOL.md`、`docs/governance/CHATGPT_REVIEW_PROTOCOL.md`
- **总结果表与阶段报告：** [`docs/research/D2-A_model_performance_matrix.md`](../research/D2-A_model_performance_matrix.md)、[`docs/stage_reviews/D2-A-R6J4_review.md`](D2-A-R6J4_review.md)、`docs/stage_reviews/D2-A-R5C_review.md`、`docs/stage_reviews/D2-A-R5C-teaching_pilot_review.md`、R6-A与R6-C至R6-I的独立阶段报告；R6-B结果见`experiments/D2-A-R6/R6-B/analysis.md`和`docs/progress.md`。R6-J4报告覆盖该轮末端物理驱动候选。
- **R6-J4冻结协议/配置/实现：** `experiments/D2-A-R6/R6-J4/protocol.md`、`config.yaml`、`scripts/run_d2a_r6j4.py`、`src/radardistill/students/r6_j4.py`、`src/radardistill/evaluation/d2a_r6j4.py`及`src/radardistill/training/d2a_r6j4.py`
- **R6-J4实际结果：** `experiments/D2-A-R6/R6-J4/metrics.json`、`analysis.md`、`preflight.json`、`g1.json`、`training_summary.json`、`runs/seed_909/{j1_joint,j4_joint}/training_history.json`与`training_record.json`、`cleanup.json`
- **蒸馏与代表架构的主要协议/结果：** `experiments/D2-A-R5/R5-C-teaching/`、`experiments/D2-A-R6/R6-C/`、`R6-D/`、`R6-F/`、`R6-G/`、`R6-H/`、`R6-I/`各自的`protocol.md`、`config.yaml`、`metrics.json`和`analysis.md`

## 希望审查的具体问题

1. 哪些学生在各自同场景开发集上检测或参数估计最好？总表是否保持了不同数据划分之间的边界，是否存在指标口径或模型角色被混淆？
2. 哪个truth-only架构/容量证据最强？现有结果能否支持选择一个后续学生平台，还是仍受单seed、训练未收敛和C3保护失败限制？
3. 现有教师B3、K3/K2-local的检测能力、教师知识质量和学生教学收益是否被正确区分？目前有没有足够证据继续教师软知识蒸馏？
4. 检测与参数估计的联合模型、分离模型或混合链哪种更有研究价值？R6-D、R6-F与R6-J4是否支持明确结论，还是仅显示任务折衷？
5. 当前低虚警评价是否充分？H0样本量、全局/pooled与C3逐格familywise判定是否足以支撑声明？仍缺哪些统计或物理证据？
6. 总体研究方向和近期路线应选CONTINUE、REFINE、REDIRECT还是STOP？请只选一个主判断，给出下一项唯一优先任务、最小可证伪方案和停止条件；若建议改变总纲，请仅提供具体建议文本、证据与影响分析，不要将建议表述为已批准变更。

## 研究方向影响与下一步建议

当前执行结论是：保留truth-only学生学习和任务级评价的研究价值；停止当前R6-J4参数化和已测K3/K2-local教学搜索；不因P7没有候选晋级而宣称所有物理驱动或蒸馏路线无效。后续唯一优先建议是在获得独立方向审查后，围绕困难C3场景的可分性与逐条件Pfa失稳提出新假设、固定强基线和独立H0/H1数据协议；在协议冻结前不启动新架构、蒸馏或盲测。

请按`docs/governance/CHATGPT_REVIEW_PROTOCOL.md`的四层流程分别审查工程正确性、实验科学性、研究主线和战略调整，并区分已证实、被反证、未决及推断。阶段门通过不自动代表总体研究方向正确。
