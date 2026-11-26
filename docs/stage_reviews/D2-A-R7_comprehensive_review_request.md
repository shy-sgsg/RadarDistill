# RadarDistill D2-A-R7 综合阶段审查请求

**请首先读取本次结果提交对应版本的 `docs/RESEARCH_CHARTER.md`，理解RadarDistill的总体研究主题、科学问题、研究目标、技术路线及新修订的单元级低虚警要求。随后结合实际代码、仿真配置、传统教师正确性证据、7kg工程参考核查、无杂波性能矩阵及先进传统方法对照，独立审查本阶段完成度、工程正确性、实验科学性和下一步研究方向。**

## 提交和总纲版本

- **结果提交 SHA：** `9ad3eac1ddcb2c19952c3f0a97d564dc8a9be430`
- **研究总纲：** `docs/RESEARCH_CHARTER.md`，v1.4，提交 `029d6c9bb54582905b3b4f4c8c558c8217856c32`
- **审查请求 SHA：** 请以包含本文件的后续提交为准，并从Git历史读取。
- **提交范围：** 上述结果提交包含R7综合报告、统一性能CSV、实际metrics/preflight、R7-E协议/适用性材料、矩阵构建器和定向测试。工作区存在未纳入提交的外部工程源码包和其他实验运行文件；它们不属于本次R7审查提交，也没有被此提交修改。
- **本阶段：** D2-A-R7，R7-A至R7-E基础性能重建、教师正确性与单元级低虚警评价。
- **综合裁决：** Codex内部判断为 **REFINE**；请独立复核该判断。

## 总体目标、科学问题和决策范围

本阶段对应的总体目标是：通过物理仿真真值和经过验证的传统教师知识，训练能够直接完成GMTI检测和参数估计的学生；为后续学生/教师比较建立正确的低虚警统计口径、可靠的基础性能参考和可辨识的训练问题。本轮重点关联总纲Q1/Q2/Q3/Q4/Q6/Q8/Q9/Q10，尤其是教师是否正确、教师知识是否相关、学生与传统方法的差距、检测与连续参数估计的折衷，以及后续知识是否值得迁移。

本阶段不把CFAR当作独立研究目标，也没有开展R7-F新模型训练、知识蒸馏或final blind。R7-A–R7-E完成后，R7-F因G1/G4缺口和cell级Pfa未验证而暂缓。历史R5/R6冻结协议和负结果保持不变。

请重点判断：

1. 历史Pfa单位及独立统计分母是否解释正确；Pfa_CPI能否被合理地区别于Pfa_cell。
2. R7-A对`Pfa_cell≈1e-6`所需样本量和当前证据缺口的陈述是否严谨。
3. R7-B能否支持“当前声明合成几何内DPCA实现正确”，以及它不能支持哪些真实设备/检测/教师结论。
4. R7-C将7kg标记为`NOT_QUALIFIED_NOT_RUN`是否有充分的版本、许可、正确性和输入边界依据。
5. R7-D统一L0/L1结果是否公平、指标是否正确；47个历史checkpoint和7种传统参考的比较是否有过度筛选/多重比较问题。
6. R7-E对ATI、dense CFAR、Kelly/AMF/ACE、STAP、知识辅助方法和7kg等项目的未运行/不适用判定是否恰当。
7. 全量矩阵是否完整保留用户要求的模型、传统教师、数据划分、SNR/条件、Pd/Pfa、参数RMSE、样本量、训练/参数量、置信区间和来源证据。
8. 现有证据是否支持延后R7-F；下一步是否应优先解决dense CUT低虚警评价，或存在更有价值且在总纲范围内的路线。

## 本阶段实际执行与结果

- **R7-A：** 32×130物理RD网格与dense/sparse/CPI统计语义完成；10项边界测试通过；主要历史Pfa口径已审计；32场景CPU预检约129.8 scenes/s生成。正式dense CUT校准/独立低虚警验证未执行，`Pfa_cell≈1e-6`仍`NOT_VERIFIED`。
- **R7-B：** 1,280个新合成场景完成DPCA纯分量诊断。纯静止杂波相干幅值中位数0.8486→0.9858；残差功率比0.02833；杂波抑制15.478 dB（95% scene-bootstrap CI 15.420–15.540）；CPU/CUDA K2响应抽样最大绝对差0。噪声差分功率约×2（+3.010 dB）。结论限于声明的共址Tx合成模型和固定DPCA路径。
- **R7-C：** 对`/home/shy/AIR/7kg`只读检查；没有可用版本历史/标签、未找到源码再分发许可证据、README自述`NOT_CERTIFIED`，且原始/完整回波与XML控制接口尚无已验证的D2-A `[3,130,32]`距离压缩IQ适配器。未复制、构建、修改或集成。
- **R7-D：** 47/47历史checkpoint及7种传统方法完成新L0/L1评价，0个checkpoint失败。AISAR / PyTorch 2.8.0+cu128 / CUDA 12.8 / RTX 3050 Laptop GPU；峰值CUDA分配516.94 MiB。L1每SNR、每方法各有2,048个H0 calibration、2,048独立H0 verification和2,048个H1；目标工作点是逐方法/SNR校准的Pfa_CPI=0.01。没有模型训练或蒸馏；没有方法产生dense CUT分数图。
- **R7-E：** 已实测B0/B1 FFT、FFT细化、lag-one及有限网格profile-ML。profile-ML每SNR仅运行预先指定的256个H1。需要不同几何、dense分数、额外背景样本或不同工程输入的方法均明确标记未运行/不适用。
- **资源清理：** 依据R7-D自身owner清单删除4个可重建IQ数组和真值数组，释放5,112,088,079 bytes；保留owner记录、scene plan、checkpoint inventory、preflight、metrics、protocol/config和报告。提交后工作目录约1.5 GiB，小于20 GiB。

## 用户要求的全量性能表

主表是单张长格式机器可读CSV，共28,058行，R7-D有485行：

- [统一性能矩阵CSV](../../experiments/D2-A-R7/unified_performance_matrix.csv)
- [字段、单位与跨阶段可比性说明](../research/D2-A-R7_UNIFIED_PERFORMANCE_MATRIX.md)
- [综合阶段报告及R7-D逐SNR重点表](D2-A-R7_review.md)
- [R7-D protocol/config/metrics/analysis](../../experiments/D2-A-R7/R7-D/)
- [R7-E protocol/config/analysis](../../experiments/D2-A-R7/R7-E/)

R7-D主要发现：R6-J4固定DPCA/FFT前端加学习读出模型在−10/−5 dB有最佳受测学生Pd/Joint点值，但它是混合方法；中高SNR下FFT/profile-ML参数精度高得多。R5-C的K3在可辨识C2+C3教师top-cell定位率81.22%，比K1高3.125个百分点，但最难部分重叠低SNR格仅4.10%。R6-G的K3蒸馏相对truth-only未达增益门；当前教师任务质量不等于教学收益。

## 阶段报告和关键证据

- [R7综合阶段报告](D2-A-R7_review.md)：逐项回答低虚警、DPCA、7kg、学生检测/参数估计、知识蒸馏、D/P/J比较及下一步共23个问题。
- [统一矩阵构建器](../../scripts/build_d2a_r7_unified_performance_matrix.py)；[定向回归测试](../../tests/test_d2a_r7_performance_matrix.py)。
- [传统教师正确性报告](../research/D2-A-R7_TEACHER_CORRECTNESS.md)；[7kg工程参考只读报告](../research/D2-A-R7_ENGINEERING_REFERENCE.md)；[传统方法适用性矩阵](../research/D2-A-R7_CLASSICAL_REFERENCE_MATRIX.md)。
- [R7-A口径/资源分析](../../experiments/D2-A-R7/R7-A/analysis.md)；[R7-B DPCA分量分析](../../experiments/D2-A-R7/R7-B/analysis.md)；[R7-D分析](../../experiments/D2-A-R7/R7-D/analysis.md)。
- 动态状态和假设：[当前状态](../CURRENT_STATUS.md)、[路线图](../RESEARCH_ROADMAP.md)、[假设登记表](../research/ASSUMPTION_REGISTER.md)、[R7研究计划](../research/D2-A-R7_RESEARCH_PLAN.md)。

## 希望ChatGPT做出的判断

请遵循[`CHATGPT_REVIEW_PROTOCOL.md`](../governance/CHATGPT_REVIEW_PROTOCOL.md)，分别评价：

1. 工程正确性：数据生成、统计、阈值、CUDA结果、报告与主表是否可追溯和自洽。
2. 实验科学性：R7-D是否公平，Pfa_CPI与Pfa_cell区分是否严格，最优值摘要如何处理多重比较，profile-ML小样本比较是否过度解释。
3. 研究主线：本阶段是否推进GMTI任务级学生及教师教学目标；哪些假设得到支持、反证或仍未决。
4. 战略调整：现有路线是否合理，是否有更有价值的替代研究方向；是否应先建立dense cell检测、改进连续估计、重新验证教师，或先做其他工作。

阶段门通过不自动代表整体研究方向正确。请主动指出证据支持的路线调整建议，并区分事实、推断、未决和未执行事项。不要把R7-D开发比较称为盲测，也不要把Pfa_CPI换算为Pfa_cell或把教师定位率称为Pd。

**请最终只选择一个阶段裁决：CONTINUE / REFINE / REDIRECT / STOP。** 给出对整体研究方向的判断、唯一下一项优先任务、最小可证伪方案、会改变判断的证据和停止条件。该审查不自动授权R7-F或final blind；若建议修改研究总纲，只提出建议供用户讨论，不要直接修改总纲正文。
