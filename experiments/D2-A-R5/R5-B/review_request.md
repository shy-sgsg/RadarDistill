# ChatGPT 阶段审查请求：D2-A-R5-B TFAS 原始 IQ 时频注意力 pilot

请首先读取本次Git提交对应版本的 `docs/RESEARCH_CHARTER.md`，理解RadarDistill的整体研究主题、研究目标、研究内容、科学问题和技术路线，然后结合这些内容审查本阶段的代码、实验、数据和研究结论。

请按 `docs/governance/CHATGPT_REVIEW_PROTOCOL.md` 进行独立审查。不要只评价代码和阶段门：还要判断本阶段是否推动了总体研究目标、科学问题是否值得继续、技术路线是否合理、是否有更有价值的替代方向，以及当前证据是否要求调整长期研究思路。阶段门通过不自动代表研究方向正确。

## 审查基准

- **结果提交 SHA：** `d847f233a5614580132d4ccad41e59e4d68b6789`
- **分支：** `main`
- **目标提交工作树：** 该提交包含的跟踪文件无未提交差异。当前共享工作区另有未跟踪的 GMTI 外部源码包、历史进度文件和运行缓存；它们未包含在该提交/审查包内，且未被本阶段暂存。
- **提交差异：** 结果提交相对 TFAS 实现提交 `55e7c7810e6cbb0277f79b1b6b4c401052b559dc` 新增阶段指标、逐场景输出、训练记录/最佳 checkpoint、分析、阶段自审和研究状态更新；可重建的 14.23 GB IQ 缓存没有提交。
- **阶段/实验：** D2-A-R5-B，单个原始 IQ 时频注意力学生 TemporalFrequencyAttentionStudent（TFAS）seed 101 E1 truth-only pilot。
- **研究总纲路径与版本：** `docs/RESEARCH_CHARTER.md`，v1.3（2026-10-27修订）；结果提交未修改总纲正文。

## 总体目标与本阶段科学问题

RadarDistill 的总体目标是通过可信物理仿真真值与传统方法知识，训练直接完成 GMTI 目标检测和连续参数估计的学生，并在后续评估精度与完整处理成本。学习模型必须保留并正确利用复数相干信息，且在明确的检测、参数、虚警保护门下评价；不能仅因替换成 Transformer 或通过阶段门就主张研究成功。

R5-B服务总纲 Q3/Q4/Q6/Q8/Q10，检验 H-ARCH-003：在三通道原始复 IQ、任务头、数据和训练预算相同的条件下，一个保留 I/Q 实虚关系、沿慢时间与距离建模的轻量非 CNN 学生，能否改善可辨识 C2/C3 检测与连续参数联合任务并满足 C3 H0 绝对虚警保护。H-LEARN-002关注物理变换是否与模型输出一致。研究对象是任务级 GMTI 学生，不是独立 Transformer 或 CFAR 项目。

## 预注册和训练设置

- **协议/配置：** `experiments/D2-A-R5/R5-B/protocol.md`、`config.yaml` v1.0；架构调查见 `architecture_survey.md`。
- **信号：** 三通道 complex IQ，PRF 1300 Hz、130 脉冲、32 距离单元、静止杂波；A/B/C可辨识条件，D仅作完全重叠边界描述。
- **模型：** 实值输入六个独立 I/Q 平面；共享 `6→56` embedding；两层慢时间 Transformer、每距离门学习注意力池化、一层距离 Transformer；输出存在、距离门/偏移和 Doppler 向量。无幅度化、固定 FFT 或固定相干投影；无严格群等变保证；参数量79,638，比Medium CNN少3.92%。
- **数据：** 复用E003/R5-A冻结的51,200训练pair、5,000 validation pair、H0校准8,192、H0 verification 8,192、C3 H0 verification 4,096、development 4,096及exact-overlap边界ID；相同R5-A pair order，训练集遍历8次。数据均已参与之前阶段，不是新独立测试或blind。
- **训练：** E1物理真值监督，seed101，12,800 optimizer updates，有效batch64，AdamW，FP32、`cuda:0`，RTX 3050 Laptop GPU；每400步验证，最低共同validation truth-task loss checkpoint用于评价。未用教师hard labels、soft responses、KD、truth-assisted输入或final blind。
- **冻结门：** G0 CUDA/数据/参数量和协议正确；G1有限梯度及16-pair/200-update overfit；G2 development AUC CI下界>0.5且range/Doppler RMSE优于真值常值baseline；G3 C2/C3 joint相对Medium增益至少+0.02且paired CI下界>0，并满足继承任务保护及TFAS自身C3 H0 Pfa≤0.10。G3失败按协议停止额外seed、KD、Large、调参和blind。

## 训练曲线、资源和知识来源

- **环境：** Python 3.13.7；PyTorch 2.8.0+cu128；CUDA 12.8；RTX 3050 Laptop GPU，4 GiB；FP32。预检CUDA空闲显存约3.15 GB，训练峰值allocated/reserved约1.014/1.283 GB。
- **预算实际执行：** 12,800/12,800 updates，microbatch16、梯度累积4，有效batch64；无OOM；训练7,505.99秒，训练命令wall time 7,532.98秒。没有实时性或端到端速度结论。
- **验证曲线：** update 0 task loss 5.466、AUC .477、range RMSE 11.62 cells、Doppler RMSE 330.25 Hz；最低validation truth-task loss出现在update 12,400，为1.71301，AUC .96188、range RMSE 4.617 cells、Doppler RMSE 81.71 Hz；update 12,800 loss为1.71355。验证末段接近平台，但一个seed不能证明收敛或架构上限。
- **知识来源：** 仅物理真值 E1。没有训练教师、教师质量审计、教师hard/soft标签、知识蒸馏或truth-assisted对照。
- **匹配参照：** Small与Medium CNN复用已完成的R5-A seed101结果；未重训。不存在KD收益或传统方法对照结论。

## 结果摘要

- **工程/G1：** G0通过。16-pair overfit task loss `5.57683→0.000867`（−99.984%），presence/range-cell accuracy均1.0，任务头梯度有限非零。R5定向测试26项通过。
- **G2：** development AUC `0.96139`（95% scene-bootstrap CI `[0.95760,0.96512]`）；全H1 range RMSE `4.595 cells`，周期 Doppler RMSE `82.00 Hz`，均优于训练真值常值baseline。仅表示基本映射可学习。
- **H0：** 独立学生校准阈值`0.0543661`，校准n=8192，目标Pfa .05；总体verification `436/8192=.05322`，单侧exact95%上限 .05748。C3 H0 verification `622/4096=.15186`，单侧exact95%上限 .16138，超过绝对 .10 门。
- **开发总体：** Pd `.8020`（95% CI `[.7900,.8142]`），joint success `.5054`（`[.4895,.5205]`）；range RMSE `4.595` cells，Doppler RMSE `82.00 Hz`。
- **课程分层：** C0 joint `.8643`；C1 `.7207`；C2 `.3516`；C3 `.0850`。C3较低Pfa伴随严重漏检和估计失败。
- **Matched CNN参照：** Small/Medium/TFAS参数量为13,383/82,887/79,638；总体AUC `.9937/.9958/.9614`；Pd `.9619/.9746/.8020`；joint `.8389/.9106/.5054`；C3 H0 Pfa `.20410/.18896/.15186`。两CNN的绝对Pfa也失败，不能把它们称为通过完整任务门的模型。
- **主要G3端点：** TFAS−Medium C2/C3 joint差`−0.60449`（paired-scene bootstrap 95% CI `[-0.62598,-0.58350]`）；远低于预设至少`+0.02`。总体Pd差`−0.17261`，总体joint差`−0.40527`；C2/C3 range和Doppler RMSE比分别2.385和4.599。TFAS相对CNN的C3 Pfa差虽为负，但TFAS自身绝对门失败。
- **训练成本：** 训练约2.09小时；峰值CUDA allocated/reserved为1,014,349,312/1,283,452,928 bytes；参数量比Medium少3.92%。这只是单次训练记录，不是推理/系统成本对比。

## 物理诊断、阶段自审及审查问题

公共相位旋转和全IQ Doppler频移诊断复用E003已使用的development/verification IDs，只作描述。32种公共相位下，H0误警率样本率范围`.02734–.07813`，H1判决一致率中位数`.9258`，Doppler输出变化p95最高`122.18 Hz`、range输出变化p95最高`14.98 cells`。频移响应不稳定，partial-overlap检测一致率平均`.695`。这些结果不构成blind，也不能唯一解释G3失败。

执行方阶段自审见 `docs/stage_reviews/D2-A-R5B_review.md`，详细科学分析见 `experiments/D2-A-R5/R5-B/analysis.md`。自审建议为总体 **REFINE**、TFAS候选依据G3 **STOP**。H-ARCH-003仅对当前TFAS候选NOT_SUPPORTED；不外推否定所有非CNN。H-LEARN-002仍INCONCLUSIVE；尚无教师知识吸收证据。研究总纲v1.3未改。

请审查并回答：

1. 工程实现、样本ID复用、阈值校准、bootstrap单位、指标与协议门是否吻合？有没有代码/统计/物理错误会改变结论？
2. G2通过与G3失败的解释是否恰当？训练末段平台能否支持架构判断？单seed和复用开发材料对结论造成哪些边界？
3. 物理一致性诊断是否有足够证据，还是可能受复用样本、扰动定义或实值输入设计影响？
4. 低于CNN的C3 Pfa能否说明有可用改善，还是任务能力退化下的检测器失效？绝对门失败应如何影响候选判断？
5. 本阶段是否推进总纲目标？H-ARCH-003/H-LEARN-002状态是否恰当？R5-C教师审计是否仍有价值，还是应先调整整体研究问题/路线？
6. 对总纲一致的整体判定只能选一个：**CONTINUE / REFINE / REDIRECT / STOP**。请给出唯一下一项优先任务、最小可证伪方案、会改变判断的证据和停止条件。阶段门通过本身不代表方向正确。

## 必读材料

- `docs/RESEARCH_CHARTER.md`
- `docs/CURRENT_STATUS.md`
- `docs/RESEARCH_ROADMAP.md`
- `docs/research/ASSUMPTION_REGISTER.md`
- `docs/governance/CHATGPT_REVIEW_PROTOCOL.md`
- `experiments/D2-A-R5/R5-B/protocol.md`、`config.yaml`、`architecture_survey.md`、`analysis.md`
- `experiments/D2-A-R5/R5-B/metrics.json`、`training.json`、训练history、逐场景预测、`physical_diagnostics.json`
- `docs/stage_reviews/D2-A-R5B_review.md`
- 实现：`src/radardistill/students/temporal_frequency_attention.py`、`scripts/run_d2a_r5_attention.py`
- 关键比较：`experiments/D2-A-R5/R5-A/capacity/analysis.md` 与 metrics；R5-B相对实现提交`55e7c7810e6cbb0277f79b1b6b4c401052b559dc`的Git差异

审查时请首先以结果 SHA `d847f233a5614580132d4ccad41e59e4d68b6789` 检查 `docs/RESEARCH_CHARTER.md`。该版本总纲为v1.3；请不要把阶段自审作为独立结论，也不要把未运行的KD、传统教师比较或blind写成证据。
