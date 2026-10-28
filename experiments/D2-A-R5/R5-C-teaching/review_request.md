# ChatGPT 阶段审查请求：R5-C T0–T3 教学内容 Pilot

请首先读取本次Git提交对应版本的 `docs/RESEARCH_CHARTER.md`，理解RadarDistill的整体研究主题、研究目标、研究内容、科学问题和技术路线，然后结合这些内容审查本阶段的代码、实验、数据和研究结论。

请再阅读 `docs/governance/CHATGPT_REVIEW_PROTOCOL.md`，按其四层流程审查本阶段。不得假定你已知仓库或本次对话历史；请核对以下 SHA、协议版本和文件。

**结果提交 SHA：** `896e7de4a20c8c0f96a8eb28760617423a2efc95`
**工作区是否干净：** 否。提交包含本阶段结果和列明的研究设计；工作区还有未跟踪的旧实验记录及用户生产源码文件，它们未纳入本提交。
**阶段：** D2-A-R5-C Teaching Content Pilot，T0–T3 单训练 seed 开发 pilot。
**研究总纲路径及版本：** `docs/RESEARCH_CHARTER.md`，v1.3（建立日期 2026-10-24，修订日期 2026-10-27；本阶段没有修改）。

## 总体目标、科学问题与预注册假设

RadarDistill 的长期目标是利用物理仿真真值和传统雷达方法训练能够直接完成 GMTI 检测与参数估计的学生网络，并检验其精度或完整处理成本优势。本阶段对应总纲 Q1、Q2、Q9、Q10：传统教师的中间响应是否能在匹配物理真值之外改善学生的最终检测、距离和 Doppler 任务？

前置 R5-C 教师质量审计中，4×零填充 K3 相对原生 K1 的 C2+C3 定位率差为 `+0.03125`（95% CI `[+0.02474,+0.03776]`），但 C3/partial-overlap/SNR 0 dB/CNR 20 dB 子组定位仅 `4.1%`。原审计 G4 PASS 保持有效，但不代表教师在该困难组可靠。本 pilot 的预注册假设 H-TEACH-002 是 K3 或 K2-local 在真值监督之外改善任务表现；本次只检验该假设的一次单 seed 教学内容 pilot。

冻结四臂为：T0 truth-only；T1 truth+K3 全 RD 响应；T2 truth+K2-local；T3 匹配的 truth-assisted 局部响应控制，T3 不是教师。主对比为 T1−T0 和 T2−T0 的 C2+C3 joint success。晋级要求点增益至少 `+0.02` 且同时单侧 97.5% 配对 bootstrap 下界大于零，并满足任务保护；每个候选还须达到 C3 H0 `Pfa≤0.10`。教师辅助损失仅作用于 H1，物理真值持续作为主任务监督。

## 模型、预算和训练充分性边界

- 学生：Medium `TaskStudent(width=96)`；训练期总参数 `120,717`，去除辅助频谱头后的推理参数 `70,277`。
- seed `303`；四臂相同初始化、同一训练场景顺序。训练/验证分别为 `51,200/5,000` 个新配对场景；每臂 12,800 updates，每 update 32 个配对场景、有效 batch 64 CPI，共固定遍历训练对 8 次。
- RTX 3050 Laptop `cuda:0`、FP32、microbatch 16，四臂顺序训练，无 CPU 回退或 OOM。AdamW、weight decay `1e-4`、梯度裁剪 `1.0`；两段余弦学习率从 `1e-3` 降到 update 6,400 的 `1e-4`，再降到 update 12,800 的 `1e-5`；每 400 updates 验证并按共同真值任务 loss 选 checkpoint。
- T0/T1/T2/T3 用时分别 `6745.77/4446.82/4295.96/3781.71 s`；最佳 checkpoint 更新为 `12800/12800/12800/12400`。逐步训练记录、检查点、恢复状态和验证指标随结果提交提供。此单 seed、固定数据集训练不能估计 seed 方差，也不能证明已收敛或容量上限。

## 数据、代码与实际结果

训练/验证新场景与 R5-A 已核对 IDs 无重叠。评价数据独立分为 H0 calibration `8,192`、总体 H0 verification `8,192`、C3 H0 verification `4,096`、H1 development `4,352`；未生成 final blind。G0/G1 均通过。

训练和评价实现：

- `scripts/run_d2a_r5c_teaching_pilot.py`
- `src/radardistill/training/d2a_r5c_teaching_pilot.py`
- `src/radardistill/distillation/r5c_teaching.py`
- `src/radardistill/evaluation/d2a_r5c_teaching_pilot.py`
- 定向测试：`tests/test_d2a_r5c_teaching.py`、`tests/test_d2a_r5c_teaching_runner.py`、`tests/test_d2a_r5c_teaching_evaluation.py`

主要结果如下；`Pfa` 均为冻结阈值下的独立验证结果，C3 上界为单侧 exact 95%：

| 臂 | 总体 H0 Pfa | C3 H0 Pfa [上界] | H1 Pd | AUC | C2+C3 joint | Range RMSE | Doppler RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| T0 | 0.0520 | 0.1943 [0.2048] | 0.9200 | 0.9531 | 0.7750 | 3.238 cells | 30.419 Hz |
| T1 | 0.0531 | 0.2031 [0.2137] | 0.9221 | 0.9524 | 0.7646 | 3.248 cells | 30.298 Hz |
| T2 | 0.0500 | 0.1824 [0.1926] | 0.9207 | 0.9522 | 0.7612 | 3.302 cells | 31.533 Hz |
| T3 | 0.0553 | 0.1973 [0.2078] | 0.9228 | 0.9530 | 0.7906 | 3.275 cells | 28.635 Hz |

T1−T0 C2+C3 joint 差 `−0.01042`（95% CI `[−0.01849,−0.00208]`）；T2−T0 `−0.01380`（`[−0.02292,−0.00391]`），均未达 `+0.02` 主增益门。T1/T2 均未通过全部保护门；四臂 C3 Pfa 均超过 `0.10`。T3−T0 为描述性 `+0.01563`（95% CI `[+0.00781,+0.02344]`），未达点增益门且属于 truth-assisted，不能归因于教师。新 H1 development 的 C3 partial-overlap/SNR 0 dB/CNR 20 dB 子组中，K3 仅 `24/512`（4.69%）正确定位。

协议/config 和实际记录：`experiments/D2-A-R5/R5-C-teaching/{protocol.md,config.yaml,metrics.json,analysis.md,training.json,preflight.json,g1_overfit_sanity.json,cleanup.json}`，训练历史/检查点及逐场景预测均在同目录。阶段审查报告：`docs/stage_reviews/D2-A-R5C-teaching_pilot_review.md`。阶段结果停止当前 K3/K2-local 教师内容分支；H-TEACH-002 在本 pilot 范围内为 NOT_SUPPORTED。总体建议 REFINE，下一项优先任务为独立预注册的 R6-A 同场景检测/参数估计基线。

清理记录显示只删除了本阶段 manifest 指定的四个 IQ memmap，共 `11,222,016,512` bytes；保留清单、真值元数据、预测和训练记录。清理时项目大小为 `15,160,472,696` bytes（约 `14.12 GiB`）。提交同时包含 R6-A/R6-B 协议与 R6 研究设计交接材料；它们是预注册/设计，不是已运行的 R6 结果。研究总纲及历史冻结结果未修改。

## 请求审查

请先核对提交 `896e7de4a20c8c0f96a8eb28760617423a2efc95` 中对应版本的研究总纲，再按 `CHATGPT_REVIEW_PROTOCOL.md` 分别审查工程正确性、实验科学性、研究主线和战略调整。请从训练记录、场景清单、逐场景预测、协议和实现验证上述结论，特别判断：

1. 单 seed、八次固定数据遍历下，负的 T1/T2 差异及停止当前教师内容分支是否得到充分支持？
2. 原 R5-C G4 PASS 与该困难子组低定位率应如何并列解释？T3 能否只作为 truth-assisted 控制而不混淆成蒸馏收益？
3. 是否有数据泄漏、教师/真值混淆、统计或阈值实现问题，或未报告的关键任务保护失败？
4. R6-A 同场景基线是否是结合总纲与本 pilot 证据后的唯一优先后续任务？什么证据会改变 REFINE 或停止判断？

阶段门通过不自动代表研究方向正确。最终只选一个：CONTINUE / REFINE / REDIRECT / STOP，并明确整体判断、唯一下一项优先任务、最小可证伪方案、改变判断的证据和停止条件。如认为总纲需调整，只提交问题、依据、建议文本与影响供用户讨论，不要修改总纲或把建议视为已批准。
