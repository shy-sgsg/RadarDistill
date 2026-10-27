# ChatGPT 阶段审查请求：D2-A-R5-A Small/Medium 容量 pilot

请首先读取本次Git提交对应版本的 `docs/RESEARCH_CHARTER.md`，理解RadarDistill的整体研究主题、研究目标、研究内容、科学问题和技术路线，然后结合这些内容审查本阶段的代码、实验、数据和研究结论。

## 审查版本与项目主线

- **阶段结果提交 SHA：** `760cb30c7ce9dafdbcb5a328d9228b0f2eb9f585`
- **执行代码/训练 SHA：** `1678a468d5c150ff734d80ad40cf75407a49651f`
- **工作区：** 该结果提交时 tracked worktree clean；既有未跟踪用户文件保留且未纳入本阶段提交。
- **研究总纲：** `docs/RESEARCH_CHARTER.md`，v1.3（本阶段未修改正文）。
- **总体研究目标：** 通过物理仿真真值及质量合格的传统教师知识，训练能直接完成 GMTI 动目标检测与连续参数估计的学生，评估精度与完整处理效率。
- **本阶段问题：** R5-A Phase 2 检验 width=32 TaskStudent 在冻结预算中是否受容量限制，以及 width=96 收益能否同时满足任务保护。
- **相关科学问题/假设：** 总纲 Q1/Q3/Q4/Q6/Q8/Q10；H-CAP-001、H-CAP-002；H-ARCH-003仍未测试。
- **冻结门槛：** `experiments/D2-A-R5/R5-A/protocol.md` v1.1 §4、§6、§8；C2/C3 joint-success 有意义增益至少0.02且 paired scene bootstrap 95% CI 下界大于0，且 G4 所有任务保护通过；C3 每模型 H0 Pfa 点估计不超过0.10。

## 本阶段设计、代码和结果

- **协议/config：** `experiments/D2-A-R5/R5-A/protocol.md` 与 `experiments/D2-A-R5/R5-A/capacity/config.yaml`。
- **代码：** `scripts/run_d2a_r5_capacity.py`；容量规格校验在 `src/radardistill/training/d2a_r5.py`；绘图复现 `scripts/plot_d2a_r5_capacity.py`。
- **模型与数据：** Small width32 / 13,383 参数，复用 R5-A S0 seed101 checkpoint；Medium width96 / 82,887 参数，从 seed101 初始化。两臂沿用 E003 train/validation IDs 与 R5 pair plan、物理真值 E1、有效 batch64、FP32、12,800 updates。校准/verification/development也是已用 E003 材料，不是新 blind。
- **设备与训练：** RTX 3050 Laptop `cuda:0` 顺序训练；Medium microbatch16，3,951.3秒，无 OOM，峰值 CUDA allocated 820,737,024 bytes。R5 定向测试18项通过。
- **主结果：** Medium−Small C2/C3 joint success `+0.13086`（95% CI `[+0.11230,+0.15088]`）；C2/C3 range/Doppler RMSE ratio `0.839/0.713`。主要收益门通过。
- **保护失败：** C3 H0 verification Pfa Small/Medium=`0.20410/0.18896`（n=4,096；单侧 exact 95% upper=`0.21474/0.19931`），均高于0.10；其余相对保护项通过。R5-A容量阶段因此判 **REFINE**，Medium不晋级，不运行Large。
- **完整结果：** `experiments/D2-A-R5/R5-A/capacity/analysis.md`、`metrics.json`、`training.json`、`result.csv`、`roc.csv`、`scene_predictions.csv`、`figures/`。
- **阶段自审：** `docs/stage_reviews/D2-A-R5A-capacity_review.md`。

## 请独立审查

请按 `docs/governance/CHATGPT_REVIEW_PROTOCOL.md` 的四层流程评价工程正确性、实验科学性、研究主线和战略调整。除门控判断外，请回答：

1. Medium 的 C2/C3 收益是否支持有限的容量限制结论，还是还存在训练/数据/阈值解释？
2. G3 因两臂 C3 H0 Pfa 均失败而未通过，是否足以停止 Large；Small/Medium 是否仍是 R5-B 有效 CNN 参照？
3. 在继续 R5-B 原始 IQ 轻量时频注意力 pilot 前，是否应先增加一项有界 C3 Pfa 诊断，还是按既定路线继续即可？
4. 单 seed、E003 已用开发集、Small 历史训练时间缺失和 bootstrap 的局限，是否在结论中得到恰当处理？
5. 本阶段是否推动了总体 GMTI 学生研究目标；是否有更高价值的替代路线或停止理由？

请区分事实、推断与尚未验证的假设；不要把 E003 开发材料称为盲测，不要推断跨 seed 结果，也不要建议修改已冻结门槛。阶段门通过本身不自动证明整体研究方向成立。最终从 CONTINUE / REFINE / REDIRECT / STOP 中选择一个，并给出唯一下一项优先任务、最小可证伪方案及会改变判断的证据。
