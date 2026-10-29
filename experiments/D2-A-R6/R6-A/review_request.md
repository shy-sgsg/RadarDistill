请首先读取本次Git提交对应版本的 `docs/RESEARCH_CHARTER.md`，理解RadarDistill的整体研究主题、研究目标、研究内容、科学问题和技术路线，然后结合这些内容审查本阶段的代码、实验、数据和研究结论。

请按 `docs/governance/CHATGPT_REVIEW_PROTOCOL.md` 审查 R6-A 及其触发的 R6-B 预注册分支选择。不要假定你知道仓库或此前对话；先检查以下提交与文件。

**结果提交 SHA：** `9baee0e` (`Archive R6-A shared-scene baseline`)
**代码/分支选择相关提交：** `58ba8bd`（选支脚本）、`b3ad8a5`（选支结果）、`f91da92`（oracle门映射更正）、`3c2ab79`（条件joint图修正）。
**工作区状态：** 结果提交后，所有 tracked changes 已提交；工作区仍有未跟踪用户数据与本地历史实验材料，未纳入该提交。R6-A prediction `.progress.json` 文件也未提交；分析使用的CSV预测、scene/seed manifests和主要环境记录已提交。
**阶段：** D2-A-R6-A同场景检测/参数估计基线；R6-B只完成预注册分支选择，尚未开展其训练与独立评价。
**总纲路径及该提交对应版本/修订日期：** `docs/RESEARCH_CHARTER.md` / v1.3 / 2026-10-27。
**本阶段总体研究目标：** 用可信物理仿真真值和经审查的传统算法知识训练能直接执行GMTI目标检测与参数估计的学生；按检测、连续参数误差、可靠性和完整处理链评价学习与混合方法。R6-A建立共同基线，不把中间统计量作为独立研究终点。
**核心科学问题：** Q1（学生与传统方法的任务能力）、Q3（学生表征）、Q6（弱目标/连续Doppler瓶颈）、Q8（保留哪些物理算子）、Q10（可辨识信息、容量和学习条件）。相关登记假设 H-R6-DET-001 按当前合取表述（困难格保有足够条件可分性、分数尺度迁移为主要因素）在 condition 10 不获支持；尺度贡献尚未独立评估。
**本次实验预注册方案：** 三通道、PRF=1300 Hz、N=130、32距离单元、静止杂波；新独立 H0 calibration=12,000、H0 verification=12,000、H1 development=6,144，12条件等额分配。所有方法使用同IQ。每方法单独在H0 calibration校准全局阈值，verification不回调阈值；条件及候选分析仅用于开发诊断，不生成final blind。
**模型参数量、训练预算与曲线：** R6-A不训练模型；仅推理冻结 R5-A Medium width96 checkpoint，训练参数82,887、推理参数70,277，原训练seed101、12,800 updates。R6-A模型没有新optimizer更新或学习率计划；请结合 R5-A capacity 的训练记录审查该checkpoint来源，勿将其表现解释为R6-A新训练或跨seed泛化。
**教师与知识迁移：** R6-A无教师训练或KD。R3 T1/T2作为未获固定阈值资格的传统检测候选；R5-C当前K3/K2-local教学内容分支的负结果和STOP仍保留。当前不据R6-A重新启动蒸馏。

## 本阶段材料

**实验协议/配置：** `experiments/D2-A-R6/R6-A/protocol.md`、`config.yaml`；R6-B冻结协议/配置：`experiments/D2-A-R6/R6-B/protocol.md`、`config.yaml`。
**关键实现：** `src/radardistill/evaluation/d2a_r6a.py`、`scripts/run_d2a_r6a.py`、`scripts/select_d2a_r6b_branch.py`；测试 `tests/test_d2a_r6a.py`。
**实际结果：** `experiments/D2-A-R6/R6-A/metrics.json`、`analysis.md`、`predictions/*.csv`、`scene_manifest.csv`、`seed_manifest.csv`、`environment_*.json` 和 `figures/*.png`。
**阶段审查报告：** `docs/stage_reviews/D2-A-R6A_review.md`。
**研究对齐与动态状态：** `docs/task_alignments/D2-A-R6_alignment.md`、`docs/CURRENT_STATUS.md`、`docs/RESEARCH_ROADMAP.md`、`docs/research/ASSUMPTION_REGISTER.md`、`docs/decisions.md`、`docs/progress.md`。
**验证：** AISAR整仓 `python -m pytest -q` 为456 passed；R6-B branch selector在AISAR完成端到端运行。
**主要指标：** 全局Pfa 0.05工作点，Medium AUC=0.9830、Pd=0.9334、joint success=0.8306（95% CI `[0.8213,0.8402]`）、直接Doppler RMSE=24.68 Hz；全局Pfa单侧exact95%上界=0.04996。condition 10（C3 partial-overlap/SNR0/CNR20）AUC=0.5956、Pfa=0.097、Pd=0.2207、joint=0.0391。R3 T1/T2总体joint分别为0.7806/0.7586，均低于Medium。

**R6-B预注册选支实际结果：** 只用R6-A H0 calibration与H1 development，明确未读取H0 verification或final blind。C3 pooled AUC=0.9316449（bootstrap 95% CI `[0.9244004,0.9387064]`）；condition 10 AUC=0.6032402（`[0.5708519,0.6329457]`），上界低于冻结0.70门，因此选择R分支：匹配truth-only Medium与training-H0 hard-negative-resampled Medium。完整记录在 `experiments/D2-A-R6/R6-B/branch_selection.json`。

## 请求审查

1. R6-A的同IQ、独立校准/验证、条件ROC、配对bootstrap和oracle/actual candidate区分是否支持报告中的结论？
2. 全局Pfa上界略低于0.05，但C3条件Pfa范围0.072–0.109，困难格条件AUC约0.60；如何限定Medium的当前证据和资格？
3. R6-B分支R的选择是否正确地依据冻结规则；哪些证据说明条件可分性不足，而不是只有全局阈值/尺度迁移？
4. 最近整数距离门oracle更正及条件joint绘图修正是否被充分隔离、记录，且对其他结果的影响描述准确？
5. 按总纲Q1/Q3/Q6/Q8/Q10，R6-B matched hard-negative训练对照是否为当前最有信息价值的唯一下一步？其保护和停止条件是否足够？

请依固定协议的四层流程审查工程、实验科学性、研究主线和战略路线，指出尚未充分支持的主张，最终只选 **CONTINUE / REFINE / REDIRECT / STOP** 之一，并给出唯一优先任务。不要将R6-A development称为blind，也不要把R6-B分支选择当成训练收益结果。研究总纲v1.3不应因本次结果自动修改。
