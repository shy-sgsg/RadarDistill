# ChatGPT 阶段审查请求：D2-A-R6-B

请首先读取本次Git提交对应版本的 `docs/RESEARCH_CHARTER.md`，理解RadarDistill的整体研究主题、研究目标、研究内容、科学问题和技术路线，然后结合这些内容审查本阶段的代码、实验、数据和研究结论。

请按 `docs/governance/CHATGPT_REVIEW_PROTOCOL.md` 的四层审查流程执行。先核对提交 SHA、总纲版本和材料路径；不要假设了解仓库或本次对话历史。

**结果生成代码 SHA：** `293f1733b417d7fdc65fd9e294c8511cef43613c`（`metrics.json`记录的评价实现版本）
**审查材料归档提交：** `ad97a46aba10b71fdb98a5c3a5cc976842a6cccf`（已推送至 `main`；请读取该提交内的研究总纲v1.3）
**工作区说明：** R6-B归档提交后目标文件无未提交改动；工作区另有未跟踪用户资料，本阶段未暂存、提交或修改。
**阶段 / 实验：** D2-A-R6-B branch R：训练期 H0 hard-negative resampling
**总纲路径及版本：** `docs/RESEARCH_CHARTER.md` v1.3，路径保持不变
**总体研究目标：** 通过物理真值和合格传统算法知识，训练任务级GMTI学生，以检测、连续参数估计和完整处理链评价其价值。
**对应科学问题：** 总纲Q1/Q4/Q6/Q9/Q10；本阶段具体检验训练期困难H0覆盖能否缓解C3困难条件的条件ROC/虚警问题。
**预注册门：** `experiments/D2-A-R6/R6-B/protocol.md` 与 `config.yaml`；选支R依据 `branch_selection.json`。
**希望审查的决策：** hard-negative R分支是否值得开展至少三seed确认；低Pfa与任务保护是否允许候选晋级。

## 模型、训练与数据

- Medium TaskStudent width=96，训练参数82,887，推理参数70,277。
- 两臂初始化seed303，12,800 optimizer updates，effective batch64（每步32 H0+32 H1），FP32 CUDA RTX 3050 Laptop。
- 学习率沿用R5-A Medium两段cosine：6400步0.001→0.0001，后6400步0.0001→0.00001；AdamW、weight decay 0.0001、gradient clip 1.0。
- H1样本顺序和条件配额相同；H0仅在HardNegative臂以每条件全程50% top-quartile困难样本、50%均匀样本抽取。训练日志按400步验证/检查点记录，不逐更新落盘。
- 样本量：H0 calibration 120,000；H0 verification 120,000；H1 development 6,144；训练51,200配对场景，验证5,000配对场景。所有新seed/scene ID与历史阶段互斥。
- 只有一个正式训练seed；无教师蒸馏、无final blind。
- 两臂validation task loss在update12,000最低（Truth-only/Hard-negative为0.228539/0.231157），update12,800为0.229891/0.232303。训练集区间loss仍下降，但终段验证loss略升；此单seed曲线不能证明收敛或延长预算收益。R5-A曾观察到1,600至12,800步验证指标改善。用户已允许未来新实验公平增加updates并探索至多两倍batch，本轮冻结的12,800步/effective batch64不变。

## 本阶段材料

- protocol/config：`experiments/D2-A-R6/R6-B/{protocol.md,config.yaml}`
- 实现：`scripts/run_d2a_r6b_low_pfa.py`、`src/radardistill/training/d2a_r6b_hard_negative.py`、`tests/test_d2a_r6b_hard_negative.py`
- 选支/G0/G1/训练/资源：`branch_selection.json`、`preflight.json`、`g1.json`、`runs/seed_303/training.json`、各臂训练历史与检查点
- 结果：`metrics.json`、`analysis.md`、`figures/`、`scene_manifest_*.csv`、`environment_*.json`
- 阶段报告：`stage_review.md`
- 结果内容SHA：`293f1733b417d7fdc65fd9e294c8511cef43613c`（指标声明的执行代码提交）
- 关键估计：HardNegative 对比 TruthOnly 的 paired bootstrap、C3/hard-cell条件指标与低Pfa exact upper bound均见 `metrics.json`。

请判断本阶段是否推动总纲目标，hard-negative问题是否值得继续，是否存在更有价值的替代方向，H-R6-DET-001及相关解释如何更新。并审查R5-A长预算改善与R6-B最后800步验证损失回升应如何解读：后续是否值得另行注册独立matched训练预算实验（增加updates，是否同时将有效batch提高到128），以及如何匹配更新数、样本暴露和checkpoint规则以避免混淆；不要把该问题与继续hard-negative分支等同。请区分单seed开发证据、阶段门资格和可推广结论。最终只选 CONTINUE / REFINE / REDIRECT / STOP，并给唯一下一项优先任务和停止条件。
