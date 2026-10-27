# D2-A-R5-A 阶段自审

## 审查基准

- **评估代码提交：** `d1d46617deac234f9456121b2adf0752b44c6f5d`；该提交包含本阶段指标门名称适配修复。阶段审查请求会给出包含完整报告和产物的证据提交 SHA。
- **研究总纲：** `docs/RESEARCH_CHARTER.md` v1.3；本阶段未修改总纲。
- **范围：** R5-A seed 101 的训练充分性 E1 truth-only pilot；不含 Small/Medium 容量对照、KD、正式多 seed 或 final blind。
- **主要证据：** `experiments/D2-A-R5/R5-A/{protocol.md,config.yaml,analysis.md,metrics.json,training.json,preflight.json,scene_manifest.csv,scene_predictions.csv}`，训练曲线和开发集分层图见 `figures/`。
- **输入/输出代码：** `scripts/run_d2a_r5.py`、`src/radardistill/` 中的 R5 训练与评估实现；R5 定向测试 13 项通过。评估门标签适配只重命名/映射继承自 E003 的 gate 字段，不改变指标、阈值或数据。

## 1. 工程正确性

**判断：本阶段所覆盖的 CUDA 训练和评估路径可复现，未发现已知实现故障。** 运行环境为 AISAR、Python 3.13.7、PyTorch 2.8.0+cu128、RTX 3050 Laptop `cuda:0`；FP32、microbatch 16、梯度累积 4、有效 batch 64，S0/S2 顺序训练，无 CPU 回退或 OOM。G1 的 16 对小样本过拟合检查通过。数据生成清单记录 81,288 个 split-disjoint base-scene IDs；train/validation ID 沿用 E003 冻结数据。

本阶段评估适配器修正了 E003 遗留门名与 R5-A 冻结协议不一致的问题：基础学习归入 `G1_basic_learning`，容量/架构价值单列为 `G3_architecture_value`。更正后在同一模型、预测、阈值和 bootstrap 配置上重评；数值计算未改。定向测试 13 passed。

用时记录不完整：训练记录覆盖了扩展调用时长，但初始 6,400 updates 的墙钟时间不可恢复，因此不能从现存 878.3 秒/1,005.4 秒推断完整 12,800-update 用时。runner 已改为后续累计记录。运行时延只记录模型输入到输出/传输诊断，不能当作完整 GMTI 系统时延。

## 2. 实验科学性

**判断：匹配单 seed 结果支持“1,600 updates 不是本次候选的充分训练预算”，并足以拒绝当前 S2 候选；不支持跨 seed、盲测或架构上限主张。** 两臂沿同一长程 schedule 从头训练至 12,800 updates（8 遍固定训练 ID），seed 101、E1 物理真值监督、相同 train/validation IDs 和配对计划；两臂均在 6,400 步满足预注册连续延长条件。验证曲线截至 12,800 步仍改善，故不作收敛判断。

同一长 schedule 下，update 1,600→12,800：S0 task loss `0.519→0.321`、range RMSE `2.844→2.203 cells`、Doppler RMSE `35.49→24.16 Hz`；S2 分别 `1.024→0.694`、`3.750→2.907 cells`、`72.51→52.13 Hz`。R5 训练 split 和 E003 开发/校准/验证材料均为已使用场景，不能充当新的独立盲测。

复用 E003 development、各自由 H0 calibration 固定阈值后的结果：S0/S2 Pd `0.9619/0.9336`、AUC `0.99374/0.98999`、全 H1 range RMSE `2.239/2.824 cells`、Doppler RMSE `25.23/54.25 Hz`、joint `0.8389/0.6055`。配对 scene bootstrap 2,000 次：S2−S0 overall joint 差 `−0.2334`（95% CI `[-0.2488,-0.2188]`），C2/C3 joint 差 `−0.3945`（`[-0.4189,-0.3696]`）。

普通 H0 verification Pfa 为 S0/S2 `0.05017/0.04834`；C3 H0 verification Pfa 为 `0.20410/0.18384`，均超过协议的 0.10 点限值。S2 任务保护和 C2/C3 主收益门失败；S0 是更强的已测任务基线，但 C3 固定阈值虚警同样不合格。G1 basic learning 两臂通过，不等于部署/教师资格。

32 个公共相位旋转诊断中 S2 预测基本不变，但整体频移干预下其 Doppler 响应仍不可靠；S0 的频移响应较合理，但存在公共相位敏感性。物理干预复用 E003 开发诊断 ID，只能作机制线索。没有运行教师知识、KD 或 final blind。

## 3. 研究主线贡献

本阶段服务于总纲 v1.3 的任务级 GMTI 学生、训练充分性、物理归纳偏置和后续知识迁移可行性目标。结果表明：延长训练预算能在本固定数据和单 seed 范围内继续改善两个候选；已测固定相干前端的相位不变性没有转化成检测/参数任务优势。停止结论仅适用于该 S2 特征、投影、初始化和训练配置，不能外推否定所有物理结构或原始 IQ 非 CNN 架构。

- **H-CAP-001：SUPPORTED（有限范围）**，仅限 seed 101、冻结数据和当前 schedule；不代表收敛或独立泛化。
- **H-CAP-002：未决。** 尚无 width-only 对照。
- **H-ARCH-003：未测试。** S2 不是所提原始 IQ 非 CNN 架构。
- **H-LEARN-002：INCONCLUSIVE。** 旋转不变性、频移干预和任务表现没有共同支持“学到任务相关物理知识”。

## 4. 战略判断

**本阶段判断：REFINE。** 保留 S0 作匹配容量基线；停止当前固定相干 S2 候选；下一项唯一优先任务为冻结协议中的 Small width=32 / Medium width=96 单因素容量对照。维持 seed、样本、真值损失、pair plan、有效 batch 和更新预算一致，显式保护 C3 H0 Pfa 与 C2/C3 joint。无预注册任务收益或任务保护失败时不运行 Large、KD 或 final blind。完成该对照仍不能解决 C3 虚警时，下一轮应比较学生检测分数与可观测尺度 nuisance 的因果机制，不能直接宣称教师知识可迁移。

相比立即进入蒸馏，容量对照能以一个变量判断当前 S0 的误差是否可能受 width=32 限制，成本与解释风险更低。外部 ChatGPT 审查材料应优先复核：开发数据复用是否足以支持当前有限判断、C3 Pfa 限值下的候选资格、继续容量实验的预期信息价值，以及是否存在更小的 C3 稳健性诊断。阶段门通过不自动证明长期路线正确；目前证据不足以建议改动研究总纲。
