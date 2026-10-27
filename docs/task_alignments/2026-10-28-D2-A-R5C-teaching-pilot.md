# D2-A-R5-C 匹配教学内容 pilot 任务对齐（草案）

## 研究主线

- **总纲：** `docs/RESEARCH_CHARTER.md` v1.3；不修改总纲。
- **总体目标：** 以物理真值和传统知识蒸馏训练可直接完成 GMTI 检测与参数估计的学生，并分别评价教师能力、知识质量与学生端教学收益。
- **科学问题：** Q1/Q2/Q4/Q6/Q9/Q10；检验 R5-C 通过质量门的完整 RD 与局部 Doppler 软响应是否改善同一学生的最终任务，以及增益是否只是额外辅助头/辅助监督的效果。
- **假设：** H-TEACH-002（局部知识对最终任务是否有用）、H-LEARN-002（学生是否吸收任务相关物理频率信息）；教师可靠性与时机假设不在本 pilot 检验。

## 证据与优先性

- R5-C 的 K1/K3/K2-local 通过冻结 G4；K3 相对 K1 的 C2+C3 定位差为 `+0.03125`（95% CI `[+0.02474,+0.03776]`）。这只证明教师响应有任务相关信息，不证明学生会吸收。
- R5-A Medium width=96（82,887 参数）在 C2/C3 joint success 上优于 Small，但 C3 H0 Pfa=`0.18896`，未过每模型 `0.10` 绝对保护；不能把 Medium 称为合格 detector。
- R5-B TFAS 按冻结 G3 停止；当前最强已测学生仍是 Medium，但其 C3 虚警风险须在每个教学臂保留。
- R5-C ChatGPT 审查请求已归档；其独立方向审查当前待返回。本文件为设计草案，不是冻结协议或实验结果。

## 最小验证实验

- 以 Medium CNN 任务干路为共同学生，所有臂从同一初始化、用相同训练样本顺序、12,800 updates、有效 batch 64、FP32 与同一优化器/学习率计划训练。
- 四臂：T0 truth-only；T1 truth + K3 全图软响应；T2 truth + IQ-only K2-local 响应；T3 truth + 真值距离门及连续 Doppler 的 truth-assisted 局部分布。四臂实例化同一 520-bin 训练期辅助头；T0 不使用辅助损失。
- 只在 H1 样本计算辅助损失，避免要求学生模仿 H0 的噪声峰。K3/K2-local 由 IQ-only CUDA 教师在线逐批生成，不保存 IQ 或教师图；T3 明确标为 oracle/truth-assisted。
- 使用 R5-A 固定 51,200 对训练场景；validation 只用于共同 checkpoint 选择。新 seed 的独立 H0 calibration/verification 与 H1 development 场景分开；development 用于 pilot 筛选，不是 blind。
- 主指标为可辨识 C2+C3 joint success（检测正确、距离误差≤1 cell、周期 Doppler 误差≤20 Hz）；同时保护 Pd、C3 H0 Pfa、距离与 Doppler 误差。按 `base_scene_id` 配对 bootstrap；该单 seed 区间只描述场景抽样。

## 决策影响与停止条件

- G4 已通过，因此有理由提出一轮有限 student-side 检验；它不替代正在等待的 R5-C 独立方向审查。外部审查返回前不冻结协议、不启动训练。
- T1/T2 相对 T0 的 C2+C3 joint 增益须至少 `+0.02`，且每个对照的 Bonferroni 同时单侧 97.5% 场景 bootstrap 下界大于0；任何晋级还须满足每臂 C3 H0 `Pfa≤0.10` 及映射至 T0 的冻结任务保护。T3 只用于区分教师局部信息与真值辅助监督，不作为可部署方法。
- 若教师臂不优于 T0、收益只体现在 KD loss、C3 虚警/其他主要任务保护失败，停止对应知识形式；不开展时机、可靠性权重、正式多 seed 或 final blind。
- 只有一个预注册教师臂通过收益与保护门后，才另立至少三 seed 确认协议；单 seed pilot 不支持跨 seed 或机制因果结论。
