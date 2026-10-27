# D2-A-R5-B：原始 IQ 时频注意力学生 pilot

**版本：** 1.0
**状态：** 预注册；模型训练、开发评价与物理干预尚未运行。
**冻结时点：** 在 TFAS 代码执行训练、加载新模型检查点或查看新候选预测前。
**总纲：** `docs/RESEARCH_CHARTER.md` v1.3；不修改研究总纲。
**对齐：** `docs/task_alignments/D2-A-R5B_alignment.md`。

## 1. 研究目标与可证伪假设

本阶段检验 H-ARCH-003：在保持三通道复 IQ、任务头、数据和训练预算一致时，显式沿慢时间建模复相干关系的轻量非 CNN 时频注意力学生，相对已测 CNN capacity baseline 能否改善可辨识 C2/C3 的检测与连续参数联合任务，同时通过 C3 H0 虚警绝对保护。该阶段服务于总纲 Q3/Q4/Q6/Q8/Q10，不检验教师知识或蒸馏。

这是单个非 CNN 候选的 one-seed pilot，不估计训练 seed 方差，不代表盲测、普遍注意力能力或架构上限。R5-A 已测 Medium 的 C2/C3 参数质量有收益但 C3 Pfa 失败；该失败是本阶段必须保留的绝对保护约束，不回调门槛。

## 2. 固定信号、样本和对照

- 信号模型固定为 D2-A-E003：三通道复 IQ、PRF 1300 Hz、130 脉冲、32 距离单元、静止杂波、A/B/C 可辨识条件；D 同流形条件只作边界描述。
- 只复用 E003/R5-A 已冻结的同一训练、验证、H0 calibration、H0 verification、C3 H0 verification、development 与 exact-overlap ID。严禁修改 E003/R5-A 数据、缓存、seed、真值、阈值或旧结果。
- 训练使用 51,200 个配对基础场景及 R5-A 冻结 `training_pair_plan.npz`；validation 为 5,000 对。每个模型看到完全相同的有序 pair plan。模型选择使用 common validation truth-task loss。
- 对照为 R5-A seed101 E1 truth-only `Small width=32` (13,383 参数) 和 `Medium width=96` (82,887 参数)；直接重用已提交 checkpoint/预测，不重新训练、不调参。TFAS 预期约79,638参数，目标范围为 Medium 参数数的 ±20%。Small/Medium 仅提供容量和任务标尺，Medium 的 C3 绝对虚警失败仍原样报告。
- 校准、verification 和 development 是已参与之前模型选择的复用材料，不是 blind。该阶段不生成新数据、final blind 或教师标签。

## 3. 冻结模型

唯一训练候选为 `TemporalFrequencyAttentionStudent` (TFAS)，实现细节见 [`architecture_survey.md`](architecture_survey.md)：

- 输入 `[B,6,130,32]`，实部/虚部作为六个独立输入平面；不能取幅度、生成频谱替代原始输入或使用 E003 28→6 固定相关投影。
- 共享每脉冲 embedding `6→56`；130 脉冲固定正弦位置编码。
- 两层、4-head、width=56、FFN=112 的时间 Transformer，在每个 32 距离门独立执行；每门用一个学习标量注意力池化。
- 32 距离门加学习距离位置嵌入后经一层同规格 range Transformer。
- 输出 `presence_logit`、`range_logits[32]`、`range_offset[32]`、`doppler_vector[32,2]`；复用原 `TaskStudent` 的 presence、range、offset、circular Doppler 真值损失和 decode。
- dropout=0，FP32。禁止在 pilot 内改变宽度、层数、位置编码、输入归一化或损失；参数量超 Medium ±20% 则 G0 失败并停止，不重新选型。

## 4. 训练计划与资源限制

| 设置 | 冻结值 |
|---|---|
| 监督 | 物理真值 E1 truth-only；不含教师 hard label、teacher soft response、KD 或 truth-assisted teacher window |
| seed | 101 |
| 更新 | 12,800 optimizer updates，匹配 R5-A 最长预算 |
| 场景暴露 | 每 update 32 个配对场景、64 个 CPI；固定51,200对训练场景完整遍历8次，共409,600对场景抽样/819,200 CPI presentation |
| pair order | 复用 R5-A 保存的同一 `training_pair_plan.npz`，不重新抽样 |
| optimizer | AdamW，weight decay `1e-4`，gradient clip `1.0` |
| 学习率 | update 1–6,400: cosine `1e-3→1e-4`; update 6,401–12,800: cosine `1e-4→1e-5`，与 R5-A continuation schedule 匹配 |
| 精度/设备 | FP32、AISAR 环境、RTX 3050 Laptop `cuda:0`；CUDA 不可见则停止，禁止 CPU fallback 与 AMP |
| microbatch | 初始16；OOM 顺序降至8、4；梯度累积保持有效batch64，不改科学预算；microbatch4仍 OOM 即停止 |
| GPU驻留 | TFAS 单模型顺序驻留，不同时加载 Small/Medium；评估也按模型顺序运行 |
| 验证/选择 | 每400 updates；最小 common validation truth-task loss 选 checkpoint；保存每验证点训练/验证四项任务 loss 与总 loss |

训练前 G0 记录提交 SHA、AISAR/Python/PyTorch/CUDA/driver、实际 GPU 名称、可用显存和磁盘余量；并核对 R5-A calibration/verification/development provenance、缓存 scene IDs 和 pair plan。代码必须已提交并推送到 `main`，输入数据/protocol 不能有未提交差异。

## 5. 阶段门和停止规则

### G0：预注册/工程/设备

协议和 config 已提交；模型参数数在 Medium ±20%；数据 ID、E003/R5-A source provenance 与训练 pair plan 正确；RTX 3050 Laptop `cuda:0` 可用；FP32、OOM fallback 与保存路径通过。任一失败不开始正式训练。

### G1：数值、梯度与可学习性

合成输入和真实 16 对训练场景前向/损失/反向皆有限；presence、range 与 Doppler 头梯度均有限非零；16 对 H0/H1 固定小批重复200 updates，E1 truth-task loss 至少下降50%，presence 准确率100%、H1 range cell 准确率≥95%。任何失败则先视为实现/优化失败，正式12,800步停止，不能加蒸馏掩盖。

### G2：基础任务学习

在 E003 development 上报告 paired AUC 及95% scene-bootstrap CI；AUC CI下界>0.5；range 与周期 Doppler RMSE 均优于只由训练真值常值构成的基线。G2 不替代虚警或 joint task gate。

### G3：注意力候选保留门

主要端点为 C2/C3 paired development joint success（检测、绝对 range error≤1 cell、周期 Doppler error≤20 Hz）。TFAS 相对 Medium CNN 的主要增益至少 `+0.02` 且 paired base-scene bootstrap 95% CI 下界>0。继承 E003 的任务保护要求同时满足：

- overall Pd 与 overall joint-success 差点估计均≥−0.01，95% CI下界均>−0.05；
- C2/C3 joint-success 差95% CI下界>−0.05；
- C2/C3 range、Doppler RMSE ratio点估计均≤1.05且95% CI上界均≤1.10；
- C3 H0 Pfa差值95% CI上界≤+0.02，且 TFAS 自身 C3 H0 verification `Pfa_CPI≤0.10`。

单独满足参数RMSE、AUC或Pfa改善不能替代主要收益及其余任务保护。TFAS 未通过 G3 时停止扩展该架构，不运行额外 seed、KD、Large、参数搜索或 final blind；完整保留其负结果。通过 G3 也只允许另行提出正式多 seed/新独立 blind 协议，不自动触发下一阶段。

## 6. 评价和物理诊断

- 检测：H0 calibration 目标 `Pfa_CPI=0.05`，higher empirical quantile，固定 `score>=threshold`；每个学生独立 calibration。verification、C3 H0 和 development 不得调阈值。报告严格 `>`/`>=`、平局数、Pfa exact upper 95% CI、总体和 C0–C3/场景/SNR/CNR分层。
- H1：Pd、ROC/AUC、range RMSE/MAE、周期 Doppler RMSE/MAE/bias、predicted-range mismatch、joint success；报告 C0–C3 及可辨识 A/B/C，D 仅作可辨识性边界。比较 TFAS−Medium 与 TFAS−Small，2000次 paired base-scene bootstrap，固定预注册seed；统计区间表达场景抽样，不表达 seed 方差。
- 物理诊断复用 E003 既有的分层 H0/H1 IDs：32个公共相位旋转、全 IQ 的 Doppler 平移、目标分量平移/重生成几何。记录 presence/range/circular-Doppler输出变化、判决一致率与真实相位/频移关系。诊断为描述性，不替代 G3，不构成独立测试。
- 成本：参数量、训练时间、峰值 allocated/reserved GPU memory、microbatch、OOM尝试次数；输入/输出边界与已有 Small/Medium 相同。当前不做实时性结论。
- 传统方法：报告已冻结 D2-A 基线可复用的同开发集结果并明确其协议/阈值资格；不得把尚未通过固定阈值教师门的 B3/T1 或失败的 T001 ACE×AMF 称为获批教师，不在本阶段新增传统基线搜索。

## 7. 结果、边界与交付

训练前先提交预注册文档、config 和任务对齐；实现后提交独立的小型代码包并推送到 `main`，之后才允许训练。结果包保留 preflight、G1、training history/checkpoint索引、training/resource、metrics、paired per-scene predictions、physical diagnostics、analysis、阶段自审和 ChatGPT review request。原始 IQ 缓存与可重建输入保持在既定位置并按原 manifest 审计；不提交训练缓存。

最终解释必须区分模型正确性、训练可学习性、参数估计、检测/Pfa、物理一致性和架构收益。未运行的教师、KD、正式多 seed、blind 与完整实时性不得写成结果。对项目长期方向的结论只限于当前静止杂波仿真和已用开发材料。
