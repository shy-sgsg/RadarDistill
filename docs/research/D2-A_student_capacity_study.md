# D2-A 学生训练预算与容量研究

**适用范围：** D2-A 静止杂波、3通道复 IQ、PRF=1300 Hz、N=130、32距离单元。本文汇总冻结的 R5-A 证据；它不修改实验协议或研究总纲。

## 研究问题

1. E003 的 1,600 updates 是否足以判断现有学生的学习能力？
2. 在匹配数据、监督和训练预算时，增大 TaskStudent 主干宽度能否改善任务估计？
3. 这些改善是否同时满足检测、参数估计和 C3 固定阈值虚警保护？

主要端点为可辨识 C2/C3 配对 development joint success；距离、Doppler、Pd/AUC、C3 H0 Pfa 和训练/显存成本共同约束阶段结论。开发指标不代替最终 blind。

## 当前证据

### 训练预算

E003 的 seed101 S0/S2 每模型只运行 1,600 updates。R5-A 在冻结 E003 train/validation ID 和 E1 truth-only 下，将两臂沿相同 schedule 训练至 12,800 updates。验证任务指标继续改善，支持 H-CAP-001：1,600 updates 对本组模型与数据不足以评价可达到的任务水平。单 seed、ID 复用和末段改善意味着不能声称收敛或独立泛化。

### 容量

Phase 2 只改变原始 IQ TaskStudent 主干宽度：Small width=32（13,383参数）复用 R5-A S0 checkpoint；Medium width=96（82,887参数）从 seed101 初始化。两者使用相同 12,800 updates、pair plan、有效 batch64、FP32、训练/验证 IDs 和 truth-only 损失。

Medium 相对 Small 的 C2/C3 joint-success 提升 `+0.13086`（paired scene bootstrap 95% CI `[+0.11230,+0.15088]`）；C2/C3 range RMSE 比 `0.839`（CI `[0.731,0.950]`），Doppler RMSE 比 `0.713`（`[0.654,0.773]`）。这表明 width=32 在已测分布/预算内存在有限容量限制证据。

但 Small/Medium 的 C3 H0 Pfa 分别为 `0.20410` 与 `0.18896`，高于冻结的0.10每模型上限。Medium 虽未相对恶化 C3 Pfa 且所有相对任务保护分量通过，仍因绝对 Pfa 门失败而未获晋级资格。模型宽度改善参数估计，没有解决检测阈值的 C3 场景迁移。

训练/评价数据沿用 E003/R5 开发材料；只有 scene bootstrap 不确定区间，没有训练 seed 不确定区间。Medium 训练约3,951秒；Small 的完整历史墙钟时间不可恢复，故成本比较不完整。细节见 [R5-A 容量分析](../../experiments/D2-A-R5/R5-A/capacity/analysis.md) 与 [阶段自审](../stage_reviews/D2-A-R5A-capacity_review.md)。

## 假设状态

- **H-CAP-001：有限支持。** 延长预算改善 seed101 的冻结数据验证指标；未证明收敛或泛化。
- **H-CAP-002：有限支持，阶段门未通过。** Width=96 提升 C2/C3 joint 和连续参数指标，但绝对 C3 false-alarm 保护失败。
- **H-ARCH-003：未测试。** CNN 宽度对照不是原始 IQ 非 CNN 架构证据。

## 决策边界

当前不运行 Large；避免在 C3 检测保护失败时以更大模型扩大计算量。下一项工作是按 R5-B 调研并实现一个轻量原始 IQ 时频注意力 pilot，以 Small/Medium CNN 作为有界参照。候选须共享任务定义和数据暴露，单独校准，但继续报告 C3 H0 Pfa 与 C2/C3 任务指标。若非 CNN 候选也无法通过预注册保护，不启动训练种子扩展、KD 或 final blind。
