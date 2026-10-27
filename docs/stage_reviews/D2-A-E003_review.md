# D2-A-E003 阶段自审

## 审查基准

- 实验：D2-A-E003，Physics-Invariant Coherent Representation for Task-Level GMTI Learning。
- 研究总纲：`docs/RESEARCH_CHARTER.md` v1.2；本阶段未修改总纲。
- 核心证据：[`protocol.md`](../../experiments/D2-A-E003/protocol.md)、[`metrics.json`](../../experiments/D2-A-E003/metrics.json)、[`analysis.md`](../../experiments/D2-A-E003/analysis.md)、[`training.json`](../../experiments/D2-A-E003/training.json)、[`physical_consistency.json`](../../experiments/D2-A-E003/physical_consistency.json)、[`compute.json`](../../experiments/D2-A-E003/compute.json)。
- 代码/训练：训练使用已推送的 E003 冻结 runner（训练记录来源 SHA `bd18df1f4fa818684bd28874c3ba96d23a1e7dde`）。评价计时的 inference-mode 缺陷在 `af0f596` 修复，测试全套 303 项通过；初始 preflight 被保留，后续重验记录训练已开始。成功指标仅来自修复后使用相同 checkpoints、数据和阈值的完整重跑。
- 工作区有用户本地未跟踪 GMTI 文件；这些文件未纳入 E003 提交，也未修改。按 E003 manifest 删除 90 个缓存文件（14,232,309,192 bytes），清单外缓存文件为 0；清理后项目 801,766,373 bytes（0.747 GiB），低于 20 GiB 限制。清理审计见 `experiments/D2-A-E003/cleanup.json`。

## 1. 工程正确性

**结论：通过已覆盖的实现与数据检查。** 物理输入为 `3×130×32` 复 IQ；固定 28 通道特征对同场景公共复相位构造不变，lag-1 映射正确，lag-4/16 周期混叠如预期，跨通道相对相位保留，距离门无互混；complex64 相对最大误差 `9.54e-8`，比预注册 `1e-4` 更小。该 `1e-4` 是前向数值比较容限，不代表 Hz、range、检测或参数精度标准。

训练用 RTX 3050 Laptop `cuda:0`、FP32、S0/S2 串行、microbatch16、有效 batch64。两臂各完成 1,600 updates，无 OOM 或 CPU 回退。参数量增加 1.30%。场景 ID 清单表明 81,288 个 E003 ID 跨数据 split 互斥；配对 H0/H1 保持在同一组；final blind 未生成。

评价首次运行在最后的 S2 CUDA timing 环节因 inference tensor 被默认梯度模式下的卷积消费而异常退出。错误已按调用栈定位；计时函数统一关闭 autograd 并加回归测试，完整测试套件为 303 passed。中断尝试没有写出最终 metrics；科学结果只来自同缓存/检查点/配置的修复后完整评价。此修复只改变推理计时上下文和训练后 preflight 留档行为。

计算统计覆盖模型输入到任务输出、数据传输和显存，并区分 S2 固定前端；没有覆盖完整距离压缩、传统前后处理或系统链，所以不作系统时延或实时性结论。

## 2. 实验科学性

**结论：匹配单 seed pilot 能支持淘汰当前候选，但不支持总体人群或跨 seed 主张。** S0/S2 使用相同初始化的 TaskStudent 权重、样本次序、真值标签、有效 batch、训练更新、优化器和验证选点规则；S2 只增加固定相干前端与 174 参数投影。单 seed 固定为 101。H0 校准与验证独立；H1 development 与训练/验证隔离；场景 bootstrap 以 `base_scene_id` 配对，RMSE 在每次重采样中重算。

总体 H0 verification Pfa 为 S0 `0.04834`、S2 `0.04749`，但专用 C3 H0 verification (n=4096) 为 `0.18140` 和 `0.15283`，均超过冻结的每臂 `0.10` 点门。S2 相对 S0 的 C3 Pfa 差为 `−0.02856`，95% CI `[−0.04444,−0.01245]`；它只说明同一 C3 mixture 中 S2 误警较少，不表示 Pfa 已可接受或跨所有条件稳定。

S2 的整体 Pd 差为 `−0.04736` `[−0.05713,−0.03759]`，joint 差为 `−0.21021` `[−0.22559,−0.19507]`，C2/C3 joint 差为 `−0.25928` `[−0.28223,−0.23682]`。C2/C3 range RMSE 比为 1.329 `[1.221,1.456]`、Doppler RMSE 比为 1.835 `[1.720,1.959]`。所有关键 G4 保护项和主要收益门失败；S2 在当前 protocol 下 STOP。S0/S2 均通过基础 G3，表示两者均比常值基线学到目标信息，不代表 S2 与 S0 等效。

强传统方法仅在 R2 的 64 场景/几何受控诊断中以描述性 CUDA reference 记录，未与 S0/S2 在完整 E003 development split 上做同阈值任务比较。因此本轮不能断言学生超过 FFT、教师、ML 或完整传统基线。知识蒸馏完全未运行。

## 3. 总体研究目标与研究主线

E003 直接服务总纲中的任务级学生、真值监督与物理先验评价（Q1/Q4，并关联 Q3/Q8）。结果回答了一个重要边界：**显式公共相位不变性可以构造和验证，但它并不自动改善复杂杂波中的最终检测/参数任务。** S2 phase score 变化约 `1e-5`、决策一致率为 1；同一模型的 C1–C3 joint、range 和 Doppler 却明显恶化。故 G5 的结构成功不能替代 G4 任务门。

相关物理结论：同通道延迟相位保留 PRF 周期频率信息；跨通道相位差被显式保留，但本轮没有角度任务标签，不能推出角度/平台参数估计能力。二阶相关会引入目标—背景交叉项，不等价于 DPCA/杂波抵消。`q(X)` 归一化部分信号尺度，`log(q)` 留下一个粗略可观测尺度通道；模型的条件 H0 Pfa 仍不稳定。28→6 投影、归一化、二阶特征和优化过程没有分别做消融，因此不能把负结果单因归到复相关公式。

创新性方面，本轮不支持“新结构已优于既有学生”的方法贡献。可复核的研究价值在于一项受控负结果：具有严格公共相位不变性的固定前端在所测任务边界下未能保护估计与联合成功；该结论限于一个 seed、一个合成模型和指定 28→6 S2 实现。

## 4. 假设状态

- **H-ARCH-D2-002：NOT_SUPPORTED（当前候选）。** 公共相位不变性支持其数学部分；任务收益/保护未达到预注册标准。
- **H-FREQ-D2-001：INCONCLUSIVE。** 多个 lag 都对频率具有预期的周期响应，但没有 lag-1-only 学生作为对照；S2 Doppler 结果较差，尚不能判断额外延迟是否提供可学习的增量任务信息。
- **H-DET-D2-002：NOT_SUPPORTED（当前候选）。** S2 C3 Pfa 相对改善但未达绝对门，Pd、参数和 joint success 明显退化。
- **H-IDENT-D2-001：INCONCLUSIVE。** R2 非退化信息增量与 S2 困难条件失败同时出现，当前数据无法区分可辨识边界、归一化、投影瓶颈和优化/seed 影响。
- H-KD、教师软知识可迁移性、跨训练 seed 稳定性与真实数据迁移均未被 E003 检验。

## 5. 战略判断与下一步

| 层级 | 判断 |
|---|---|
| S2 固定相干特征 + 28→6 投影候选 | **STOP**：冻结 G4 与主收益门失败；不做正式三 seed、E2/E3 或 final blind |
| D2-A 任务级学生路线 | **REFINE**：保留 S0 的可学习性证据，先解释 C3 Pfa 与物理/参数失效，不继续扩展 S2 架构 |
| RadarDistill 长期研究使命 | 总纲 v1.2 仍适用；本结果不足以建议修改总纲，也没有推翻真值驱动学生与蒸馏研究的长期目标 |

唯一下一项优先工作是将本阶段送独立 ChatGPT 审查，重点判断是否值得做一个不增加模型/不训练的表示归因诊断，区分固定特征、场景归一化和 28→6 投影中的信息损失。只有审查后形成清楚、可证伪且更有价值的假设，才另行预注册该诊断。当前不恢复教师搜索或 KD，也不继续 S2 多 seed。

独立审查请求见 [`review_request.md`](../../experiments/D2-A-E003/review_request.md)。
