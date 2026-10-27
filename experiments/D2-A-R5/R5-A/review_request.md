请首先读取本次Git提交对应版本的 `docs/RESEARCH_CHARTER.md`，理解RadarDistill的整体研究主题、研究目标、研究内容、科学问题和技术路线，然后结合这些内容审查本阶段的代码、实验、数据和研究结论。

# ChatGPT 阶段审查请求：D2-A-R5-A

请按 `docs/governance/CHATGPT_REVIEW_PROTOCOL.md` 完成四层审查。先检出并核对以下证据提交及提交内的总纲；不要用其他提交中的文件替代。

## 审查基准

- **阶段证据提交 SHA：** `51e10a9041c9d62b83229699a703c81562d16f03`
- **该提交的工作树：** 提交内工作树干净；本地另外有未跟踪的用户 GMTI 源文件、D1 进度记录、D2-A-T001 审计文件、R5 可重建缓存等，不属于本证据提交、未被修改或纳入审查版本。
- **训练代码 SHA：** `d996b66418221e027c29927789ff84f3c1df3c89`
- **评估代码 SHA：** `d1d46617deac234f9456121b2adf0752b44c6f5d`
- **总纲：** `docs/RESEARCH_CHARTER.md`，版本 v1.3，修订日期 2026-10-27。
- **阶段：** D2-A-R5-A；seed 101、E1 物理真值监督下的训练充分性 pilot。

请注意：v1.3 总纲路线表记载“R5 尚无运行结果”，那是该版总纲形成时的状态。当前阶段事实以本证据提交中的 `CURRENT_STATUS.md`、路线图和 R5-A 材料为准。不要据此假设已修改或覆盖总纲；本阶段没有改动总纲。

## 研究主线映射

总纲主题是以物理模型和渐进式知识蒸馏驱动任务级 GMTI 目标检测与参数估计。本阶段只检验学生训练预算和一个固定相干候选是否影响最终检测/距离/Doppler 任务，不研究单独的中间特征目标，也没有开展知识蒸馏。

- **总体目标关联：** 总纲总体目标 1、2、6、7、8、9；当前直接支持 Q10（训练预算、容量与架构归纳偏置），并为 Q3/Q4/Q6/Q8 提供有限证据。
- **阶段问题/假设：** 1,600-update E003 pilot 是否欠训练（H-CAP-001）；width=32 是否存在容量限制（H-CAP-002）；显式相干归纳偏置是否有益（H-ARCH-003 的相关但非决定性证据）。
- **冻结判断：** 两臂在 6,400 步都满足预注册连续延长规则，所以按同一设置延至 12,800；H-CAP-001 只可在该 seed、数据和 schedule 范围内判断。12,800 步仍改善时不得声称收敛或容量上限。

## 预注册方案与实际训练

- **协议/配置：** `experiments/D2-A-R5/R5-A/protocol.md` v1.1、`config.yaml`。均在 R5 数据和训练前冻结。
- **输入：** 三通道复 IQ，PRF 1300 Hz、130 脉冲、32 距离单元，静止杂波仿真。
- **模型：** S0 原始 IQ `TaskStudent(width=32)`，13,383 参数；S2 固定相干前端加 28→6 投影及同宽主干，13,557 参数。R5-A 不包括非 CNN 模型。
- **监督：** 仅物理仿真真值 E1；没有教师硬标签、软频谱响应或真值辅助 KD 损失。
- **训练数据：** 51,200 对 H0/H1 配对场景；validation 5,000 对。沿用 E003 冻结 scene IDs 和同一 pair plan。每 1,600 updates 为重复遍历同一训练集一遍；12,800 updates=8 遍，没有新增独立训练场景。每次更新32对场景、有效 batch 64 CPI，microbatch 16、梯度累积4。
- **学习率/预算：** 首段 6,400 updates 的 cosine `1e-3→1e-4`；满足冻结条件后连续延长至12,800，扩展段 cosine `1e-4→1e-5`；AdamW，weight decay `1e-4`，梯度裁剪1.0。选 checkpoint 的规则为最小 common truth-task validation loss；本次 S0/S2 分别选 update 12,400/12,000。
- **设备：** RTX 3050 Laptop GPU `cuda:0`，FP32，按模型顺序驻留；无 CPU 回退、无 OOM。G1 16 对小样本过拟合检查两臂通过。

## 主要结果

所有开发/校准/verification 及物理干预材料都来自此前用于 E003 的场景；它们不是新的独立 blind。最终盲测未生成。

在同一 R5 长程 schedule 内，update 1,600→12,800 的验证指标持续改善：

| 指标 | S0 raw-IQ | S2 固定相干 |
|---|---:|---:|
| task loss | 0.519→0.321 | 1.024→0.694 |
| range RMSE (cell) | 2.844→2.203 | 3.750→2.907 |
| Doppler RMSE (Hz) | 35.49→24.16 | 72.51→52.13 |

复用的 E003 development 上，以各自独立校准阈值评价：

| 指标 | S0 | S2 |
|---|---:|---:|
| Pd | 0.9619 | 0.9336 |
| AUC | 0.99374 | 0.98999 |
| 全 H1 range RMSE (cell) | 2.239 | 2.824 |
| 全 H1 circular Doppler RMSE (Hz) | 25.23 | 54.25 |
| joint success | 0.8389 | 0.6055 |
| C3 H0 verification Pfa | 0.2041 | 0.1838 |

S2−S0 overall joint 差 `−0.2334`（paired base-scene bootstrap 95% CI `[-0.2488,-0.2188]`）；C2/C3 joint 差 `−0.3945`（`[-0.4189,-0.3696]`）。bootstrap 区间反映固定 seed 下场景抽样不确定性，不表示训练 seed 方差。两臂 basic-learning G1 通过；S2 的任务保护、C2/C3 主收益和 G3 architecture-value 门失败。S0 的 C3 Pfa 同样超过0.10保护门，不能称为合格检测器或教师。

S2 在公共相位旋转诊断中数值稳定，但其频移干预未表现出可靠的任务相关频率平移响应；S0 频移响应相对合理但存在公共相位敏感。此诊断复用 E003 ID，只是开发材料。完整曲线、分场景指标、统计区间和资源记录见 `analysis.md` 与 `metrics.json`。

## 结论边界与待审决策

本阶段报告的结论是 **REFINE**：H-CAP-001 在单 seed/固定数据/当前 schedule 的有限范围内获支持；停止当前固定相干 S2 候选；H-CAP-002 和原始 IQ 非 CNN H-ARCH-003 仍未检验。未运行教师质量审计、教师知识 pilot、KD、多 seed 或 final blind。完整前后处理链的成本未测；训练总墙钟时间因初始段计时未保存而无法完整还原。

Codex 建议下一步按既有 R5-A protocol 做 Small width=32 / Medium width=96 的单因素对照，再决定是否扩大架构；无预注册收益或任务保护失败则不运行 Large。请审查这一优先级是否仍有足够信息价值，尤其考虑两臂 C3 H0 Pfa 均失败；可比较一个更小的 C3 阈值稳定性/尺度 nuisance 诊断作为替代，但不要把建议当作已批准的路线更改。若认为研究总纲需要调整，只提交证据、建议文本和影响分析供用户讨论。

## 必读材料

- `docs/RESEARCH_CHARTER.md`
- `docs/CURRENT_STATUS.md`、`docs/RESEARCH_ROADMAP.md`、`docs/research/ASSUMPTION_REGISTER.md`、`docs/decisions.md`
- `docs/governance/CHATGPT_REVIEW_PROTOCOL.md`
- `experiments/D2-A-R5/R5-A/protocol.md`、`config.yaml`、`analysis.md`、`metrics.json`、`training.json`、`preflight.json`、`data_generation.json`、`cache_manifest.json`
- `experiments/D2-A-R5/R5-A/scene_manifest.csv`、`scene_predictions.csv`、`runs/seed_101/{S0,S2}/training_history.csv`、阶段曲线图
- `docs/stage_reviews/D2-A-R5A_review.md`
- 训练/评价实现：`scripts/run_d2a_r5.py` 和其调用的 `src/radardistill/` 模块

## 审查交付

请按 review protocol 的四层流程评价工程正确性、实验科学性、研究主线贡献和战略调整。明确哪些结论是证实、反证、未决或推断；检查数据重用、单 seed、不确定性、C3 阈值失败、门控映射、缺失传统基线比较、完整成本和训练计时边界。

最终只选一个主判断：**CONTINUE / REFINE / REDIRECT / STOP**。给出整体研究方向是否仍有价值、唯一下一项优先任务、最小可证伪方案、会改变判断的证据和停止条件。不要把本阶段门通过等同于研究方向正确，也不要把未运行的实验写成结果。
