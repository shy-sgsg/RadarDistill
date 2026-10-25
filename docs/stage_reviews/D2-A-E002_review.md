# D2-A-E002 阶段审查材料

**请首先读取本次Git提交对应版本的 `docs/RESEARCH_CHARTER.md`，理解RadarDistill的整体研究主题、研究目标、研究内容、科学问题和技术路线，然后结合这些内容审查本阶段的代码、实验、数据和研究结论。**

- 结果与代码冻结提交 SHA：`1f822470b906c733706c1a10f031e7a627e140a8`。
- 研究总纲：`docs/RESEARCH_CHARTER.md` v1.1；本阶段未修改总纲。
- 本阶段协议：`experiments/D2-A-E002/protocol.md` v2.1。

## 与总体研究目标的关系

本阶段检验物理仿真真值监督的任务级学生能否直接从三通道复数 IQ 输出目标存在、距离和 Doppler，并在相同数据多重集与 1,600 次更新预算下比较全分布随机混合、由易到难课程顺序及课程加 IQ-only K2 软响应。工作对应研究总纲 Q1（从复数观测学习物理参数）、Q2（教师是否转移有效信息）、Q3（仿真真值与传统教师如何支持训练）和 Q4（三通道联合参数任务的可行性）。本阶段不评价后期部署计算开销 Q6。

## 协议、代码、数据和结果

- 协议/配置：`experiments/D2-A-E002/protocol.md`、`experiments/D2-A-E002/config.yaml`。
- 数据生成及训练入口：`scripts/run_d2a_e002.py`；配对缓存、IQ-only K2、损失、CUDA 训练器和评价模块位于 `src/radardistill/`。
- 汇总：`experiments/D2-A-E002/metrics.json`、`result.csv`、`D_boundary.csv`、`development_by_group.csv`、`h0_verification_by_group.csv`、`development_per_scene.csv`。
- 阈值校准和验证：`calibration_thresholds.csv` 与 `metrics.json` 中的精确单侧 Clopper–Pearson 上限。
- K2 与物理诊断：`k2_audit.json`、`k2_audit_scenes.csv` 及 `metrics.json`。
- 阶段分析和图表：`experiments/D2-A-E002/analysis.md`、`experiments/D2-A-E002/figures/`。

## 观察到的证据

- 实际训练臂：A, B, C；每臂只有训练 seed 101。
- development AUC 最好的学生臂：A（AUC=0.9846）。
- E0 能量检测基线：AUC=0.6103，Pd=0.0538；完整基线对比见 `result.csv`。
- K2 质量审查是否允许 C 臂：True。
- G4 三臂均通过；A/B/C 的 AUC 为 0.9846/0.9787/0.9815，Pd 为 0.9158/0.8996/0.9000。A 的主 A/10/10 dB joint success 为 0.8489（单侧下限 0.8201）。
- G3 总体 Pfa 点估计与精确上限子门通过，但所有学生的预注册分层门均失败；最差子组精确上限分别为 A=0.1953、B=0.2045、C=0.2022。
- G5 数值由保存的逐场景开发 CSV 按预注册 seed/2000 次 base-scene paired bootstrap 重算并保留：B−A joint-success 增益 `−0.1254`（95% CI `[−0.1386,−0.1122]`）；C−B 为 `+0.0036`（`[−0.0088,+0.0162]`）。B/C 都未在预算内达到 A 的最佳 validation loss，故课程与 K2 均无 G5 pilot signal。
- G6 公共相位决策一致率为 0.943–0.950，Doppler 变化 p95 为 61.7–71.9 Hz，频移等变残差 p95 最大 59.6–73.4 Hz；等变门失败。通道独立相位扰乱使 AUC 下降 0.176–0.208，表明模型使用相干信息但相位/频移表示不稳定。
- **训练—推理距离门诊断：** 选定 validation checkpoint 的预测 range cell 与真值 cell 不一致率 A/B/C 为 10.30%/11.14%/10.96%；同一模型改在真值 cell 读取 Doppler 后 RMSE 为 40.95/56.03/59.55 Hz，相比正常预测 cell 解码的 46.66/58.55/60.03 Hz。development 中预测门错配为 502/5000、551/5000、565/5000。错门会增加部分 Doppler 误差，但真值门下的大残差表明它不足以解释主要 Doppler 误差；该单 seed 汇总诊断没有区间估计，不能作因果或跨种子结论。详见架构/损失审查。
- K2 总体 audit 允许 C，但 C3 partial-overlap / SNR=0 dB 的 teacher top-cell localization 仅 4.7%、target-window mass lift 为 0.000673；分条件弱响应未触发冻结的 C3 pooled 取消规则，应作为 K2 局部可靠性限制保留。
- IQ 生成缓存已在分析完成后清理；保留的 split manifests、训练记录、模型 checkpoint、审计和结果见 `experiments/D2-A-E002/`。删除记录见 `docs/maintenance/CACHE_CLEANUP_2026-10-25.md`。
- 阶段门与决策：**REFINE**；具体 G1–G6 数值见 `metrics.json`。
- exact-overlap D 只作为不可辨识参考，不用于来源恢复门或模型选择。

## 对整体研究方向的审查问题

请判断本阶段是否真正推进三通道任务级学生目标，而不是只优化中间教师统计量；真值监督是否足以学习可辨识物理量；课程顺序和 K2 相对 B 臂是否给出有价值的 pilot 信号；学生是否优于常值参数预测与 E0 检测基线；R3 固定阈值结果是否与 K2 软响应资格正确区分；物理一致性证据是否支持学生使用通道相干信息；普通实值 TaskStudent 是否仍适合作为后续学生基线，或证据是否支持引入显式复数/几何归纳偏置；训练—推理 range 门错配只是次要误差还是值得单独干预。

## 统计边界与下一步建议

本轮为单 seed 开发 pilot；按 base_scene_id 的 bootstrap 区间只反映场景抽样不确定性，不代表跨 seed 不确定性。验证集用于选 checkpoint，development 用于有限的阶段门诊断；未生成 final blind。不得据此声称课程或蒸馏已跨 seed 证实。请结合总体研究总纲和预注册门决定下一阶段；如进入正式确认，应先冻结独立训练 seed、统计功效、模型选择以及全新的盲测协议。
