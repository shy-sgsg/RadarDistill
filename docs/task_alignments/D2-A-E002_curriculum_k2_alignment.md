# D2-A-E002 课程学习与 K2 教师辅助任务对齐及完成记录

**日期：** 2026-10-25
**状态：** 用户批准的单 seed A/B/C pilot 已完成；阶段决定 REFINE。
**研究总纲：** `docs/RESEARCH_CHARTER.md` v1.1；本任务没有修改总纲。
**协议：** `experiments/D2-A-E002/protocol.md` v2.1，运行前冻结。
**模型、运行及指标冻结 SHA：** `1f822470b906c733706c1a10f031e7a627e140a8`。
**缓存清理与动态状态归档提交：** `3743c7b906b29a0f1db4dfefa252997829789d35`。本次交付汇报同时提供包含本记录更新的最新仓库 HEAD SHA。

## 总纲映射

- **总体目标：** 用物理仿真真值和传统算法知识训练任务级 GMTI 学生，直接从多通道复数回波完成目标检测、距离和连续 Doppler 估计，并在任务质量约束下评价后续精度与完整处理成本。
- **科学问题：** 对应 Q1（学生能否从复数观测学习物理量）、Q2（教师是否转移有效知识）、Q3（仿真真值与传统教师如何形成训练方法）和 Q4（三通道相干信息能否支持检测与参数估计）。本轮不研究部署成本 Q6。
- **路线关系：** D2-A-R3 固定阈值 G2 未通过，T1/T2 没有获得固定阈值教师资格；此事不等价于 IQ-only 中间软响应不能用于受控训练。本轮同时比较直接真值、课程真值及课程加 K2，并保留真值任务损失，直接检验总体学生目标。
- **授权及边界：** 用户任务附件明确批准调整动态路线并运行本轮 A/B/C pilot。此前 E002 v1 truth-only 提案未执行；v2.1 已取代它。D1、D2-A-E001、D2-A-R3 和 P0 冻结结论均未改写。

## 已完成实验

沿用 D2-A-E001 三通道静止杂波模型和 D1-E001 `TaskStudent`，训练 A（全难度随机混合真值）、B（C0→C4 课程真值）、C（同 B 的课程和样本顺序，加 H1-only IQ 生成 K2 soft RD response）。每臂训练 seed 101、1,600 updates、有效 batch 64、FP32；在 RTX 3050 Laptop GPU `cuda:0` 顺序运行。train、validation、教师审查、H0 calibration/verification、development 和 exact-overlap 边界按 base-scene seed 隔离；没有生成 final blind。

### 关键结果

- G1 工程 sanity 通过；固定小样本真值 loss 从 7.1634 降至 0.02834。K2 数值、尺度和信息隔离审计通过，故按冻结规则运行 C。C3 pooled K2 审计允许 C，但 partial-overlap / SNR=0 dB 的 top-cell localization 仅 4.7%，属于 K2 条件性弱项。
- G4 任务学习门三臂通过。开发集 A/B/C AUC 为 `0.9846/0.9787/0.9815`，Pd 为 `0.9158/0.8996/0.9000`；A 主 A/10/10 dB joint success 为 `0.8489`（单侧下限 `0.8201`）。E0 AUC=`0.6103`。
- G3 总体学生 Pfa 点估计为 `0.0457–0.0495`，总体区间子门通过；分层门失败，最差子组单侧精确上限为 A/B/C=`0.1953/0.2045/0.2022`。
- G5 没有课程或教师 pilot signal：B−A joint-success 差为 `−0.1254`（paired 95% CI `[−0.1386,−0.1122]`）；C−B 为 `+0.0036`（`[−0.0088,+0.0162]`）。B、C 均未在 1,600 updates 内达到 A 的最佳 validation task loss。
- G6 公共相位和频移等变门失败。公共相位决策一致率 A/B/C=`0.950/0.947/0.943`；Doppler 变化 p95=`61.7/71.9/62.4 Hz`；频移等变残差 p95 最大=`73.4 Hz`。独立通道相位扰乱使 AUC 下降 `0.176–0.208`，显示模型使用通道相干信息，但没有学到要求的稳定变换行为。
- 训练—推理 range 门错配率约为 10–11%。验证集同模型真值门 Doppler RMSE A/B/C=`40.95/56.03/59.55 Hz`，预测门为 `46.66/58.55/60.03 Hz`。错门会增加部分误差，但真实门下仍有 `41–60 Hz` RMSE，现有证据不支持错门解释主要 Doppler 残差；H-OPT-D2-001 保持 INCONCLUSIVE。

## 研究影响与结论

本阶段表明，同一普通实值 `TaskStudent` 在当前合成、可辨识开发分布上能学习部分任务级检测与参数映射，直接支持总体目标继续接受检验。结果不支持课程排序或本轮 K2 软响应带来预注册收益；也没有满足分组虚警和公共相位/频移物理一致性要求。通道相干消融显示模型会利用通道关系，但没有替代架构对照，因此不能据此宣称显式复数或几何架构更优。

本阶段决策为 **REFINE**：不启动正式三 seed、进一步蒸馏确认或 final blind。单 seed 的 A/B/C 和开发结果只作筛选依据，不作统计显著或跨 seed 泛化结论。K2 软响应审计不改变 R3 的固定阈值门；T1 仍未获得正式固定阈值教师资格。

## 文件与复现

- 协议/配置：[`experiments/D2-A-E002/protocol.md`](../../experiments/D2-A-E002/protocol.md)、[`config.yaml`](../../experiments/D2-A-E002/config.yaml)
- 实际运行代码：[`scripts/run_d2a_e002.py`](../../scripts/run_d2a_e002.py)、`src/radardistill/` 下的数据、K2、训练、损失及评价模块
- 指标和结果：[`metrics.json`](../../experiments/D2-A-E002/metrics.json)、[`result.csv`](../../experiments/D2-A-E002/result.csv)、逐场景 CSV、训练历史、checkpoint、图表
- 研究分析：[`analysis.md`](../../experiments/D2-A-E002/analysis.md)、[`architecture_and_loss_review.md`](../../experiments/D2-A-E002/architecture_and_loss_review.md)
- 阶段审查：[`D2-A-E002_review.md`](../stage_reviews/D2-A-E002_review.md)
- 缓存清理审计：[`CACHE_CLEANUP_2026-10-25.md`](../maintenance/CACHE_CLEANUP_2026-10-25.md)

复现入口见协议第 8 节。执行环境为 AISAR / PyTorch 2.8.0+cu128 / CUDA 12.8 / RTX 3050 Laptop GPU。缓存清理后整个项目约 `8.8 GiB`，低于用户设定的 `20 GiB` 上限；没有为追求更低占用而删减仍需审查的模型、结果或审计材料。

## 下一步唯一优先任务

预注册一个有界诊断，区分 C3 条件虚警失稳与公共相位/频移等变失败的表示来源，并在协议、模型、预算和指标冻结及审查前不启动新训练。研究总纲保持不变。
