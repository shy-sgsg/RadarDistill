# 当前研究状态

**更新日期：** 2026-10-25
**工作分支：** `main`
**最新完成工作：** D2-A-E001 静止地杂波仿真、R0/R1 基线和阶段审查；修复并重跑 RMSE bootstrap 区间；R2 已对齐 D1-E001 学生架构，仍未执行训练。
**阶段判断：** D1-E001 的 GO → D2 保持；D1-E002 的 AWGN Doppler 结构搜索 STOP / REDIRECT；D2-A 为 CONDITIONAL GO。

## 主要发现

在 D2-A 的合成静止杂波场景中，T1–T12 全部通过。B3 几何 DPCA 的分离目标杂波抑制中位数为 15.41 dB；A/B 检测明显，C 部分重叠时退化，D 的同流形源归因接近随机。开发主条件 A 的 B3 `Pfa_CPI=0.0898`，高于独立标定经验值 0.0508；不得据此宣称阈值已可靠迁移，也不得用开发 H0 回调阈值。

Range/Doppler RMSE 置信区间已修正为对每次 scene-cluster bootstrap 样本重新计算 RMSE 后取 percentile；更新后的 `analysis.md` 同时报出两类条件 RMSE 区间。R2 固定采用 D1-E001 `TaskStudent`，匹配 1,600 次更新和有效 batch 64；K1/K2 仅为训练期辅助图，不改 K3 推理结构。

D1-E002 的 S2 lag-one 复相干学生在 0/5/10/20 dB 的等权 Doppler RMSE 为 23.352 Hz，较 S0 改善 10.86%，但低 SNR Pd 非劣门失败；四倍 FFT 抛物插值、APX15 和有限网格 profile-ML 为 0.104、0.122、0.100 Hz。D1-E001 的 C/D 满足原冻结联合指标 G2，但没有蒸馏收敛加速证据；旧阶段结论均保留。

## 唯一优先任务

先预注册并解决 **D2-A 独立阈值迁移/虚警稳健性**：明确检验固定标定阈值在独立场景及幅度变化下的 `Pfa_CPI`，不得用当前开发集重新调参。通过后再启动 R2 的 E1 真值监督学生 pilot；E1 未证明基础任务学习前，不开展 E2/E3 蒸馏，也不生成最终 blind。

## 关键材料

- 总纲/路线：[研究总纲](RESEARCH_CHARTER.md)、[动态路线图](RESEARCH_ROADMAP.md)、[假设登记](research/ASSUMPTION_REGISTER.md)
- 审查：[D2-A-E001](stage_reviews/D2-A-E001_review.md)、[D0/D1 E002 后总审查](stage_reviews/D0-D1_post_E002_review.md)、[D1-R1 Doppler 诊断](stage_reviews/D1_R1_doppler_diagnosis.md)、[D1-E001](stage_reviews/D1-E001_review.md)、[D1-E002](stage_reviews/D1-E002_review.md)
- 数据与结果：[`experiments/D2-A-E001/`](../experiments/D2-A-E001/)、`experiments/D0-E001/`、`experiments/D1-E001/`、`experiments/D1-R1/`、`experiments/D1-E002/`
- D2-A 物理模型、可辨识性与传统基线审查：[`D2-A_physical_model.md`](research/D2-A_physical_model.md)、[`D2-A_identifiability.md`](research/D2-A_identifiability.md)、[`D2-A_baseline_review.md`](research/D2-A_baseline_review.md)
- 环境：AISAR / PyTorch 2.8.0+cu128 / CUDA 12.8 / RTX 3050 Laptop GPU，4 GiB；正式训练必须使用可见 CUDA。
