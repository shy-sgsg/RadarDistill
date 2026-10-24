# 当前研究状态

**更新：** 2026-10-25
**D1-R1 诊断代码依据：** 4f50df0（metrics 中记录的诊断代码 SHA；诊断报告/表格在 70d4c6c 提交）。D1-E002 的训练提交、选择器修正和实验产物提交分别记录在对应 run manifest 与 Git 历史中。
**最新完成阶段：** D1-E002 独立盲测与阶段审查完成；D1-E001 原阶段 GO → D2（G1/G2 PASS、G3 FAIL）保持不变。
**最重要发现：** S2 lag-one 复相干读出相对 S0 的高 SNR Doppler RMSE 改善 10.86%，且 95% 区间低于零；但低 SNR 检测保护门未通过。S1 未改善主指标。两个学生都远落后四倍 FFT 插值、APX15 和有限网格 profile-ML。
**主要瓶颈：** 频率误差和公共相位/频移一致性仍不足；D1-E002 按预注册协议为 **STOP / REDIRECT**，不继续增加本轮架构或训练变体。D1-E001 的既有结论不被覆盖。
**当前状态：** D1-E002 使用了 50,000-H0 calibration、40,960 条新盲测（32,768 H0 + 8,192 H1），每个校准分数经验 Pfa=0.00998。首次盲测在 profile-ML 配置读取错误处中断，随后用同一 blind cache 和已保存预测完成重放；没有重生成数据、重训模型或重校准阈值。
**关键证据：** experiments/D1-R1/analysis.md；experiments/D1-E002/protocol.md、config.yaml、pilot/、formal/、calibration/、blind/；docs/progress.md、docs/decisions.md；docs/stage_reviews/D1-E002_review.md；D1-E001 与 D0-D1 历史审查。
**GPU：** AISAR / PyTorch 2.8.0+cu128，当前会话可见 RTX 3050 Laptop GPU（4 GiB）；模型训练不得 CPU 回退。
