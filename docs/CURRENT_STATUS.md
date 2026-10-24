# 当前研究状态

**更新：** 2026-10-24
**D1-R1 诊断代码依据：** 4f50df0（metrics 中记录的诊断代码 SHA；诊断报告/表格在 70d4c6c 提交）。D1-E002 的训练提交、选择器修正和实验产物提交分别记录在对应 run manifest 与 Git 历史中。
**最新完成阶段：** D1-R1 诊断及 D1-E002 pilot、S1/S2 三 seed formal CUDA 训练完成；D1-E001 原阶段 GO → D2（G1/G2 PASS、G3 FAIL）保持不变。
**最重要发现：** 距离压缩三通道复 IQ 学生能学习最终任务；部分蒸馏方法改善冻结最终联合指标，但没有收敛加速证据。
**主要瓶颈：** 低 SNR 检测弱；高 SNR 连续 Doppler RMSE 仍约 25–35 Hz，明显差于传统估计器。完整链速度优势未证明。
**当前唯一优先任务：** 提交 50,000-H0 calibration 分数、阈值和环境记录；之后按冻结阈值生成一次 final blind 并完成统计/误差分析。
**下一阶段批准：** 用户已批准 D1-E002 路线。pilot 按原门筛入 S1/S2，S3 停止；两个入选候选均已完成三个正式种子。12-checkpoint manifest 已冻结，独立 calibration 已完成；final blind 尚未生成。
**关键证据：** experiments/D1-R1/analysis.md；experiments/D1-E002/protocol.md、config.yaml、pilot/、formal/；docs/progress.md、docs/decisions.md；docs/stage_reviews/D0-D1_overall_report.md；docs/stage_reviews/D1-E001_review.md。
**GPU：** AISAR / PyTorch 2.8.0+cu128，当前会话可见 RTX 3050 Laptop GPU（4 GiB）；模型训练不得 CPU 回退。
