# 当前研究状态

**更新：** 2026-10-24
**当前实验依据提交：** 4f50df0（D1-R1 metrics 记录的诊断代码 SHA；诊断报告/表格在 70d4c6c 提交，D0/D1 冻结结果来自各自实验提交）
**最新完成阶段：** D1-R1 诊断完成；D1-E002 protocol/config 已冻结但尚未训练；D1-E001 原阶段 GO → D2（G1/G2 PASS、G3 FAIL）保持不变。
**最重要发现：** 距离压缩三通道复 IQ 学生能学习最终任务；部分蒸馏方法改善冻结最终联合指标，但没有收敛加速证据。
**主要瓶颈：** 低 SNR 检测弱；高 SNR 连续 Doppler RMSE 仍约 25–35 Hz，明显差于传统估计器。完整链速度优势未证明。
**当前唯一优先任务：** 严格按冻结协议顺序运行 S1/S2/S3 单 seed pilot，并执行预设筛选门；没有候选过门时不训练 formal、不生成新 blind。
**下一阶段批准：** 用户已批准 D1-E002 路线。S2 为主纯 IQ 候选，S1 为一次性反证，S3 是独立 FFT 混合方法；formal 三种子、calibration 和全新 blind 只有在各自 pilot 门通过后才开展。
**关键证据：** experiments/D1-R1/analysis.md；experiments/D1-E002/protocol.md 与 config.yaml；docs/stage_reviews/D0-D1_overall_report.md；docs/stage_reviews/D1-E001_review.md。
**GPU：** AISAR / PyTorch 2.8.0+cu128，当前会话可见 RTX 3050 Laptop GPU（4 GiB）；模型训练不得 CPU 回退。
