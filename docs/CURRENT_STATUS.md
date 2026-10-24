# 当前研究状态

**更新：** 2026-10-24
**当前实验依据提交：** 4f50df0（D1-R1 诊断代码版本；D0/D1 冻结结果来自各自实验提交）
**最新完成阶段：** D1-R1 诊断完成；D1-E001 原阶段 GO → D2（G1/G2 PASS、G3 FAIL）保持不变。
**最重要发现：** 距离压缩三通道复 IQ 学生能学习最终任务；部分蒸馏方法改善冻结最终联合指标，但没有收敛加速证据。
**主要瓶颈：** 低 SNR 检测弱；高 SNR 连续 Doppler RMSE 仍约 25–35 Hz，明显差于传统估计器。完整链速度优势未证明。
**当前唯一优先任务：** 冻结 D1-E002 protocol/config，随后按协议先做单 seed pilot；R1 已定位候选且未读取旧盲测。
**下一阶段批准：** 用户已批准 D1-E002 路线；R1 后优先 pilot S2 低成本复相干特征，S1 作单 seed pooling 反证，S3 仅作 FFT 混合诊断；正式门槛与新盲测仍待冻结。
**关键证据：** experiments/D1-R1/analysis.md、metrics.json、result.csv；docs/stage_reviews/D0-D1_overall_report.md；docs/stage_reviews/D1-E001_review.md。
**GPU：** AISAR / PyTorch 2.8.0+cu128，当前会话可见 RTX 3050 Laptop GPU（4 GiB）；模型训练不得 CPU 回退。
