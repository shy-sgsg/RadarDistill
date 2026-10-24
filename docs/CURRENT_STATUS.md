# 当前研究状态

**更新：** 2026-10-24
**当前实验依据提交：** 38d73b6（D0/D1 结果与阶段判断对应的仓库版本）
**最新完成阶段：** D1-E001；D1 阶段 GO → D2（G1/G2 PASS、G3 FAIL）。
**最重要发现：** 距离压缩三通道复 IQ 学生能学习最终任务；部分蒸馏方法改善冻结最终联合指标，但没有收敛加速证据。
**主要瓶颈：** 低 SNR 检测弱；高 SNR 连续 Doppler RMSE 仍约 25–35 Hz，明显差于传统估计器。完整链速度优势未证明。
**当前唯一优先任务：** D1-R1，使用训练/验证数据诊断 Doppler 误差来源，不触碰已用作最终评价的盲测预测。
**下一阶段批准：** 用户已批准 D1-R1 及其后 D1-E002 的整体路线；D1-E002 的具体结构、预算、统计门和独立盲测协议尚待 R1 后冻结。
**关键证据：** docs/stage_reviews/D0-D1_overall_report.md；docs/stage_reviews/D1-E001_review.md；experiments/D1-E001/analysis.md。
**GPU：** AISAR / PyTorch 2.8.0+cu128，当前会话可见 RTX 3050 Laptop GPU（4 GiB）；模型训练不得 CPU 回退。
