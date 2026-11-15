# D2-A-R6-G 阶段报告

**内部阶段结论：REFINE**

**实验协议：** [`experiments/D2-A-R6/R6-G/protocol.md`](../../experiments/D2-A-R6/R6-G/protocol.md) v1.0.1

**实际指标：** [`metrics.json`](../../experiments/D2-A-R6/R6-G/metrics.json)
**分析和训练曲线：** [`analysis.md`](../../experiments/D2-A-R6/R6-G/analysis.md)、[`training_task_curves.png`](../../experiments/D2-A-R6/R6-G/figures/training_task_curves.png)

## 研究主线

R6-G 服务于 RadarDistill 总纲关于任务级学生、教师知识迁移和学生学习机制的目标。本阶段比较物理真值监督、真值加 K3 全 RD 教师软响应、以及匹配形状的真值辅助响应；要判断教师是否真正帮助学生，必须看独立场景上的检测、距离、Doppler 和 joint 结果。

## 执行和证据

- G0/G1 通过；RTX 3050 Laptop `cuda:0`、FP32 顺序训练，三臂均完成 51,200 updates、effective batch64、microbatch4，无 OOM 或 CPU 回退。
- 训练数据为 51,200 个配对场景，H0 calibration 和独立 verification 各 120,000，H1 development 6,144；使用单个训练 seed 707，不生成 final blind。
- T1−T0 的 C2+C3 joint 差为 `−0.00130`（95% paired CI `[−0.00781,+0.00521]`），预注册 `+0.02` 增益门未通过；Range 与 Doppler RMSE 保护上界也未通过。
- T1−T3 joint 差为 `−0.01302`（95% CI `[−0.01953,−0.00673]`）。K3 教师没有胜过 truth-assisted 对照。
- T3−T0 为 `+0.01172`（95% CI `[+0.00499,+0.01823]`），低于 `+0.02` 门，且 Range 保护上界为 1.0576；该组是真值辅助控制，不属于教师蒸馏。
- 三臂独立 H0 verification 的全局 Pfa 均接近 0.05，但 C3 pooled Pfa 约为 0.0995–0.1005，C3 六格 familywise 上界门全部失败。
- 在 C3 / partial-overlap / 0 dB / CNR20 困难格，K3 定位正确率为 `25/512 = 4.88%`；T1 joint 为 0.1738，T0 为 0.1934。该差作为风险信号报告，不外推为单格显著性结论。
- 三臂 validation truth loss 及注册点任务指标在 12,800 到 51,200 步总体改善，但最佳 checkpoint 早于末步；51,200 不是已证实的收敛上限。

## 工程、科学性与方向判断

训练、数据分组、独立校准/验证和配对评价按冻结设置完成。评价指标中的运行 HEAD 为 `737245afe33a5a677bcfb42bee6b8e80a28adca6`；训练 owner/G0/G1 绑定的 SHA 为 `3a986699f52836293f159ec850033de7c5424f69`。协议、配置和训练源哈希匹配冻结预检；后续提交增加的是评价入口及阶段文档，训练核心没有变化。

科学结论是：**本次固定的 K3 全图响应没有改善 CE-TCN 的最终任务表现，也没有支持教师知识带来更快且有效的学习。** T3 小幅 joint 正差不能归因于教师，且未满足既定增益和参数保护合取门。教师在困难 C3 条件下的定位错误仍是明确风险。R6-G 不支持继续搜索当前 K3 权重、温度、时机或可靠性策略。

本结果不推翻教师蒸馏总方向，也不证明所有教师形式无效；它缩小了当前 K3/CE-TCN/单 seed 配置的可行范围。绝对 C3 虚警资格仍未满足。保留 R5-C、R6-E/F 的所有历史结论，不修改研究总纲 v1.3。

## 停止条件和后续

停止本轮 K3 全图教学分支、单臂追训、权重/温度搜索和 final blind。R6-H 的复数对角 SSM/混合架构实验仍是 `execution_authorized: false` 草案，须先完成正式场景 ID 审计、冻结协议和 G0/G1；当前结果不自动批准该训练。P7 在路线内下一阶段完成或按其预注册门停止后再决定。

训练缓存按 owner 清单清除；保留 scene manifest、metrics、本地逐场景预测、训练 history/records、检查点、协议和本报告。cleanup audit 见 [`cleanup.json`](../../experiments/D2-A-R6/R6-G/cleanup.json)。没有清理其他阶段或用户未跟踪文件。
