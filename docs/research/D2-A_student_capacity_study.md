# D2-A 学生训练充分性与容量研究

## 问题

E003 每次更新消费 32 对场景，51,200 对场景在 1,600 updates 中恰好遍历一次。其单 seed S0/S2 是 1-epoch pilot。R5-A 将“固定样本重复训练”和“新增场景覆盖”分开，先测训练曲线，再测容量。

## 预注册对照

| 对照 | 主干宽度 | 参数量 | 变量 |
|---|---:|---:|---|
| Small | 32 | 13,383 | 当前 TaskStudent |
| Medium | 96 | 82,887 | 仅 width 改变 |

训练充分性先对 E003 S0 与 S2 从头训练 seed 101，使用同一 51,200 对固定 train、5,000 对 validation、匹配初始化/顺序/loss/AdamW 与 CUDA FP32。主预算 6,400 updates（4 次遍历），每 400 updates 留训练与验证记录。训练从头开始并采用预注册完整 cosine LR；不续接旧 E003 checkpoint。

若 4,800–6,400 之间验证真值任务损失相对下降至少 1%，且末 5 个预定点至少 3 段连续改善，同时无数值异常或明确任务保护退化，则同一训练状态允许 S0/S2 两臂都延长至 12,800 updates。扩展时不重置 optimizer、scheduler、RNG 或数据游标；不满足条件则停止于 6,400。更新预算是合理观察上限，不等同理论收敛证明。

容量对照先用 Small 与 Medium 在匹配预算下比较。训练集 loss 持续高且随 Medium 改善支持容量/拟合限制；训练好而验证退化支持过拟合；训练曲线异常支持优化困难；预算与容量增加后任务不变且物理信息可观测，才支持任务/表示瓶颈。所有判断并列 Pd/Pfa、range/Doppler、joint success、C0–C3、训练/推理资源。

只在 Medium 对至少一个预注册主任务显示有意义改善、且没有主要任务明显退化时考虑 Large；Large 需要独立协议和显存预算，不做宽度网格搜索。

## 评估和限制

检测阈值用独立 H0 calibration，H0 verification 不用于调阈值。H1 development 按 scene 配对 bootstrap；single seed 不表示 seed 方差。旧 E003 development 曾参与选择，不能作为最终 blind。固定数据重复训练的泛化与在线生成新场景的泛化是不同问题，只有观察到过拟合才另行预注册新场景对照。
