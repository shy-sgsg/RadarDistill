# D2-A-R5-A Small/Medium 容量阶段自审

## 审查基准

- **执行代码提交：** `1678a468d5c150ff734d80ad40cf75407a49651f`；容量 config、runner、规格校验和回归用例均在该提交中。
- **研究总纲：** `docs/RESEARCH_CHARTER.md` v1.3；未修改总纲正文。
- **范围：** seed 101 的 Small width=32 与 Medium width=96 单因素 E1 truth-only capacity pilot。没有 KD、多 seed、Large 或 final blind。
- **主要材料：** `experiments/D2-A-R5/R5-A/capacity/{config.yaml,preflight.json,g1_overfit_sanity.json,training.json,metrics.json,result.csv,analysis.md}`，两模型 history、预测表、ROC 与图；训练入口 `scripts/run_d2a_r5_capacity.py`。
- **检验：** R5 定向测试 18 项通过；脚本语法/入口检查通过；两模型实际训练与评价均使用 `cuda:0`。

## 阶段判断

**REFINE。** Medium 相对 Small 的 C2/C3 joint success 提升 `+0.13086`（paired scene bootstrap 95% CI `[+0.11230,+0.15088]`），且范围和 Doppler RMSE 比均低于1；但两模型 C3 H0 Pfa 分别 `0.20410` 与 `0.18896`，都超过预注册的 0.10 每模型上限。G3 任务保护失败，因此 Medium 尚不能晋级为合格候选，也不授权 Large。

## 1. 工程正确性

容量 runner 复用 R5-A 的 split loader、pair plan、训练实现、阈值校准和 paired-bootstrap 评价；Small 使用 R5-A S0 最佳 checkpoint，Medium 独立初始化 width=96。训练代码只改变 width；有效 batch 64、FP32、AdamW、schedule 和 12,800 updates 保持一致。两个模型顺序驻留 CUDA。

RTX 3050 Laptop GPU 上 Medium 实际以 microbatch 16 完成训练，没有 OOM。G1 小批过拟合 loss 降至初值的 `0.00057`，presence/range-cell 准确率为1.0，任务头梯度有限且非零。容量评估中每模型各自用 calibration 阈值，verification 与 development 数据继续分开记录。未发现本阶段执行器、模型维度、 checkpoint 或 threshold comparator 的已知错误。

Small 的历史训练记录没有累计墙钟时间；因此只能报告 Medium 的 `3,951.3 s`，不能声称 Medium 比 Small 快/慢多少。单 seed 未估计训练 seed 方差。训练和评估均使用 E003 已使用 ID，不能称为 blind。

## 2. 实验科学性

两模型都完成冻结预算，最佳验证任务损失为 Small `0.32059`（update 12,400）与 Medium `0.24346`（update 12,800）。Medium 的全部 H1 development 指标总体改善：Pd `0.96191→0.97461`、AUC `0.99374→0.99583`、range RMSE `2.239→1.879 cells`、Doppler RMSE `25.23→17.86 Hz`、joint success `0.83887→0.91064`。

C2/C3 primary joint-success 差达到预注册的最小 `0.02` 增益且区间下界大于0。Pd、overall joint、C2/C3 joint、RMSE ratio 和相对 C3 Pfa 差的保护分量均通过。失败来自两个模型共同的 C3 绝对 Pfa：Small `836/4096=0.20410`、Medium `774/4096=0.18896`；单侧 exact 95% 上限分别 `0.21474`、`0.19931`。Medium 的差值 CI 未显示相对恶化，但阈值跨 C3 场景的迁移仍明显失败。

这组结果显示容量对可辨识困难场景的参数估计有实际影响，但不能把单次 development bootstrap 解释成跨 seed 泛化或独立测试。C3 Pfa 也不是宽度增加能解决的现象；训练增加与容量增加均未令其接近0.10限值。

## 3. 研究主线贡献

- **H-CAP-001：** 仍为限定支持。更大训练预算提升了本 seed 的学习曲线；不代表收敛或新场景泛化。
- **H-CAP-002：** 更新为有限支持：width=96 在本开发分布的 C2/C3 joint 与参数误差上明显优于 width=32；完整 G3 未通过，因为绝对 C3 Pfa 保护失败。
- **H-ARCH-003：** 未测试。宽度不同的 CNN 对照不能支持或否定原始 IQ 非 CNN 架构。
- **知识迁移假设：** 本阶段未训练教师或蒸馏学生，不能对教师知识价值作结论。

本阶段服务于总纲任务级 GMTI 学生的训练充分性和容量问题，同时揭示 joint accuracy 与目标检测虚警稳定性之间仍需共同约束。项目长期目标仍未由这一单 seed development pilot 证明。

## 4. 战略判断

不运行 Large、KD 或 blind。Medium 的容量结果足以作为下一架构 pilot 的 CNN 参照，但未达到候选晋级门。下一项唯一优先任务为 R5-B：完成针对当前三通道静止杂波 GMTI 的非 CNN 架构调查，随后只实现一个轻量、保留原始复 IQ 的时频注意力候选，先过物理正确性、参数量、CUDA 显存和小批可学习性门，再与 Small/Medium 的有界开发材料比较。R5-B 继续保留每模型 C3 H0 Pfa ≤0.10 的保护要求；不得因为 Medium 的 C3 Pfa 失败而回调门槛或使用已看过的开发数据调阈值。

教师质量审查与蒸馏继续等待 R5-C 前置门。当前结果没有证据要求修订研究总纲；状态与路线写入动态文件。
