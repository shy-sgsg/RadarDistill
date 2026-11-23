# D2-A-R6-H 阶段内部审查

**审查类型：** 内部阶段验收；不是 ChatGPT 中途审查请求。
**审查基准：** 训练/评价协议与代码来自冻结提交 `70a3e6a0b84d3be1613a7653ac018171b11b0cf1`；结果记录于 `experiments/D2-A-R6/R6-H/metrics.json`。
**研究总纲：** `docs/RESEARCH_CHARTER.md` v1.3（本阶段未修改）。
**阶段结论：** **REFINE**。

## 研究主线

本阶段对应总纲 Q3、Q6、Q8、Q10：检验复数相干慢时间输入下的 SSM 和局部卷积+SSM 是否具备任务学习能力，并判断是否产生可进入后续知识迁移或正式确认的学生候选。R6-H 使用物理真值监督，**没有测试教师知识或蒸馏收益**；它不能作为“教师教得更好”的证据。

## 工程与执行审查

- 冻结的 G0/G1 均通过；三臂按 CE-TCN→SSM→CNN-SSM 顺序，在 AISAR CUDA 环境和 RTX 3050 Laptop `cuda:0` 上各完成 25,600 updates。
- 配置、数据划分、有效 batch 64、FP32、更新预算和 checkpoint 选择保持冻结；三臂都没有静默 CPU 回退或 OOM 记录。
- 独立 H0 calibration/verification 各 120,000，H1 development 6,144；`metrics.json` 状态为 evaluated，`final_blind_generated=false`。
- 第一次独立评价在 SIGTERM 后中断，使用原冻结入口完成重跑；已完成的 CE-TCN 和 SSM 两臂与中断前 partial 逐项一致。恢复记录保存在对应 `attempts/` 目录。
- CE-TCN/SSM `training_record.json` 的显存字段与同臂 `training_history.csv` 峰值不一致。报告采用 history 的逐检查点最大值并保留原记录；此项不影响已经完成的模型评价，但需避免把不一致的 record 显存字段引用为峰值。
- 训练/验证缓存依冻结协议 owner 清理，共删除 11,233,318,401 bytes（约10.46 GiB）；结果、训练 history、scene manifest 和阶段审计记录保留。清理后工作区实测约1.27 GiB，明细见 `cleanup.json`。

## 冻结阶段门与结果

| 项目 | 结果 | 判断 |
|---|---|---|
| G0：设备、场景划分、资源、参数量 | PASS | RTX 3050 Laptop，CUDA 设备符合协议，scene-ID 审计通过 |
| G1：模型正确性和小样本学习 | PASS | 三臂通过冻结前置检查 |
| SSM−CE-TCN C2+C3 joint | +0.01931，97.5% CI `[+0.01085,+0.02778]` | 点增益低于 +0.02，主筛选未过 |
| CNN-SSM−CE-TCN C2+C3 joint | +0.03016，97.5% CI `[+0.02214,+0.03798]` | 主端点通过 |
| Pd / Doppler 保护 | 两候选均通过 | 仅这些保护项通过 |
| Range RMSE 保护 | SSM ratio 1.037、上界1.127；CNN-SSM ratio1.070、上界1.165 | 均未通过冻结的上界1.05 |
| C3 pooled H0 Pfa | 0.09715 / 0.09875 / 0.09877 | 点值接近0.10；SSM/CNN-SSM 单侧 exact 上界超过0.10 |
| C3 六格 familywise Pfa | 三臂均失败；最差格上界0.10596 / 0.10905 / 0.11235 | 未获 C3 低虚警资格；CNN-SSM 最难格点 Pfa=0.1048 |
| 公共相位旋转 | 最差决策一致率≥0.99967 | 近似不变性得到支持 |
| `±130 Hz` 全场景频移 | Doppler 残差 p95 405.7–531.0 Hz，Range p95 变化约20.5–21.0 cells | 未显示频移等变性 |

完整条件分层、置信区间、训练曲线和物理诊断见 [`R6-H analysis`](../../experiments/D2-A-R6/R6-H/analysis.md) 与 [`metrics.json`](../../experiments/D2-A-R6/R6-H/metrics.json)。

## 科学解释与限制

CNN-SSM 的 C2+C3 joint point gain 达到预注册主端点，但同时未通过 Range RMSE 保护；它在最难 C3 条件的 H0 Pfa 点估计为 0.1048，H1 Pd 0.414、joint 0.219。故不能按总体 joint 提升晋级为可靠低虚警模型。纯 SSM 未达到主端点阈值，也未通过 range 保护。

三臂 validation best 都落在 25,600 update 末端，且 22,400→25,600 仍有约 1.0%–1.9% 的净下降。训练预算没有证明平台；本阶段不声称模型极限，也不单臂追训。结果是 seed808 单次开发评价，只适用于冻结架构、仿真分布与预算；不能推广至所有 SSM/Transformer，也不是 final blind。

## 阶段决定与后续

R6-H 的真值监督架构筛选完成。当前三个候选没有一个同时达到 joint 收益和任务保护条件，且没有一臂通过 C3 六格 familywise 虚警门；因此不进入本候选组的 P7 多种子正式晋级，也不生成 final blind。该决定是对当前候选和证据的 **REFINE**，不是对整个 SSM/混合架构方向的 STOP。

下一项唯一优先研究为：冻结一个小范围、只使用可观测输入的 C3 检测分数/背景尺度稳健化实验，重点检查 C3 partial-overlap 的 H0 分数漂移与 H1 可分性；必须保留 CE-TCN、CNN-SSM 与 B3 强基线，并独立校准/验证。该实验须另行预注册后执行，不修改 R6-H 指标或结论。待它产生合格学生候选后，再评估是否启动多种子 P7。
