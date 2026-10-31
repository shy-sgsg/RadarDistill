# D2-A-R6-C 阶段审查

**阶段结论：REFINE；继续 R6-D 已授权路线。**

**证据提交：** `a34feea10822ac4ef4dfbe9b64771826aa1e80df`。协议 `experiments/D2-A-R6/R6-C/protocol.md` v1.2，研究总纲 `docs/RESEARCH_CHARTER.md` v1.3。所有 H1 结果均为 development，未生成 final blind；单训练种子不估计初始化方差。

## 总纲对齐与问题

R6-C服务于 RadarDistill 的任务级 GMTI 学生目标，主要对应总纲 Q3（模型与表示）、Q4（训练优化）、Q6（困难检测和参数估计）及 Q10（学生学习机制）。它将容量、独立训练场景数量和训练延长分开考察，没有引入教师、蒸馏或物理前端。

## 执行与工程审查

- 按协议 v1.2，在 RTX 3050 Laptop `cuda:0`、FP32、seed 404、有效 batch 64 下顺序训练 Medium/Large/XLarge；活动参数分别为 70,277/292,005/1,012,197。history 每 400 updates 记录，resume 状态每 800 updates 保存，没有逐 update 追加日志。
- 三模型规模臂各运行 12,800 updates。总训练时长分别为 4,239/7,405/17,156 秒，峰值 CUDA allocated 分别约 0.41/0.88/1.63 GiB；均无 CPU 回退或 OOM。Medium 在 update 12,000、Large 与 XLarge 在 update 12,400 获得最低共同 validation task loss。
- Medium 的数据规模对照使用同 seed、同有效 batch、同更新数和相同总样本暴露：51,200 与 102,400 个唯一场景对；训练时长 4,239/4,679 秒，峰值 allocated 均约 0.41 GiB。
- 每臂使用独立 H0 calibration 120,000、H0 verification 120,000、H1 development 6,144 个场景。评价 split 的 12 份预测归档均已生成。第一次统计汇总因 NumPy 不支持布尔向量直接相减而失败；修复为两端先转 float64 再求有符号差，并复用了全部 12 份归档完成统计，没有重复推理。回归用例及审计见 `experiments/D2-A-R6/R6-C/attempts/evaluation_aggregation_fix.json`。
- R6-C owner 清理器删除了其清单拥有的缓存 21,467,222,779 bytes；清理后项目为 1,205,698,287 bytes，小于 20 GiB。记录见 `experiments/D2-A-R6/R6-C/cache_cleanup.json`，未删除用户未跟踪文件。

## 主要结果

主要工作点为各模型独立 H0 calibration 阈值对应的 global `Pfa_CPI=0.05`，主任务指标为可辨识 C2+C3 joint success。

| 模型 | C2+C3 joint | 相对 Medium joint 差（配对 CI） | C3 pooled Pfa 单侧 95% 上界 | C3 六格 familywise 门 |
|---|---:|---:|---:|---|
| Medium | 0.8164 | 参照 | 0.10162 | FAIL |
| Large | 0.8628 | +0.04644，97.5% CI [0.03646, 0.05643] | 0.10207 | FAIL |
| XLarge | 0.8704 | +0.05404，97.5% CI [0.04449, 0.06402] | 0.10266 | FAIL |

Large 与 XLarge 均越过预注册的相对 joint 增益和配对区间门；总体 Pd、range 与 Doppler RMSE 配对保护也通过。但两者 C3 pooled 精确上界均高于 0.10，六个 C3 子格的 familywise 上界门也未全部通过。因此，两个大模型均未通过正式 seed 晋级门；不能据相对任务收益宣称获得可靠检测器，也不进入多 seed 或 final blind。

Medium 从 51,200 扩至 102,400 唯一训练场景对、保持 12,800 updates 和有效 batch 64 后，C2+C3 joint 差为 +0.00673，95% CI [−0.00066, +0.01455]，没有达到 +0.02 实质增益且区间跨零。Doppler RMSE 比值为 0.953，95% CI [0.917, 0.993]；range RMSE 比值为 0.973，95% CI [0.921, 1.025]。C3 pooled Pfa 单侧上界为 0.10412，保护门失败。故数据规模对照只提供有限的 Doppler 次级信号，不支持主要 joint 泛化假设。

## 训练预算解释

按冻结 v1.2，selected extension pair 为 Medium 与仅依 validation loss 选出的 XLarge。Medium 在 11,200–12,800 窗口的 task loss 相对降幅 0.047%，且四段中只有两段满足 nonworsening；XLarge 窗口内 task loss 增加约 1.17%，也只有两段 nonworsening，训练—验证 gap 增加 0.01442，超过 0.01 上限。两臂未共同通过续训门，故没有训练到 25,600 或 51,200 updates。

此结果表示 R6-C 的预注册延长条件未触发，不表示 12,800 是模型极限，也没有直接测得更长训练的因果收益。后续新实验应根据科学问题预注册更充分的共同更新预算，并让比较臂匹配有效 batch、训练数据和样本暴露、学习率计划、验证及 checkpoint 选择；所有日志继续限于固定间隔/检查点，不逐 update 落盘。若训练至充分预算后验证曲线仍改善，应报告预算内未观察平台，不宣称收敛。

## 假设与研究影响

- **H-CAP-003（容量）：有限支持。** 在本单 seed、静止杂波 development 范围内，扩大同拓扑主干显著提高 C2+C3 joint 并改善参数误差；绝对 C3 虚警保护失败，不能晋级为正式学生候选。
- **H-CAP-004（更多唯一训练场景）：主假设未获支持。** joint 收益未达门；较低 Doppler RMSE 为次级证据，且 C3 Pfa 上界恶化，需限制解释。
- **H-CAP-001（延长训练）：R6-C 未检验更长预算。** 两臂未过门即停止，所以不能据此支持或否定 25,600/51,200 updates 下的学习能力。
- 阶段推进了容量与数据多样性归因，但 C3 false alarm 仍是候选保护瓶颈。没有证据要求修改研究总纲 v1.3。

## 下一步

按 R6 既定顺序进入 R6-D：先完成 J1 D/P/J 与 J2 软候选双向交互的实现、CUDA 资源和正确性门，再冻结 development pilot。其新 protocol 应明确比 12,800 更充分的匹配更新预算，并保持 D-only、P-only、J1/J2 对照的有效 batch、训练样本暴露和 checkpoint 规则一致；不得把训练预算差异混入联合任务/交互收益。继续保留 C3 H0 独立保护；R6-C不追加训练、不进入多 seed或blind。R6整项 P0–P7 完成后再提交一次综合 ChatGPT 审查材料。
