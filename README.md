# RadarDistill

**研究主题：** 物理模型与渐进式知识蒸馏驱动的高效雷达动目标检测及参数估计。

RadarDistill 研究怎样用物理仿真真值和传统雷达教师训练学生，从多通道复数回波完成 GMTI 目标检测与参数估计。教师知识用于训练和对照；学生推理不必复制教师。最终以任务质量约束下的精度、可靠性和完整处理成本判断纯学习或混合方法是否有价值。

## 核心问题

- 教师知识能否改善最终检测和参数估计，还是只拟合中间响应？
- 哪些复数相干物理先验能帮助低 SNR 检测和连续参数估计？
- 学生何时能采用不同于传统处理链的表示？哪些传统算子应保留？
- 在同硬件、同输入边界和完整链路下，学习方法能否带来精度或成本优势？

## 研究路线

- **D0：** 物理仿真、真值、传统教师与评价正确性。
- **D1：** 任务级检测、距离/Doppler 估计和渐进蒸馏。
- **D2：** 逐项增加三通道参数或杂波/弱目标场景。
- **D3：** 从距离压缩数据迁移到原始回波。
- **D4：** 任务可靠后优化结构与计算。
- **D5：** 同条件完整 GMTI 系统验证。

## 当前状态

D0-E001 简化仿真与 FFT 子链正确性检查 12/12 通过。D1-E001 的 G1/G2 通过、G3 未通过，冻结决定为 GO → D2。D1-E002 因低 SNR 检测保护门失败停止当前 AWGN Doppler 结构搜索；历史结论均保留。D2-A-E001 的 R0 T1–T12 全通过；后续 R3 固定阈值迁移决定为 **REFINE**，T1 未获固定阈值教师资格。

**当前状态：** D2-A-E002 v2.1 的 A/B/C 单 seed pilot 已完成，使用 RTX 3050 Laptop GPU CUDA 顺序训练。三个学生臂均通过 G4 任务学习门，但 G3 分层虚警门、G6 公共相位/频移一致性门失败，课程与 K2 未达到 G5 增益；阶段决定 **REFINE**。没有开展正式三 seed，也没有生成 final blind。原始生成缓存已清理，配置、manifests、检查点、训练日志、审计与高价值结果保留。

## 复现入口

使用 AISAR Conda 环境与 `cuda:0` 复现 D1-E001：

```bash
conda run --no-capture-output -n AISAR python scripts/run_d1_e001.py --config experiments/D1-E001/config.yaml --output experiments/D1-E001-reproduction --phase formal --device cuda:0
```

复核已保存的 D1-E001 blind predictions：

```bash
conda run --no-capture-output -n AISAR python scripts/evaluate_d1_e001.py --phase blind --replay-existing-predictions
```

## 资料入口

- 长期使命：[研究总纲](docs/RESEARCH_CHARTER.md)
- 当前与路线：[CURRENT_STATUS](docs/CURRENT_STATUS.md)、[RESEARCH_ROADMAP](docs/RESEARCH_ROADMAP.md)、[假设登记](docs/research/ASSUMPTION_REGISTER.md)
- 阶段审查：[D2-A-E001](docs/stage_reviews/D2-A-E001_review.md)、[D0/D1 E002 后总审查](docs/stage_reviews/D0-D1_post_E002_review.md)、[D1-R1](docs/stage_reviews/D1_R1_doppler_diagnosis.md)、[D1-E001](docs/stage_reviews/D1-E001_review.md)、[D1-E002](docs/stage_reviews/D1-E002_review.md)
- 实验档案：`experiments/D0-E001/`、`experiments/D1-E001/`、`experiments/D1-R1/`、`experiments/D1-E002/`
- D2-A-E002 阶段：[协议](experiments/D2-A-E002/protocol.md)、[分析](experiments/D2-A-E002/analysis.md)、[指标](experiments/D2-A-E002/metrics.json)、[审查](docs/stage_reviews/D2-A-E002_review.md)
- D2-A 复现：`python scripts/run_d2a_e001.py --config experiments/D2-A-E001/config.yaml --output-dir experiments/D2-A-E001 --batch-size 16`
- D2-A 物理模型、可辨识性与基线审查：[物理模型](docs/research/D2-A_physical_model.md)、[可辨识性](docs/research/D2-A_identifiability.md)、[基线审查](docs/research/D2-A_baseline_review.md)
- D2-A 测试：`pytest -q tests/test_d2a_simulation.py tests/test_d2a_baselines.py tests/test_d2a_evaluation.py tests/test_d0_simulation.py`
- E002 全套测试：`conda run --no-capture-output -n AISAR python -m pytest -q`（本次清理后 250 passed）
- 科研治理、任务执行与审查模板：[`docs/governance/`](docs/governance/)
- 详细历史计划：[research_plan.md](docs/research_plan.md)、[progress.md](docs/progress.md)、[decisions.md](docs/decisions.md)
