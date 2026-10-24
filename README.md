# RadarDistill

**物理模型与渐进式知识蒸馏驱动的高效雷达动目标检测及参数估计方法研究**

RadarDistill 研究如何结合可信物理仿真真值和传统雷达教师，让学生从多通道复数回波完成 GMTI 目标检测与参数估计。训练时可以使用教师知识；最终推理不必复制教师。研究不预设 AI 优于传统算法，目标是在任务质量约束下找到有价值的学习或混合方案。

## 当前状态

D0-E001 仿真/教师子链正确性检查 12/12 通过。D1-E001 的阶段决定为 GO：学生从距离压缩三通道 IQ 学习了检测、距离和 Doppler，部分蒸馏策略改善最终联合指标，但未改善收敛速度。低 SNR 检测较弱，高 SNR Doppler RMSE 仍约 25–35 Hz，明显差于传统精估计器；完整链速度优势尚未证明。

D1-R1 Doppler 误差诊断已完成。D1-E002 已完成 CUDA pilot、S1/S2 三 seed formal、独立 calibration 和一次 final blind。S2 的高 SNR Doppler RMSE 相对 S0 改善 10.86%，但低 SNR 检测非劣门未通过；S1、S2 均远落后四倍 FFT 插值和 profile-ML。按冻结协议，D1-E002 决策为 **STOP / REDIRECT**，不再扩展本轮结构搜索。首次盲测的 profile-ML 配置读取错误已通过同一盲测缓存重放修正；没有重生成盲测数据。D1-E001 的阶段结论保持不变。详见 [D1-E002 阶段审查](docs/stage_reviews/D1-E002_review.md)。

## 研究主线

- **D0：** 可信物理仿真、真值、传统教师和统一评价。
- **D1：** 任务级检测与参数估计、渐进式蒸馏及 Doppler 诊断。
- **D2：** 逐项增加三通道参数或复杂场景难度。
- **D3：** 从距离压缩输入迁移到原始回波。
- **D4：** 在任务可靠后优化结构和计算。
- **D5：** 同硬件、同输入边界验证完整 GMTI 处理链。

核心科学问题包括教师知识是否真正改善最终任务、哪些中间知识有效、学生能否采用不同于传统链的表示、如何提高低 SNR 检测和连续参数估计，以及学习或混合方法何时有完整成本优势。

## 快速入口

- 长期使命、范围、假设和成功标准：[docs/RESEARCH_CHARTER.md](docs/RESEARCH_CHARTER.md)
- 当前状态：[docs/CURRENT_STATUS.md](docs/CURRENT_STATUS.md)
- 动态路线图：[docs/RESEARCH_ROADMAP.md](docs/RESEARCH_ROADMAP.md)
- 假设登记：[docs/research/ASSUMPTION_REGISTER.md](docs/research/ASSUMPTION_REGISTER.md)
- D0/D1 汇总与阶段审查：[D0-D1 总结](docs/stage_reviews/D0-D1_overall_report.md)、[D1-E001 审查](docs/stage_reviews/D1-E001_review.md)
- D1-R1 诊断：[协议](experiments/D1-R1/protocol.md)、[结果分析](experiments/D1-R1/analysis.md)、metrics.json 和 result.csv
- D1-E002 预注册：[协议](experiments/D1-E002/protocol.md)、[配置](experiments/D1-E002/config.yaml)
- 冻结配置和实验分析：[D0-E001](experiments/D0-E001/)、[D1-E001](experiments/D1-E001/)
- 编码和审查协议：[docs/governance/](docs/governance/)
- 研究方案细节与历史：[research_plan.md](docs/research_plan.md)、[progress.md](docs/progress.md)、[decisions.md](docs/decisions.md)

## 复现入口

使用 AISAR Conda 环境和可见 CUDA 复现 D1：

    conda run --no-capture-output -n AISAR python scripts/run_d1_e001.py --config experiments/D1-E001/config.yaml --output experiments/D1-E001-reproduction --phase formal --device cuda:0

复核已有盲测预测（不生成新测试样本或改变阈值）：

    conda run --no-capture-output -n AISAR python scripts/evaluate_d1_e001.py --phase blind --replay-existing-predictions

D1-E002 的盲测结果和保存预测位于 `experiments/D1-E002/blind/`。此前发生的单次失败重放命令如下；结果已完成，脚本会拒绝覆盖已有盲测产物：

    conda run --no-capture-output -n AISAR python scripts/replay_d1_e002_blind.py --batch-size 64

D0 的仿真与教师正确性复现：

    conda run --no-capture-output -n AISAR python scripts/run_d0_e001.py --config experiments/D0-E001/config.yaml

测试命令和依赖见 requirements-dev.txt。正式实验输出、协议、指标和限定条件均以各实验目录及阶段审查为准。
