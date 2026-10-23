# RadarDistill

**物理模型与知识蒸馏驱动的三通道 GMTI 动目标检测及参数估计方法研究**  
Physics-Guided Knowledge Distillation for Three-Channel GMTI Target Detection and Parameter Estimation

本项目研究轻量神经网络能否从复数雷达观测中学习传统信号处理提取的多普勒、通道相位和相干信息，并逐步评估教师蒸馏、物理约束与自监督方法。研究路线从单目标频率估计开始，不预设学习方法优于正确实现的传统基线。

## 当前阶段

**P0 已实施并运行，尚未通过继续进入 P1 的性能门。** 仿真使用 PRF=1300 Hz、N=130 和复高斯白噪声；比较 FFT 峰值、插值、相位差、连续频率最大似然，以及真值监督、教师软目标和联合训练三种学生策略。

主实验 `P0-E001` 使用三个独立训练种子、每个测试 SNR 1024 个独立样本。20 dB 时 E1 学生 RMSE 为 15.00 Hz，FFT 峰值为 2.96 Hz，FFT 插值为 1.23 Hz，相位差为 0.206 Hz，连续 ML 为 0.0332 Hz。E1 在多个 SNR 点优于常数预测，说明存在基本频率学习；但从 -5 到 20 dB 均未超过 FFT 峰值。延长训练和两种位置保留 CNN 诊断均未消除差距。结论是 P0 代码和数据管线可用，学习有效性仅获部分支持，继续门未通过，因此未开始 P1。完整分析见各实验的 `analysis.md` 和 `docs/progress.md`。

项目初始运行环境没有 PyTorch，也没有可用的 NVIDIA GPU，因此当前 P0 学生以 NumPy 实现前向传播和 Adam 更新；网络结构保持为小型 Conv1D。指标和训练输出不能被解释为 GPU/PyTorch 性能结果。

## 环境与运行

Python 3.10+、NumPy、PyYAML；测试另需 pytest。安装依赖：

```bash
python -m pip install -r requirements-dev.txt
```

运行物理一致性/行为测试：

```bash
pytest -q
```

运行完整 P0 训练、评估与报告：

```bash
python scripts/run_p0.py --config experiments/P0-E001/config.yaml
```

完整实验会生成独立数据划分、教师估计、模型权重、预测和分析文件，运行时间取决于 CPU。配置中的训练、验证和测试种子彼此独立；训练未使用测试数据。

## 项目结构

- `AGENTS.md`：科研规范和研究边界。
- `docs/`：研究方案、阶段计划、实验规范、文献综述、研究空白与决策记录。
- `src/radardistill/`：仿真器、经典基线、学生模型和实验管理。
- `tests/`：信号模型、频率基线、周期误差、训练接口和实验产物检查。
- `experiments/`：配置、种子数据、模型、预测、指标和报告。
- `scripts/`：可复现实验入口。

## P0 复核位置

- 配置：`experiments/P0-E001/config.yaml`
- 汇总指标：`experiments/P0-E001/metrics.json`
- 分组及逐 SNR 指标：`experiments/P0-E001/result.csv`
- 结果解释：`experiments/P0-E001/analysis.md`
- 数据与模型：`experiments/P0-E001/datasets/`、`experiments/P0-E001/models/`
- 训练时长诊断：`experiments/P0-E002/analysis.md`
- 位置保留模型诊断：`experiments/P0-E003/analysis.md`、`experiments/P0-E004/analysis.md`

下一步保持测试集固定，只做一个预先定义的 P0 诊断，针对位置读出/相位增量归纳偏置；避免盲目扩大模型搜索。只有学生在正常 SNR 范围的部分点稳定超过 FFT 峰值后，才进入 P1（三通道相位关系估计）。阶段目标和验收门见 `docs/research_plan.md` 与 `docs/experiment_protocol.md`。
