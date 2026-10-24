# RadarDistill

**物理模型与知识蒸馏驱动的三通道 GMTI 动目标检测及参数估计方法研究**  
Physics-Guided Knowledge Distillation for Three-Channel GMTI Target Detection and Parameter Estimation

本项目研究轻量神经网络能否从复数雷达观测中学习传统信号处理提取的多普勒、通道相位和相干信息，并逐步评估教师蒸馏、物理约束与自监督方法。研究路线从单目标频率估计开始，不预设学习方法优于正确实现的传统基线。

## 当前阶段

**P0-E007 决策：GO（G1、G2 通过，G3 未通过）。P1-A-E001 决策：STOP / REDIRECT（G1 未通过）。** E007 使用 RTX 3050 Laptop GPU 对 M1 循环等变频谱模型、M2 FFT 子栅格修正和 M3 小型 IQ Transformer 先筛选后正式训练。M1 在 20 dB 的三种子 E1 RMSE 为 0.0372、0.0437、0.0361 Hz；五个 0–20 dB 主点均显著优于 N 点 FFT 峰值、N 点插值、相位差和 16N FFT 峰值，但未胜过 4N/16N 局部抛物线插值或连续 profile-ML。M1 对功率输入消融没有显著优势，E3 联合蒸馏未通过 G3。P1-A-E001 使用三通道共享单音与复 AWGN，在 RTX 3050 上顺序训练三个相关特征 MLP；0/3 种子通过无噪声精度门。20 dB 学生 RMSE 为 0.023–0.026 rad，明显落后于接近 CRLB 的互相关/profile-ML（约 0.0097–0.0100 rad）。当前不进入 P1-B 或蒸馏。E006 历史 STOP 结论仍完整保留。

**历史结论保留：** P0-E006 于 2026-10-23 的独立阶段审查为 STOP / REDIRECT，G1、G2、G3 均未通过。E007 是新的预注册 P0 诊断，更新当前阶段判断，但不改写 E006 的历史结果。E007 的模型、协议、完整指标和阶段审查见 `experiments/P0-E007/`、`docs/superpowers/specs/2026-10-23-p0-e007-design.md` 和 `docs/stage_reviews/P0-E007_review.md`。

历史主实验 `P0-E001` 使用三个独立训练种子、每个测试 SNR 1024 个独立样本。20 dB 时 E1 学生 RMSE 为 15.00 Hz，FFT 峰值为 2.96 Hz，FFT 插值为 1.23 Hz，相位差为 0.206 Hz，连续局部 ML 为 0.0332 Hz。E001–E005 说明旧学生在多个 SNR 点优于常数预测，但未超过 FFT 峰值。E006 增加了相关特征 MLP、频率响应混合模型、匹配物理前端基线和独立最终盲测；结果表明 S1 优于旧 CNN，但三个阶段门均未通过。详见实验 `analysis.md` 和 P0 最终审查。

E001–E005 使用当时可用的 NumPy/CPU 实现；这段历史结果不是 GPU 训练结果。P0-E006 使用 Conda `AISAR` 环境中的 PyTorch CUDA 训练，正式入口在启动前强制检测 CUDA，设备不可用时会停止，不回退 CPU。已确认的当前设备为 NVIDIA GeForce RTX 3050 Laptop GPU；训练结果不会被解释为部署速度或工程加速证据。

## 环境与运行

Python 3.10+、NumPy、PyYAML；测试另需 pytest。安装依赖：

```bash
python -m pip install -r requirements-dev.txt
```

运行物理一致性/行为测试：

```bash
pytest -q
```

复现历史 NumPy P0 入口：

```bash
python scripts/run_p0.py --config experiments/P0-E001/config.yaml
```

运行冻结的 P0-E006（需要可见 CUDA 与 AISAR 环境）：

```bash
conda run --no-capture-output -n AISAR python scripts/run_p0_e006.py
```

E006 会生成训练/验证和最终盲测数据、教师估计、模型权重、逐样本预测、指标、图表和数据清单。模型选择只使用预注册验证协议；旧 E001–E005 测试集仅作开发基准。

运行 P0-E007 的完整流程（请使用新输出目录进行独立复现；需要 AISAR 和 CUDA）：

```bash
conda run --no-capture-output -n AISAR python scripts/run_p0_e007.py --phase all --output experiments/P0-E007-reproduction
```

P1-A-E001 的正式三种子训练与一次性独立盲测已完成，结果在 `experiments/P1-A-E001/`。下方命令用新的输出目录复现相同固定种子和盲测样本，不构成新的独立盲测，也不会覆盖正式产物：

```bash
conda run --no-capture-output -n AISAR python scripts/run_p1_a_e001.py --output experiments/P1-A-E001-reproduction
```

Git 保留代码、配置、种子清单、指标、报告、预测和最终盲测样本；可按配置与种子重建的训练/验证/无噪声数据，以及评估器作废运行的大型原始输出，由 `.gitignore` 排除在提交之外并保留在本机。这样克隆仓库后可重建训练数据，同时避免把数百 MB 的临时/可再生文件推入 Git。

从已保存的盲测预测重建报告与曲线：

```bash
conda run --no-capture-output -n AISAR python scripts/run_p0_e007.py --phase summarize
```

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
- P0-E006 协议和结果：`experiments/P0-E006/protocol.md`、`experiments/P0-E006/analysis.md`
- P0-E007 协议和结果：`experiments/P0-E007/protocol.md`、`experiments/P0-E007/analysis.md`、`experiments/P0-E007/rmse_snr.png`
- P0-E007 汇总指标：`experiments/P0-E007/metrics.json`、`final_metrics.json`、`result.csv`
- E006 历史阶段审查：`docs/stage_reviews/P0_final_review.md`
- 当前 P0-E007 阶段审查：`docs/stage_reviews/P0-E007_review.md`
- P1-A 设计与协议：`docs/stage_reviews/P1_A_plan.md`、`experiments/P1-A-E001/protocol.md`
- P1-A-E001 完整结果和阶段审查：`experiments/P1-A-E001/analysis.md`、`docs/stage_reviews/P1-A-E001_review.md`

P1-A-E001 的冻结阶段门未通过，当前相对相位 MLP 路线停止；不通过继续扩展网络、蒸馏或进入 P1-B 来绕过该结果。若将来重新启动，需要新的可证伪问题和独立预注册。阶段门和完整证据见 `docs/research_plan.md`、`docs/experiment_protocol.md` 与上述阶段审查。
