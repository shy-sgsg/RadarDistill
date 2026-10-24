# D0/D1 任务级蒸馏阶段总结

**日期：** 2026-10-24
**阶段决定：** D0 PASS → D1；D1 G1/G2 PASS、G3 FAIL，按冻结门槛 GO → D2。
**证据范围：** 物理仿真与 APX15 慢时间 FFT 子链；三种子 D1 训练、独立 H0 校准和独立盲测。未使用实测 GMTI 数据。

## 一、研究主题调整

此前的 P0/P1 先导工作以单个物理量（主要是 Doppler 或相对相位）的学习精度为中心。2026-10-24 批准的新主线把目标改为最终检测与参数估计任务：用物理仿真真值和传统算法教师共同训练学生，逐步减少教师约束，最终在需要时移除教师辅助头。学生可以使用不同于教师的内部表示，不要求逐模块复刻传统处理链。

这一调整来自 P0-E006 的 STOP/REDIRECT 和 P1-A-E001 的 G1 未通过：单独估计器没有达到预设精度，不能作为进入整链研究的必要前置门。P0-E007 的 GO、E006 的负结论以及 P1-A 的停止结论均保留为各自阶段的历史结果，不互相覆盖。

本阶段的近期目标是验证轻量学生能否在可控条件下直接检测目标并估计距离与 Doppler，蒸馏是否提高最终任务指标或收敛。部署速度和实测泛化不是 D0/D1 的验收目标。

## 二、研究方案

### D0：仿真真值与传统教师

冻结信号为单目标三通道复数距离压缩慢时间 IQ，形状 `[B,3,130,32]`，`PRF=1300 Hz`、`N=130`；目标具有连续距离和 Doppler、随机初相/幅度，背景为圆对称复 AWGN。D0 以离散 LFM 匹配滤波、噪声统计、通道 steering、频轴和边界检查验证生成器。

教师适配 APX15 的 `ColFFTTranspose` 慢时间 FFT 子链，输出 130×32 软功率响应；连续估计另用 4 倍补零 FFT 和局部插值。适配器只使用 IQ，不读真值或噪声功率。它不是完整 APX15/CoreX 生产链。

### D1：直接任务学生与四种训练方式

学生以 IQ 实部/虚部六个平面为输入，采用宽度 32、七个时间膨胀残差块的轻量二维 TCN，输出存在性、连续距离单元/偏移和周期 Doppler 向量。学生前向不运行 FFT、CFAR 或教师；最终推理导出会移除训练用频谱辅助头。

四组共用主干、样本、种子和每组 1600 次更新：A 仅物理真值监督；B 教师软频谱预训练后真值微调；C 真值与软频谱联合训练；D 教师预训练、联合训练、教师权重退火和真值微调。教师硬距离/Doppler 估计只作为基线，不当作物理标签；仿真真值和教师输出分别保存。

三组正式种子按序在 RTX 3050 Laptop GPU 上以 FP32、有效 batch 64 训练。训练、独立验证、50,000 条 H0 阈值校准和 40,960 条最终盲测分开；最终盲测每个 SNR 含 4096 条 H0、1024 条 H1。盲测每方法共 32,768 条 H0，只报告预注册 `Pfa=1e-2/1e-3`，不外推 `Pfa=1e-6`。

## 三、代码与文档

| 路径 | 内容 |
|---|---|
| [`docs/research/task_level_progressive_distillation.md`](../research/task_level_progressive_distillation.md)、[`docs/research/D0_D1_design.md`](../research/D0_D1_design.md) | D0–D5 任务级路线及 D0/D1 方法摘要 |
| [`experiments/D0-E001/config.yaml`](../../experiments/D0-E001/config.yaml)、[`protocol.md`](../../experiments/D0-E001/protocol.md) | D0 物理模型、种子、检查和验收门 |
| [`src/radardistill/simulation/d0.py`](../../src/radardistill/simulation/d0.py)、[`src/radardistill/teachers/apx15_doppler.py`](../../src/radardistill/teachers/apx15_doppler.py) | 仿真生成器及 APX15 FFT 教师适配器 |
| [`experiments/D1-E001/config.yaml`](../../experiments/D1-E001/config.yaml)、[`protocol.md`](../../experiments/D1-E001/protocol.md) | D1 冻结架构、A/B/C/D 预算、基线、统计方法和阶段门 |
| [`src/radardistill/students/task_student.py`](../../src/radardistill/students/task_student.py)、`src/radardistill/distillation/`、`src/radardistill/training/d1_runner.py` | 学生、教师软响应损失和 CUDA 训练器 |
| `src/radardistill/evaluation/d1_evaluation.py`、`scripts/evaluate_d1_e001.py` | 冻结校准、盲测评估与已保存预测重放 |
| [`scripts/benchmark_d1_e001.py`](../../scripts/benchmark_d1_e001.py)、[`inference_benchmark.py`](../../src/radardistill/evaluation/inference_benchmark.py) | 初步推理成本测量及统计摘要 |
| [`experiments/D0-E001/`](../../experiments/D0-E001/)、[`experiments/D1-E001/`](../../experiments/D1-E001/) | 配置快照、训练历史、权重、预测、指标、图表和分析 |
| [`docs/stage_reviews/D0_review.md`](D0_review.md)、[`D1-E001_review.md`](D1-E001_review.md) | 分阶段证据、门槛判定和边界 |

完整 APX15 包和可重建 D1 训练缓存保留在本机，未纳入该研究交付提交。

## 四、D0 结果

12/12 项冻结检查通过。离散 LFM PSF 最大归一化幅度误差为 `1.2662e-4`（门槛 `2e-4`）；复 AWGN 功率相对误差 `0.1187%`（门槛 `0.5%`）；32 条重放样本字段逐元素一致；256 个通道方向的 steering 相位误差为 0。APX15 频轴在 130 个原生 DFT 栅格音调上全部正确。

4096 条无噪 H1 上教师距离 RMSE 为 `7.17e-10 cell`，Doppler RMSE 为 `0.02904 Hz`，最大 Doppler 误差 `0.04052 Hz`。含噪教师在 `-10 dB` 的 AUC 为 `0.98856`（95% CI `[0.98414,0.99261]`），达到冻结教师可用性门；在 `-20/-15 dB` 的 AUC 分别为 `0.48675/0.64430`，低 SNR 教师输出并不可靠，不能替代真值。

**D0 判定 GO → D1** 仅表示仿真和教师子链通过正确性检查；D0 不训练学生，也不证明完整 CoreX 链或真实数据上的性能。

## 五、D1 结果

### 训练与稳定性

A/B/C/D 每个正式 seed 都进行 1600 次优化更新，四组共用数据和训练预算。模型训练阶段 13,383 个参数，峰值 CUDA 分配显存约 `1.10 GB`、保留显存约 `1.21 GB`；每个方法/种子的训练耗时约 `132.79–133.96 s`。每个策略有 3 个正式种子。

| 策略 | 20 dB range RMSE 均值 (cell) | 20 dB Doppler RMSE 均值 (Hz) | 相对 A 的 ΔJ；配对层级 bootstrap 95% CI | 改善种子 | 收敛加速门 |
|---|---:|---:|---|---:|---|
| A 真值监督 | 0.06517 | 29.59 | 基准 | — | — |
| B 教师预训练→真值微调 | 0.07473 | 25.98 | `-0.00000047`；`[-0.00019825,0.00022190]` | 2/3 | 未通过 |
| C 真值与教师联合 | 0.06712 | 27.63 | `-0.00013472`；`[-0.00025944,-0.00000810]` | 3/3 | 未通过 |
| D 渐进蒸馏 | 0.06756 | 25.54 | `-0.00086207`；`[-0.00214046,-0.00004611]` | 3/3 | 未通过 |

B/C/D 均未在至少 2/3 个种子上提前 20% 达到 A 的最佳验证损失。C 与 D 按冻结最终联合指标 J 的规则通过；D 的平均 J 改善约 7.6%，但收益幅度受 seed 1 影响明显。B 没有可重复最终收益。所有蒸馏策略都不能声称改善收敛速度。

### 检测、参数误差和传统基线

A 组跨种子平均 AUC 在 `-20/-15/-10/-5 dB` 约为 `0.486/0.514/0.621/0.941`，校准 `Pfa=1e-2` 下 Pd 约为 `0.015/0.014/0.047/0.607`；0 dB 及以上 AUC 和 Pd 均为 1.0。D 组相同低 SNR 点 AUC 约为 `0.491/0.511/0.607/0.900`，Pd 约为 `0.007/0.013/0.027/0.412`。因此高 SNR 检测成功不等于低 SNR 检测已解决；APX15 教师在 `-10 dB` 的 AUC/Pd 为 `0.989/0.911`，低 SNR 检测明显强于学生。

20 dB 下，D 全部 1024 条 H1 的 range/Doppler RMSE 均值为 `0.06756 cell/25.54 Hz`；APX15 教师为 `0.04064 cell/0.03624 Hz`。4× FFT 抛物线 Doppler RMSE 为 `0.02080 Hz`。profile-ML 只在预注册的 128 条子集上运行：D 为 `0.0689 cell/25.42 Hz`，profile-ML 为 `0.01994 cell/0.01906 Hz`。这些数字表明学生能学习任务映射，但连续 Doppler 精度仍远落后强传统估计器。分 SNR 结果和误差分组见 `experiments/D1-E001/result.csv`、`blind/metrics.json` 与 `rmse_snr.png`。

公共相位旋转和频移等变性诊断只在验证集进行；当前输出仍有非零变化，未证明严格不变/等变。带噪 Nyquist 邻域样本量较少，不能单独支撑高精度边界结论。

### 初步推理成本

成本记录在 [`experiments/D1-E001/benchmark/inference_cost.json`](../../experiments/D1-E001/benchmark/inference_cost.json)。固定合成输入为 10 dB H1、seed `2026102410`，输入边界是**已经距离压缩的** `[B,3,130,32]` 复 IQ；每个 batch 预热 5 次、计时 30 次。学生 GPU 计时包含 host/device 搬运、前向、解码和结果回传；CPU 线程数为 1。

| Batch | 学生 CUDA 平均 / P95 (ms) | 学生 CPU 平均 (ms) | APX15 FFT 子链 CPU 平均 (ms) | 学生 CUDA 峰值分配显存 |
|---:|---:|---:|---:|---:|
| 1 | 1.53 / 1.91 | 3.92 | 0.83 | 2.81 MB |
| 32 | 15.77 / 18.01 | 235.47 | 50.60 | 89.18 MB |
| 128 | 59.75 / 62.11 | 2491.09 | 185.66 | 358.86 MB |

单样本学生检查点为 53,033 bytes，导出推理模型为 9,093 参数。APX15 数字是 CPU 上 FFT 子链加 D0 硬估计；学生 CUDA 与教师 CPU 的时延不能用于公平速度结论。这不是完整原始回波到输出的端到端测时，也没有进行实时验收。

## 六、八项科研判断

1. **学生能否直接完成目标检测？** 在冻结仿真高 SNR（0 dB 及以上）可以达到本协议的检测水平；`-20` 至 `-10 dB` 明显退化，D 组 `-5 dB` Pd 约 `0.412`。因此结论是高 SNR 可行，广泛工作区间检测尚不可靠。
2. **学生能否直接完成参数估计？** 能从 IQ 学到距离与周期 Doppler 的映射，并显著优于常值预测；range 亚栅格误差较小，但 Doppler 在 20 dB 仍约 `25–35 Hz`，不能称为高精度或传统方法替代。
3. **是否需要显式 Doppler FFT？** D1 的学生没有显式 FFT，仍可输出 Doppler；但其精度远低于 FFT 插值和 profile-ML。D1 没有做“保留固定 FFT 前端”的学生消融，故当前既不能证明 FFT 必须保留，也不能证明推理可安全删除它。
4. **教师预训练是否改善收敛？** 没有。B 未达到最终收益门；B/C/D 都未通过预注册的收敛加速标准。
5. **联合蒸馏是否改善最终性能？** C 与 D 相对 A 的主指标 J 均有 3/3 种子同向改善，配对 95% CI 完全低于零，满足冻结 G2 最终指标规则。C 的改善较小，D 的幅度对 seed 1 较敏感。
6. **去教师化是否造成性能下降？** D 的最终推理学生已去除频谱辅助头；其 J 相对 A 仍改善，未显示明显的去教师化失败。但协议没有对 D 与 C 做单独的非劣/等效门，因此不能下“去教师化无损”的结论；该项仍需在后续预注册比较中检验。
7. **学生与传统处理链差距多大？** 高 SNR 检测可达到 Pd=1.0，但低 SNR 学生不及 APX15。20 dB D 的全 H1 Doppler RMSE `25.54 Hz`，远高于 4× FFT 插值 `0.02080 Hz`、APX15 `0.03624 Hz`；profile-ML 子集 RMSE `0.01906 Hz`。G3 未通过，学生没有证明参数估计竞争力。
8. **下一步应优先提精度还是优化计算？** 先提精度。学生在 20 dB 的参数误差仍大，且当前时延测量跨设备、未覆盖完整输入链，不能据此宣称已有速度优势。

## 七、唯一优先后续任务

建议的下一项是**冻结一个精度优先的 Doppler 表示对照实验**：仅改变学生的 Doppler 表示/物理前端这一因素，比较直接 IQ 学生与固定 FFT 频谱辅助的连续修正学生，并同时保留相同 FFT 输入下的非学习插值器、APX15 与 profile-ML。继续使用单目标三通道复 AWGN、原距离和检测头；本轮不同时加杂波、多目标或通道失配。

- **原因：** D1 已通过基础学习和部分蒸馏收益门，但 G3 未通过，最大差距是 Doppler 精度；先诊断频率表征瓶颈，比压缩一个尚未达精度门的模型更直接。
- **输入与实施：** 采用 D0 信号模型和独立训练/验证 seed；候选只增加固定 FFT 频谱作为 Doppler 分支输入，网络只预测连续修正。比较 E1 真值监督与冻结的 D1-A/D 基线；三种子正式训练，更新数及数据预算匹配。
- **评价：** 预先冻结各 SNR 的周期 Doppler RMSE、分频率/Nyquist 误差、检测 Pd/Pfa、与匹配插值和 profile-ML 的配对区间、推理需保留的算子和同设备输入边界成本。模型冻结后生成新的最终盲测。
- **改变判断的结果：** 若新学生在至少两个相邻正常 SNR 点、跨种子稳定优于匹配非学习插值且接近 profile-ML，同时不损害检测，才继续研究该混合表示；届时应把固定 FFT 的贡献单独列出。
- **停止条件：** 若它不能超过相同 FFT 输入的非学习插值，或收益只来自个别 seed/SNR，就停止增加 Doppler 网络复杂度，保留传统估计器并重新评估纯学生 Doppler 头的研究价值。

这是一项待预注册的建议，不是已批准或已运行的 D2 实验。

## 八、复现入口

环境为 Conda `AISAR`、Python 3.13.7、PyTorch `2.8.0+cu128`、CUDA 12.8、RTX 3050 Laptop GPU 4 GiB。正式数据、模型、种子和图表位于 `experiments/D0-E001/`、`experiments/D1-E001/`。

```bash
# D0 重跑请使用新输出目录，保留正式审计产物
conda run --no-capture-output -n AISAR python scripts/run_d0_e001.py \
  --config experiments/D0-E001/config.yaml \
  --output experiments/D0-E001-reproduction

# D1 三种子正式训练，CUDA 不可用时入口会停止
conda run --no-capture-output -n AISAR python scripts/run_d1_e001.py \
  --config experiments/D1-E001/config.yaml \
  --phase formal --device cuda:0 \
  --output experiments/D1-E001-reproduction

# 在相同 reproduction 目录独立校准与生成盲测
conda run --no-capture-output -n AISAR python scripts/evaluate_d1_e001.py \
  --config experiments/D1-E001-reproduction/config.yaml \
  --output experiments/D1-E001-reproduction --phase calibration --device cuda:0
conda run --no-capture-output -n AISAR python scripts/evaluate_d1_e001.py \
  --config experiments/D1-E001-reproduction/config.yaml \
  --output experiments/D1-E001-reproduction --phase blind --device cuda:0

# 从已保存盲测预测重放最终指标，不重新生成数据
conda run --no-capture-output -n AISAR python scripts/evaluate_d1_e001.py \
  --phase blind --replay-existing-predictions

# 初步成本测量；另指定输出可保留现有记录
conda run --no-capture-output -n AISAR python scripts/benchmark_d1_e001.py \
  --warmup 5 --repeats 30 \
  --output experiments/D1-E001-reproduction/benchmark/inference_cost.json
```

协议为 `experiments/D0-E001/protocol.md`、`experiments/D1-E001/protocol.md`；主要指标为各自 `metrics.json` 和 `experiments/D1-E001/result.csv`；曲线为 `experiments/D1-E001/rmse_snr.png` 与 `detection_roc.png`；阶段结论见本报告及 `D0_review.md`、`D1-E001_review.md`。
