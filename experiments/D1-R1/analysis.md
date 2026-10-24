# D1-R1：任务级学生连续 Doppler 误差来源诊断

**日期：** 2026-10-24
**结论用途：** D1-E001 开发/验证材料上的诊断，不是新最终测试，不改变 D1-E001 阶段结论。
**代码提交：** 4f50df0c9249948b1f2ef5a0961e746e4e0883da
**主要指标：** metrics.json；逐 SNR 表：result.csv。

## 1. 数据与执行

保持 D1-E001 单目标、PRF=1300 Hz、N=130、三通道、32 距离门及复 AWGN 条件。共同开发集 pilot_validation 共 2048 条，8 个固定 SNR，每个 SNR 有 128 个 H0 和 128 个 H1。另使用三个正式训练缓存和相应验证缓存拟合/评估线性 probe、计算梯度。所有这些数据都是 D1 开发材料。

对每个正式训练 seed，评估冻结 A_truth_only 与 D_progressive_distillation 的 inference checkpoint；A 为主诊断，D 只用于训练策略敏感性比较。模型推理和特征提取使用 AISAR、PyTorch 2.8.0+cu128、CUDA 12.8、RTX 3050 Laptop GPU（4 GiB，驱动 580.178.04），FP32，batch 32；峰值已分配显存约 548 MiB。此轮没有训练模型，也没有读取 D1-E001 blind predictions 或校准数据。指标记录的代码 SHA 与实际诊断实现一致。

重现命令（将结果写入新目录）：

    conda run --no-capture-output -n AISAR python scripts/diagnose_d1_r1.py --output experiments/D1-R1-reproduction --device cuda:0 --batch-size 32

指标由入口直接写入 metrics.json。result.csv 从 metrics.json 展开得到；该 CSV 的各行仍是同一份诊断指标，不是独立样本或新的显著性分析。

## 2. 误差分解

### 2.1 预测距离门与 oracle 距离门

normal decoder 始终用 range_logits 的 argmax 选择 Doppler 向量。oracle 分析对同一输出的 Doppler 向量改用 floor(true_range_cell) 索引；真值没有送入模型，也没有改变 decoder。下表为 A 真值监督组在共同开发集上的跨种子均值和标准差，每个 seed、每个 SNR 有 128 条 H1。

| SNR (dB) | 预测门 Doppler RMSE (Hz) | 真值门 oracle RMSE (Hz) | AUC 均值 |
|---:|---:|---:|---:|
| -20 | 381.56 ± 6.76 | 365.32 ± 5.95 | 0.501 |
| -15 | 379.17 ± 5.62 | 312.56 ± 7.39 | 0.515 |
| -10 | 310.95 ± 5.92 | 186.16 ± 10.81 | 0.592 |
| -5 | 80.08 ± 15.54 | 71.30 ± 8.66 | 0.919 |
| 0 | 32.86 ± 3.42 | 32.83 ± 2.29 | 1.000 |
| 5 | 22.13 ± 1.28 | 22.14 ± 1.29 | 1.000 |
| 10 | 19.68 ± 1.80 | 19.43 ± 2.21 | 1.000 |
| 20 | 32.36 ± 6.45 | 31.87 ± 7.07 | 1.000 |

D 的对应预测门 Doppler RMSE 在 0/5/10/20 dB 为 29.14/21.11/19.72/26.13 Hz，oracle 为 29.62/21.06/19.35/26.04 Hz；没有显示距离门选择是高 SNR 精度瓶颈。-10 dB 的 oracle 误差比预测门低约 125 Hz，说明在低 SNR 距离选择会额外传播误差，但 oracle 仍有约 186 Hz 残差。-20 与 -15 dB 即使 oracle 也很差。

共同开发集 128 条/点的离散度不等于最终统计区间。D1-E001 冻结验证记录在 20 dB 的 A range RMSE 为 0.067–0.076 cell、Doppler RMSE 为 25.40–32.13 Hz，presence AUC 均为 1.0；相同高 SNR 下仍存在很大的连续 Doppler 残差。此处只复核保存的 D1 验证历史，不重算或更改 blind 结论。

### 2.2 检测、距离与低 SNR 教师质量

在共同开发集上，A/D 平均 AUC（每 seed、每 SNR 的 H0/H1 各 128 条）如下。表中 teacher AUC 来自同一份 D1 teacher 缓存。logit≥0 的 Pd 是未校准描述值，不能解释为固定 Pfa 工作点。

| SNR (dB) | A AUC | D AUC | 教师 AUC | 教师 Doppler RMSE (Hz) |
|---:|---:|---:|---:|---:|
| -20 | 0.501 | 0.503 | 0.511 | 369.59 |
| -15 | 0.515 | 0.519 | 0.666 | 332.42 |
| -10 | 0.592 | 0.584 | 0.982 | 286.50 |
| -5 | 0.919 | 0.877 | 1.000 | 76.47 |
| 0 | 1.000 | 1.000 | 1.000 | 0.238 |
| 5 | 1.000 | 1.000 | 1.000 | 0.138 |
| 10 | 1.000 | 1.000 | 1.000 | 0.080 |
| 20 | 1.000 | 1.000 | 1.000 | 0.038 |

A 的共同开发集全 SNR range-bin accuracy 为 0.664–0.673，continuous range RMSE 为 6.08–6.24 cell；该聚合受低 SNR 影响，不能与只看 20 dB 的 D1-E001 range 结果混为一谈。真值门 offset RMSE 为 0.200–0.206 cell。

教师在 -20/-15 dB 的检测接近随机，且 Doppler 很差；-10 dB 时检测 AUC 已高，但 Doppler RMSE 仍约 287 Hz；-5 dB 检测 AUC 达 1，Doppler 仍约 76 Hz。教师检测知识和连续 Doppler 知识的可靠区间并不相同。该结果说明低 SNR 全图蒸馏存在误导风险，但不能证明教师 KL 实际损害了学生。

### 2.3 分组误差与训练种子

metrics.json 对频率四分区、Nyquist 邻域、初相八分区、目标幅度四分位和距离块分别给出 A/D、每个训练 seed 的预测门与 oracle Doppler 样本数、RMSE、MAE、bias。A 的 oracle 误差跨 seed 汇总示例：

- 四个频率区间 RMSE 均值为 149.7–202.7 Hz（混合全部 SNR）；频率分布和低 SNR 共同影响该数值。
- |f|≥585 Hz Nyquist 邻域为 154.6 ± 26.8 Hz，内部频率为 188.3 ± 5.9 Hz；该开发集没有显示边界附近更差，但边界组每 seed 只有 111 条，不足以形成普遍结论。
- 八个初相组的 RMSE 均值为 154.8–213.7 Hz；四个幅度组为 181.3–189.8 Hz；四个距离块为 168.9–205.7 Hz。
- 三个 A 训练 seed 的预测门 Doppler RMSE 在 -20 dB 为 376–389 Hz、-10 dB 为 306–318 Hz、20 dB 为 28.56–39.81 Hz。高 SNR 下存在种子差异，但每点只有 128 条开发 H1，不据此作精确总体推断。

这些组覆盖了初相、幅度、频率、距离及 Nyquist 邻域；其指标是诊断统计，不是盲测门槛。

## 3. 表征与损失诊断

### 3.1 时间均值池化前后

对每个 A seed，从 temporal mean pooling 前按真值距离门读取 32×130 表征，展平为 4160 维；池化后读取 32 维表示。每个 probe 用对应训练缓存的 1024 个 H1 拟合，按 SNR 区间分层，每区间 128 条；在对应验证缓存的 1024 个 H1 上评价。标准化参数只由训练特征拟合，圆周目标为 cos/sin，ridge α=1 固定。

| Seed | 池化前 probe RMSE (Hz) | 池化后 probe RMSE (Hz) | 池化后 0/5/10/20 dB RMSE (Hz) |
|---:|---:|---:|---|
| 1 | 258.3 | 180.7 | 35.8 / 23.5 / 28.6 / 40.7 |
| 2 | 272.4 | 187.7 | 36.5 / 23.8 / 26.2 / 38.0 |
| 3 | 267.0 | 191.4 | 28.9 / 24.2 / 24.6 / 46.7 |

在本固定样本数和正则化下，池化后线性 probe 更好；因此当前结果不支持“频率信息只因时间均值池化而丢失”的简单解释。但池化前特征维数是池化后的 130 倍，样本数仅 1024，较差结果也可能来自高维读出估计不稳定。该 probe 只说明固定线性读出的可读性，不能单独证明池化的因果作用。H-ARCH-001 仍为 INCONCLUSIVE。

### 3.2 任务损失、梯度与训练历史

A/D 的训练历史显示 Doppler loss 从初始约 0.87–1.04 降至末期约 0.17–0.32；验证汇总 Doppler RMSE 仍在约 215–230 Hz（该值混合全部 SNR）。高 SNR detection 与 range 已达到很强水平时，连续 Doppler 仍保留明显残差。

在三个正式训练 seed、八个 5 dB 训练 SNR 区间、冻结 A/D checkpoint 上，用每组 16 H0+16 H1 计算共享 stem+temporal blocks 的任务梯度。跨 seed/SNR 的两两 cosine 均值接近 0：

| 梯度对 | A mean cosine | D mean cosine |
|---|---:|---:|
| presence / range cell | -0.005 | 0.001 |
| presence / range offset | -0.011 | -0.011 |
| presence / Doppler | -0.003 | 0.014 |
| range cell / range offset | -0.010 | -0.011 |
| range cell / Doppler | 0.001 | -0.006 |
| range offset / Doppler | 0.019 | 0.007 |

单个区间仍有正负波动（A 约 -0.26 至 0.47，D 约 -0.20 至 0.29）。梯度只在最终 checkpoint 和小型确定性 batch 上测量，不能否定训练早期、不同数据抽样或损失标度造成的冲突；现有证据不支持“持续强烈负梯度对齐是主要原因”。

### 3.3 全图教师 KL

只保存了导出的 inference checkpoint，里面没有训练期 spectrum_head 权重。因此无法在同一模型参数和 batch 上重算 teacher-KL 对共享主干的梯度；metrics.json 明确记录为 unavailable。A 的 teacher-KL loss 为 0；D 的训练历史 teacher-KL 标量最高为 1.58–2.07，退火末期为 0。该标量不能替代梯度证据，也不能量化其占用表征能力。

结合教师分组表现，只能得出：全图响应在低 SNR 可能含有不可靠的 Doppler 结构；本轮没有直接证据表明它造成了学生误差。D1-E001 的冻结 G2 仍然 PASS，G3 仍然 FAIL；本诊断没有重算或改写它们。

## 4. 主要判断、竞争解释与最小修改

### 当前主要瓶颈

高 SNR 的 Doppler 残差在真值距离门下几乎不变，排除了“主要由预测 range bin 错误造成”的解释。池化后 probe 并不比池化前差，任务梯度均值也没有系统性负对齐。因此本轮没有证据支持单纯增加时间特征维度或修改多任务损失权重即可解决主要问题。

可复核证据是：A 在 0/5/10/20 dB 的预测门与 oracle RMSE 几乎相同；同时 D1 正式历史在 20 dB 有很高检测 AUC和较小距离误差，但 Doppler RMSE 仍约 25–32 Hz。模型的 Doppler 输出头对 32 维时间池化特征作低维圆周回归，而验证 probe 也只能以约 24–47 Hz 精度读出高 SNR 频率。

### 单一优先结构候选

D1-E002 的优先 pilot 候选为 **S2：在原 IQ TCN 旁增加归一化 lag-one 复相干统计，并仅接入 per-range Doppler readout**。候选特征为每个通道/距离门的相邻脉冲共轭积归一化和；其相位直接编码脉冲间相位增量，不调用完整 Doppler FFT。该候选针对高 SNR oracle 下仍存在的精频率读出瓶颈，同时保留 IQ、检测和距离任务。

这是一项可证伪候选，不是已证明的修复。必须与仅使用同一 lag-one 统计量的非学习估计器比较，并报告新增算子成本、检测/距离保护指标及跨 seed 稳定性。

S1（仅用于单 seed pilot 的反证检查）：在 Doppler 头前以四个时间区间保留粗时间结构，保持共享 TCN、检测/距离头和监督不变。该候选的证据弱于 S2，因为本轮池化前 probe 更差且存在维数/样本数竞争解释；若 pilot 没有实质改善，不进入正式种子训练。

S3 固定 FFT 混合估计只作诊断参考，并与相同 FFT 前端的非学习局部插值比较；其结果不能作为纯学习学生成功证据。

## 5. 竞争性解释与停止边界

1. 线性 probe 的失败可能源于高维样本不足或 α 固定，而非池化本身。
2. 当前 TCN 训练目标和正则化可能偏向检测/粗估；checkpoint 选择以综合 validation_task_loss 而非 Doppler 单项为主。
3. 低 SNR 下噪声与距离门竞争仍会产生 oracle 之后的巨大误差；R1 的 oracle 路径并非合法推理方案。
4. 教师 detection quality 与教师 Doppler quality 在同一 SNR 下不一致；没有 KL 梯度 checkpoint，不能声称蒸馏导致误差。
5. D1 validation 用于 checkpoint 选择，pilot_validation 是开发材料。上述证据不代表独立最终盲测，也不改变 D1 G1/G2/G3。

若 S2 pilot 未显示可复现的 Doppler 改善，或改善同时造成预注册检测/距离保护指标退化，则停止该纯 IQ 结构搜索；不得靠增加复杂损失绕过基础映射或扩展未经证据支持的模型复杂度。
