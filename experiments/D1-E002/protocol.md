# D1-E002 预注册协议：受控 Doppler 读出比较

**日期：** 2026-10-24
**状态：** 在 E002 候选实现、训练和新盲测生成前冻结。
**唯一数值配置：** [`config.yaml`](config.yaml)。
**设计依据：** [D1-R1 诊断](../D1-R1/analysis.md)。
**研究范围：** D1-E001 的单目标、三通道、距离压缩复慢时间 AWGN 仿真；不改变信号模型或既有 D1 阶段结论。

## 1. 问题与假设

**问题：** D1-A 的高 SNR Doppler 残差在真值距离门下仍约 20–32 Hz。只增加一个读出归纳偏置，是否能改善 0/5/10/20 dB 的连续 Doppler 估计，同时保留检测和距离性能？低成本相邻脉冲相干统计能否让学生学到足够的频率信息？

R1 排除了预测距离门作为高 SNR 主要误差源的解释；固定 ridge 诊断没有证明均值池化造成信息丢失；最终 checkpoint 梯度均值没有显示持续强烈冲突。故 S2 是唯一优先候选，S1 是较弱证据下的单 seed 反证。S3 单独衡量经典 FFT 前端加小型学习修正的作用。

预注册假设 **H-ARCH-002**：在三通道复 AWGN 条件下，向原 TCN Doppler 读出增加归一化 lag-one 复相干量，能比冻结 D1-A 在高 SNR 降低连续 Doppler RMSE，且不使检测或距离精度超出冻结的非劣界。候选通过前，假设只是待检验，不视为事实。

## 2. 固定观测模型与研究边界

输入为 `[B,3,130,32]` 复数慢时间 IQ，PRF 为 1300 Hz；网络输入为实部/虚部平面 `[B,6,130,32]`。目标数为 0 或 1，目标连续 Doppler 位于 `[-650,650) Hz`，目标距离位于 `[4,28)` 理论带宽分辨单元，噪声为圆对称复高斯白噪声。H0/H1 的幅度、SNR、随机初相、通道方向、噪声先验和数据生成路径沿用 D1-E001；训练 SNR 在 `[-20,20] dB` 均匀抽取。

本实验不加入杂波、多目标、目标加速度、运动补偿误差、通道失配、角度输出或原始快时间回波。物理标签由仿真器保存；教师输出不覆盖标签。学生推理前向不能接收真值、教师输出、噪声功率或 SNR。频率误差按 PRF 周期回绕至 `[-PRF/2,PRF/2)`。

## 3. 候选结构与可归因范围

所有可训练候选共享 D1-A 的宽度 32、七层时间 dilation `[1,2,4,8,16,32,64]` 的深度可分离 TCN、GroupNorm(8)、SiLU、范围 stem 和检测/距离头。候选之间只改 Doppler 路径。S0/S0-D 为历史冻结锚点，不在本实验重训。

### S0：冻结真值监督基线

使用 D1-E001 的三个 A_truth_only formal inference checkpoints，分别与 E002 的三个正式种子编号配对。它保留 TCN 的时间均值池化 Doppler 头。D1-D progressive-distillation 的三个冻结 checkpoints 作为次要历史参考，不代替 S0，也不改变 D1-E001 的 G1/G2/G3 决定。

### S1：四段时间读出（纯 IQ，单 seed 反证）

只在 TCN temporal blocks 后增加四段时间均值：脉冲索引 `[0,32)`、`[32,64)`、`[64,96)`、`[96,130)`。将每个距离门的四个 32 维向量拼成 128 维，通过 `128→2` 线性层输出圆周 Doppler 向量。检测、距离和 offset 头继续使用原始 130 脉冲均值特征。相对 S0 多 192 个参数，总计 9,285。该设计只检验保留粗时间位置是否有用，不增加卷积层或调参。

### S2：lag-one 复相干读出（纯 IQ，主候选）

对每个通道 `c` 和距离门 `r`，由 IQ 计算

`z[c,r] = sum(n=0..128, x[c,n+1,r] * conj(x[c,n,r])) / (sqrt(sum|x[c,n+1,r]|^2 * sum|x[c,n,r]|^2) + 1e-8)`。

将每个 `z[c,r]` 的实部、虚部作为六个 per-range 特征，与原 TCN 的 32 维时间均值特征拼接，仅送入 Doppler `38→2` 圆周读出；检测、距离和 offset 头保持不变。特征不使用标签，不调用 FFT。相对 S0 多 12 个参数，总计 9,105。

同时报告匹配的非学习 lag-one 估计：在方法自身预测的距离门，对三通道 `z` 等权相加，取复数角度并乘 `PRF/(2π)` 后周期回绕。该对照使用与 S2 相同的相干统计，不含训练参数，用于区分物理特征与神经读出的贡献。

### S3：固定 FFT 加小型残差网络（混合方法）

对每个通道/距离门施加矩形窗，将 130 个脉冲右补零至 `NFFT=520`，作中心化 FFT。三个通道功率非相干相加，逐距离门找峰值 `k0`；取峰值循环邻域 `[-2,-1,0,1,2]` 五点功率的 `log(P+1e-12)` 并减去该五点均值，作为 5 维输入。一个 `5→16→16→1` SiLU MLP 经过 `tanh` 输出限制在 `[-0.5,0.5]` FFT 栅格内的修正量。输出为 `wrap((k0+δ)×2.5 Hz)`。

训练仅以真值 Doppler 计算最近局部 FFT 栅格的周期残差，并裁剪至 `[-0.5,0.5]` 栅格；目标距离类别仅用于选取训练监督位置，绝不进入模型前向。推理时由 S3 自身 range logits 选距离门，再读取该门的 FFT patch。TCN 检测/距离/offset 头和 MLP 一起做 E1 真值监督；Doppler 分支不含 TCN Doppler 头。总参数 9,412，比 S0 多 319。

必须同步评估相同 NFFT、相同预测距离门的非学习方法：三点 `log(power)` 抛物插值，以及以相同五点 patch 计算的 excess-power 质心（局部功率减 patch 最小值，负值置零，质心偏移裁剪到 ±2 栅格）。S3 的全部结果标为 **FFT-assisted hybrid**；即使优于纯 IQ 也不能作为纯端到端网络成功证据。另报告真值距离门上的 FFT peak-capture 率，用来区分峰值错选与局部修正。

## 4. 保留的传统方法

在相同开发验证及最终 blind sample IDs 上报告冻结 D1 APX15 慢时间 FFT 教师、原生 130 点 FFT peak、520 点四倍补零 peak、520 点三点插值、S2 的等权 lag-one 估计、S3 匹配非学习估计器。profile-ML 沿用冻结 D1 实现，只对预先固定的每 SNR 前 128 个 blind H1 运行，用作参考，不参与训练、pilot 选择或阈值设定。该 ML 数值实现、采样子集和其计算边界须在结果中说明。

所有方法共用新验证和 blind 样本。传统方法只按自身可见 IQ 估计，不得读取物理真值。仅真值距离门误差和 FFT capture 计算可将真值当作离线诊断分组，不能作为正常推理输出。

## 5. 数据、随机种子与盲测隔离

完整随机种子和数据量以 `config.yaml` 为准。E002 预留独立命名空间：pilot `202621xxxx`；正式 train/validation/initialization `202622xxxx`；检测 calibration `202623xxxx`；final blind `202624xxxx`；bootstrap 和物理一致性 `202625xxxx`。训练块和逐 SNR 流按固定 stride 1009 派生。配置测试会展开各个派生种子，确认它们彼此无重复，且与 D0/D1 已登记的训练、验证、calibration、blind 和物理一致性种子无交集。

pilot 和每个正式 seed 使用各 4096 个 H0、4096 个 H1 训练样本；每个验证 SNR 有 128 个 H0 和 128 个 H1，共 2048 条。每个候选同 seed 使用同一训练/验证 sample IDs、样本顺序、有效 batch、骨干及公共任务头初始权重。新 calibration 使用 50,000 个 H0（8 个 SNR 各 6,250）。所有 checkpoint、模型列表、代码和阈值规则冻结后，才可生成 final blind；每 SNR 4096 个 H0、1024 个 H1，共 40,960 条。profile-ML 子集为每 SNR 前 128 个 H1。所有方法和 seed 模型按同一 blind sample IDs 评价。

在本协议提交前不生成 E002 train/validation、calibration 或 blind 数据；旧 D1 blind 和 calibration 不打开作选择。校准/盲测按配置重新生成，不缓存到 GPU；正式 blind 的模型选择用途为 false。若 pilot 无候选达筛选门槛，不做正式训练，也不生成 E002 final blind。

## 6. 训练预算与候选筛选

S1/S2/S3 均先做同一个 pilot seed；每个候选最多 1,600 次 AdamW 更新、有效 batch 64、每 100 步验证一次，不早停。Formal 入选候选使用三个独立 train/validation/initialization seed，仍是相同 1,600 更新和有效 batch。学习率 `1e-3→1e-4` cosine、weight decay `1e-4`、gradient clipping `1.0`。各任务损失权重为 1；presence BCE 对全样本，range class/offset/Doppler 对 H1 真值位置。只用物理仿真真值，E2 教师硬标签和 E3 教师软响应均不参与本实验。

候选和种子顺序执行；同一时刻只保留一个训练模型在 CUDA。模型初始化采用相同初始化 seed 先建立公共 TCN、presence/range heads；共享参数从同一初始状态复制，新增 Doppler 参数用固定候选 seed 初始化。训练样本留在 CPU，按微批搬到 RTX 3050 `cuda:0`。FP32；OOM 时依次用微批 32、16 加梯度累积，保持有效 batch 和更新次数不变；16 仍 OOM 则停止该候选并记录，不改数据、损失或训练预算。禁止 CPU 回退。

checkpoint 以验证集 0/5/10/20 dB 每点等权的 Doppler RMSE 最低为准，平局选较早更新；不依赖盲测或教师验证指标。

每个 pilot 候选必须同时满足以下规则才进入正式候选集：

1. S1/S2 的验证集高 SNR 平均 Doppler RMSE 至少比 S0 pilot 低 10%；S3 的同指标至少比其匹配三点 FFT 插值低 10%。
2. 主 SNR 每个点的 Doppler RMSE 相对各自 comparator 不得恶化超过 5%。
3. 低 SNR（-20/-15/-10/-5 dB）平均 AUC 相对 S0 不低超过 0.05；高 SNR 平均 range RMSE 相对 S0 增加不超过 0.05 cell。
4. 输出有限，Doppler 周期解码正确，CUDA 训练按固定预算完成。

pilot 门槛只筛明显不成立的候选，不作为正式效应证据。S1 是反证候选，没通过即停止，不增加其它时间结构变体。S2 是主候选，但同样必须通过。S3 可作为混合候选单独进入三 seed formal；即使通过也不改变纯 IQ 主结论。未进入 formal 的 pilot 权重、验证日志和负结果仍保存。

## 7. 正式指标、阈值与不确定性

**主指标：** 每个方法在 0/5/10/20 dB 的 per-SNR H1 Doppler RMSE 等权平均，单位 Hz。候选相对其 seed 配对 S0 的差记作 `ΔRMSE = RMSE_candidate − RMSE_S0`。同时逐 SNR 报告结果、MAE、bias、`|error|>50 Hz` 比率和检测正确条件误差。

**检测阈值：** 对每个方法单独使用新 50,000 H0 calibration，在全 SNR 平衡混合流上校准 `Pfa=1e-2`。把 H0 分数升序排列，阈值取索引 `ceil((N+1)×(1−Pfa))−1` 对应的分数；判决为 `score > threshold`，相等计为未检测以保持保守。盲测不重估阈值。报告校准实际 Pfa、blind Pfa、各 SNR Pd/AUC、漏检和 95% 二项区间。

**检测保护门：** 低 SNR `[-20,-15,-10,-5] dB` 等权平均 Pd 的 paired 95% CI 下界必须高于 `−0.05`；任何一个低 SNR 点的 Pd 点估计不得比 S0 低超过 0.10。

**距离保护门：** 0/5/10/20 dB 等权 mean range RMSE 的候选减 S0 paired 95% CI 上界必须低于 `+0.05 cell`。

**纯 IQ 正向判据：** S1 或 S2 的主指标点估计至少改善 10%；paired hierarchical bootstrap 的 `ΔRMSE` 95% CI 上界小于 0；三个 seed 中至少两个方向改善；且检测、距离保护门全通过。S3 不能通过纯 IQ 判据。S3 只在相对匹配非学习 FFT 估计达到同等 10%、CI 和 seed 方向门槛时，记为“学习式 FFT 残差修正有增益”。

95% 区间使用 2,000 次 paired hierarchical bootstrap：先以放回方式抽三对 formal training seed，再在每个 SNR 抽相同 blind sample IDs；所有方法在每次重复内使用相同 ID 重采样。seed 方向、每 seed 指标和样本数量同时列出，不能只报 pooled 区间。任何未通过或区间跨零的结果不得称为已证明改善。

报告不同 Doppler 四分段/正负频率/Nyquist 邻域、初相八分段、幅度四分位、范围区间、SNR 和训练 seed。另用独立开发诊断 seed 的 256 条 H1（0/10 dB 各 128 条）执行物理一致性检查：对所有通道/脉冲/距离统一乘 `exp(jθ)`，`θ∈{0.37,1.41,2.83}`；再用 `exp(j2πΔf n/PRF)` 调制，`Δf∈{±0.625,±5.0} Hz`。报告 Doppler 输出的公共相位敏感度、频移等变残差、检测分数/距离输出变化；这些是诊断指标，不增加事后通过门。还报告 range、教师/FFT/profile-ML 差距、S2 非学习统计 baseline、S3 相同 FFT baseline、条件 CRLB、参数量、FFT/相干操作边界、训练时间和 GPU 峰值分配/保留显存。统计和物理一致性只针对该仿真条件，不外推到真实 GMTI。

## 8. 决定与停止规则

- 至少一个纯 IQ 候选满足主指标与两项保护门，才支持 H-ARCH-002 或保留 S1 的结构假设为该协议下的有界发现。
- 纯 IQ 候选主指标不达 10%、paired CI 跨零、少于 2/3 seeds 同向，或检测/距离保护门失败，均不得宣称 D1-E002 纯 IQ 结构成功；不追加结构变体或训练损失来挽救结果。
- 只有 S3 成功时，结论限于 FFT 辅助混合方法相对已匹配传统 FFT 插值的变化；纯 IQ 结构搜索停止。
- 所有候选 pilot 失败时，停止正式训练及新 blind 生成，保留负结果；不得通过看旧 blind、降低门槛或增加搜索轮数改变选择。
- D1-E001 的 G1/G2/G3 冻结结论一律不追改。E002 不做完整链实时性验收，不声称边缘部署优势。

## 9. 复现与执行顺序

1. `python scripts/validate_d1_e002_protocol.py --config experiments/D1-E002/config.yaml`：验证配置/派生种子/门槛（实现后）。
2. protocol/config/tests 提交后，按 `S1 → S2 → S3` 在 CUDA 顺序运行 pilot；pilot 只读取指定 train/validation split。
3. 按本协议机械决定 formal 入选集合；所有 pilot negatives 保留。
4. 对入选候选按 seed 顺序训练 formal；保存初始/最佳 checkpoint、训练历史、环境和资源占用。
5. 冻结候选 checkpoint、代码和模型 manifest；新建独立 calibration 并冻结检测阈值；随后才生成一次 final blind。
6. 对所有方法使用同一 blind IDs 评估，运行冻结 profile-ML 子集、bootstrap 和误差分组；写出结果和阶段审查。

协议冻结后若发现配置/代码错误，暂停实验，记录变更原因；不得沿用已查看的 pilot 或 blind 作为新盲测。盲测只可在模型和门槛未接触 blind 数据前生成。
