# D2-A-E002：独立真值监督 TaskStudent 可行性 pilot

**协议版本：** 1.0-proposal
**日期：** 2026-10-25
**状态：** 方案已设计，待 ChatGPT 阶段审查与用户执行批准
**当前修订范围：** 协议与配置设计；本提交不实现训练入口、不生成数据、不训练、不生成 final blind。
**基准物理模型：** `experiments/D2-A-E001/config.yaml` 及其冻结协议；R3 结果和门槛保持不变。

## 1. 研究目标与任务对齐

本实验只回答一个任务级问题：不使用教师硬标签、软频谱、DPCA 输出或教师阈值，已有 `TaskStudent` 能否只凭三通道复数慢时间 IQ 和物理真值，在静止杂波下联合学习 H0/H1 目标检测、距离及连续 Doppler？

该问题服务于 RadarDistill 的总体目标：先验证物理真值监督能否形成可用的 GMTI 任务学生，再判断可靠传统教师的知识是否能额外改善学生。它对应总纲 Q1、Q3，并为 Q2 的后续蒸馏研究建立基础。R3 的 T0/T1/T2 固定阈值 G2 未通过，因此当前没有获准的传统教师；这并不等于学生自身的 presence 分数无法校准。本协议以学生自己的独立 H0 校准/验证作为独立假设检验。

E002 不更改、重解释或豁免 D2-A-E001、D2-A-R3 的冻结结果与前置门；不代表 R3 通过，不把 T1 改称正式教师，也不以该 pilot 结果声称教师蒸馏有效。E002 提议在 A/B/C 可辨识场景上训练一个现有 `TaskStudent`；这是新的数据分布提案，不会覆盖原 R2 的 A-only E1 记录。

### 假设 H-LEARN-D2-001

在 D2-A 当前物理可辨识的静止杂波条件下，D1-E001 的 `TaskStudent` 可只用物理真值监督，从三通道复数 IQ 学到目标存在、距离和连续 Doppler 的联合映射，并在独立开发数据上明显优于随机/常值与简单能量先验。

若数据生成、标签、输出编码、训练及 CUDA 执行检查通过，但 G2/G4 不通过，则本假设在本协议、单 seed 和所测仿真条件下不获支持；停止 E1 后续扩展和 KD，不以增大网络或加入教师损失补救。若 G2/G4 通过，只支持单 seed 的基础可学习性，不能据此声称跨种子稳定、优于强传统方法、具有真实数据有效性或已经具备蒸馏收益。

## 2. 冻结条件与实验范围

### 2.1 观测与信号模型

- 沿用 E001：3 通道、PRF 1300 Hz、130 脉冲、32 距离单元、距离压缩复数 IQ。
- 沿用三通道理想标定、静止地面点散射杂波、圆对称复高斯白噪声、E001 的目标与平台几何、波形和距离响应。
- H0 不含目标；H1 含一个目标。每个配对 scene 的 H0/H1 共享完全相同的杂波、噪声和背景参数，只在 H1 中叠加目标分量。
- SNR/CNR 使用 E001 的五个工作点：`(0,10)`、`(10,0)`、`(10,10)`、`(10,20)`、`(20,10)` dB。目标参考幅度范围为 `[-6,6] dB`，初相位均匀覆盖 `[0,2π)`。
- A/B/C 对应 separated、near-clutter-edge、partial-overlap；D exact-overlap 仅做不可辨识边界诊断，不要求恢复唯一目标来源标签。
- 不增加多目标、加速度、通道误差、原始 ADC、新杂波模型或真实数据条件。

### 2.2 模型与监督

使用 `src/radardistill/students/task_student.py` 中既有 D1-E001 `TaskStudent`，不读取 D1 checkpoint，不搜索宽度、深度或结构。输入为 `[B,6,130,32]`，通道排列固定为 `[I_ch0,I_ch1,I_ch2,Q_ch0,Q_ch1,Q_ch2]`。使用 width 32、时间 dilation `[1,2,4,8,16,32,64]` 和既有 K3 输出：presence logit、32 个距离单元 logits、单元内距离偏移、每个距离单元的单位 Doppler `(cos,sin)` 向量。

实例化时关闭 D1 的可选 spectrum head；这不改变骨干或 K3 推理头，只移除 E1 不使用的教师辅助输出。模型推理只读原始复 IQ 的 I/Q 平面，不调用 FFT、DPCA、CFAR 或教师输出。监督损失只用物理真值：presence BCE 对 H0/H1 都计算；range cell CE、offset SmoothL1 (`beta=0.1`) 和圆周 Doppler cosine loss 只对 H1 计算，四项权重均为 1。H0 参数占位符只在 loss 层屏蔽，不能拼入模型输入。

## 3. 数据拆分、样本量与防泄漏

所有 split 以 `base_scene_id` 为单位。配对 H0/H1、同场景派生量及反事实变换必须完整留在同一个 split。数据生成使用稳定的 background/noise/target nuisance 子随机流；每条样本保存 scene ID、拆分、SNR/CNR、目标幅度、目标真值和生成参数到独立 sidecar，不把这些 sidecar 字段传给 detector。

| 分组 | 数量 | 用途 | seed 范围 |
|---|---:|---|---:|
| Train | 50,000 对 H0/H1 base scenes | A/B/C、五个 SNR/CNR 点近似等权训练 | 17,000,000–17,049,999 |
| Validation | 5,000 对 | 每 100 updates 选择 truth-task validation loss 最低 checkpoint | 17,100,000–17,104,999 |
| 学生 H0 calibration | 15,000 个 H0 scene | 建立学生自己的一条全局阈值 | 17,200,000–17,214,999 |
| 独立 H0 verification | 15,000 个 H0 scene | 只报告阈值迁移 Pfa，不调阈值 | 17,300,000–17,314,999 |
| Development | 5,000 对 H0/H1 | A/B/C 检测、参数和联合任务指标 | 17,400,000–17,404,999 |
| D 边界诊断 | 1,024 对 H0/H1 | 描述 exact-overlap 的不可辨识边界 | 17,500,000–17,501,023 |

Train、validation 和 development 在 A/B/C 与五个 SNR/CNR 点上按固定权重平衡采样，目标幅度从 E001 的均匀范围抽样；development 使用 3×5 平衡网格，按 scenario 顺序和配置中的工作点顺序排列，前五个格各 334 对，其余十格各 333 对。每次训练更新抽取 32 个不同 base scenes，每个贡献 H0/H1 一对，共 64 条 IQ；按固定打乱顺序遍历，数据集耗尽后重新洗牌。1600 updates 对应 51,200 对呈现，约为 50,000 对训练集的一次遍历加 1,200 对重复样本。

15,000 个 H0 calibration 与 15,000 个 H0 verification 各自按相同的固定 IID 混合分布独立抽样：五个 SNR/CNR 点等权，每个 scene 的幅度分箱 `[-6,-3),[-3,0),[0,3),[3,6] dB` 等权抽样。每个 scene 独立抽取混合层级，故总体 Pfa 的边际观测为 IID Bernoulli；实际各层样本数必须记录。该分布定义 E002 的全局运行工作点，不代表未知真实环境分布。

样本总量按观测记录计为约 152,048 个 CPI（约 15.18 GB 的未压缩 complex64 IQ；训练成对计入 H0 与 H1）。不得把全数据集载入 GPU 或提交重建数据。按 CPU 流式生成；每批只把当前 microbatch 放到 GPU。旧 E001/R3 数据不得用于训练、校准、阈值或 E002 开发模型选择。E002 development 是开发评估数据而非 final blind；看过结果后不能在同一开发集上进行无界调参。

### 3.1 Pfa 样本量与验收数值

目标校准工作点为全局 `Pfa_CPI=0.05`；工程容许上限预注册为 `0.06`。对 15,000 个独立 H0 verification scene 使用单侧 95% Clopper–Pearson 上限。通过 G3 的条件是：`Pfa` 点估计不高于 0.06，且单侧精确 95% 上限不高于 0.06。此规则检查“是否有证据把超限风险压到 6% 以下”，不要求双侧区间必须覆盖精确 0.05，也不把校准点估计误当验证结果。

样本量核算（独立同分布二项参考）：n=15,000、真实 `p=0.05` 时，单侧 95% 精确上限规则 `U95≤0.06` 的通过概率约 99.99%；真实 `p=0.06` 时误通过概率约 4.68%。若额外要求观测 `p_hat≤0.05`，即使模型真实 `Pfa=0.05`，该点估计条件也只有约 50.97% 通过概率，故不将它设为硬门。n=15,000 时，`k=750` 对应 `p_hat=0.05`、CP 上限约 0.05303；`k=851` 是仍满足 `U95≤0.06` 的最大错误数（`p_hat≈0.05673`）。精确门限须由验证实现按 `beta.ppf(0.95,k+1,n-k)` 核对。

校准 95th-percentile order statistic 的 rank 标准差约 `sqrt(15000×0.95×0.05)=26.7`，正态近似 95% rank 区间约 ±52 个顺序统计量，即约 ±0.35 个百分点的尾概率位置；分数单位上的阈值波动未知，运行后用 2,000 次 calibration bootstrap 另报阈值区间。此 bootstrap 仅描述阈值估计不确定性，不能替代独立 verification。

这项计算假设独立 base scenes 和冻结的 IID 混合分布；若未来执行器未能实现 IID 混合抽样，必须在运行前修改协议并改用适合分层抽样的加权区间，不能事后沿用二项区间。另按五个 SNR/CNR 条件和四个幅度箱报告条件 Pfa、样本数及精确区间。任一二级分组若单侧 95% 上限超过 0.10，标为严重分组失稳并判 G3 未通过；这些分组门是边际诊断，不声称多重比较校正后的同时覆盖。不得按 SNR/CNR、幅度或目标真值在推理时选择阈值。

## 4. 训练与运行预算

- 单 seed `101`；新建初始化；不重用 D1/E001/R3 的权重。
- AdamW，1600 optimizer updates；有效 batch 64；初始学习率 `1e-3`，余弦衰减至 `1e-4`；weight decay `1e-4`；全局梯度范数裁剪 1.0；FP32。
- 默认 microbatch 16、梯度累积 4 次；若 OOM，microbatch 降到 8、累积 8 次，保持有效 batch 与 optimizer update 数不变。再次 OOM 即停止并记录；不得静默改 batch、训练步数、精度或 CPU 回退。
- 每 100 updates 在 validation 计算四项物理真值任务损失的既定等权总和并选最低者。H0/H1 validation 同时用于 presence 与 H1 参数任务；不得查看 calibration、verification 或 development 后换 checkpoint。
- AISAR/CUDA 必须可见；检查设备名为 RTX 3050 Laptop GPU，记录 PyTorch/CUDA/驱动、`nvidia-smi`、初末显存和峰值分配。没有 GPU 则 preflight 失败，不启动训练。一次只允许一个训练进程。
- 先报告 CUDA preflight、数据接口、标签与 split 审计；执行授权后才训练。训练耗时没有本阶段实测值，不能预先声称运行时；实际运行另记录墙钟时间。

## 5. 学生阈值与比较公平性

最终 checkpoint 冻结后，学生 presence logit 是唯一学生检测分数。只用 15,000 个独立 H0 calibration scenes 计算全局经验 95th 百分位阈值（`higher` order-statistic），阈值应用于独立 H0 verification 和 development H0/H1。每种传统分数各自用相同 calibration scene IDs 形成自己的阈值；不借用 R3/B3/T1 阈值，不给学生传入教师阈值。

所有方法使用相同 verification/development 场景与 H0/H1 配对；报告阈值、`Pfa_CPI`、单侧精确上限、ROC/AUC、固定阈值 Pd 和场景分层结果。阈值决策比较符冻结为 `score >= threshold`；同时列出 `score > threshold`、恰好平局数及平局比例。若量化/饱和造成大量平局，按 G1 失败停止评估并先修复评分/精度方案，不临时改比较符。

传统比较至少包括：

- **E0：** 简单三通道 CPI 功率先验，分数为 `max_r sum_(channel,pulse)|IQ|²`。
- **B0：** E001 冻结的通道 0 FFT peak 检测。
- **B1：** E001 冻结的三通道非相干 RD peak。
- **B2：** E001 冻结的几何感知相干方位搜索基线。
- **B3：** E001 冻结 raw DPCA。
- **T1：** R3 冻结的 scene-wide 归一化 DPCA 研究候选；在正式教师批准前不得称作获准教师。

参数估计统一从各方法自己的 RD peak/输出中解码 range、Doppler；学生用 K3 输出。Range error 报 cell 与米；Doppler error 按 PRF 周期回绕到 `[-PRF/2,PRF/2)`。同时报告各方法自身 calibration threshold 下的 Pfa/Pd，以及在 development ROC 上插值到相同经验 FPR=0.05 的 Pd；G2 的 Pd 非劣检查只比较共同经验 FPR=0.05 下的学生与 E0，不混用不同实际 Pfa。若某方法无法产生合法估计，记为失败，不丢样本。

## 6. 指标与不确定性

### 检测

- H0 verification 总体 Pfa、精确单侧 95% 上限、阈值和 tied scores。
- 五个 SNR/CNR、四个目标参考幅度箱、A/B/C 场景的条件 Pfa；开发集报告 AUC/ROC、固定阈值 Pd、共同 FPR=0.05 的 Pd。
- 学生相对 E0 的 pooled A/B/C AUC 差及配对 95% base-scene bootstrap 区间；同时列 B0/B1/B3/T1，不以 T1 替代学生真值监督。

### 参数与联合任务

- A/B/C pooled 及分组的 Range RMSE/bias、圆周 Doppler RMSE/bias、错误输出率；分别报告所有 H1 输出及正确检测子集，不用命中子集掩盖漏检。
- 仅从 train truth 得到比较先验：range 训练中位数、Doppler 训练圆周均值。禁止用 validation/development 真值选择常数。
- `joint_success = detected AND |range_error|≤1 bin AND |wrapped_doppler_error|≤20 Hz`；所有漏检或非法估计均记失败。报告 A/B/C pooled 以及每条件成功率。
- AUC、Pd 差、RMSE 差、joint success 使用 2,000 次配对 base-scene bootstrap；每次重采样完整 scene pair 并重新计算统计量，RMSE 每次对平方误差均值开方。Pfa 主区间用精确单侧二项区间。D exact-overlap 只描述，不纳入 G2/G4。

## 7. 数据捷径与物理一致性诊断

以下诊断使用 development 的固定样本生成受控副本，不重估检测阈值。真实物理标签、场景条件和 transformation 参数不进入模型输入。

1. **H0/H1 一致性：** 同一 `base_scene_id` 的 H0/H1 输入除 H1 目标分量外须完全一致；校验噪声、杂波、形状、sidecar 标签屏蔽、无重复 scene IDs。批次采样不泄露成对关系标识。
2. **输入审计：** detector 接受 IQ 张量；目标存在、range、Doppler、SNR、CNR、幅度、scenario、随机 seed 和 `base_scene_id` 只保存在目标端监督或分析 sidecar；不以真实 range 选取网络输入窗口。
3. **公共复相位旋转：** 对整条 IQ 乘 `exp(jφ)`，至少 32 个预先生成随机相位。理论上 detection/range/Doppler 不变。G5 要求固定阈值决策一致率≥99%，range 改变量绝对值的 95 百分位≤0.25 bin，圆周 Doppler 改变量 95 百分位≤5 Hz，AUC 绝对改变量≤0.01。
4. **整体复幅度缩放：** 对同一场景乘正实 `0.5` 和 `2.0`。不重新校准阈值，报告 Pfa、Pd、参数与分数变化。此为受控尺度扰动，不等同真实 AGC/幅相链路；若 H0 Pfa 点估计超过 0.10 或 H1 Pd 绝对下降超过 0.10，标记“幅度尺度敏感”。此标记限制尺度鲁棒性结论，不单独否定定义分布内的基础学习门。
5. **目标慢时间相位斜率：** 只对已知 H1 target component 乘 `exp(j2πδn/PRF)`，`δ∈{-25,-10,10,25} Hz`，真值按 Doppler 周期同步平移。构造变换时使用真值仅为生成诊断样本，不作为模型输入。G5 要求 Doppler 平移残差 95 百分位≤10 Hz、range 中位变化≤0.25 bin，并报告固定阈值 presence 决策变化。
6. **通道相干消融：** 每 CPI 对三个通道各乘独立常相位，保留单通道幅度但扰乱几何空间相位关系。报告 ROC/Pd/range/Doppler/joint 的配对变化，不要求输出不变。若表现无变化，则不能声称模型确实利用了跨通道相干信息；若大幅下降也需与输入分布偏移区分。

公共相位与目标 Doppler 变换有明确物理对应；全局幅度和通道相位消融是受控干预。G5 的通过/失败只说明当前诊断，不外推到未建模硬件误差。

## 8. 阶段门与停止规则

| 门 | 预注册判据 | 失败时处理 |
|---|---|---|
| G1 工程正确性 | E001 相关 T1–T12 与 split/配对检查通过；模型只读 IQ；loss/输出编码符合冻结实现；CUDA 设备有效；保存 seed/config/environment | 停止训练或评价，修复后另审；不解释为科学假设被否定 |
| G2 检测学习 | pooled A/B/C 学生 AUC 的配对 bootstrap 95% 下界 `>0.5`；相对 E0 AUC 点增益至少 `0.02` 且差值区间下界 `>0`；共同经验 FPR=0.05 时 Pd 差值（学生−E0）配对区间下界 `>-0.05` | H-LEARN-D2-001 不获完整支持；停止 KD，不用 development 调参 |
| G3 虚警约束 | 独立 15,000 H0 verification 上 `Pfa≤0.06` 且单侧 exact 95% CP 上限 `≤0.06`；五个 SNR/CNR 层及四个幅度层均报告；任一层 CP 上限 `>0.10` 视为严重失稳 | 学生不能作为当前工作点下可靠任务检测器；不套用教师阈值或事后修阈值 |
| G4 参数学习 | pooled A/B/C 中 range 与 circular Doppler RMSE 均优于训练常值先验，配对 95% 区间上界均 `<0`；主 A/10/10 的 joint success 点估计 `≥0.50` 且单侧 95% bootstrap 下界 `>0.40` | 停止后续蒸馏；先检查可辨识性、标签、输出和损失 |
| G5 物理信息 | 公共相位旋转与目标 Doppler 平移通过第 7 节数值门；幅度及通道相干干预完成并如实报告 | 精确对称/等变检查失败则不进入 KD；幅度标记限制尺度鲁棒性主张；相干消融无变化则不声称使用跨通道相干 |

pilot 只有在 G1–G5 全通过时，才构成“该 seed / 该仿真条件下值得提出后续正式 E1 多 seed 方案”的证据；仍不能直接批准 E2/E3、final blind、教师选择或 P1 推进。若各网络任务能力仍明显落后传统基线但通过 G1/G3/G4，可在阶段审查中讨论学生路线是否有科研价值；不得把门槛事后降低。

## 9. 复现与执行边界

本次没有 E002 runner 或 E002 训练代码；以下是用户另行批准后实现并通过审查的命令草案，**当前不可执行**：

```bash
python scripts/run_d2a_e002.py --config experiments/D2-A-E002/config.yaml --stage preflight
python scripts/run_d2a_e002.py --config experiments/D2-A-E002/config.yaml --stage pilot --seed 101 --device cuda:0
```

runner 实现后须先检查完整 split seed 区间无重叠、固定 IID H0 混合、scene pair 相同背景/噪声、模型输入仅 IQ 和 CUDA 型号；preflight 失败时不得训练。训练完成后应额外保存 checkpoint、逐 update train/validation log、指标/阈值、bootstrap seed、系统/依赖信息、峰值 GPU memory、`nvidia-smi` 和 wall-clock。新场景 seed 区间 `17M` 系列仅供 E002 proposal；final blind 不分配、不生成。

## 10. 本提案不授权的工作

- 不运行训练或下载/生成本协议场景；不生成 final blind。
- 不运行 E2/E3、K1/K2/K3 蒸馏或重新训练 D1 网络。
- 不改 `docs/RESEARCH_CHARTER.md`、E001/R3 protocol/config、已有数据、结果或冻结阶段结论。
- 不因单 seed pilot 通过而声称稳定性、真实回波泛化、教师有效或整体系统效率提升。

执行前待审查并批准：统计门及 IID 混合实现；G2/G4/G5 定量门；新增 seed 命名空间；E002 runner/preflight 的实现与数据泄漏审计；用户对单 seed RTX 3050 truth-only pilot 的明确执行批准。
