# P0-E007 预注册协议

**状态：** 2026-10-23，用户已批准设计；在任何 E007 模型代码或训练前冻结。本协议新增一次有界 P0 架构诊断，不覆盖 E006 的 STOP/REDIRECT 结论。

## 研究问题

在单目标复数慢时间频率估计中，比较（M1）固定 16N 复 FFT + 相干循环等变网络、（M2）固定 4N FFT + 子栅格修正 MLP、（M3）直接处理 IQ 的小型 Transformer。验证物理归纳偏置、经典频谱辅助学习、非 CNN 序列学习及其蒸馏表现。

## 信号模型

沿用 x[n]=A exp(j(2π f n/PRF+φ))+w[n]，PRF=1300 Hz、N=130、f 在 [-PRF/2, PRF/2) 连续均匀分布、φ 在 [-π,π) 均匀分布、A 在 [0.5,1.5] 均匀分布、圆对称复高斯白噪声。训练/验证 SNR 为 [-15,-5,0,10,20] dB；测试 SNR 为 [-15,-10,-5,0,5,10,15,20] dB。不加入杂波、通道失配、多目标、加速度或运动补偿误差。

## 数据角色与种子

| 角色 | 数据种子 | 数量 | 用途 |
|---|---:|---:|---|
| Screen train | 17117 | 6000 | 三种模型及 M1 功率消融使用同一批数据 |
| Screen validation | 27117 | 1200 | screen checkpoint 选择 |
| Formal train | 17101, 17202, 17303 | 每种子 6000 | 入选模型正式训练 |
| Formal validation | 37101, 37202, 37303 | 每种子 1200 | 与相应训练种子配对，独立模型选择 |
| Development benchmark | 7070 | 沿用 E005 test.npz | 架构筛选、错误诊断；不是最终盲测 |
| Noiseless correctness | 989897 | 2048 | 连续离栅格频率与 Nyquist 周边 |
| Final blind test | 999983 + 每个 SNR 1000×序号 | 每 SNR 1024 | 仅在模型筛选、训练规则和阶段门冻结后生成 |

Screen Torch seeds：M1=80101，M1-power=80102，M2=80201，M3=80301。正式训练 Torch seed 由 81000 基数、架构索引、策略索引和训练种子索引确定，写入每个 checkpoint 与 history。不同数据角色不共享随机状态。

## 模型和固定前端

### M1：循环等变相干频谱网络

- 固定 PyTorch 复数 FFT，长度 16N=2080；不可学习。
- 输入为归一化 log-power 和相邻复互谱归一化实/虚部；M1-power 消融只输入 log-power。
- 三层小型 Conv1d，circular padding、无频率轴池化/展平，输出 2080 点循环 logits。
- E1 目标为标准差 0.5 个输出栅格的圆周高斯分布；解码采用圆周概率均值。
- 输入频移一个 16N 频率栅格时，输入特征和输出 logits 应相应循环滚动。共同相位旋转应在相干特征中严格抵消。

### M2：FFT 子栅格修正网络

- 固定长度 4N=520 复 FFT；粗频率为最大功率 bin。
- 取粗峰 ±4 bin 共 9 点，输入归一化 log-power 及每点相对粗峰的归一化复互谱实/虚部。
- MLP 为 27→64→32→1，tanh 输出限制在 ±0.5 个 4N bin。物理真值和硬教师残差均按 PRF 圆周差计算并裁剪至该范围；同时报告裁剪/饱和比例。
- 相同局部窗口的非学习基线为中心及相邻 bin 上的三点 log-power 抛物线插值。该模型明确标记为物理前端加学习修正。

### M3：轻量 IQ Transformer

- 输入为 RMS 归一化的 130×2 float32 I/Q token，保留公共相位信息；固定正弦位置编码。
- 2 层、d_model=32、4 attention heads、FFN=64、dropout=0.05、batch-first encoder 和汇聚 token；输出 2080 点循环 logits。
- 可训练参数量需记录；不得加载预训练权重。

## 筛选、正式纳入与停止

Screen 只运行 E1 真值监督；M1-power 作为同族输入消融。各模型使用相同数据、batch、优化器、训练上限和验证选择方式。Screen checkpoint 以验证 E1 损失最小为准。

模型进入三种子正式阶段，当且仅当同时满足：

1. 所有损失和梯度有限，验证损失相对第 1 轮下降至少 20%；
2. development benchmark 的 20 dB E1 RMSE ≤5 Hz，或在主 SNR 的 0、5、10、15、20 dB 中至少两个相邻点优于 N 点 FFT 峰值；
3. 结构声称的确定性变换满足行为测试：M1 频移循环等变、M2 局部相干特征与输出修正在整 4N 栅格平移下不变。M3 不预设严格等变，实测后报告。

未通过时保留 checkpoint、训练曲线、开发指标、物理诊断和停止原因；不得为挽救单个模型单独增加轮数或调参。若无模型进入正式阶段，不生成最终盲测集，不开展 KD。

## 正式训练与教师目标

所有入选模型在 3 个正式训练种子上先运行 E1。正式 E1 的 development 结果再次满足筛选准确度条件（跨种子平均 20 dB RMSE≤5 Hz，或跨种子平均 E1 在至少两个相邻主 SNR 点优于 N 点 FFT 峰值）后，该模型才进入 KD；否则保留三个 E1 结果并停止该模型的 KD。M1 入选时，M1-power 只做相同三种子的 E1 输入消融，不做 KD。正式 E1 与 KD 均完成后，才生成最终盲测。

- **E1 truth：** 物理生成的频率真值。
- **E2-H hard：** FFT 初始化局部 profile-ML 连续点估计，作为硬点目标。学生分布头用相同的圆周高斯编码，M2 用粗频率相对残差。
- **E2-S soft：** M1/M3 使用 16N FFT 网格上的 profile likelihood 响应；M2 使用粗峰 ±0.5 个 4N bin 内的 65 点局部 profile response，并将其归一化期望修正量作为软目标。profile score 为 |Σ x[n]exp(-j2πfn/PRF)|²/(N·noise_power)，先减最大值、再除固定温度 2.0、softmax 归一化。
- **E3-S：** 真值损失 + 0.5×教师软响应损失。系数不调参。

E2-H 的早停只看对应教师硬目标验证损失；E2-S 只看教师软目标验证损失；E1 和 E3-S 以物理真值验证损失选择 checkpoint。真值、硬教师、软教师响应分别存储和评价，错误教师输出不覆盖真值。记录教师相对真值的误差及软响应熵。

## 优化、显存与记录

- Python/PyTorch 使用现有 AISAR Conda 环境；CUDA 必须可用，否则 runner 在训练前失败，禁止 CPU fallback。
- PyTorch 2.8.0+cu128，RTX 3050 Laptop GPU，4 GiB 显存。FP32，Adam、learning rate 0.001、batch 64、最多 160 epochs、early-stop patience 40。
- 所有模型和种子顺序训练；原始数据、标签和特征保存在主机内存，按 batch 送入 CUDA，不缓存整套数据到 GPU。
- 记录峰值已分配显存、设备名称、驱动/CUDA/PyTorch/Python 版本、GPU 训练耗时。每个 best checkpoint 保存架构配置、策略、数据/torch 种子、轮数和验证选择损失。
- 若 OOM，释放当前模型并将该阶段所有架构的 batch 统一改为 32 后重跑，以保持比较条件一致；不能单独降低某个模型的预算。

## 基线、指标与统计

比较 N 点 FFT 峰、N 点 FFT 插值、相位差估计、局部 profile-ML、全局网格 profile-ML、4N FFT 峰、16N FFT 峰、4N 局部三点 log-parabolic 插值。报告每个 SNR 的 circular RMSE/MAE/bias/outlier rate、95% paired bootstrap CI、训练种子分散、CRLB 比值、频率区间、Nyquist 邻域、初相扇区、幅度四分位；M2 另报粗峰错误率、修正饱和率。

物理诊断分别测 common phase rotation、整体复尺度、频移及 Nyquist wrap。对 M1/M2 报告结构性等变测试与学后输出测试；对 M3 只报告实测偏差。相位旋转、频移均在相同 IQ 上成对变换。

主要 P0 门保持不变：

- **G1：** 至少一个纯相干学习路线（M1 或 M3）在至少 2/3 种子上 noiseless MAE<1 Hz 且 P95<5 Hz，并且 20 dB E1 RMSE 至少比零频预测低 50%，同时通过周期边界与数据隔离检查。
- **G2：** N 点 FFT 峰值主门的配对 95% CI 在至少两个相邻主 SNR 点完全低于零。M1 另需与 16N FFT 峰值比较，M2 另需与同局部频谱三点插值比较；各需要两个相邻主 SNR 点 CI 完全低于零，才能宣称该混合前端本身带来学习增益。M3 不要求超过 ML，但完整报告差距。
- **G3：** 入选模型的 E3-S 相对 E1 在两个相邻主 SNR 点 paired 95% CI 完全低于零，且每点至少 2/3 训练种子方向改善。

所有 CI 使用 2000 次配对 bootstrap，按种子及测试样本重采样。G1/G2/G3 独立判定。G1 失败时 STOP/REDIRECT；G1 通过而 G2 未通过时按现行阶段规范判断 CONDITIONAL GO；不得把仿真结论外推到真实 GMTI 或实时性能。

## 指标分组的操作定义

- 异常率定义为 `|wrapped error| > 5 Hz`。
- 频率分箱为连续无模糊区间的 8 个等宽区间；Nyquist 边界组定义为距 `±PRF/2` 不超过 `PRF/N`（10 Hz），同时报告区间内部。
- 初相分为 8 个等宽扇区；幅度按最终盲测数据的秩分成 4 个四分位。以上分组均按 SNR 报 RMSE、MAE、偏差、异常率和样本数。
- 最终盲测主要方法间的配对 bootstrap 同时按训练种子和同一测试记录进行层级重采样；candidate-minus-baseline 的 95% 区间上界低于 0 才记作胜出。每个模型比较使用相同盲测信号。
- noiseless correctness 集共 2048 例：一半连续均匀频率，一半位于 Nyquist 两侧各一个 `PRF/N` 邻域，包含 `-PRF/2` 端点。G1 的 MAE/P95 门槛同时应用于全频率集和该边界子集。

## 最终盲测

正式模型、checkpoint 选择规则、正式 E1 与 KD 纳入名单、指标和阶段门确定后，以新 seed 999983 为 final blind base，每个 SNR 使用 base+1000×序号生成 1024 个样本。只运行一次，不据此调参。若无模型通过 screen，不生成该数据并明确说明。

## 产物与复现

保存 config.yaml、protocol.md、screening.json、selection.json、metrics.json、result.csv、training_history.csv、analysis.md、图表、datasets/、models/ 和运行日志。入口为 scripts/run_p0_e007.py；报告记录 git HEAD/dirty 状态和实际 AISAR 硬件/软件版本。
