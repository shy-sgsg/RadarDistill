# D2-A-E001 预注册协议：静止地杂波、传统基线与后续学生任务

**冻结日期：** 2026-10-25
**状态：** R0/R1 开发实验协议；实施后只允许记录偏差，不得按结果改目标、场景或门槛。
**数据性质：** 全部为合成数据。
**当前授权边界：** 可实施 R0/R1 并运行开发基线；本协议不授权在前门验收前开展正式学生训练或生成最终 blind。

## 1. 研究问题与任务边界

本实验在既有 P0/D0 单目标复 AWGN 距离压缩 IQ 上只增加静止地面散射杂波。目标是验证物理生成、H0/H1 配对、可辨识条件及经典检测上限，为三通道学生与蒸馏建立可信任务。

系统沿用 PRF=1300 Hz、N=130、3 个研究接收通道和 32 个距离单元。主仿真使用生产场景 XML 的 15.8 GHz 载频、50 MHz 带宽；距离单元取 D0 的 `c/(2B)=2.99792458 m`。这是距离压缩 IQ 的近似仿真，不是生产 ADC/匹配滤波链或实测数据。D0 的无杂波回归使用 D0 原始 16 GHz 配置，不更改旧结果。

本轮不训练学生、不调参追求网络性能、不生成/解封 final blind、不将开发集结果称为盲测。D1-E001/E002 历史数据和结论保持不变。

## 2. R0 信号模型

每个背景由 32 个固定于平地的点散射体构成，方位角在 `[88.86°,91.14°]` 均匀分布、距离单元在 `[0,32)` 均匀分布；复散射系数为独立 `CN(0,1)`。每个脉冲重新计算平台、发射相位中心、三个接收相位中心到散射点的传播距离。坐标为 x 北、y 东、z 上；平台参考场景位置 `(0,0,6000) m`、速度 `(60,0,0) m/s`。传播总路程 `L=Rtx+Rrx`，复载波相位 `exp(-j 2π L/λ)`；静止地杂波多普勒完全由运动几何形成，不强制为零。

波束主瓣假设为仅依方位角变化、半功率全宽 2.28° 的 Gaussian 幅度响应，不含旁瓣。距离压缩响应采用 D0 的 sinc PSF。发射相位中心假定与平台参考原点重合，生产通道校准、实测阵列相位和实际方向图未知。所有这些是仿真假设，限制外推到真实 APX15 的结论。

目标复幅度参考值在 `[-6,6] dB` 均匀抽取，初相在 `[0,2π)` 均匀抽取。输入 `snr_db` 继续按 D0 的 `A²/Pn` 连续 PSF 峰值定义噪声方差。`compute_scene_power_metrics()` 则另行从完整 range-compressed CPI 的 target/clutter/noise component array 计算同域均方功率：

```text
SNR = Ps / Pn
CNR = Pc / Pn
SCNR = Ps / (Pc + Pn)
```

仅 SNR 与 CNR 两个比值独立。结果同时记录 D0-compatible nominal SNR 与实际 CPI-mean power ratios，不混用其名称。CNR 通过背景场景整体缩放到目标值；噪声为独立圆对称复高斯，通道协方差为单位阵。通道幅相/时延误差未建模。

## 3. 配对、标识和拆分

每个 base scene 只生成一次背景散射位置/幅相及噪声。H0=`clutter+noise`，H1=`clutter+noise+target`；二者 shape、背景和噪声完全相同。H0 中目标真值字段遮蔽，counterfactual target 条件仅供分析员计算配对功率/误差，不传入检测器。场景、背景、目标和噪声 realization 分别持有稳定 ID；随机数流按 background/noise/target nuisance 分开。

任何数据划分均按 `base_scene_id` 分组；同一背景的 H0/H1 和所有噪声/目标 counterfactual 不得跨 split。规定 seed 范围：

| 用途 | Seed 范围 | 样本数/规则 |
|---|---:|---|
| 未来学生 train | 1,000,000–1,999,999 | 计划 50,000 base scenes |
| 未来学生 validation | 2,000,000–2,499,999 | 计划 5,000 base scenes |
| H0 threshold calibration | 3,000,000–3,099,999 | 每个唯一 SNR/CNR 条件 512 H0 scenes |
| 开发评估 | 4,000,000–4,099,999 | 每个冻结条件 256 个配对 H0/H1 scenes |
| final blind | 9,000,000–9,999,999 | 仅保留 seed 空间，不生成；模型和门槛冻结后方可使用 |

测试/开发 H0 不可参与阈值或超参数选择之外的再调参；最终 blind 必须来自全新场景，不得复用多次看过的 D0/D1 blind。bootstrap 以 base-scene 为 cluster，配对样本共同重采样。

## 4. R0 正确性验收

12 项物理测试固定为：

1. T1 零杂波适配器逐字段回归冻结 D0 输出。
2. T2 无噪声静止散射体的距离 sinc、载波相位、几何多普勒和通道相干符合双程路径公式。
3. T3 静止散射体 Doppler 随平台速度按几何改变，不能被置零。
4. T4 方向图随 beam pointing 移动而朝指向方向增强。
5. T5 发射/接收相位中心逐通道路径与显式坐标计算一致。
6. T6 配对残差 `H1-H0` 等于 target component。
7. T7 `Ps/Pn`、`Pc/Pn`、`Ps/(Pc+Pn)` 数值定义和代数关系成立。
8. T8 同 seed 重生场景、noise、truth 和 ID 完全一致。
9. T9 改目标位置/速度不改变配对 H0、clutter 或 noise（真值无泄漏）。
10. T10 整体坐标平移和公共复相位旋转保持观测关系不变。
11. T11 沿飞行方向的已知运动产生符合几何值的距离徙动。
12. T12 H0/H1 使用同一完整观测/噪声/杂波接口，不能按标签生成不同形状或空背景捷径。

T1–T12 必须全部通过，才接受 R0。任一失败先修正物理/实现错误；不得用训练性能替代 R0。

## 5. R0 可辨识性场景

正向径向速度表示目标远离平台。A/B/C 使用均匀抽取的 beam 内方位和 `[4,28)` fractional range cell；D 将目标放在该 CPI 的一个已有杂波散射点上、速度为零、未知复幅相。

| 场景 | 目标径向速度 | 含义 | 可辨识性预期 |
|---|---:|---|---|
| A separated | 4.0 m/s | 目标 Doppler 与主瓣内静止地杂波 Doppler 有明显间隔 | 有限快拍下仍受 clutter/noise，但不属支撑精确重叠 |
| B near edge | 1.2 m/s | 靠近平台 60 m/s、主瓣边缘静止杂波的 Doppler 上界 | 预期检测/估计明显退化；作为边缘区，不要求 AI 恢复被淹没信息 |
| C partial overlap | 0.4 m/s | Doppler 支撑重叠；连续角/距离散射流形一般不与某个随机点完全相同 | 分布可能可辨，单 CPI 能力由实测基线说明 |
| D exact overlap | 0 m/s | 目标与现有某个杂波散射点共享 range/angle/slow-time manifold | 未知复散射幅相下，目标来源标签在单 CPI 中不唯一；不得作为网络必须恢复的门槛 |

对 D，若 target scatter coefficient 与同址 clutter coefficient 均未知，则观测只含二者的复和；单 CPI 无法唯一拆分标签。统计检测若依赖 RCS/先验，必须明确其依赖，不能称为观测本身可辨。

## 6. R1 传统检测器和比较

四种开发基线共享观测场景和相同 CPI：

- **B0：** channel 0 的矩形窗慢时间 FFT，RD 最大功率。
- **B1：** 三通道非相干 RD 功率求和，取全图最大值。
- **B2：** 17 点方位网格的相干波束搜索，steering vector 由实际 D2 接收几何和载频计算；仰角取配置固定参考，不读取 truth angle；对每个角波束求 RD 图并取最大值。此为物理前端辅助基线。
- **B3：** 只用 channel 0/1 沿航向接收基线做 DPCA。候选连续时间对齐为 `dt=−(b1−b0)·v/(2||v||²)`；用 band-limited fractional slow-time shift 后相减。此处 `b` 是接收位置，Tx 共址是假设。只有静止 clutter 中位功率抑制超过 0 dB、分离场景目标保留率中位数不低于 0.5 且几何测试通过，才纳入有效基线；否则保留失败结果并排除性能排序。
- **Full STAP：** 不实现。3×130 输入为 390 个 space-time DOF，而局部 32 range bins 不能支持满秩 IID clutter covariance；欠定/强正则解不作为公平强基线。

B3 `clutter_rejection_db` 定义为 `10log10(Pc_channel0_before/Pc_difference_after)`；`target_retention` 定义为 `Ptarget_difference_after/Ptarget_channel0_before`。DPCA 差分未除以 2，因此移动目标可相干增强，功率比可能大于 1；此量是相对输入单通道的诊断，不代表能量效率。

**工作点：** H0 独立校准的 `Pfa_CPI=0.05`。每个 RD map 的最大统计量对应一次 CPI 级检测，故 `Pfa_CPI` 是主阈值指标；本协议的 `false_alarms_per_cpi` 是该单峰检测器每 CPI 的 0/1 误警数均值，理论上等于 `Pfa_CPI`。另外报告 `Pfa_CUT`（超过同阈值的 RD cell 比例）和 `suprathreshold_rd_cells_per_cpi`（每幅图超过阈值的 cell 数均值），不能把这三类量混为一谈。按 `snr_db × cnr_db` 五个唯一条件分别用 512 个 calibration H0 设置经验 95th 百分位阈值；每条件 256 个独立开发配对 scene 评估。A–D 主比较为 SNR=10 dB、CNR=10 dB；另在 A/C 做 SNR 与 CNR 0/20 dB 敏感性条件，完整矩阵见 `config.yaml`。

报告每个方法/条件的校准阈值、开发 `Pfa_CPI` 及按场景聚类 bootstrap 区间、`Pfa_CUT`、false alarms/CPI、超阈值 RD cell 数、`Pd` 及 cluster-bootstrap 区间、ROC/AUC、命中时 range/Doppler error、包含 misses 的 joint success（range error≤1 bin 且 Doppler error≤20 Hz）、SCNR、B3 clutter rejection/target retention，以及每 CPI CPU 时间和复杂度。每次检测只输出全图一个 peak；超阈值 RD cell 数只是旁瓣/峰邻近诊断，不冒充业务目标数。不外推到多候选业务处理链。

开发样本量只足以检查较高误警工作点，不足以证明 `Pfa_CUT=1e-6`。后续检测阶段须另行实现/验证适用于低虚警的 CFAR 与足够尾部校准。

## 7. 统计和成功/停止判据

- H0 calibration 与开发 test seed 独立；阈值只用 calibration H0，不在开发 H0 上重校准。
- `Pd`, RMSE/joint success 和 H0/H1 配对差异使用 2,000 次 base-scene cluster bootstrap；ROC AUC 以 scene 为单位计算。报告点估计及 95% percentile interval。miss 在 joint success 中记 0；RMSE 同时报告命中条件值和把 miss 计为失败的 joint metric。
- R0 通过条件：T1–T12 全部通过。
- R1 阈值/检测实现通过条件：校准分布按所设 0.05 CPI 工作点达到经验分位数；独立开发集报告实际 `Pfa_CPI` 区间，不靠它重新调阈值。分离场景 A 的 20 dB SNR、10 dB CNR 条件下，至少一个有效传统基线需满足 AUC 的 95% CI 下界高于 0.5，且 Pd 的 95% CI 下界高于 Pfa 的 95% CI 上界；否则停止后续学生任务并分析模型/基线/场景。
- B/C/D 不设置“网络必须检测成功”门槛；逐条件报告可辨识边界。若 D 显示同一流形上的源分解不稳定，应视为理论/模型边界而非 AI 缺陷。
- 若 R0/R1 通过，可冻结 R2 数据协议；是否进入学生实验由阶段审查报告按证据给出 GO / CONDITIONAL GO / STOP / REDIRECT。若 R1 不通过，不开展教师蒸馏或以更复杂损失掩盖失败。

## 8. R2 学生与蒸馏协议（冻结设计，尚未执行）

学生输入是原始 `[3,130,32]` complex IQ，拆为 6 个实值 I/Q 通道。固定一个小型 2D CNN：8 个残差卷积块、宽度 32、GroupNorm+GELU；所有消融架构完全相同。三个头为：K1 每接收通道 complex RD response；K2 clutter-suppressed target saliency map；K3 detection logit、fractional range 和 Doppler。模型参数、优化器步数和数据预算不得按实验臂改变。

训练集 50,000 scenes，validation 5,000，独立 H0 calibration 5,000，开发评估 5,000，按 base scene 分割。正式训练种子 `[101,202,303]`；RTX 3050 Laptop 4 GiB 顺序训练，batch=32，从 CPU on-demand 读取，不把训练集缓存到 GPU。AdamW、lr=3e-4、120 epoch；FP32 为默认，只有实际支持并记录数值差异后才用 AMP。保存配置、环境、种子、训练历史和最优/最终权重。

监督臂：

- **A truth-only：** 120 epoch，物理真值监督 K3 和适用的物理可计算辅助量，不使用 teacher。
- **B truth + K1：** K1 teacher soft response 预训练 40 epoch，再与真值 K3 联合 80 epoch。
- **C truth + K2：** K2 teacher saliency 预训练 40 epoch，再与真值 K3 联合 80 epoch。
- **D progressive：** K1/K2 teacher 预训练 40 → truth 与任务联合 40 → teacher loss 线性退火 20 → truth-only finetune 20。

Teacher hard labels、teacher soft map、physical truth 分别保存；teacher 不覆盖真值，不可用真值构造 teacher prediction。Teacher 固定为 R1 中满足协议门槛的最强已验证方法及其输出；若没有方法过 R1 门，则取消 B/C/D，不训练。K1/K2 teacher maps 每个 scene 保存 source method、阈值/归一化和置信度；A/B/C/D 使用相同场景、同预算和相同随机训练种子。

E1 任务学习 gate：A 必须在独立 validation 高 SNR 分层上高于常值预测器，且 range/Doppler error 与 H0 false detection 不出现严重退化；若失败，不启动 KD。E2 仅比较 B/C teacher transfer；E3 评价 D progressive 是否在独立开发集、三个训练种子上重复优于 A。最终性能只在模型/协议/门槛冻结后使用新的 final blind seed 范围评估。

## 9. 外部算法依据

- DPCA 需要按有效相位中心/平台运动对齐通道，并在三通道 SAR-ATI GMTI 中作运动目标检测验证；本研究的双程几何公式是按当前 Tx/Rx 约定重新推导，见 [Barker & Barber, Processing Three-Channel SAR-ATI for GMTI](https://udrc.eng.ed.ac.uk/sites/udrc.eng.ed.ac.uk/files/publications/0022.pdf) 和 [MIT Lincoln Laboratory, Displaced Phase Center Antenna Technique](https://archive.ll.mit.edu/publications/journal/pdf/vol12_no2/12_2displaced.pdf)。
- 常规自适应 STAP 对独立同分布训练快拍有样本支持要求，异质背景会限制可用样本；本协议额外以 `390 DOF > 32` 的有限样本维数事实排除满秩全维协方差比较，见 [IET Radar, Sonar & Navigation 论文 10.1049/rsn2.12146](https://ietresearch.onlinelibrary.wiley.com/doi/10.1049/rsn2.12146) 及 [10.1049/rsn2.12034](https://ietresearch.onlinelibrary.wiley.com/doi/10.1049/rsn2.12034)。

## 10. 结果文件和阶段决定

实际运行后至少交付：`config_used.yaml`、`environment.json`、`metrics.json`、`result.csv`、`analysis.md`、RMSE/Pd/Pfa 图和复现命令。`analysis.md` 必须逐项回答物理生成是否正确、每个基线作用、A–D 可辨识性、teacher 适用性、未证明的结论和研究阶段门。

最终阶段审查分别给出工程、科学、研究创新三个判断及 GO / CONDITIONAL GO / STOP / REDIRECT。所有负结果保留；不因当前样本量声称稀有虚警、真实数据泛化、生产校准有效或网络优于传统算法。
