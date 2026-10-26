# D2-A 教师瓶颈归因

**证据截止：** 2026-10-26。只复核已冻结的 D2-A-E001、D2-A-R3、D2-A-E002 开发结果；本报告没有新运行，也不改写旧结果。
**所据资料：** E001 `protocol.md`/`analysis.md`/`metrics.json`；R3 `protocol.md`/`analysis.md`/`metrics.json`；E002 v2.1 `protocol.md`、`metrics.json`、`k2_audit.json`、`architecture_and_loss_review.md` 及逐场景开发 CSV；D2-A 物理模型和可辨识性分析。

## 研究主线定位

本诊断服务于研究总纲 v1.2 的 Q1、Q2、Q3、Q4 和 Q9：识别传统教师的任务误差与知识质量，并独立评价提升的教师是否有利于学生。它不是单独扩展 CFAR/STAP 算法清单，也不声称 T1 已是正式教师。总纲未修改。

## 当前教师与学生基线证据

- **B3** 是三通道 IQ 的几何 fractional-time alignment DPCA，但检测差分只使用通道 0/1，之后取全图 FFT 功率峰值。E001 在分离 A、近边缘 B 检测很强；部分重叠 C 开发 Pd/joint success 约 `0.53/0.53`；D 同流形不可辨识，不作算法失败门。B3 Doppler/距离均为峰值 bin 输出，没有连续参数联合似然。
- **T1** 是 B3 RD 图最大功率除以固定 guard 外的全图中位数。它改变 detection score 和阈值统计，不改变 argmax：因此 T1 的 range/Doppler 估计与 B3 完全相同。R3 CNR=10 dB partial-overlap C 中 T1 Pd/joint=`0.982/0.982`，优于 B3=`0.530/0.530`；但 R3 固定阈值 G2 未通过，不能称为稳定可部署检测器。E002 新阈值/混合开发上 T1 AUC=`0.9863`、Pd=`0.9206`、joint=`0.9010`，参数估计仍继承 B3，且 C3 条件退化明显。
- **K2** 是 B3 的归一化完整 RD power map。它不等于检测阈值教师，也不能继承 T1 的强检测结果。E002 K2 audit 的 pooled C3 identifiable A/B/C target-window mass lift=`0.03623`，top-cell localization=`0.7227`；但 C3 partial-overlap/SNR=0 子组 lift=`0.000673`，top-cell localization=`0.0471`。正 pooled lift 遮住了目标最需要软知识的困难子组。
- **E002 学生**在 5,000 对开发场景上 A/B/C AUC=`0.9846/0.9787/0.9815`，但 joint success=`0.6934/0.5680/0.5716`，T1=`0.9010`。课程真值 B 与课程+K2 C 的 joint success 差=`+0.0036`，95% 场景 bootstrap CI=`[-0.0088,+0.0162]`；没有 K2 pilot 增益。三个学生虽通过 pooled task gate，C3/SNR0 分层 H0 上限约 `0.195–0.204`，公共相位/频移门也失败。

## 分层瓶颈归因

| 来源 | 观测证据 | 归因判断 | 不能据此断言 |
|---|---|---|---|
| 杂波抑制 | B3 对 A/B 有效；对 C 的 DPCA 差分目标保留较弱。E001 抑杂中位数约 15.406 dB，但该数不能代表所有 target/clutter manifold。E001 中位目标差分功率比约 3.614，因差分未归一化可大于 1。 | DPCA 的两通道差分对 C 的目标/杂波流形重叠不够充分；它也丢弃了第 3 通道。改善完整三通道相干利用值得一次有界检验。 | 不证明 DPCA 本身错误，也不证明 full-STAP 一定有用；A/B/C 目标频率重叠之外，精确 D 的来源归因物理上不可辨识。 |
| 检测统计量/尺度 | R3 确认 H0 绝对分数尺度与潜在参考幅度耦合；T1/T2 消除受控整体尺度变化，但不满足所有独立固定阈值 G2。E002 pooled Pfa 可在阈值下控制，而 C3/SNR0 子组上限高。 | 旧 T1 能改善全局排序/检测，但一个全局阈值在困难混合和分组里不保证稳定；按分布混合训练/校准的均值不足以保证低SNR、高CNR子组。 | 不能把 R3 固定 5% G2 与 E002 混合子组门混为同一统计条件；也不能把尺度不变等同条件 Pfa 稳定。 |
| 距离与 Doppler 估计 | T1 和 B3 使用同一峰值 index，故误差逐场景相同。E002 C3 组 T1/B3 range RMSE=`5.80 cells`、Doppler RMSE=`93.03 Hz`；C3 partial-overlap/SNR0 在旧开发集 n=208 上 T1/B3 joint success=`0.0096/0.0192`，估计误差约 `10.76 cells/162.79 Hz`。 | 当前 range-Doppler peak 是共同参数瓶颈；T1 只重标定检测，不做连续频率插值、目标模板拟合或 nuisance-aware估计。单个较差 peak 可同时导致检测和两项估计失败。 | 旧开发单条件 n=208 只作诊断，不是 T001 主要结果；不能据此调参数或估算新的方法收益。 |
| K2 软监督失真 | K2 在 pooled C3 看似有用，但 C3/C/SNR0 top cell 仅 4.7%，目标邻域 mass lift 仅 0.000673；软图可能把杂波峰交给学生。 | “均值软图质量”不足以代表教学价值；困难层需要分别测 localization、target mass、H0响应/熵和错误率。 | 当前 C−B 无收益不否定其他教师；本轮没有证明 K2 对所有难度都有害。 |
| 可辨识性 | C 是部分 Doppler overlap，但 target range/angle仍可区分；D 共享静止杂波点的同一流形，来源不可辨识。 | C 是合理的困难主门；D 只描述，不纳入性能筛选或要求无先验标签恢复。 | 任何模型在 D 上失败都不能作为 C3/C 的教师失败或学生失败证据。 |

## 竞争的机制解释

1. **分数/阈值机制：** T1 的 AUC 和 detection 提升主要来自可观测尺度归一化，不代表目标参数信息增加。通过共同新测试同时比较 AUC、固定 Pfa Pd、range/Doppler 与 joint success，可以拆开排序、阈值和估计。
2. **单通道差分信息损失：** B3 用通道0/1，剩余第三通道没有进入其目标统计。C 条件下完整三通道空间相干可能区分部分具有相似 Doppler 的 target/clutter；但是否可区分由具体阵列 steering 与噪声/杂波协方差决定，需实验。
3. **训练快拍不足：** 完整 390 维 SMI 不可由 32 个 range cells 充分支撑；post-Doppler 维数 3 有统计可行性，但单 CPI range cells 可能相关、杂化或含目标。T001 用独立 H0 合成 CPI 离线拟合 3×3 covariance bank，不假设一个测试 CPI 的32个门都 IID；代价是 teacher 依赖仿真分布协方差，存在模拟到真实失配。
4. **物理模型失配：** 目标模板/接收 steering 依赖 Tx 共址、理想标定和平地模型；当前没有实机通道标定。T001 只能验证在冻结仿真物理模型下的实现，不给出现场部署结论。

## 最优先教师缺口

第一优先不是继续搜索 CFAR 窗，而是在 C3 partial overlap/SNR0 中同时检验：保留完整三通道相干的目标导向检测图，是否能改善检测/连续 Doppler，并形成比 K2 更可靠的目标邻域软响应。一个 post-Doppler 三维 ACE 候选满足可实现和样本支持边界；它不通过任一阶段门时保留负结果并停止，不再扩展第二个 advanced teacher。

## 证据不确定性

1. E001/R3/E002 使用的物理散射、AWGN、Tx phase center和通道校准均为仿真假设；没有真实数据或完整 APX15 实测回放。
2. 各旧结果来自开发场景，且 E002 的开发集已经用于模型/路线诊断；T001 将重新生成独立场景。旧指标只用于候选选择和预期困难区间，不进入 T001 新统计。
3. 旧 K2 的软知识质量不能由硬检测 Pd 取代；T001 需要独立评价 H0/H1 map、目标窗口 mass、定位、错峰、熵与 paired differences。
