# D2-A-R7-E：先进传统参考适用性与实测结果

## 结论

R7-E按冻结范围完成传统方法适用性检查，并整合R7-B物理诊断、R7-D共同场景结果及7kg只读审查。没有改变方法、窗口、阈值或数据，也未生成额外场景。可在L0/L1上公平运行的FFT、FFT参数细化、lag-one与profile-ML参考均已纳入R7-D；需要dense CUT输出、额外几何/训练背景或新运行接口的算法明确标为未运行/不适用。

逐方法、逐SNR结果见[统一性能矩阵](../../../experiments/D2-A-R7/unified_performance_matrix.csv)，原始条件和适用性依据见[传统方法适用性矩阵](../../../docs/research/D2-A-R7_CLASSICAL_REFERENCE_MATRIX.md)。

## 本轮已实测参考

| 方法 | 输出类型 | 本轮数据与指标 | 解释边界 |
|---|---|---|---|
| B0单通道FFT峰值 | CPI存在分数、距离/Doppler | R7-D L0/L1；L1每SNR 2,048 H0校准、2,048 H0验证、2,048 H1；按Pfa_CPI=0.01分层校准 | 单个CPI分数，不输出逐CUT图；单通道传统参考 |
| B1三通道非相干FFT峰值 | CPI存在分数、距离/Doppler | 同一R7-D场景与阈值流程 | 功率非相干融合，不使用跨通道复相干关系 |
| B1峰值的原生频率、log-parabolic和4×零填充抛物线细化 | 与B1共享检测头的参数估计 | 同一R7-D场景；区别参数读出贡献 | 4×频率栅格不增加独立观测；不扩大物理CUT分母 |
| lag-one相位差 | 参数-only | 全部R7-D L1 H1；无检测分数 | 不报告Pd/Pfa；其周期相位估计随SNR变化 |
| 有限网格profile-ML | 参数-only | 每SNR固定前256个H1；0.25-cell距离网格、16×Doppler FFT | 使用单目标模型和已知搜索边界；估计误差不能与n=2,048的估计结果忽略样本量差异 |

## 同场景结果要点

- −20至−15 dB时检测能力接近低值区。−10和−5 dB时R6-J4固定DPCA/FFT候选学生的点值Pd/Joint高于本轮B1参考；R6-J4包含固定物理前端，属于混合学生，不是纯raw-IQ端到端结果。
- 0 dB及以上，B0/B1传统FFT参考在Pd或Joint上总体更强。具体最佳checkpoint与Pfa_CPI均逐SNR列在R7-D分析和总表中；若实际验证Pfa偏离0.01，必须结合Pfa置信区间阅读Pd。
- 连续Doppler估计方面，传统FFT及profile-ML远强于受测学生。profile-ML在中高SNR的RMSE较低，但它每SNR只处理固定256个H1，且只作参数估计，不提供检测门限。
- 所有R7-D传统检测参考输出CPI级标量分数，`Pfa_cell`均不适用。此轮没有通过将CPI概率换算为cell Pfa来宣称低虚警资格。

## 未运行或不适用方法

| 方法 | 状态 | 本阶段依据 |
|---|---|---|
| ATI相位干涉 | `NOT_RUN_REQUIRES_SEPARATE_GEOMETRY_PROTOCOL` | 当前没有独立的相位中心标定、速度模糊映射及对应输入协议；ATI不能用DPCA或非相干融合代替 |
| CA-CFAR / OS-CFAR逐CUT检测 | `NOT_RUN_REQUIRES_DENSE_CELL_PROTOCOL` | R7-D模型只输出CPI分数；本轮没有为RD局部参考窗、guard、边界/目标污染和dense分母冻结新协议，历史T2仅称CFAR-like |
| Kelly GLRT、AMF/ACE | `NOT_RUN_ASSUMPTIONS_NOT_QUALIFIED` | 目标导向矢量和独立同分布secondary数据条件未建立；未用H1真值补造运行期协变量 |
| 全秩390维STAP/SMI | `NOT_APPLICABLE_CURRENT_TRAINING_SUPPORT` | 3×130维观测与仅32个距离训练门不支持常规全秩样本协方差 |
| Post-Doppler/降秩STAP | `DEFERRED` | 需要先识别并冻结杂波子空间、训练门、秩/加载规则；L0/L1无杂波结果不能代替L2/L3杂波协方差评价 |
| 知识辅助协方差 | `NOT_APPLICABLE_CURRENT_INPUTS` | 当前没有运行期可验证的地形、平台或独立杂波协方差先验 |
| APX15 7kg工程链 | `NOT_QUALIFIED_NOT_RUN` | 多个源码快照缺少可用提交/标签及许可证证据，README自述`NOT_CERTIFIED`，与D2-A距离压缩IQ接口不匹配；本轮未构建/改动/接入 |

## 阶段判断

适合当前统一无杂波对照的传统估计与检测参考已通过R7-D实际运行；方法适用性清单也已明确区分实际数值结果与未运行项。R7-E不构成复杂杂波、低虚警CFAR或7kg整链路资格。G4对于**当前可比无杂波参考**完成，对于复杂杂波自适应方法和工程链仍为未验证；不能据此宣称传统方法全集已覆盖。
