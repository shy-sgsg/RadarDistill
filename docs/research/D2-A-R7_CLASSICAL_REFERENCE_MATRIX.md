# D2-A-R7：传统方法适用性矩阵

**用途：** 为GMTI任务级学生建立可解释、条件明确的传统比较与教师资格边界。该矩阵不是传统算法排行榜，也不把论文在不同雷达/杂波/样本条件下的结果移植到D2-A。

**冻结范围：** 三通道、PRF 1300 Hz、130脉冲、32距离单元；R7-D L0无杂波无噪声，L1无杂波复AWGN。检测工作点是分SNR的Pfa_CPI=0.01，Pfa_cell仍未验证。R7-B另有静止杂波分量诊断。每类结果和原始指标均在[完整统一性能矩阵](../../experiments/D2-A-R7/unified_performance_matrix.csv)中逐条件、逐方法记录。

| 方法族 / 参考 | 方法输入与背景训练条件 | 32距离门及当前仿真适配性 | 本项目实际运行证据 | 复杂度/限制 | 处置 |
|---|---|---|---|---|---|
| DPCA / MTI：几何分数时移差分 | 共轭相位中心几何、PRF、平台速度及通道相干；本身不是单元级门限 | 适合诊断合成的静止杂波抵消；真实Tx共址、通道相位和运动补偿仍未实测确认 | R7-B新独立1280场景。静止杂波相干幅值中位数0.8486→0.9858；残差功率比0.02833；抑制15.478 dB (scene bootstrap 95% CI 15.420–15.540)。噪声残差功率约增至2倍（+3.010 dB）。 | 每CPI约一次两通道分数DFT及差分，O(N log N × range)；结果限定声明仿真几何 | `RUN_PHYSICS_DIAGNOSTIC`; not a detection/estimation ranking row |
| ATI / 相位干涉 | 至少两个经相位标定的空间/时间相干相位中心、基线几何、相位模糊规则 | 潜在可从通道干涉相位估速度；本阶段无独立标定/无模糊运行合同，且L0/L1没有杂波背景用途 | 本轮未作为独立算法运行；与DPCA差分、三通道非相干求和不等同 | 低计算量，但系统标定与模糊区间关键 | `NOT_RUN_REQUIRES_SEPARATE_GEOMETRY_PROTOCOL` |
| B0 单通道FFT峰值检测 | 每CPI单通道IQ；无需背景快拍，需H0校准CPI分数阈值 | L0/L1适用；单通道相干信息 | R7-D在同一组L0/L1场景复评；8个SNR，每格H0校准/验证及H1各2048 | O(Npulse×Nrange log Npulse)；稀疏单峰检测，不是dense CUT输出 | `RUN_R7D_PFA_CPI_0.01`; Pfa_cell N/A |
| B1 三通道非相干FFT峰值 | 每通道IQ；对通道功率求和，不估协方差 | L0/L1适用；可检验通道融合而不使用跨通道相干 | R7-D相同场景评估 | 约三次FFT加通道功率和；不建模相位关系 | `RUN_R7D_PFA_CPI_0.01` |
| FFT原生频率、log-parabolic、4×零填充抛物线 | IQ/FFT局部谱；插值不增加独立观测 | 适合当前连续Doppler的基本估计参考；频率误差用PRF周期回绕 | R7-D检测输出由B1峰定位，再比较原生、对数抛物线、4×零填充抛物线参数 | 后两者增加少量插值工作；4×栅格不可扩大CUT统计分母 | `RUN_R7D_MATCHED_SCENES`; 与B1检测头配套，区分前端贡献 |
| Lag-one相位差 | 慢时间相位增量；可在选定距离门求估计 | 无杂波连续单目标适配；没有独立检测输出，不应报告Pd/Pfa | R7-D对每个L1 SNR全部H1的参数-only估计 | O(Npulse×Nrange)；多目标/低SNR/幅度接近零时易受噪声影响 | `RUN_R7D_PARAMETER_ONLY`; detection N/A |
| 有限网格profile ML | 单目标复数IQ、已知波形/噪声模型和搜索区间；R7-D将距离0.25 cell网格、16×Doppler FFT | 适合作为模型匹配的强参数估计参考；模型正确时接近最优，但不是连续空间精确全局MLE | R7-D每SNR预定前256个H1；不输出检测统计量 | 搜索成本约与距离假设数、FFT因子、脉冲数成正比；只对固定256样本 | `RUN_R7D_PARAMETER_ONLY_256`; detection N/A; 与全2048样本RMSE需标样本量 |
| CA/OS-CFAR (Rohling) | 每个CUT的局部参考单元；OS排序对训练窗中干扰目标污染比CA平均更稳健，但依赖背景统计和训练窗口 | 理论上适用RD图，当前32 range门、Doppler循环边界、窗口跨目标/杂波边缘等影响须另冻结；AWGN-only不是杂波边缘压力集 | R5/R6有CFAR-like历史结果；当前R7-D scalar peak没有dense cell map，不能验证Pfa_cell | 每CUT窗口求和或部分排序；OS需排序。密集相关CPI需要场景聚类CI | `NOT_RUN_AS_CELL_DETECTOR`; independent protocol/data required |
| 历史T2局部中位CFAR-like分数 | B3功率图与局部参考窗；R5/R6冻结窗宽、guard及边界 | 仅对其原协议定义的D2-A杂波场景有历史意义；不能改称标准OS-CFAR | 历史R5/R6行留在CSV；需查看各自stage指标；不作为R7-D新数据的结果 | 背景中位数比均值更抗稀疏强点，但不自动保证恒虚警 | `HISTORICAL_ONLY`; no R7-D transfer claim |
| Kelly GLRT / AMF / ACE | CUT目标导向矢量、多个独立无目标secondary样本，以及在样本支持下估计协方差；ACE尺度不变 | full 390D时只有32个range cells，协方差最多秩32，且无IID/无目标污染保证；当前真实通道标定/目标流形也未实测 | R7-D未运行；ACE/AMF结果不可从相关研究或H1已知真值反推 | 全秩样本协方差 O(p²K+p³); Kelly/AMF依赖可逆或有条件协方差 | `NOT_APPLICABLE_FULL_390D_CURRENT_SUPPORT`; reduced model requires new preregistration |
| full STAP / SMI | 三通道×130脉冲空时向量 (p=390) 与大量同分布背景训练向量，目标导向/杂波谱模型 | 32 range cells远少于390维，训练样本还相关、异质及可能被目标污染；不满足常规SMI训练支持 | R7-D未运行 | 估计复杂度至少约 O(p²K+p³); covariance rank ≤32 if only 32 snapshots | `NOT_APPLICABLE_CURRENT_TRAINING_SUPPORT`; 不以伪逆/强正则冒充标准STAP |
| post-Doppler / 降秩STAP | Doppler滤波后的小维空间协方差或可识别低秩结构；仍需训练门、无污染和秩/加载规则 | 维度降到3有统计可行性，但32 range样本同质/独立性未知；当前没有确认低秩/杂波谱先验 | R7-D未运行；R6文献调研仅是候选理论 | 计算可降为低维矩阵运算；模型选择会引入新自由度 | `DEFERRED`; first validate clutter covariance structure in L2-L3 |
| Knowledge-aided covariance / terrain priors | 经授权的地形/平台/阵列/杂波先验及相符运行数据 | 当前没有可验证的环境协变量或独立真实协方差先验；不能把生成器内部参数输入检测器 | R7-D未运行 | 数据与先验维护成本高；先验失配可损伤Pfa/Pd | `NOT_APPLICABLE_CURRENT_INPUTS` |
| 7kg APX15 engineering pipeline | 可复现构建版本、明确输入数据布局与校准合同、接口适配和同场景输出 | R7-C只读检查未找到本项目D2-A IQ直接适配/可证明版本界面；源码规模不等于可比输入输出 | 未集成、未修改、未声称运行 | 完整链路成本、传感器语义和接口未验证 | `NOT_QUALIFIED_NOT_RUN`; 保留为后续工程核查对象 |

## 传统参考实测结果位置

R7-D的逐SNR实际指标、置信区间、训练检查点/种子、参数量、数据数及源文件指针均存于[统一长格式CSV](../../experiments/D2-A-R7/unified_performance_matrix.csv)。该文件以“方法 × 数据集 × SNR/条件 × 检查点”为行单位，完整保留各历史分组。R7-D所有模型和标量传统分数按`Pfa_CPI`，绝不映射成`Pfa_cell`。R7-B物理诊断按分量/条件列在同一矩阵，也不混入检测Pd排名。

传统估计器评价不替代教师教学收益；R5/R6教师知识质量和对应学生的实际蒸馏收益仍须分列读取。7kg、ATI、CFAR cell-Pfa、ACE/Kelly/AMF、full/reduced STAP均没有本轮R7-D可比检测输出，因此其指标在完整表中明确为`NOT_RUN`或`NOT_APPLICABLE`，不填造值。

## 来源

- Muehe & Labitt, *The Displaced-Phase-Center Antenna Technique*, MIT Lincoln Laboratory Journal, [PDF](https://archive.ll.mit.edu/publications/journal/pdf/vol12_no2/12_2displaced.pdf)。DPCA是运动平台MTI杂波抵消技术，真实效果依赖相位中心/平台运动补偿。
- H. Rohling, “Radar CFAR Thresholding in Clutter and Multiple Target Situations,” *IEEE TAES*, 1983, [DOI:10.1109/TAES.1983.309350](https://doi.org/10.1109/TAES.1983.309350)。OS-CFAR通过排序参考窗单元构造局部自适应阈值；论文优势仅适用于其噪声/干扰假设。
- E. J. Kelly, “An Adaptive Detection Algorithm,” *IEEE TAES*, 1986, [DOI:10.1109/TAES.1986.310745](https://doi.org/10.1109/TAES.1986.310745)。其原始检测模型需要独立且与CUT共享未知协方差的无信号secondary向量。
- L. E. Brennan & L. S. Reed, “Theory of Adaptive Radar,” *IEEE TAES*, 1973, [DOI:10.1109/TAES.1973.309792](https://doi.org/10.1109/TAES.1973.309792)。支撑有限训练样本与空时自适应向量维数的样本支持讨论。
- 更广泛的GMTI教师算法文献与原始论文来源见[`D2-A_teacher_literature_review.md`](D2-A_teacher_literature_review.md)。
