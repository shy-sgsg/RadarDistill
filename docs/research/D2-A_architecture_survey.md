# D2-A 学生架构调研与适用性设计

**范围：** D2-A 3 通道复 IQ、130 脉冲、32 距离单元、静止杂波、检测/距离/连续 Doppler；论文任务与当前体制不完全相同，引用只支持结构动机，不作为性能证据。

| 架构 | 与当前信号的适配 | 主要风险 | R5 决定 |
|---|---|---|---|
| CNN/TCN | 时间卷积和 dilation 能覆盖 130 个脉冲，逐距离输出自然；现有 TaskStudent 是基线 | 普通实值卷积隐式学习相位/频率；有限预算下可能欠训 | Small/Medium 控制组 |
| 雷达时频 Transformer | 长程 token 交互可表达跨脉冲峰结构；FFT/时频坐标适合距离-Doppler图 | 处理完整复 IQ 的 token 与位置敏感性；不同 FMCW/RAD数据不能直接外推到三通道 GMTI | 先评估“慢时间×距离轴分解注意力”，原始 IQ 输入，单 seed pilot |
| 复值网络/复注意力 | 复数线性运算和相位结构更贴近 IQ，可减少 I/Q 结构损失 | complex normalization、复数注意力和数值稳定实现复杂；已有文献多为 DoA/成像/FMCW，不等价于 GMTI | 调研保留，不与首个 attention pilot 并行训练 |
| 状态空间模型 | 递归状态可表示长时序结构，理论上适合脉冲序列 | 该任务序列仅 130，复杂度优势未知；雷达检测论文体制不同 | 设计备选，不做首轮实现 |
| 物理模型驱动深度展开 | 可把似然、稀疏目标、杂波子空间或 GLRT 步骤显式展开 | 依赖精确前向模型、可辨识性、噪声/杂波协方差与可微算子；错模会编码错误 | 只做模型/算子兼容审查，不自动启动训练 |
| 固定物理前端 + 学习器 | 可保留 DFT、相干积累等可验证结构并解释物理前端贡献 | 混合模型不能作为纯端到端成功证据；E003 已证明物理特征不必然带来任务收益 | 报告前端和学习器的独立指标/消融 |

## 首个非 CNN 候选设计

Temporal-Frequency Attention Student 直接接收 `[B,3,130,32]` 复 IQ，以六个实虚平面表示。逐样本/距离进行线性 token embedding；沿 130 个脉冲做轻量多头 self-attention，再沿 32 个距离位置做轴向 attention；二维轴分解避免 4,160-token 全局注意力的平方显存。慢时间 token 加周期频率/时间位置编码，range 轴加位置编码；最终按距离聚合慢时间表示，输出与 TaskStudent 相同的 presence、range、offset、unit Doppler 向量及训练期辅助头。

第一版参数目标约 80K–100K，与 width=96 TaskStudent 接近；设计冻结后报告实际参数量、峰值 CUDA 显存、FLOPs/前向计算及过拟合结果。输入保留原始复数采样，不把 E003 相干特征作为唯一输入，不使用预训练权重；不以未验证速度优势为目标。

## 代表性原始研究

- Giroux et al., **T-FFTRadNet: Object Detection with Swin Vision Transformers from Raw ADC Radar Signals**, ICCV Workshops 2023。用雷达变换层及 Transformer 处理 ADC/RD；体制为汽车 FMCW，不能直接证明 GMTI 适用：[论文 PDF](https://openaccess.thecvf.com/content/ICCV2023W/BRAVO/papers/Giroux_T-FFTRadNet_Object_Detection_with_Swin_Vision_Transformers_from_Raw_ADC_ICCVW_2023_paper.pdf)。
- **Cross-Attention Transformer for Coherent Detection in Radar Under Low-SNR Conditions** (2025)。研究脉冲相干积累与 Transformer，但使用其特定脉冲雷达信号模型：[论文](https://www.mdpi.com/1424-8220/25/24/7588)。
- **Complex-Valued Neural Networks for Millimeter Wave FMCW-Radar Angle Estimations**。比较复数与实值网络的 DoA 估计，支撑复数运算动机，不等同于慢时间 GMTI：[机构论文页](https://tore.tuhh.de/entities/publication/817282b3-4e9f-4747-9e0d-201705a60aeb)。
- **Enhanced Radar Imaging Using a Complex-Valued Convolutional Neural Network**。展示复值 CNN 用于雷达成像：[IEEE Xplore](https://ieeexplore.ieee.org/document/8458209/)。
- **Deep unfolding based space-time adaptive processing method for airborne radar**。体制与机载杂波更近，提供模型驱动展开方向的相关先例；其具体数据/假设仍须与本项目核对：[论文页](https://rhhz.xml-journal.net/en/article/doi/10.12000/JR22051)。
- **Unfolding Target Detection with State Space Model**。结合 CFAR 展开与状态空间网络的目标检测研究；其摘要不足以证明对本项目非均匀杂波/连续参数输出适用：[arXiv](https://arxiv.org/abs/2410.22774)。

## 物理展开设计边界

候选设计可从“已知目标/杂波字典的稀疏目标系数 + 低维杂波/噪声子空间”的复数 MAP/GLRT 或近端优化展开开始。启动实现前必须从 `simulation/d2a.py` 明确算子、噪声/杂波协方差、参数可辨识范围、H0 退化情况及梯度；不能把局部 CFAR展开直接当作 GMTI 任务教师。当前仅列为设计审查，不运行该分支。
