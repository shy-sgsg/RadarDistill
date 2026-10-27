# D2-A-R5-B：非 CNN 架构适用性调查与结构设计

**调查日期：** 2026-10-28
**适用信号：** 三通道复 IQ、PRF=1300 Hz、130 脉冲、32 距离单元、D2-A 静止杂波及 A/B/C 可辨识场景；D 为同流形边界参考。
**用途：** 为 H-ARCH-003 选择一个有界、预算匹配的非 CNN 学生候选；不是各类模型的性能排名。

## 研究问题与现有证据

本调查服务于研究总纲 v1.3 的 Q3/Q4/Q6/Q8/Q10，以及 H-ARCH-003 与 H-LEARN-002。R5-A 已表明，在相同 seed101、E1 真值监督和 12,800 updates 下，Medium width=96 CNN 相比 Small width=32 在 C2/C3 联合成功和连续参数误差上更好；但 Small 与 Medium 的 C3 H0 Pfa 分别为 0.2041 和 0.1890，均未过 0.10 门。R5-A 没有检验非 CNN 架构。

当前输入为 `[B,3,130,32]` 复 IQ。目标 Doppler 由跨脉冲相位旋进编码，PRF 周期为 1300 Hz；距离标签来自 32 个离散距离单元及单元内偏移。网络须同时保留目标与静止杂波的三通道相干关系，不能只拟合目标幅度或把频谱峰位置当成连续参数真值。当前任务直接预测 H0/H1、距离单元/偏移及周期 Doppler。

## 架构族比较

| 架构族 | 与本任务的适配点 | 关键限制 | 本轮判断 |
|---|---|---|---|
| CNN/TCN | TaskStudent 已用慢时间膨胀卷积处理 130 脉冲，并在 32 个距离单元上输出检测和参数；Small/Medium 给出已训练、预算明确的参照。 | 普通实值卷积没有严格公共相位等变保证；Medium 的 C3 虚警门失败。 | 必须保留为匹配基线，不预设其弱于注意力。 |
| 时频/时空 Transformer | 可直接比较不同时刻的相干特征，显式建立长脉冲依赖；把时间轴与距离轴分解可避免在全部 130×32=4160 个 IQ 位置做全局注意力。脉冲雷达 RD-Transformer 将脉冲相干积累建模为注意力融合，但同时使用 Tx 参考并聚焦检测；另一项 TrDet/MVTrDet 在 R-D 图上利用二维位置注意力做杂波建模。 | 注意力位置编码及实值 Q/K/V 不保证频移等变或公共相位不变；池化可能丢失频率相位轨迹。已发表的工作与当前的三通道机载静止杂波、直接连续参数估计并不相同。 | 选择一个原始 IQ、时间/距离分解的轻量注意力 pilot；对相位、频移和 C3 Pfa 单独作诊断。 |
| 复数网络/复数注意力 | 复数线性算子可保留 I/Q 旋转结构；FMCW 雷达复数网络在二维 DOA 任务上展示过相对实值网络的收益，复数时间注意力也用于微动目标时序识别。 | 相关论文分别是 FMCW DOA 或微动分类，不能推断对当前 GMTI 检测、距离和连续 Doppler 有收益。严格复数注意力还需控制公共相位群作用、归一化与非线性，增加实现变量。 | 本轮候选采用显式 I/Q 双实通道 embedding，避免频谱/幅度输入；不宣称它是复数等变网络。若 pilot 显示相位干预问题，后续再单独冻结复数等变结构。 |
| 状态空间模型 (SSM) | 选择性 SSM 提供线性序列复杂度，可处理长序列；已有雷达工作把 CFAR 结构展开为轻量 SSM 检测器。 | 该检测器针对 FMCW 人体检测和 CFAR，参数约 260K；Mamba 主要在语言、音频、基因序列上验证。当前时间长度仅 130，尚无证据表明需优先承担 scan 顺序、CUDA kernel 和复数输入适配成本。 | 暂不训练；保留为后续替代方案，若 attention 的训练/显存或长程结构成为瓶颈再考虑。 |
| 物理模型驱动深度展开 | 稀疏模型/ISTA/C-LISTA 可将目标稀疏、Doppler focusing 等先验写入有限迭代；DURAS 及异质杂波展开检测工作证明此路线可行。 | DURAS 面向 sub-Nyquist Doppler focusing，Stevens 等工作主要输入 R-D 网格图并估计稀疏目标图；与当前稠密 130 脉冲三通道 IQ 及任务级连续参数头不同。 | 只完成数学适配审查，不做第二个完整模型 pilot。 |
| 物理前端 + 学习器 | DFT/相干积累、匹配滤波或似然搜索可以降低频率学习难度；每种保留算子可与同前端的非学习读出匹配比较。 | E003 的固定相干相关＋投影 S2 在任务保护和 C2/C3 joint 门失败；重复其 28→6 窄投影无法区分前端信息损失与学习器能力。FFT/R-D 前端还会改变输入边界，不能当作纯端到端优势。 | 仅分析可能的可微 Doppler profile-likelihood/有限频率格点展开；没有单因素证据前不另开训练臂。 |

## 文献与适用范围

以下均引用原始研究或出版方/作者机构页面；归纳只覆盖其实际体制和任务。

1. Lu, Pan, Zhou, “Cross-Attention Transformer for Coherent Detection in Radar Under Low-SNR Conditions,” *Sensors* 25(24), 7588 (2025), [DOI 10.3390/s25247588](https://doi.org/10.3390/s25247588). 面向脉冲雷达低 SNR 相干检测，以 Tx/Rx cross-attention 学习相干融合；支持“注意力可以作用于脉冲相干关系”的动机，但其已知 Tx 参考、检测任务和场景不能替代本实验的 Rx-only 参数估计验证。
2. Shi et al., “Masked variational transformer for complex clutter modeling and target detection,” *Signal Processing* 239, 110236 (2026), [DOI 10.1016/j.sigpro.2025.110236](https://doi.org/10.1016/j.sigpro.2025.110236). 使用二维位置嵌入建模 R-D 谱杂波，并在 measured/simulated 数据评价；它是谱图上的无监督检测工作，不等价于三通道原始慢时间 IQ 学生。
3. Kaiser et al., “Complex-Valued Neural Networks for Millimeter Wave FMCW-Radar Angle Estimations,” EuRAD 2022, [DOI 10.23919/EuRAD54643.2022.9924767](https://doi.org/10.23919/EuRAD54643.2022.9924767). 在十种真实 FMCW 设备数据上比较 CVNN/RVNN，研究对象是二维 DOA。它支持保留复数结构值得验证，不证明 Doppler/GMTI 任务收益。
4. Bai et al., “Recognition of Micromotion Space Targets at Low SNR Based on Complex-Valued Time Convolutional Attention Denoising Recognition Network,” *IEEE Transactions on Radar Systems* 3 (2025), 193–202, [DOI 10.1109/TRS.2025.3527209](https://doi.org/10.1109/TRS.2025.3527209). 复数时序注意力和 CV-TCN 用于微动识别；分类标签及回波设置不同于本任务的检测/距离/连续 Doppler 联合输出。
5. Gu and Dao, “Mamba: Linear-Time Sequence Modeling with Selective State Spaces,” [arXiv:2312.00752](https://arxiv.org/abs/2312.00752). 提出输入依赖的 selective SSM，重点证据来自通用长序列、多模态任务；不作为当前雷达收益证据。
6. Yu and Wu, “Unfolding Target Detection with State Space Model,” [arXiv:2410.22774](https://arxiv.org/abs/2410.22774). 将 CFAR 处理结构与 SSM 组合，在 FMCW 人体检测上报告约 260K 参数模型；可作为雷达 SSM 先例，但任务和分布不同。
7. Goyal et al., “DURAS: Deep Unfolded Radar Sensing Using Doppler Focusing,” ICASSP 2021, [DOI 10.1109/ICASSP39728.2021.9414967](https://doi.org/10.1109/ICASSP39728.2021.9414967). 将 partial Doppler focusing、CNN 和 C-LISTA 结合，处理 sub-Nyquist pulse-Doppler 信号重建/检测；与当前完整慢时间观测不是同一估计问题。
8. Stevens, Tigrek, van Sloun, “Deep Learning for Radar Target Detection in Non-Homogeneous Clutter,” [作者公开论文](https://tristan-deep.github.io/publications/cfar/cfar_paper.pdf). 比较小型 CNN 与基于稀疏 R-D 模型的 10 层展开 ISTA；其展开器约 986 参数，目标是网格 R-D 检测和后续 centroid localization，不能直接作为连续 Doppler 学生的结果。

## 候选结构：Temporal-Frequency Attention Student (TFAS)

“Frequency”指由脉冲间相干相位旋进承载的 Doppler 信息，本候选不做 FFT，也不把离散频率图作为输入。结构固定如下：

1. 输入为缓存既有 `iq_ri` 布局 `[B,6,130,32]`：三路实部后接三路虚部。仅做布局变换为 `[B,32,130,6]`，不取幅度、不归一化相位、不使用 E003 的 28→6 固定相关投影。
2. 每脉冲以共享线性层 `6→56` 嵌入；加确定性正弦脉冲位置编码，以标出 130 个有序慢时间位置。
3. 对每个距离单元独立沿 130 脉冲执行 2 层 pre-norm Transformer encoder（width=56、4 heads、FFN=112、dropout=0）。每距离门用可学习标量 attention pooling 汇成一个时序特征。
4. 32 个距离特征加可学习距离位置嵌入，再经 1 层同规格 range Transformer encoder，提供距离轴上下文。
5. 输出接口与 `TaskStudent` 一致：全局 presence logit、32 距离 logit、32 个单元内 offset、每距离门的单位圆 Doppler 向量。Doppler 仍由原有 circular decode/loss 处理。

参数量按 PyTorch `TransformerEncoderLayer(d=56, nhead=4, dim_feedforward=112, norm_first=True, dropout=0)` 逐项预计为 **79,638**，比 Medium CNN 的 82,887 少约 3.9%。实现后在运行 preflight 中核实；目标硬窗为 Medium 参数量的 ±20%，超窗则不启动训练，也不现场调宽/深。

### 物理关系和预期风险

简化目标项满足 `x_c[n,r] ≈ a_c(r) exp(j(2π f_D n/PRF + φ_c))`。原始 I/Q 与时间位置使注意力可以访问同一距离门内跨脉冲复相位关系；三通道共享嵌入和距离上下文可联合利用通道间相干信息。输出采用原有 `cos/sin` Doppler 表示，处理 `−PRF/2` 与 `+PRF/2` 周期边界。时间/距离分解把注意力规模限制为 `32×130² + 130×32²`，避免全局 `4160²` token attention。

该结构是实值注意力网络，不具备严格公共相位旋转不变性或 Doppler 频移等变性；位置编码/池化可能削弱窄频率结构，attention 可能学会依赖场景幅度或过拟合已看开发条件。公共相位旋转、整体 Doppler 频移、C0–C3、边界/误差分层及 C3 H0 Pfa 均按 R5-B protocol 报告。这些诊断复用既有开发 ID，只用于物理解释和筛选，不构成 blind 证据。

## 可微物理替代方向的数学检查（仅设计审查）

如果后续证据指向频率读出而非检测/距离，可用当前已校准的距离门 `r` 对每通道做复相干积累：

`A_c(f,r) = Σ[n=0..129] x_c[n,r] exp(−j 2π f n/1300)`，

并构造多通道非相干统计量 `Λ(f,r)=Σ_c |A_c(f,r)|² / (σ̂²(r)+ε)`。在单目标复 AWGN、距离门正确、每通道复振幅未知且噪声方差已知时，profile likelihood 对复振幅取最大后与最大化上述相干能量等价；对有限样本的实际背景功率估计会使其不再是精确似然。频率搜索按 `PRF=1300 Hz` 周期回绕，DFT 格距为 `PRF/N=10 Hz`、最近格点的最大量化误差为 `PRF/(2N)=5 Hz`，sub-bin 修正再与 FFT 插值/profile-ML 比较。当前 D2-A 背景是有结构的静止杂波，`σ̂²` 与通道背景并非已知白噪声；若把它直接当 ML 或恒虚警统计量会错误。因此本轮不将公式作为前端、教师或训练输入，只将其作为后续物理展开设计的可证伪起点。

## 结论

证据支持测试一个小型时频分解注意力模型：脉冲雷达文献提供相干注意力先例，复数雷达模型说明相位算子有值得检验的归纳偏置，而当前注意力长度很短，且已存在同任务预算匹配 CNN 参照。证据不足以直接选择复杂数值注意力、SSM 或深度展开，也不足以说 Transformer 更优。若 TFAS 在固定 12,800-update 单 seed 内未通过基础可学习性/任务保护/C3 H0 绝对虚警门，停止该候选，保留负结果，不追加 Transformer 深度、调参网格、知识蒸馏或 blind。
