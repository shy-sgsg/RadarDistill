# D2-A-T001 教师候选选择

**选择状态：** 已冻结；依据 D2-A-E001/R3/E002 和文献记录，仅允许这一项主要先进教师候选。2026-10-26 在任何 T001 新数据生成前，根据数学/物理测试设计审查修正 score 定义，详见协议 v1.1。
**研究总纲：** `docs/RESEARCH_CHARTER.md` v1.2（提交 `289b812e8bc93fbdabca0cfb3722ad24f045351b`）。未修改总纲正文。

## 选择

主候选为 **三通道 post-Doppler、模型辅助 ACE×AMF（PD-ACE/AMF）**：用独立 H0 合成 CPI 离线拟合 520 个四倍零填充 Doppler bin 的 3×3 通道协方差；保留 ACE 空间相干度诊断，并以 `ACE × whitened_energy / observable_CPI_mean_power` 作为唯一检测/估计 score。该式代数上为逐 CPI 总能量归一化的 adaptive matched filter (AMF)，既保持全图公共复尺度不变，也保留不同 Doppler bin 的目标能量剖面。再用同一 score map 生成与旧 K2 同形的 `[130,32]` 归一化软响应；以四倍零填充频率网格及局部抛物线插值输出连续 Doppler 估计。每个 H0/H1 检测/估计函数只接收 IQ、已拟合 covariance bank 和公开冻结几何配置。

**数据前勘误：** v1.0 的纯 ACE 对固定空间 steering 的单目标复正弦会消掉各 Doppler bin 的幅值差，故无法定位该目标的频率峰。该问题在任何 T001 场景生成前被发现；v1.1 将唯一候选修正为归一化 ACE×白化能量（AMF），没有增加候选、搜索参数或改变样本、门槛和预算。

这是对 post-Doppler 降维、自适应相干检测理论的有界仿真适配，不声称复现某论文的完整 KRYLOV、空间 AR 或 SBL 方法，也不预称其 SOTA。

## 为什么现在选择它

1. **命中教师实际瓶颈。** T1 重标化 B3 detection score 但保留原 B3 argmax，因此没有改变参数误差。E002 C3 全组 range/Doppler RMSE 仍为 `5.80 cells/93.03 Hz`；C3 部分重叠/SNR0 的旧开发样本上，K2 localization 仅 `4.7%`，joint success 接近 1–2%。
2. **使用完整三通道相干信息。** B3 的物理差分只使用两通道；ACE 的向量 CUT 为每个 Doppler/range cell 的三个复通道，显式匹配已知接收 steering 并对干扰协方差白化。
3. **控制样本支持。** full-STAP 向量维度 `3×130=390`，32 个 range cells 的协方差 rank≤32，不足以支撑 full-rank SMI。候选在 post-Doppler 后只对 3 个通道估计协方差；离线用 4096 个独立、纯 H0 合成 CPI 拟合，使样本数远大于 3 维，但不把独立性、同分布或真实数据迁移视为已证明。
4. **检测、估计与软知识共用一个统计图。** 与 T1/B3 一样避免报告多个不相关子模块的增益；归一化 ACE×AMF 图保留 Doppler 能量剖面，用于检测、range/Doppler 和 K2 形态对照。
5. **资源与实现有界。** covariance bank 只有 `520×3×3` 复数矩阵；每 CPI 的处理由 FFT、3×3 线性变换和17方向 steering scan组成。教师先 CPU 执行；若学生分支通过门控，只训练一个新学生臂并复用已有匹配 S0/S1。

## 备选方案比较

| 方案 | 与证据的关系 | 拒绝或暂缓原因 |
|---|---|---|
| full 390-D SMI/AMF STAP | 经典自适应检测直接扩展到全通道×慢时间 | 每 CPI 32 range cells，协方差秩≤32；忽略理论样本支持与 range 异质性会造成伪精度。 |
| 高阶 post-Doppler AR(P) parametric AMF / Hao-style GLRT | 直接建模时间相关并联合定位，理论上更接近 task-level教师 | 3 通道、32 range cells 下 AR 阶数及 secondary-cell 同协方差假设未得到诊断；一个实验同时估计 AR、spatial covariance、信号参数会混合过多机制。 |
| SBL beam-Doppler sparse reduced-rank STAP | 文献展示低样本自适应协方差路线 | 文献使用不同阵列/字典和 angle-Doppler 辅助通道；当前仅三实体通道且字典模型没有适配证据，算法/超参数较多。 |
| Krylov ACE | 低训练支持、理论上具有 covariance-aware detection | 需要迭代子空间阶数/对角加载规则；本轮先检验 ACE 的核心维度和 covariance 学习是否有收益，不叠加第二种降维机制。 |
| 低秩鲁棒/SIRV或K-distribution covariance | 异质/重尾杂波可用的协方差模型 | D2-A 当前是复高斯点散射体加 AWGN，暂无支持 SIRV/K distribution，也不知道有效 rank；容易改变科学问题。 |
| 更复杂多教师或 router | 可能在不同场景互补 | K2 的困难层错误已知，但没有独立教师互补证据；多教师会混淆“更强教师”与“路由器”。 |
| 纯真值直接新训练网络 | 有利于学生表征学习 | 本阶段重点为教师质量与知识传递；E002 已证实 truth-only 学生基础任务可学，但相位/频移与分层虚警仍有问题。 |

## 实际输入/统计模型及边界

- 固定每个 Doppler bin `k` 的三复通道观测向量 `x[k,r]`。在独立 H0 training CPI 上由三通道 IQ 生成 zero-padded slow-time FFT。以每个 CPI 的可观测总能量归一化其外积，再跨独立场景和所有 range cell 聚合成 `R[k]`，只用 H0 IQ；不读潜在目标振幅、SNR/CNR sidecar、H1 truth、scene 标签或组件功率。
- 目标导向量 `a(az)` 从 E001 冻结 channel geometry 计算，搜索既有17点 azimuth grid；测试 CPI 的 range/frequency/angle 真值不作为输入。
- 空间 ACE 诊断为 `|aᴴR⁻¹x|² / ((aᴴR⁻¹a)(xᴴR⁻¹x))`；唯一检测 score 为 ACE 乘 `xᴴR⁻¹x / e_CPI`，等价于 `|aᴴR⁻¹x|² / ((aᴴR⁻¹a)e_CPI)`，其中 `e_CPI` 是 IQ 可观测总平均功率。这样保留空间匹配且不会把单目标复正弦的 Doppler 能量信息归一掉。固定 covariance 的模型辅助 detector 不继承依赖在线 secondary 的解析 CFAR 定理，阈值必须独立校准/验证。
- 新方法在 `[130,32]` native score map 的归一化形式直接对照 K2。该分布仅为 soft saliency，不解释为校准后验概率。
- 频率估计用520点零填充图和周期三点 parabola；range 用 score map 的整 bin 峰值加三点局部抛物插值（约束在峰值±0.5 cell）。不把结果称为 ML/CRLB 达界。
- 精确 overlap D 只作为可辨识性边界，不设来源标签恢复门。

## 风险、停止条件和主张边界

主要风险包括：covariance bank 学到的是当前仿真混合先验，不代表真实地杂波；range cells 和 CPI 之间的统计依赖/异质尚未知；steering 有 Tx 共址与理想通道相位假设；ACE 对固定模板失配敏感；乘回白化能量后，统计量还受到错误协方差谱形的影响；零填充提高插值精度但不增加独立信息；全图归一化响应可能仍向 H0 最大偶然峰分配质量。

如果 G1 正确性失败，停止统计比较并保留可复现失败；如果 G2 没有预注册的困难条件增益或保护条件失败，停止教师分支；如果 G3 软知识可靠性失败，不运行学生新教师蒸馏。每个失败结论均限于本仿真和本冻结协议。不得追加第二种高级教师或在这批开发结果上搜索 covariance shrinkage、角度网格、温度或背景窗参数。
