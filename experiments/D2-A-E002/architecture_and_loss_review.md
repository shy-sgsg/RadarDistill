# D2-A-E002 学生架构、损失与教师设计审查

**状态：** 训练前静态审查与 pilot 后证据审查；第 1–5 节保留冻结前判断，第 6 节记录实际运行结果。
**基线提交：** `1e4300c9b81dbc500216d1b355bbcb618d35f253`。
**研究总纲：** `docs/RESEARCH_CHARTER.md` v1.1。
**冻结输入/模型来源：** D2-A-E001 的三通道静止杂波物理模型；D1-E001 `TaskStudent` 和 `task_loss`。

## 1. 三通道复数相干信息

`TaskStudent` 把 `[B,3,130,32]` IQ 映射成六个实数通道 `[I0,I1,I2,Q0,Q1,Q2]`。首层 `Conv2d(6,32)` 可混合通道和实/虚部，后续非线性网络在数学上可以从数据拟合通道间交互；因此不能因为没有复数算子就断言它无法学习相干信息。另一方面，权重没有复数乘法的共享约束，GroupNorm/SiLU 也不保证公共复相位旋转等变，模型没有显式计算 `x_i x_j*`、通道时延对齐或 DPCA。相干关系可能需要由训练数据隐式学会，课程和 K2 是本轮受控候选。

本轮不向 A/B/C 主干增加复数层或 DPCA；三个臂只比较训练呈现和 K2 监督。相位旋转与通道相干消融会检查模型是否实际依赖相关结构。若三组均失败，且数据和优化检查有效，再提出一个轻量、物理含义明确的 coherence feature 对照；不立即扩展网络宽度或搜索 Transformer。

## 2. 慢时间频率信息

七个宽度 32 的时间卷积残差块使用 dilation `[1,2,4,8,16,32,64]`，时间感受野 `1+2×127=255`，覆盖 130 脉冲。卷积在均值池化前对不同时间位置进行非线性变换，所以时间均值不等同于对原始 IQ 直接求平均。不过模型没有显式频移等变性或复相位周期先验。D1-R1 的固定 ridge probe 在池化后表现更好，但特征维度从 4160 降到 32 且样本仅 1024，不能作池化因果结论；R1 也没有证实池化是主要瓶颈。D1-E002 S2 有限改善高 SNR Doppler，却未通过低 SNR 检测保护门并远落后传统精估计器，故本轮不把 S2 结论扩写为通用频率结构收益。

当前 Doppler 头按每个距离门输出单位 `(cos,sin)` 向量，以 cosine loss 和 PRF 周期回绕误差处理连续频率。该表示不受线性角度边界不连续影响，但网络本身仍须学会相位—频率映射。

## 3. 检测、距离、Doppler 耦合和损失

当前前向含全局 presence logit、range-cell logits、cell 内 range offset 和逐 cell Doppler 向量。训练按真实 range cell 选取 range offset 与 Doppler 头监督；推理 `decode_task_outputs` 先取预测 range argmax，再从该 cell 读 Doppler。若 range 分类错误，训练时监督的 Doppler 向量与推理读取的向量不一致，强杂波下可能放大参数误差。协议要求在新 validation 与最终开发 split 上记录 true-cell/predicted-cell Doppler 误差及错门比例，确认影响后再决定是否另立最小 loss 修正；本轮冻结 A/B/C 共用现有 loss，不边看结果边更改。

presence BCE 对全部 H0/H1 计算；range CE、offset Smooth-L1(`beta=0.1`) 和 Doppler circular cosine 只对 H1 计算。四项默认等权。D1 协议结果说明网络曾在简化 AWGN 中完成任务，但 D1-R1 没有持续强任务梯度冲突证据；当前提交没有记录 D2-A 训练数据上的梯度范数。因此 E002 preflight 会记录初始各项损失与单任务梯度范数，并在训练过程中记录每项损失/验证曲线。出现具体数值失衡时如实分析，不在本轮加入未预注册的自适应权重。

## 4. K2 定义及教师风险

K2 是 E001 B3 DPCA 残差的完整非阈值 RD power 图，按全图和归一化成软分布；它只由可观察 IQ 计算，不读取真值或元数据。与 T1 全局背景归一化检测分数不同，K2 是中间 soft response，不使用固定阈值。R3 G2 未通过仍意味着 T1/T2 不能获批为固定阈值教师；它不直接裁决 K2 软响应是否能作为受控训练信号。

已知风险来自 D2-A 的可辨识性：A 与静止杂波频带分离，B 靠近谱边缘，C 部分重叠，D 同流形不能唯一归因。B3/K2 可能削弱低速 C 的响应，教师审查用配对 H0/H1 和目标邻域质量量化；H0 map 只作诊断。K2 loss 只作用于 H1，并与始终存在的物理真值 task loss 联合，权重固定从 0.05 退火至 0。

## 5. 执行前冻结的经验诊断

协议冻结后完成了极小固定 H0/H1 过拟合、真值损失梯度、K2 H0/H1/尺度与目标窗口 audit，以及验证集上的真实 range cell 与预测 range cell Doppler 对照。A/B/C 训练和评价按 v2.1 执行；本节原先列出的诊断计划由第 6 节中的实际记录取代。没有重构输出结构或修改冻结损失。

## 6. Pilot 后架构、损失与门控诊断

### 6.1 实现和训练有效性

三臂均使用同一个 13,383 参数训练版 `TaskStudent`、相同初始化和数据多重集；inference export 移除 spectrum/K2 辅助头。训练损失按冻结方案对 presence、range cell、offset 和 Doppler 等权；C 的 H1-only K2 权重从 0.05 线性降至 0。RTX 3050 Laptop GPU 上顺序运行，microbatch 16、有效 batch 64、FP32，每臂 1,600 updates，无 CPU 回退或 OOM。

G1 micro-overfit 的真值任务损失从 7.1634 降至 0.02834（100 updates），各冻结任务分支梯度检查通过。K2 的 H0/H1 图均为有限、非负且归一化；最大尺度变换图差为 0，信息隔离检查通过。因此本轮 Doppler 与阶段门失败不能归因于训练入口未运行、K2 数值异常或基础梯度缺失。

### 6.2 K2 的条件性质量

按冻结规则 K2 audit 允许运行 C：C3 可辨识 A/B/C 合并的目标窗口质量 lift 均值为 0.03623，95% 区间 `[0.03200, 0.04073]`，teacher top-cell localization 为 0.7227，未触发取消 C 的预注册严重误导规则。但分条件看，C3 的 partial-overlap / SNR=0 dB top-cell localization 仅 0.0471，target-window lift 为 0.000673（95% 区间 `[0.000544, 0.000796]`），表明这类输入的 K2 位置知识很弱。此结果支持“允许一次有界 C 臂”，不支持 K2 对所有困难目标都可靠，也不改变 R3 固定阈值教师资格。

### 6.3 训练 range 门与推理 range 门

冻结损失在真实 range cell 监督 Doppler，解码在预测 range argmax 读取 Doppler。选中的 validation checkpoint 上，预测 cell 与真实 cell 的不一致率为 A/B/C=`10.30%/11.14%/10.96%`。同一 checkpoint、同一 validation H1 上，用真实 cell 读取的 Doppler RMSE 与正常预测 cell 解码 RMSE 分别为：A `40.95/46.66 Hz`，B `56.03/58.55 Hz`，C `59.55/60.03 Hz`。在独立 development 逐场景记录中，按 `floor(range_estimate_cell)` 恢复预测分类门后，不一致数为 A `502/5000`、B `551/5000`、C `565/5000`（`10.04%/11.02%/11.30%`）。

这组同模型反事实读数说明错误 range 门会额外增大 Doppler 误差，A 臂差值最大约 5.71 Hz；但即使使用真实 range cell，RMSE 仍为约 41–60 Hz。因此该门不一致是可测的次要误差来源，现有证据不支持它解释大部分 Doppler 误差。比较只有一个训练 seed，验证聚合值也没有逐场景 bootstrap 区间；不据此作因果强度或跨 seed 推断。冻结的训练损失保持不变。

### 6.4 阶段解释

TaskStudent 在可辨识仿真开发分布上能够学习检测和联合任务（G4 通过），且通道独立相位扰乱会使 AUC 下降，说明任务表现会利用通道相干；但公共相位与频率平移等变门失败，课程/K2 也没有预注册增益。range 门诊断没有揭示足以解释整体 Doppler 误差的训练—推理错配。现阶段应保留普通实值 TaskStudent 作为比较锚点；由于本实验没有比较替代架构，不能据此认定复数/几何网络更优。阶段决定仍为 REFINE，不启动正式三 seed、后续蒸馏确认或 final blind。
