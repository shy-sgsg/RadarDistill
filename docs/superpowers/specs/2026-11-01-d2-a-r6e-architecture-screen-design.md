# D2-A-R6-E 代表性架构筛选设计

**状态：** 按已批准的 R6 P4 路线和当前 `RESEARCH_ROADMAP.md` 执行的有界阶段设计。当前设计选择是在既有授权范围内作出的实施决定，不更改研究总纲或 R6-A 至 R6-D 冻结结果。

## 目标

比较显式保留复数相干关系的卷积学生、能沿慢时间和距离轴交换信息的轻量 Transformer，以及现有 J1-J 联合学生基线，判断在相同训练数据、更新预算和任务目标下，显式相位结构或全局序列交互是否带来可重复的频率估计或联合任务信号。

该阶段服务于研究总纲 v1.3 的 Q3、Q4、Q6、Q10；它是单训练种子的架构筛选，不是正式多种子确认、最终盲测或低虚警部署资格声明。

## 当前证据和代码边界

- R6-D 的 J1-J 是直接输入六平面实/虚 IQ 的共享任务基线，推理参数量 70,277；其卷积主干使用跨距离 stem 和慢时间扩张残差块，没有固定 FFT 或其他物理频谱前端。
- R6-D 中 J1-J 的整体 Doppler RMSE 为 20.67 Hz，优于 J2-J 的 27.01 Hz；J2-J 的 joint/range 收益伴随 Doppler 保护失败。J1-D/J2-J 验证目标在 12,800 后继续改善，J1-J 则在 14,400 达到最佳。故 R6-E 使用比 25,600 更长的**统一 51,200 updates 上限**，但按共同验证目标选择 checkpoint；不将训练长度本身解释为收敛。
- 用户允许后续新实验在比较公平时适当增加训练更新。R6-E 的 51,200 是每臂相同的固定预算（是常见12,800预算的4倍、R6-D的2倍）；本阶段不根据开发评价为单臂追补训练。后续若需更长训练，应在新协议中预注册全比较组共同更新数、数据暴露、学习率计划和checkpoint规则。
- 现有 TFAS 将六个 I/Q 平面用实值线性嵌入成 56 维特征，加入绝对正弦时间位置编码，经两层时间 Transformer 后先对时间做注意力池化，再仅在距离轴做一层 Transformer。R5-B 的负结果只适用于该模型、预算和数据。
- 新候选不使用教师、FFT 响应或标签作为模型输入；两种候选都从同一原始三通道 IQ 学习。物理结构和可学习部分分别记录。

## 考虑的做法

1. **五类候选同时实现并筛查：** 多尺度 CNN、复数 CNN、改进 Transformer、SSM、CNN+序列混合。覆盖范围最大，但实现机制、参数规模和训练结果难以归因，也会扩展单卡筛选预算。
2. **只比较复数 CNN 和 J1-J：** 直接检验相位归纳偏置，成本最低；但无法判断 TFAS 的负结果能否被结构不同的序列模型改善。
3. **J1-J、复数等变 TCN、完整轴向注意力 Transformer 三臂匹配比较（采用）：** 以一个紧凑实验区分实值 TCN、显式公共相位等变卷积和保留完整慢时间—距离网格的注意力交互。暂不加入 SSM/混合臂；本阶段不把单个候选的负结果外推到整个架构家族。若数据为后续 SSM 提供明确问题，可另行立项。

## 三个模型

### A：J1-J 联合基线

重新按 seed 606 在 R6-E 新数据上训练 R6-D 的 `R6J1Student` 联合臂，width=96，保持现有输出字典和 `compute_r6d_losses` 真值任务损失。此基线不复用 R6-D checkpoint，避免训练数据和场景不同带来的比较混淆。

### B：CE-TCN 复数等变卷积

- 输入从 `[B,6,130,32]` 还原为三通道复数慢时间数据；stem、慢时间深度卷积、复数 pointwise 混合和残差块均采用可学习复数权重，不使用固定 FFT/匹配滤波前端。
- 结构固定为 `(1,3)` range stem、7个慢时间 depthwise `(3,1)`残差块（dilation `1,2,4,8,16,32,64`）、复数 pointwise通道混合；复Group RMSNorm仅使用模长RMS做实数缩放，径向modReLU，复卷积不设偏置。
- 复Group RMSNorm用8组，在每样本每组的通道、时间和距离位置计算`|z|²`均值，`eps=1e-8`，仅有实数通道scale而无偏移；modReLU用可学习每通道实阈值，按复数模长径向缩放。任务输出先取复模长并沿时间均值，再产生与J1-J一致的四项任务输出。
- 复数线性层无加性复数偏置；归一化若启用，只依赖相位不变的模长统计；径向激活只按复数模长缩放。由此中间复特征对所有通道/脉冲共同相位旋转应保持等变，模长读出后的检测和参数输出应保持不变。实现测试直接验证这一性质。
- Doppler 平移等变不作为结构保证；频移扰动只作独立诊断。该候选的物理贡献是复数线性运算与公共相位对称性，学习器贡献是有限深度 TCN 对局部/扩张慢时间和距离关系的拟合。
- 从 width=64 的复数卷积候选开始；按实数自由参数总量（每个复系数计两个实参数）确认推理参数在 J1-J 的 ±20% 内。只允许在此阶段做一次容量配平修订。

### C：RAX-Transformer 完整轴向注意力

- 原始 I/Q 六平面实值嵌入为 48 维 token，在完整 `[time=130, range=32]` 网格上执行两个时间轴/距离轴交替注意力 block。时间轴使用相对慢时间 lag 编码，距离轴使用相对距离偏置；保持时间 token 直到所有轴向交互完成后再池化。
- 每个block依次执行时间轴注意力、距离轴注意力和token-wise FFN；采用4头、FFN维96、GELU、pre-LayerNorm、dropout=0、无绝对位置项；每轴使用每head的可学习精确相对偏移bias（时间−129…129、距离−31…31）。
- 两轴注意力均为全双向非因果attention；每个token以LayerNorm后做注意力/FFN residual更新，不使用attention mask。池化后用与J1-J相同的range/presence聚合和周期Doppler输出契约。
- 相对位置设计只表达采样位置差异，不声称频率平移等变或公共相位不变；输出以单位圆上的 Doppler `(cos,sin)` 表示 PRF 周期性。
- 与 R5-B TFAS 的实质差异是保留整个时间—距离网格进行交替轴向交互，并使用相对 lag，而非先做时间注意力池化后才处理距离。它仍是完全可学习的 raw-IQ 模型，不含固定物理频谱层。
- 采用 48 维、4 头、两个轴向 block 的初始结构；实数推理参数需在 J1-J 的 ±20% 内。FP32 前向/反向峰值由 CUDA G1 实测；如果显存不足，先减 microbatch 并保持有效 batch64。

## 公平性与训练边界

- 三臂共享新生成、互不重叠的训练、验证、H0 calibration、H0 verification 和 H1 development scene IDs；跨臂使用相同场景、配对 sampler 顺序和物理标签。R5-C、R6-A/B/C/D 的 scene IDs 均不得复用。
- seed=606；E1 物理真值监督；每臂最多 51,200 optimizer updates、有效 batch64（H0/H1 各32）、FP32、同一 AdamW/cosine 计划、权重衰减、梯度裁剪、验证频率和 checkpoint 选择规则。训练样本固定为51,200个配对场景并重复采样，清楚区分更新数与场景访问次数。
- checkpoint 每800 updates保存，history 每400 updates记录；不逐 optimizer update 写日志。按 J1-J joint truth validation loss 的对应实现选择每臂最佳 checkpoint；不得查看 H0 verification 或 H1 development 选模型。
- 按顺序训练，一次只驻留一个模型；有效 batch 固定64以隔离架构因素。此阶段不同时搜索 batch、宽度、损失、训练种子、温度或蒸馏权重。
- H0 calibration 与 verification 各120,000 CPI；H1 development 6,144。阈值只由 H0 calibration 的全局 `Pfa_CPI=0.05` higher 分位数确定；verification 与 development 只作独立评价。0.01/0.001可以报告描述性分层统计，但不扩展资格声明；不声称 1e-4 或 1e-6。
- 不生成 final blind，不开展知识蒸馏、模型蒸馏或推理时延测试。训练缓存在全部报告写完后仅按 R6-E owner manifest 清理；保留 protocol/config、scene manifests、checkpoint、history、预测、metrics、图和分析。

## 指标与判读

主要架构筛选端点为全局 calibration `Pfa_CPI=0.05` 下可辨识 C2+C3 的配对 joint success。两个新候选分别对J1-J比较，按scene-level paired bootstrap取双侧97.5%区间，对两项候选比较做Bonferroni控制；候选须点差至少`+0.02`且区间下界大于0，才视作有实质相对收益。Pd/RMSE任务保护使用同一两项比较的97.5%配对区间。另报告每个C0–C3条件及困难C3 partial-overlap/SNR=0 dB/CNR=20 dB的Pd、Pfa、joint、range RMSE、Doppler RMSE、AUC。提供overall/all-H1、检测命中和正确候选条件下参数误差，区分条件估计器与完整任务。

任务收益筛选沿用 `+0.02` joint 点差且配对区间下界大于0；Pd 非劣界为 `−0.02`，range 与 Doppler RMSE ratio 的保护上界为1.05。C3 六格 familywise exact Pfa 上界 `≤0.10` 是**低虚警资格保护**，与架构学习信号分开判读：未过该绝对门的候选不能称为合格低虚警检测器，但其满足预注册相对任务筛选的证据仍可支持受限研究，不自动等同于架构无价值。

所有比较均为单训练 seed 开发筛选；即使通过，也只决定是否进入后续 R6-F/P7 选模，不是跨 seed 优势或盲测结论。若模型曲线在 51,200 末端仍改善，记录为未达平台，不能称能力上限；本协议不因末段曲线事后追加单臂更新。

## 主要停止条件

1. G0 数据、协议 hash、seed/scene 独立性、CUDA 设备和缓存预算不通过时不生成正式数据。
2. G1 输出/梯度/相位物理性质错误时先定位实现原因；修复后仍不满足预设性质或16-pair/200-update loss 下降至少10%时，停止具体候选并保留负结果。
3. 全臂使用共同 51,200-update 预算和 checkpoint 规则。development 相对任务收益、Doppler/range/Pd 保护及 C3 绝对虚警资格分开报告；不得因为只通过其中一项就宣称完整成功。
4. 若所有新架构候选均无任务增益或出现严重保护退化，停止本轮候选；不自动扩大到 SSM/混合/多层搜索，先据证据决定下一条 R6 路线。

## 文献依据及范围

- Trabelsi et al., *Deep Complex Networks*, ICLR 2018：提供复卷积、复归一化和复值网络构件；其视觉/音频任务不证明本 GMTI 任务受益。[arXiv:1705.09792](https://arxiv.org/abs/1705.09792)
- Su et al., *RoFormer: Enhanced Transformer with Rotary Position Embedding*, 2021：展示相对位置的 rotary attention 构造；证据来自语言任务，R6-E 只借用相对位置编码思路，不宣称已验证雷达性能。[arXiv:2104.09864](https://arxiv.org/abs/2104.09864)
- Wang et al., *Axial-DeepLab: Stand-Alone Axial-Attention for Panoptic Segmentation*, ECCV 2020：以两个一维轴向 attention 分解二维交互；其图像分割任务和位置设计不同于本实验。[作者论文 PDF](https://www.cs.jhu.edu/~alanlab/Pubs20/wang2020axial.pdf)
- Gu & Dao, *Mamba: Linear-Time Sequence Modeling with Selective State Spaces*, 2023：研究重点是长序列建模和效率，而当前慢时间 CPI 仅130脉冲且本阶段不以推理加速为目标，因此 SSM 不列入首轮 matched pilot；这只是优先级选择，不是否定 SSM。[arXiv:2312.00752](https://arxiv.org/abs/2312.00752)

## 项目边界

保持 `docs/RESEARCH_CHARTER.md` v1.3、R5-C T0–T3 与 R6-A/B/C/D 的所有协议和结论不变。R6-E 完成后只写内部阶段报告和动态状态记录。依据 `AGENTS.md` 和用户当前要求，R6 P0–P7 全部目标完成之前不创建或提交新的 ChatGPT 审查请求；中间阶段不因缺少审查而阻塞。
