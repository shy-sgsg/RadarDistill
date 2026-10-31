# R6-D 联合检测与参数估计模型设计与正确性门

**状态：** 设计方案，不是实现或训练结果。R6-D使用RTX 3050 Laptop GPU顺序训练；正式场景生成前须冻结独立protocol。

## 1. 目标

比较检测优先D、参数优先P与真正联合J。首轮用共享主干多任务网络J1作为联合基线，再设计一个软候选双向交互模型J2。问题是检测证据与参数估计证据能否互相改善最终Pd/Pfa、连续参数与joint success，而非仅降低一个辅助loss。

输入统一为原始 `X ∈ C^{B×3×130×32}`。训练和推理只能读IQ及冻结物理配置；目标有无、距离和Doppler truth只在训练loss或离线评价路径使用。H0不产生目标参数监督。

R6-C扩大模型在单seed开发集上有收益，但其训练扩展筛选门未通过，因此没有证据说明继续训练必然无收益，也没有证据说明受测网络已到学习极限。R6-D是独立架构比较，四臂统一使用25,600次optimizer updates和effective batch64。该预算是本阶段匹配比较的固定预算，不解释为收敛或容量上限。若验证曲线到25,600步仍显示稳定改善，结果将如实保留并交由后续独立匹配训练预算实验判断，不在本阶段只给个别模型追加更新。

## 2. J1：共享主干多任务基线

复数IQ按实部/虚部两平面输入当前 residual CNN，输出共享表示 `F`。检测头输出H0/H1分数；参数头输出候选距离、连续range offset与PRF周期Doppler。J1的检测和参数头不互相传递预测值，是共享表示多任务基线。

J1须在同一主干初始化、训练数据/scene order、任务loss权重、optimizer/update budget下，和D-only、P-only比较。它用于量化共享的收益/冲突，不被当作双向交互模型。

## 3. J2：软候选双向交互

### 3.1 候选与参数表示

检测头在一个冻结的 `[520 Doppler × 32 range]`候选网格上产生 `L_det`，另含 `l_null`。softmax在null和全部候选上归一化，得到 `p_det(c|X)`；检测分数是非null总概率或其等价log-odds，阈值按H0 calibration独立确定。softmax支持是模型坐标，不是实际输出候选个数。

参数头为每个候选预测局部连续 Doppler残差、range亚格点残差及条件不确定度，得到归一化 `q_param(θ|c,X)`。Doppler表示须按PRF周期：圆周概率/相位向量或等价wrap-aware输出，不在Nyquist端点使用普通线性差。

### 3.2 一次有界双向信息交换

仅在共享主干后做一次交换，不堆叠循环反馈：

1. **检测→参数：** 用 `p_det(c|X)` 对候选特征池化成软候选表示；参数头使用该表示输出加权参数，不使用 `argmax`/hard ROI。因此检测误差仍能通过连续权重传播。
2. **参数→检测：** 每个候选参数头输出range残差、周期Doppler残差的相位向量及可选不确定度。一个共享候选MLP接收`[候选特征, Δrange, sin(2πΔf/PRF), cos(2πΔf/PRF), 不确定度]`并输出`e_param(c,X)=tanh(g(·))`；末层采用标准差约`1e-3`的小幅非零初始化、bias为零，使初始feedback接近零且反向传播从首步即可回到参数分支。严格零初始化会令首步跨分支梯度为零。证据被限定在[-1,1]。将该学习到的参数候选证据加到候选检测logit，再与`l_null`共同归一化。它是可学习的任务交互量，不预先称为物理似然或可靠性分数。第一版不直接使用目标真值、教师输出或已知场景SNR/CNR构造此证据。

最终参数由更新后的非null软候选后验与局部参数分布联合得到；若null概率高，输出“无可靠候选”，不得把随机杂波最大峰强制解释为目标。训练参数loss只在物理H1样本有效；检测loss覆盖H0/H1。

### 3.3 初始损失

J2第一版保持最小：

`L = L_presence + L_candidate/range + L_range_offset + L_circular_doppler`

所有项仍按R6-D冻结config规定权重。首轮不加入蒸馏、置信度权重、额外一致性惩罚或复杂时机。这样先检验双向候选信息能否改善最终任务；如输出不稳定或梯度不合理，停止J2，不增加loss掩盖问题。

## 3.4 训练预算和对照公平性

四臂D-only、P-only、J1-joint与J2-joint使用相同训练seed、训练场景与H0/H1采样顺序、effective batch64、25,600次optimizer updates、优化器和学习率计划、验证频率及checkpoint规则。每400步保存训练/验证汇总，每800步保存恢复checkpoint，不逐optimizer update记录。各臂按同一固定预算完成，不能根据单臂损失趋势单独提前停止或追加更新。

25,600步为相对R6-C初始12,800步筛选加倍的匹配预算，目的是给新联合架构较充分的单seed筛选机会。训练曲线、数据重复遍历次数、训练-验证差距和验证任务指标均须报告。末端仍下降只能说明该固定预算下未观察到平台，不能宣称已达到能力上限；若需更长训练，应另行冻结所有对照臂共同的更新预算。

## 4. 预实现正确性测试

### 数值与结构

- 对随机和边界shape，检查 `p_det`、`q_param` 有限、非负且按定义和为1；null项始终存在。
- 极端logits、全零IQ、非常小IQ幅度和均匀候选logits下不得出现NaN/Inf；数值稳定用log-softmax与有界feedback。
- H0时输出null/检测分数仍定义，参数输出标记为无目标条件值；H0不计算参数真值loss。
- H1只有一个可辨识目标的最小场景下，测试候选输出和回归头形状，不对D exact-overlap要求唯一源归因。

### 周期物理关系

- 真值 `+PRF/2`与`−PRF/2`产生相同wrapped Doppler目标；相位误差使用 `wrap_frequency`。
- Doppler轴由冻结520点centered frequency axis建立；端点、bin/亚bin残差、range gate索引须与配置一致。
- IQ公共复尺度、公共相位旋转与慢时间频移仅做契约测试；未预注册的物理不变性不得写成已获得的性能属性。

### 双向梯度

- 检测loss必须经参数反馈对候选检测logit产生有限、非零梯度；H1参数loss必须经 `p_det`→软池化对候选分数产生有限、非零梯度。
- 分别detach `p_det`与`e_param`的对照必须切断预期的那一侧跨任务梯度，验证双向路径真实存在。
- `detach`探针、均匀分布、单候选和高null场景不得造成梯度爆炸/全零异常。

### 候选和信息隔离

- `forward(iq)`仅接收IQ；模型调用签名不含presence/range/Doppler truth、H0/C3标签或teacher map。
- 真值只进入loss builder；oracle正确range/candidate指标由独立evaluation wrapper后算，不回传模型。
- 测试模型对输入批次顺序置换后输出相同置换；候选概率汇总与直接presence输出在预定定义下相符。

## 5. 资源与执行门

候选图为520×32，feature heads共享主干；启动GPU前记录J1/J2推理参数数、训练参数数、forward/backward峰值显存。RTX 3050 Laptop 4 GiB，单模型驻留，FP32，effective batch64，microbatch从16开始，只在OOM时减至8/4并累积保持有效batch和update不变。数据按batch搬运，不缓存全量IQ或响应图。

- **G0：** 冻结实现、参数数、CUDA/显存预算和场景seed审计通过。
- **G1：** 上述数值、信息隔离、周期边界和双向梯度测试全部通过；16-pair/200-update小样本过拟合达到冻结最小任务学习门。
- **G2：** 在R6-A/B之后的新matched development pilot上，J2对J1/D/P的最终Pd、Pfa、range/Doppler和joint success形成预注册配对差值；不能凭训练loss通过。

R6-D不与正在运行的R5-C争GPU。G1失败则修复实现或停止该具体候选；G1通过但R6-A/B显示其不匹配当前瓶颈时，不训练大样本。R6-D不运行final blind。

## 6. 对照与成果解释

核心对照是同容量/近似计算的D-only、P-only、J1共享主干、J2双向交互，以及已冻结Medium学生。混合系统的神经检测+传统profile-ML按R6-F相同scene ID单列。报告检测命中、正确候选和oracle候选三套参数表现；完整系统joint success把漏检、错误候选和参数失准均计为失败。

J2即使优于某个CNN，也只说明当前协议和复杂度范围的有限收益；不能外推为所有联合模型优于级联，也不能把物理loss或软候选图本身当作创新性证据。
