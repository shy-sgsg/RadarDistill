# D2-A-R7-D：统一无杂波历史模型复评

**状态：** 实验执行前冻结。仅复用冻结历史checkpoint；本阶段不训练模型、不蒸馏、不生成final blind。

## 目标

在同一套三通道、130脉冲、32距离单元的D2-A观测接口下，复评可恢复的D1/R5/R6学生，分开回答检测和距离/Doppler估计在无杂波条件的表现。结果是新的历史checkpoint迁移诊断，不能替代原阶段结果或被称为对原训练分布的独立泛化证明。D0/D1旧指标另按各自冻结数据原样进入统一矩阵。

## 数据定义

使用`experiments/D2-A-E001/config.yaml`中的平台、波形、通道几何、32距离格、PRF=1300 Hz和N=130。仿真场景采用新种子块3,700,000,000起，SNR分层为`[-20,-15,-10,-5,0,5,10,20] dB`。SNR定义沿用D2-A生成器：目标参考复振幅平方/连续距离PSF峰值处复噪声功率。参考目标幅度与初相沿用配置的均匀分布；距离连续采样于`[4,28)` cell、方位沿用窄波束区间，径向速度均匀采样`[-15,15] m/s`，Doppler按PRF=1300 Hz周期回绕。

- **L0：** H0为全零复数IQ；H1只包含物理目标回波，噪声和杂波分量均精确置零。每类2,048个场景。用于基础映射诊断；H0没有随机噪声分布，所以不校准或报告可部署Pfa。
- **L1：** 无杂波、复高斯白噪声。每个SNR分别生成2,048个H0 calibration、2,048个独立H0 verification和2,048个H1场景。H0只保留模拟器产生的noise分量；H1只保留target+noise。即使调用共用配对生成器，最终IQ中也精确剔除clutter分量，配置及指标记录`clutter_power=0`。
- H0/H1、校准/验证使用互不相同的种子块；每个方法使用完全相同IQ和真值。仅保存流式紧凑分数/估计量、场景计划和汇总指标；临时IQ缓存须由R7-D owner记录并在结果验收后删除。
- SNR分层阈值仅用于给出**每SNR条件Pfa=1e-2**比较，不表示未知SNR下的单一全局工作点。阈值从独立H0 calibration中按SNR、按模型分别取`higher`经验分位数；H0 verification只用于评价。L0不计算阈值。

## 目标与指标

学生模型只输出一个CPI存在分数和每CPI单一距离/Doppler参数，因此报告Pfa_CPI、Pd、ROC AUC、每CPI候选输出率及全H1/检测命中样本的距离RMSE、周期Doppler RMSE和MAE；`Pfa_cell`标记为`NOT_APPLICABLE`，不把稀疏单候选概率换算为逐CUT虚警。学生参数误差对全部H1报告，漏检不会从联合成功率分母移除。联合成功定义为检测命中、距离绝对误差≤1 cell且PRF回绕Doppler绝对误差≤20 Hz。

传统参考包括单/三通道非相干FFT峰值检测、原生FFT峰值与抛物线插值、4×零填充FFT峰值/插值、lag-one相位差，以及profile-ML有限网格参数估计。单独的相位差/profile-ML估计器无检测输出时，检测列标`NOT_APPLICABLE`。profile-ML使用每SNR前256个固定H1索引；其他方法用全部H1。所有传统阈值只根据相同的独立H0 calibration估计。

置信信息：Pd/Pfa用精确二项95%区间；参数RMSE用场景级bootstrap 95%区间。报告SNR、实际H1 Doppler范围、样本量、参数量、历史训练种子/更新数、checkpoint路径和SHA-256、代码/配置来源及推理设备。训练或验证数据量缺失时写`NOT_RECORDED`。

## checkpoint纳入规则

在运行前冻结并保存`checkpoint_inventory.csv`。纳入D1-E001 formal各arm×seed的`best_inference.pt`、D1-E002 formal S1/S2×seed的`best_inference.pt`、R5-A/R5-B/R5-C-teaching、R6-B至R6-J4中已报告模型的正式`best_checkpoint.pt`。排除pilot、initial/last checkpoint、attempt/debug副本、仅含训练中间状态的重复文件；R6-C带`_12800`重复副本不重复计作新模型。R6-F派生混合系统依赖冻结学生与外部处理链，留在其历史协议行，当前不另训或假装为独立checkpoint。

缺失checkpoint保留为`MISSING_CHECKPOINT`；可读但接口不能正确装载/缺乏输出的模型记录具体失败并停止该模型，不静默替换或重训。检测专用/参数专用arm的非训练任务头不计为相应能力。

## 环境与停止条件

顺序使用AISAR环境和CUDA `cuda:0`，峰值批量预注册为16，单次只让一个模型驻留GPU；禁止CPU回退。出现OOM时降至8再4，只改变推理微批、不改变场景/模型/阈值。记录GPU型号、VRAM峰值、耗时和磁盘。若模型输入尺寸或载入参数不匹配，标明该checkpoint未能复评；不修改checkpoint或原stage协议。

若L0/L1模型参数输出普遍呈常值、非有限或极端失效，保留结果并定位接口/模型适用性，不通过新训练修补。R7-D不是Pfa=1e-6资格实验；所有方法当前只给Pfa约1e-2的有界样本估计。
