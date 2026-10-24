# D1-R1：任务级学生连续 Doppler 误差来源诊断协议

**状态：** 开发诊断协议，2026-10-24 冻结；不构成最终测试或新阶段性能门。
**对应假设：** H-ARCH-001、H-OPT-001；辅助审查 H-KD-002。
**基线：** D1-E001 的代码、配置与阶段审查，仓库提交 09724a4。
**研究问题：** D1 TCN 的 Doppler 误差主要来自距离索引耦合、时间均值池化后的频率可读性、任务梯度冲突，还是其它数据/优化因素？

## 1. 适用范围与数据

保持 D1-E001 条件：PRF=1300 Hz、N=130、三通道、32 距离单元、单目标、圆对称复 AWGN、距离压缩复 IQ。此协议不训练模型、不改 D1-E001 结论，也不加入杂波、通道误差、多目标或原始快时间回波。

只读取以下本地开发缓存和正式训练记录：

- 共同开发集：experiments/D1-E001/data/cache/pilot_validation，2048 条，8 个固定 SNR，每类每 SNR 128 条；用于三个正式种子 checkpoint 的配对误差比较。
- 每个正式种子的 seed_i_train 与 seed_i_validation，训练缓存各 8192 条，验证缓存各 2048 条；用于梯度诊断和池化前/后 probe。probe 只用 H1 训练特征，使用同 seed validation H1 评估。
- checkpoint：formal/seed_i/A_truth_only/best_inference.pt；D_progressive_distillation/best_inference.pt 只作策略敏感性对照。
- A/D 训练历史：formal/seed_i/<strategy>/training_history.csv。

这些缓存和验证集均是 D1 的开发材料，不是新盲测。禁止读取 blind 目录、blind predictions、校准数据或最终指标来选择结构、调阈值或构造本诊断。true range 只可用于对照分析，不得进入正常模型 forward 或推理 decoder。

若只导出了 inference checkpoint，training-only spectrum head 不存在，则不能重算该头的 teacher-KL 共享主干梯度。保存的 training_history 中 KL 标量仅用于描述训练轨迹，不能伪称梯度证据。

## 2. 误差分解

对 A_truth_only 三个正式模型在共同 pilot_validation 上运行 CUDA 推理；对 D 模型做策略敏感性对照。报告每个种子和汇总结果：

- H0/H1 检测 score 的 AUC；按固定 SNR 分组。
- H1 距离门分类准确率、距离门索引错误率、预测连续距离与 offset 误差。
- 正常输出：按模型预测 argmax range bin 取 Doppler 向量。
- Oracle 诊断：同一个 Doppler head 在 floor(true range cell) 取向量。真值距离不得输入模型或正式 decoder。
- 两个 Doppler 结果均按 PRF 周期回绕；报告全 H1 误差、检测 score≥0.5 子集误差及各自样本数。
- 分组边界预先固定：频率四等宽区间 [-650,-325)、[-325,0)、[0,325)、[325,650) Hz；Nyquist 邻域 |f|≥585 Hz；初相八个等宽圆周区间；距离门 0–7、8–15、16–23、24–31；幅度四分位边界只由 seed_1_train 的 H1 目标幅度确定，再用于所有验证集。
- 按 SNR、checkpoint seed 和上述物理量分组报告 n、RMSE、bias；小样本组保留 n，不隐藏不稳定区间。三 seed 均值/标准差作描述，不将其称为总体置信区间。

## 3. 池化前后线性 probe

对每个 A_truth_only seed：

1. 从对应 seed_i_train 中抽取 1024 个 H1，按 8 个 5 dB SNR 区间各取 128 个，抽样由固定 seed 决定。
2. 在固定 checkpoint 的 temporal mean pooling 前，从真实距离门取 [channel,time] 特征并展平；池化后从同一真实距离门取 channel 特征。
3. 仅用训练样本的均值/标准差标准化特征。以固定 ridge α=1.0 对 [cos(2πf/PRF), sin(2πf/PRF)] 做线性回归，不调参。
4. 在对应 seed_i_validation 的全部 H1 上评估；真值距离只用于读出 probe 特征。还报告各 SNR 的回绕 Doppler RMSE 和 n。
5. 报告训练样本数、验证样本数、特征维度、正则化、solver 与环境。该 probe 表示“信息可线性读出”的证据，不单独证明池化造成因果损失；特征按时间排列，不做可视化因果主张。

## 4. 任务损失与共享梯度

- 解析 A/D 保存训练历史中的 presence、range-cell、range-offset、Doppler、teacher-KL 训练损失及验证主任务趋势。
- 在每个 seed_i_train 的 8 个 5 dB 区间各用确定性采样的 16 个 H0 和 16 个 H1，针对 A 与 D checkpoint 分别计算 presence、range-cell、range-offset、Doppler 损失对共享 stem+temporal blocks 的梯度范数和两两 cosine。
- H0 不参与参数损失的梯度；各任务独立求梯度后才计算余弦。teacher-KL 梯度若无含 spectrum head 的训练 checkpoint，报告 NOT AVAILABLE。
- 梯度余弦仅反映所选训练样本与冻结模型位置，不单独证明训练过程中的因果冲突。结合损失轨迹、误差和 probe 解读。

## 5. 执行与停止

使用 AISAR / PyTorch CUDA，不训练新模型。数据以 mmap 读取，分 batch 传输；模型顺序运行。保存 git SHA、配置路径、checkpoint 路径、数据 split/tag、运行环境、峰值显存、样本数和所有诊断边界。

输出：scripts/diagnose_d1_r1.py、experiments/D1-R1/metrics.json、analysis.md。若模型、缓存或特征出现非有限值，停止并报告，不通过删样本或更改指标修饰结果。

该结果只能决定一个最小 E002 结构候选。正式 E002 的结构、训练预算、保护门、选择规则和新盲测种子另行冻结，不能由本开发验证集充当最终验收。
