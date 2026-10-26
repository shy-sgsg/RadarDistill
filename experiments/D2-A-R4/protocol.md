# D2-A-R4 预注册协议：教师—学生物理信息瓶颈诊断

**协议版本：** 1.0
**状态：** 运行前冻结；执行代码与本协议先提交到 main。
**实验性质：** 合成仿真开发诊断；非正式 blind，不改变 T001 或 E002 冻结结论。
**总纲：** docs/RESEARCH_CHARTER.md v1.2；本实验不修改总纲正文。

## 1. 研究目标与硬阶段门

目标是区分：(a) T001 复数尺度诊断中的输入量化与 CUDA 精度误差；(b) ACE×AMF 教师是否丢失目标位置/频率信息；(c) 冻结 E002 学生的公共相位、频移、标签范围门和条件虚警表现；(d) 如相位问题在新场景复现，单一公共相位增强的 E1 truth-only pilot 是否可行。

T001 冻结状态仍为 G1 FAIL、G2 FAIL、G3 NOT EVALUATED、候选 STOP。历史 G1 的 1e-6 门槛和判定不变。R4 对新数值诊断另设绝对差 1e-5 的诊断门，同时报告相对误差、图级差、峰位置和阈值决策。新门槛不得追溯替换 T001 门槛。

T001 代码曾先评价教师 H0/H1 结果，再在末尾完成 G1 物理诊断；它没有在 G1 FAIL 后运行 S2，但计算顺序未把 G1 设为评分硬门。R4 所有输出标签为 exploratory_only_due_to_T001_G1_G2_fail。任何教师/学生分数仅作诊断，不能宣称通过旧阶段门。不得训练 S2/KD，不生成 final blind。R4-3 只允许条件触发的一个 phase-augmented E1 truth-only pilot；不运行正式学生三种子实验。

## 2. 冻结信号模型与分组

沿用 D2-A-E001：三通道、PRF=1300 Hz、N=130、32 距离单元、静止地面杂波、理想通道、复 AWGN、H0/H1 共用背景和噪声、H1−H0 为目标 IQ。仿真标签、潜在噪声/杂波分量不得作为检测器或学生输入。

所有场景使用新种子块 41M–48M，与 D2-A-E001/R3/E002/T001 及 final-blind reserve 分离。数据生成分块不超过16 CPI，流式送入 CUDA 教师/学生，不把全量 IQ 存到 GPU。场景ID和实际计数记录到 metrics.json。

| Split | 数量 | 预注册构成 | 用途 |
|---|---:|---|---|
| 数值精度 | 64 配对场景 | A/B/C/D 各16；复尺度 0.5、2、0.6+1.2j 和公共相位旋转 | CPU complex128 对 CUDA complex64；同一原始 IQ 的尺度构造必须先在 complex128 完成 |
| H0 calibration | 4096 | C0/C1/C2/C3 各1024；C3 中 SNR 0/10 dB 各512；CNR 使用 E002 固定值 | 仅为 B3/T1/ACE_AMF 计算新 R4 诊断阈值 |
| H0 verification | 4096 | 同 calibration mixture，独立新种子 | 固定 R4 阈值验证 Pfa 与分层差异 |
| H1 diagnostic development | 2048 配对 | 4 个冻结场景单元各512：C0/separated/20/-5；C1/separated/15/0；C2/partial_overlap/10/10；C3/partial_overlap/0/20 | 教师/学生信息损失、估计误差和条件指标 |

另用同一场景 seed 在四个 scenario label 下各生成16对 SNR=10/CNR=10 场景，逐元素核对 H0 是否与目标 scenario label 无关；这项实现审计不并入独立场景统计样本。

4096 指独立 CPI 场景数；每个 CPI 的32个距离单元不是独立 H0 样本。T001 使用逐 Doppler-bin 的 3×3 协方差银行，并非 390 维 STAP。

## 3. 数值精度审计

载入冻结 T001 covariance_bank.npz 和同一 T001 教师实现，不改写冻结源文件。报告 CPU complex128 与 CUDA complex64 基线输出差异，以及公共相位/复尺度变换差异。对于每一项分别记录：

- 最大与均方 score 绝对差；相对差以 CPU 参考量级归一化并保护零分母；
- score map 和归一化 soft map 的最大绝对差；
- argmax 频率/距离是否变化；
- 用冻结 T001 ACE_AMF 阈值作出的决策是否变化；
- 变换因子及其复数精度。

输入构造先将 IQ 提升为 complex128，再乘复尺度，之后才交给 CPU/CUDA 路径。同步复现“先转 complex64 再乘”的历史构造作为量化对照。R4 新诊断绝对差门为 1e-5；相对误差、argmax 和决策结果分别报告。任何结果都不重写 T001 的 4.58e-5 / 1e-6 历史记录。

## 4. 教师信息损失诊断

固定 B3、R3 T1、T001 ACE×AMF 定义，不搜索参数。三种方法各自用 R4 H0 calibration 的第95百分位高侧经验分位数（method=higher）校准，使用 >= 判决；H0 verification 和 H1 development 只评价、不回调阈值。报告总体及预注册条件 Pfa、精确单侧95% Clopper–Pearson 上限、Pd/AUC、range/Doppler MAE/RMSE、检出条件误差和 misses-as-failure joint success。

对 ACE×AMF 图报告 target window mass、top-cell 与目标位置误差、目标峰/最近竞争峰差，并按峰位置正确与错误分组。Oracle S=H1−H0 只用于事后分析可观察目标增量是否被教师响应保留；绝不送入教师、检测器、校准器或学生训练。单独记录协方差条件数/特征值摘要与分数行为，不把32个 range cell 当作训练协方差所需的4096个独立场景。

T001 物理频移诊断另报告“只对 H1 目标分量施加频移”与“重新生成不同物理目标速度”两类操作，避免把整幅 IQ 人工平移误称真实运动。

## 5. 冻结学生诊断

逐一评估 E002 seed 101 的 A/B/C best_inference.pt，禁止权重更新。每学生 H0 阈值沿用冻结 experiments/D2-A-E002/calibration_thresholds.csv；报告新 H0 verification 的总体和 C0–C3/SNR/CNR 分层 Pfa、精确单侧95%上限。报告与 E0/B3/T1 的同场景对照，但 R4 不据此重新选择正式模型。

诊断包括：32 个公共相位旋转的一致性；公共相位下 score、range、Doppler 变化；全 IQ 的慢时间相位斜坡与 H1 目标分量频移分开；重生成目标运动场景作真实速度对照；range cell/offset 解码与物理真值标签范围门的一致性；presence 命中/漏检条件下的 range/Doppler 误差。不得用学生分数/阈值重新校准来掩盖条件漂移。

S1 触发条件冻结为：新 H1 development 上 A/B/C 中至少两个模型的公共相位 presence 决策一致率 <0.99，或 Doppler 旋转绝对变化 p95 >10 Hz；并且 G1 工程/数据门通过。若触发，仅可做一次单种子公共相位增强 E1 truth-only exploratory pilot。不得加 K1/K2/ACE 软图或教师硬标签。

## 6. 可选 S1 pilot（仅触发时）

采用 E002 原始 TaskStudent、seed 101 的共同初始化、同一 E002 A arm 数据 seed/批次计划、FP32、有效 batch64、1600次更新、RTX 3050 cuda:0，顺序运行。S0 对照采用已冻结 E002 A 记录，不重训；仅在 S1 中对同一 H0/H1 pair 施加一个公共均匀相位旋转，二者共享旋转，真值标签不变。E1 仅用物理真值 BCE/range/Doppler 任务损失。每步记录学习率、损失、阈值不变的验证表现、显存峰值与耗时。OOM 时仅降低 microbatch 至8或4并保持 effective batch、更新预算和采样计划。

pilot 使用独立 seed base 46M 的 H0 calibration、47M 的 H0 verification、48M 的 paired H1 development 评价；相位增强 RNG seed=2026102809。其结果仍为 exploratory，不改变 E002 或 T001 门。停止条件：CUDA/G1 失败、非有限损失、microbatch4仍 OOM、验证任务损失不降，或训练后公共相位一致性及真值任务指标未改善。不得以训练损失下降单独判定成功。

## 7. 统计、环境、复现与阶段门

固定阈值比较符 >=；同时报告 > 计数、平局数和校准样本数。H0 Pfa 使用精确二项区间。H1 paired 方法差异按 base_scene_id 配对 bootstrap 2000次，seed=2026102801；区间表达场景不确定性，不代表训练种子不确定性。明确报告 H1 样本量及其开发/选择用途。

预注册门：

- G1：新场景 split/seed 不重叠；H0/H1 配对关系正确；设备确认为 RTX 3050 Laptop cuda:0；教师/学生 API 仅使用 observable IQ；数值差异、有限值和索引检查完整。
- G2：教师/学生诊断复现时不得取代历史门。R4 只判断异常机制是否有新证据解释，必须标出仍失败的层级。
- G3：触发条件满足方可运行 S1；未满足则记录不训练理由。
- G4：pilot 若运行，至少要求 E1 验证 AUC 高于0.5、range/Doppler 误差优于训练真值常数基线、公共相位决策一致率达到0.99且 Pfa 不恶化超过0.02；否则停止增强方向，不蒸馏。

环境记录 Python、NumPy、PyTorch/CUDA、驱动、GPU、可用/峰值显存、CPU、训练/推理耗时和运行 Git SHA。结果保存配置、指标、精简 CSV、分析、绘图及模型检查点；不保存重复全量 IQ。数据可由冻结配置和种子重建。实验后移除 R4 临时数据与无用中间文件，项目总占用须小于20GB；不得删除冻结实验、用户生产文件或既有审计。

## 8. 复现入口

- protocol 与代码提交并推送后：python scripts/run_d2a_r4.py --stage preflight --device cuda:0
- 诊断：python scripts/run_d2a_r4.py --stage diagnostics --device cuda:0
- 条件触发的 E1 pilot：python scripts/run_d2a_r4.py --stage pilot --device cuda:0

final_blind: false；正式多种子学生训练、KD 和路线级 GO 均不在 R4 授权范围内。
