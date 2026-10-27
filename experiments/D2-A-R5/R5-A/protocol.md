# D2-A-R5-A：训练充分性与首轮容量 pilot

**版本：** 1.0
**冻结状态：** Phase 0 文件先提交并推送；本 protocol 冻结后才生成/加载 R5 缓存、训练模型或查看 R5 结果。
**本次范围：** E003 S0/S2 训练充分性；之后 Small/Medium 容量对照。蒸馏、正式多 seed 和 final blind 未获此 protocol 授权。

## 1. 研究问题与总纲映射

E003 每次 optimizer update 使用 32 个配对 H0/H1 scene，固定训练集包含 51,200 对，故 1,600 updates = 1 epoch。E003 单 seed S0/S2 的 1-epoch 结果不能区分训练不足、容量限制、优化问题、表示限制或 seed 波动。

本试验对应研究总纲 v1.3 的 Q1/Q3/Q4/Q6/Q8/Q10 及 H-CAP-001/H-CAP-002/H-ARCH-003：在固定数据与架构下先增加优化训练预算；随后固定预算只改变 TaskStudent 主干宽度。目标是判断“已测候选和预算内”的学习趋势，不宣称得到全局架构上限。

## 2. 冻结仿真、任务和数据

- 信号：三通道复 IQ；PRF=1300 Hz；130 脉冲；32 距离单元；现有 D2-A 静止杂波模型及 A/B/C 可辨识、D 同流形边界定义不变。
- R5-A 复用 E003 train（51,200 paired base scenes）与 validation（5,000 pairs）的同一 scene ID、物理标签和配对关系。缓存可重建；只由 E003 seed/config/manifest 确定性生成，不覆盖或改写 E003 目录。
- E003 calibration、verification、C3 H0 和 development 数据只供诊断/模型筛选；均为已使用的开发材料，不能称作 R5 blind，也不得用于再次调门槛。
- 最终独立 H0/H1 测试集尚未生成；只能在最终结构、loss、阈值和 checkpoint 规则冻结后，由独立提交的协议授权生成。

## 3. 比较 A：训练充分性

### 对照与训练

| 项目 | S0 | S2 |
|---|---|---|
| 结构 | 原始复 IQ 的 `TaskStudent(width=32)` | 冻结 E003 固定相干前端、28→6 投影、相同主干 |
| 初始化 | seed 101 | 所有共享 TaskStudent 张量与 S0 bit-identical；projection 按同一确定性 seed 初始化 |
| 监督 | 物理真值 E1 | 物理真值 E1 |
| 优化 | AdamW，lr 1e-3→1e-4 cosine，weight decay 1e-4，clip 1.0 | 同 S0 |
| 数据/顺序 | 相同 train IDs；每 epoch 独立确定性 permutation，pair 不拆分；两模型共用 epoch permutations | 同 S0 |
| 预算 | 6,400 updates 主分析 | 6,400 updates 主分析 |
| precision/device | FP32，RTX 3050 Laptop `cuda:0` 强制 | 相同，顺序运行 |

每次更新有效 batch=64 个 CPI（32 个配对 H0/H1 scenes）。每 epoch 1,600 updates；主预算 4 epochs。训练从头开始，使用单个完整 6,400-update cosine schedule；不续接 E003 checkpoint。每 400 updates 验证和保存 checkpoint。旧 E003 独立 1,600-step schedule 与 R5 的 update-1600 checkpoint 不视为严格等预算比较；旧结果只作历史锚点。

### 是否延长至 12,800

6,400 updates 是首轮上限；不自动运行 12,800。只在 S0 或 S2 任一臂满足全部条件时，S0/S2 两臂一起延长，保持公平：

1. update 4,800 至 6,400 的 common validation truth-task loss 相对下降至少 1.0%；
2. 末 5 个预定验证点中，至少 3/4 相邻区间 loss 不升高超过 0.5%；
3. 无非有限 loss/gradient、训练崩溃或预先冻结任务保护指标的明确恶化。

扩展时保持模型、AdamW moments、scheduler 状态、RNG、pair sampler/epoch 和累计 updates 原样连续；不重置 optimizer。更新 6,400 后开始预注册扩展段 cosine `1e-4→1e-5`，结束于 12,800。最终 checkpoint 按所有预定验证点中的最小 common truth-task validation loss 选择。否则在 6,400 停止并说明原因。扩展结果不用于声称理论收敛。

## 4. 比较 B：容量

训练充分性阶段完成审阅后，比较 Small width=32（13,383 params）与 Medium width=96（82,887 params）。只改变主干宽度；dilation、stem、heads、输入、truth loss、optimizer、数据、训练 seed 和有效 batch 不变。两者使用同一固定 train/validation IDs 和相同初始化/样本顺序规则；CUDA 顺序运行。首轮预算 6,400 updates；若 R5-A 的冻结训练曲线规则触发延长，Small/Medium 同预算执行，仍保留 optimizer 状态。Large（约 500K–1M）不在本 protocol 授权范围；只有 Medium 显示有意义任务收益且存在容量证据时另行提出有界协议。

## 5. 停止窗口与诊断分类

- 最小预算 6,400 updates；只有 G0/G1 数值、数据或设备安全门失败才会提前中止。
- 验证间隔固定 400 updates；报告 train/validation truth-task loss、presence、range、Doppler、joint、AUC 及每个 C0–C3/预定场景条件。
- **仍在改善：** 符合上节 extension 条件，继续至 12,800；即使不满足，也如实报告训练末趋势。
- **平台：** 6,400 末段不满足 extension 条件，且主要 train/validation 任务曲线变化小；只称在本预算/数据内观察到平台，不称全局收敛。
- **过拟合：** train task loss/任务指标持续改善，validation 连续至少 3 个验证点变差，且变化超出 scene bootstrap 区间或预设相对阈值。记录证据，不延长。
- **优化异常：** 梯度/损失非有限、训练指标震荡或任务头明显不学习。停止受影响臂，排查代码；修复需先追加协议版本，保留原失败日志。
- **可能的表示/任务瓶颈：** 只有在充分预算、无明显过拟合/优化异常、train/validation 一致受限并有独立物理信息诊断时才提出；不以低验证分数单独判定。

## 6. 结果指标和门

- 主要任务端点：可辨识 C2/C3 配对 development joint success。定义为检测成功、绝对距离误差≤1 cell、圆周 Doppler 误差≤20 Hz；漏检计失败。
- 共同报告：H0 Pfa（calibration/verification 分开）、Pd、ROC/AUC、range MAE/RMSE、圆周 Doppler MAE/RMSE/bias、true-range-gated Doppler、predicted-range mismatch、joint、C0–C3 与 SNR/CNR 条件、参数量、训练时长、CUDA 峰值显存、输入到输出成本。
- 物理一致性对照复用 E003 冻结的 256 个分层 H0/H1 diagnostic IDs、32 个公共相位旋转及 E003 分开的整体 IQ 频移/目标分量反事实/重新生成目标几何干预；按 S0/S2 checkpoint 逐个报告 presence/range/圆周 Doppler 变化。由于这些样本已参与先前诊断，只作开发诊断，不作为新盲测。
- 阈值从每个模型独立 H0 calibration 的第 95 百分位 `higher` 得到，固定 `score >= threshold`；不在 verification/development 调阈值。
- 以 `base_scene_id` 为单位 2,000 次 paired bootstrap，重采样内重算 RMSE。CI 只表达场景不确定性；单 seed 不估计训练 seed 方差。
- 基础学习 G1：每臂 finite forward/backward，非退化梯度，小样本过拟合测试通过；独立 development AUC 的 95% 场景 bootstrap 下界>0.5，且 range/Doppler RMSE 优于 train-truth 常值预测。
- **小样本工程过拟合测试：** 从 train 固定抽取 16 个配对 scenes（32 CPI），重复同一小 batch 训练 200 updates；排除于科研指标。每个模型要求 loss 有限且至少下降 50%，presence 在该批准确率为 100%、H1 range-cell 准确率≥95%，range/Doppler 任务头梯度均有限且非零。未通过则停正式预算，先定位实现/优化问题；不得改科学数据预算来绕过失败。
- 任务保护继承 E003 pilot 屏幕规则作比较：Pd/joint S2−S0 点差≥−0.01 且 95% CI 下界>−0.05；C2/C3 joint 差 CI 下界>−0.05；C2/C3 range/Doppler RMSE 比点估计≤1.05 且 CI 上界≤1.10；C3 两臂 Pfa≤0.10，且差值 CI 上界≤+0.02。该门仅作 pilot screening，不是正式非劣效声明。
- 有意义收益：候选主要端点的增益点估计≥0.02 且场景 bootstrap 95% CI 下界>0，其他主要任务不触发保护失败。无收益就停止扩大该候选；不得以某一辅助 loss 下降替代任务收益。

## 7. 训练/CUDA/资源约束

仅使用 RTX 3050 Laptop `cuda:0`，先 FP32，不用 AMP。启动前记录 driver、PyTorch/CUDA、GPU 显存、`cuda:0` 可见性和磁盘余量；不可见/不匹配则不开始训练且禁止 CPU fallback。模型顺序驻留。microbatch 从 16 开始，OOM 依次降至 8、4，梯度累积保持 effective batch 64 和 updates；microbatch 4 仍 OOM 则停止。训练数据用 CPU memmap/按 batch 拷入，不将全量样本缓存到 GPU。

每个 checkpoint 保存模型、optimizer、scheduler、Python/NumPy/Torch/CUDA RNG、global update、epoch、已消费 pair indices、config 和 source SHA。训练后记录峰值 allocated/reserved、耗时和实际 microbatch。

## 8. 假设与阶段门

- **H-CAP-001：** 1-epoch E003 结果受到训练预算不足影响。R5-A 曲线/任务指标可支持或反驳（限本数据和候选）。
- **H-CAP-002：** width=32 的 TaskStudent 在某些任务/场景存在容量限制；Medium 增宽可在任务保护下改善指标。
- **H-ARCH-003：** 改变学习归纳偏置/学生架构可改善复数慢时间 GMTI 任务；后续 TF attention pilot 检验，不由单个架构否定整个命题。
- **G0：** protocol/config、split/seed、设备、CUDA 与磁盘 preflight 通过，冻结 E003 文件无变更。
- **G1：** 数据/标签/损失、梯度、过拟合测试、输出与 CUDA 路径正确。
- **G2：** 能区分改善、平台、过拟合、优化/表示限制；如曲线明显改善，不得宣称容量上限。
- **G3：** 新宽度/架构只有在一个预注册任务指标有意义改善且主要任务无明显退化时才保留，并报告成本。
- **G7：** 正式三 seed 需 pilot 任务保护与收益过门，并单独冻结协议；本协议不生成 final blind。

## 9. 复现与交付

代码放 `src/radardistill/` 和 `scripts/`；R5-A 指标、训练记录、图和失败日志放 `experiments/D2-A-R5/R5-A/`。输出含 config、seed/data manifest、environment、resource、checkpoint 索引、每 400-update history、per-scene evaluation、analysis。可重建缓存不进 Git；阶段后只按本阶段 manifest 清理。命令、提交 SHA 和测试覆盖写入分析及审查材料。
