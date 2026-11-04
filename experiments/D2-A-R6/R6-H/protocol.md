# D2-A-R6-H：复数对角 SSM 与局部卷积混合架构筛选

**协议版本：** 0.1 draft（未冻结）
**状态：** 等待 R6-G 安全完成和全仓 scene ID 审计；不得生成正式数据或启动训练。
**研究总纲：** `docs/RESEARCH_CHARTER.md` v1.3。
**路线位置：** R6 P4 的 SSM/混合架构覆盖补充。
**训练监督：** E1 物理真值；本阶段不做知识蒸馏。
**最终盲测：** 未授权生成。

## 1. 研究问题与边界

R6-E 已比较普通任务卷积基线、公共相位等变 CE-TCN 和 RAX 轴向 Transformer；RAX 当前实现明显退化，但不能据此否定所有 Transformer 或序列架构。SSM 尚未在本项目 GMTI 输入上实测。本阶段检验：在相同原始三通道复 IQ、任务头、数据和更新预算下，复数对角 SSM 及局部复数卷积加 SSM 是否能学到有用的慢时间频率信息，并改善 C2/C3 检测与连续参数任务。

本阶段候选是**固定、可学习的复数对角状态核**，不是完整的 selective Mamba：状态核在时间上固定，输入依赖的 Δ/B/C 或 selective scan 均未实现。用零填充 FFT 计算有限长可学习线性卷积只是 SSM 数值实现，不是固定雷达频谱前端。模型不保证 Doppler 频移等变；相位等变性来自复线性、幅度门控和模长任务读出，仍由测试验证。负结果限定于此参数化与预算。

## 2. 假设与对照模型

- **H-ARCH-SSM-001：** 复数对角 SSM 学生在可辨识 C2/C3 上达到或超过 CE-TCN 的任务表现，且不损害 Pd、距离和 Doppler 保护指标。
- **H-ARCH-SSM-002：** 在同一 SSM 主体前增加局部复数时间卷积，比纯 SSM 学生取得额外任务收益。
- **停止解释：** 若正确性门通过后，SSM 候选在固定筛选预算下仍明显落后或未学会频率映射，保留负结果并停止当前参数化；不外推为所有 SSM/Mamba 无效。

| 臂 | 模型 | 物理/学习前端贡献 | 默认推理参数量 |
|---|---|---|---:|
| C | R6-E CE-TCN，重新初始化并在新数据上重训 | 无固定物理前端；复数卷积核可学习，具有公共相位等变结构 | 63,429 |
| S | 复数双向对角 SSM | 无固定物理频谱层；可学习稳定复极点状态核、复数通道映射和幅度门控 | 65,077 |
| CS | 复数局部卷积 + 双向对角 SSM | 无固定物理频谱层；一个局部时间卷积块与 SSM 均可学习 | 71,909 |

上述参数量来自当前独立实现；正式 G0 必须从冻结提交重新计数，三臂候选均须落在 CE-TCN 参数量 ±20% 范围内。三臂均从头训练，不加载 R6-E/R6-G checkpoint；相同随机种子只控制可比初始化规则，不声称异构权重逐元素相同。

## 3. 数据与划分

保持 D2-A-E001 的三通道静止杂波与单目标复 AWGN 模型：PRF 1300 Hz、130 脉冲、32 距离单元和 R6-A 的 12 个条件格。训练/验证使用配对 H0/H1 场景；独立评价使用 H0 calibration、H0 verification 和 H1 development。模型仅接收复 IQ 的六平面实数表示；真值只进入 E1 loss 和离线评价。

| Split | 场景数 | 用途 |
|---|---:|---|
| Train | 51,200 对 | 三臂共同 truth-only 训练 |
| Validation | 5,000 对 | 训练中固定间隔验证、checkpoint 选择 |
| H0 calibration | 120,000 | 每臂独立设置总体 `Pfa_CPI=0.05` 阈值 |
| H0 verification | 120,000 | 独立阈值迁移与条件 Pfa 评价 |
| H1 development | 6,144 | 一次开发评价；不作 final blind |

候选 seed block 暂定采用 R6-G 已登记区间之后的独立范围：train `[2608000000,2613000000,2618000000,2623000000]`，validation `[2628000000,2633000000,2638000000,2643000000]`，H0 calibration `2708000000`，H0 verification `2728000000`，H1 development `2748000000`，G1 `2798000000`。**这些 seed 仍是草案**；正式冻结前须扫描所有已提交与运行阶段 manifests，确认与历史、R6-G 和各 split 均无 scene ID 重叠，否则先更换整个 seed block 再冻结。

训练/验证缓存预计约 11.23 GB，只在 R6-H owner 清单下临时保存；评价 IQ 流式生成且不持久化。数据生成前需有缓存空间加 4 GiB 余量。阶段结束仅删除通过 R6-H owner manifest 识别的可重建缓存；不触碰用户未跟踪数据或其他阶段缓存。最终仓库目录仍须低于 20 GiB。

## 4. 训练预算与公平性

- 三臂采用 E1 truth-only，同一配对 train/validation scene ID、条件顺序、H0/H1 比例、loss、优化器、checkpoint 规则和更新预算。
- 首轮筛选固定为 **25,600 optimizer updates/臂**，effective batch 64（每 update 32 H0 + 32 H1），FP32，AdamW，初始学习率 `1e-3`、最终 `1e-5`、单 cosine、weight decay `1e-4`、gradient clip `1.0`。本轮固定 batch64以保持与 R6-E 可解释的训练条件；新 batch128 比较应作为独立匹配实验，不与本架构效应混在一起。
- 初始 microbatch 4；OOM 时按 `[2,1]` 递减，并用梯度累积维持 effective batch64、updates 和样本暴露。禁止 AMP 与静默 CPU 回退；RTX 3050 Laptop `cuda:0` 不可用则停止。
- 每 400 updates 写训练/验证 history，每 800 updates 保存可恢复 checkpoint；不逐 update 落盘。三臂按 C→S→CS 顺序训练，每次仅一模型驻留 GPU。
- 25,600 是筛选预算，不是收敛声明。若末段验证仍持续改善，报告“预算内未观察到平台”，随后依据已保存曲线另行冻结 matched 延长协议；不得只追训表现较差的单臂，也不因一次 H1 development 结果临时延长。
- 预计每个 H0/H1 presence 暴露量为 `25,600×32=819,200` CPI/臂；对 51,200 个唯一 train scene，每个 presence 平均重访 16 次。必须分别报告唯一场景数、曝光次数、训练集遍历次数和更新数。

## 5. 正确性门与正式执行门

**G0（冻结与资源）：** 本 protocol、config、代码和测试已提交并推送 `main`；当前 tracked worktree clean；全仓 scene ID 审计零重叠；磁盘余量满足缓存要求；实际 GPU 为 RTX 3050 Laptop 且明确使用 `cuda:0`；记录 Python/PyTorch/CUDA/driver、Git SHA、GPU、空闲显存和磁盘。参数量满足预设范围。

**G1（实现学习能力）：** 三臂输出 shape、有限输出、有限非零梯度、任务 loss 与 sampler contract 通过；S/CS 的 FFT 线性卷积对照、CPI 首尾、range 轴隔离、complex64/complex128 和公共相位测试通过；各臂在相同 16 对小样本上 200 updates 的任务 loss 至少下降 10%。G1 失败时停止 formal data generation 和训练，先排除实现错误。

R6-G 完成前不运行上述 CUDA preflight、G1 或正式数据生成，以免与当前阶段争用设备或读取正在写入的 scene manifests。R6-G 结果只用于路线衔接，不会改写本阶段已冻结的物理条件或任务定义。

## 6. 指标与预注册判定

每臂仅从自己的 H0 calibration scores 按 `higher` 经验 95th quantile 设全局阈值，比较符 `score >= threshold`；独立 H0 verification 验证迁移，H1 development 做一次性任务评价。主要端点为可辨识 C2+C3 的 paired joint success；S−C 和 CS−C 两项比较采用双侧 97.5% scene-paired bootstrap 区间（控制两项比较的 familywise 95%）。候选筛选门为 joint 差点估计至少 `+0.02` 且区间下界大于0。

任务保护报告 overall Pd 差、全 H1 range RMSE ratio、PRF-wrapped Doppler RMSE ratio，保护界分别为 Pd 差区间下界 `>= -0.02`、两类 RMSE ratio 上界 `<=1.05`。检测与低虚警资格分开判定：H0 verification overall 和 12 个条件逐格报告单侧 exact binomial CI；C3 六格采用 Bonferroni familywise exact upper bound，六格均 `<=0.10` 才通过 C3 保护门。

报告 AUC、Pd/Pfa、条件分层 Range/Doppler RMSE 与 bias、correct-detection 参数误差、correct-candidate 参数误差、漏检计入的 joint success，以及模型参数、训练时间、训练样本暴露、峰值显存、训练环境。物理诊断包括 16 个公共相位旋转和 `±130 Hz` 受控慢时间频移；频移响应是诊断而非预设保证。可复用 `d2a_r6e` 的流式预测、阈值统计、paired bootstrap 与物理诊断函数，避免重复实现评价定义。

该阶段不研究 `10^-3` 或更低 Pfa，不作低虚警资格、跨训练 seed 泛化、教师增益、实时性或 final blind 声明。只有 truth-only 架构筛选显示可靠任务学习后，才可依 R6-G 的实际结果决定是否为入选架构设计匹配教师实验。

## 7. 当前执行状态

本文件是可审阅草案，不是已冻结 protocol；配置状态为 draft。已完成模型骨架、三臂工厂、split-plan静态检查和 CPU 单元测试；新模型与 R6-E 架构测试及协议测试合计21项通过。暂定 R6-H scene IDs 与 R6-E/R6-G 配置中登记的 seed plans 无重叠；全仓历史 manifests 的正式审计仍待 G0。尚无 R6-H 数据、CUDA preflight、G1、训练、指标或阶段结论。下一步在 R6-G 安全结束后执行 scene ID/资源审计，再将协议、配置与 runner 一并冻结和推送，然后按 G0→G1→数据生成→顺序训练→独立评价执行。
