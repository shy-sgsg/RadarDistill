# D2-A-R5-C 匹配教学内容 pilot 设计

**状态：** 设计草案；R5-C 独立 ChatGPT 阶段审查返回后再冻结实验协议。
**总纲：** `docs/RESEARCH_CHARTER.md` v1.3；本设计不修改总纲。
**范围：** 仅设计一次 T0–T3 教学内容筛选 pilot；不包含教学时机、可靠性权重、正式多 seed、final blind 或实时性。

## 意图

R5-C 已证明 K1/K3/K2-local 软响应在可辨识 C2+C3 中具有可测任务相关信息，但尚未证明这种信息能进入学生最终检测与参数估计。pilot 用匹配对照回答：同一 Medium 学生在保留物理真值监督时，完整 K3 或局部 K2-local 响应是否带来超出 truth-only 的任务增益；匹配的 truth-assisted 局部监督用于判断收益是否只是额外辅助头或 oracle 局部监督。

## 目标与边界

- 对齐总纲 Q1/Q2/Q4/Q6/Q9/Q10，服务最终 GMTI 学生任务，不另建检测统计量研究。
- 保持三通道、PRF=1300 Hz、N=130、32 距离单元、静止杂波与 D2-A-E001 物理生成模型。
- 只选 R5-A Medium width=96 CNN 作为首轮学生。R5-B TFAS 按 G3 STOP；Medium 的 C3 H0 Pfa=`0.18896` 仍是已知绝对保护失败。
- 物理真值始终监督 presence/range/Doppler。教师硬标签、软响应、truth-assisted 局部分布和学生输出分开；学生推理不调用教师。
- 本 pilot 是单训练 seed 的开发筛选，不估计 seed 方差，不构成 detector qualification 或 blind。

## 考虑过的方案

1. **保留现有 130-bin 辅助头并把 K3 软图降采样。** 改动最少，但把 2.5 Hz 子栅格响应压回 10 Hz 网格，无法检验 R5-C K3 的核心增量。
2. **保留 Medium 任务干路，只将训练期辅助响应头扩展到 520 Doppler bins（选用）。** 主干与 presence/range/continuous-Doppler 任务头不变；四臂使用同一响应头、参数初始化和训练预算。该设计能表达 K3 的固定 2.5 Hz 网格，响应头在推理导出时移除；额外训练参数与耗时如实记录。
3. **只蒸馏一个教师 Doppler 点估计。** 目标维度小，但无法比较完整 RD 与局部频率软响应，且会把教师峰值错误压成硬参数标签，因此不适合本轮问题。

## 选定的四臂与损失

- **共同模型：** `TaskStudent(width=96)` 原有任务干路、presence/range/range-offset/circular-Doppler heads 保持不变；仅训练期 spectrum head 输出 520 bins。根据当前实现估算训练模型约 120,717 参数（需 preflight 实测），移除辅助头后的任务模型约 70,277 参数；不得把训练期辅助头计入部署学生。
- **T0：** 物理真值任务损失，辅助响应头实例化但不计算、不使用其输出。
- **T1：** 物理真值任务损失 + 完整 K3 `[520,32]` 归一化相对功率支持的全图 KL。
- **T2：** 物理真值任务损失 + K2-local 局部频谱 KL。教师仅以 H1 IQ 运行 K3；最大响应距离门由 IQ-only 全图 argmax 确定，学生只在该门上作 520-bin 条件分布匹配，不将此输出说成独立 range estimator。
- **T3：** 物理真值任务损失 + truth-assisted 局部频谱 KL。使用真值 range cell（`floor(truth_range)`）及 N=130、PRF=1300 Hz 矩形相干积累的解析 Dirichlet 功率响应，在同一 520-bin Doppler 轴 `f_k=-650+2.5k Hz, k=0…519` 上计算 `q_k ∝ |Σ_{n=0}^{129} exp(2πi(f_truth−f_k)n/1300)|²`，再归一化。有限和本身按 PRF 周期重复，因此自然处理 Nyquist 边界；该核不调温度或窗宽。T3 是 oracle 辅助监督控制，不是教师输出。
- 辅助损失只在 H1 行计算；H0 只进入 truth presence loss。T1/T2 的 teacher input 仅 IQ 与冻结配置；truth 不传入教师。所有辅助臂使用同一固定 `lambda=0.05`（沿用既有 K2 的初始权重，不搜索温度/权重），训练期间保持常数，以隔离知识内容而不混入教学时机研究。

### 比较器与门限映射

- T1/T2 的教师收益比较均以新训练的匹配 **T0** 为参照；R5-B 相对历史 Medium 的差值不拿来充当本 pilot 的对照。
- 主收益仅有两项：T1−T0、T2−T0 的可辨识 C2+C3 joint success。每项点估计须至少 `+0.02`；为控制两个主对照的 familywise error，各对照使用 paired-scene bootstrap 的单侧 97.5% percentile lower bound（2,000 次重抽样，Bonferroni alpha=`0.05/2`），该下界须 `>0`。另报告每项名义 95% bootstrap CI。T3 对比仅作次要、描述性结果。
- 保护门也均相对 T0：overall Pd 与 overall joint 差的点估计 `≥−0.01` 且名义 95% CI 下界 `>−0.05`；C2+C3 joint 差名义 95% CI 下界 `>−0.05`；C2+C3 range 与 Doppler RMSE ratio 点估计各 `≤1.05`、名义 95% CI 上界各 `≤1.10`；C3 H0 Pfa 差名义 95% CI 上界 `≤+0.02`。每个候选臂自身的 C3 H0 Pfa 点估计还须 `≤0.10` 才能晋级。Bonferroni 同时下界只用于两个主要收益对照；保护区间为 pilot screening，不作正式非劣效声明。

## 数据、优化和公平性

- 训练只复用 R5-A 的 51,200 个固定 paired scenes 和冻结 `training_pair_plan.npz`（seed base `2026102711`）；四臂都从头训练，不接续 R5-A checkpoint。模型初始化 seed `303`，四臂 state dict bit-identical；场景多重集、顺序和每个 update 的 32 个 H0/H1 pair 完全相同。
- 每臂 12,800 optimizer updates（固定训练 cohort 八次遍历），每 update 有效 batch=64 CPI、gradient accumulation；FP32、AdamW 和 R5-A 两段 cosine schedule（1e-3→1e-4，再 1e-4→1e-5）保持一致。microbatch 从 16 开始，OOM 时按 8、4 递降并保持有效 batch；microbatch 4 仍 OOM 则停止。
- 使用 R5-A 固定 5,000 validation IDs/labels 只作共同 validation truth-task-loss checkpoint 选择；该数据已用于开发，不作为 blind。训练 K3/K2-local maps 每 update 对 H1 batch 在线经 RTX 3050 Laptop `cuda:0` / complex128 教师计算，不落盘、不缓存；四模型顺序驻留 GPU。
- 使用新场景 seed blocks，`chunk_size=16`、`chunk_seed_stride=1009`、`condition_stride=1,000,000`；split bases 为 calibration `728,000,000`、overall verification `748,000,000`、独立 C3 verification `768,000,000`、H1 development `788,000,000`。每个 split 内按上文 condition 列表从 index 0 固定编号，再分配 `split_base + condition_index×condition_stride + chunk_index×1009`；C3-only verification 只用其六个条件，index 0–5。任意冲突即停止，不自动换 seed；无需随机 condition assignment。
- H0 calibration 与 overall verification 各 `8,192` scenes，均按四个 course 各 2,048 等权分层：C0 separated 20/−5 为 2,048；C1 separated 与 near-edge 15/0 各 1,024；C2 三个场景各 683、683、682；C3 六个场景/SNR cells 各 342、342、341、341、341、341。独立 C3 H0 verification 共 `4,096`，六个 C3 cells 依固定顺序分配 `683,683,683,683,682,682`。H1 development 共 `4,352`，沿用 R5-C audit 的 cell allocation（C0 256；C1 各128；C2各256；C3各512），但全部 scene seeds 全新。calibration/verification/dev ID 互斥；每臂分别用 calibration 的 higher empirical quantile 设 `Pfa_CPI=0.05`、判决 `score>=threshold`。overall Pfa 只解释为本 protocol 定义的等 course benchmark mixture，不外推为 operational prevalence。bootstrap seed 固定 `2026102828`、按 `base_scene_id` 重抽样 2,000 次。
- 当前规模预计远低于最终 20 GiB 上限；不生成完整 IQ / teacher-map 文件，只保留 split manifest、训练 history、模型 checkpoint、逐场景预测和指标。

## 指标、统计和阶段门

- **主要端点：** 可辨识 C2+C3 joint success：固定 calibration threshold 下检测正确、绝对 range error≤1 cell、PRF 周期 Doppler error≤20 Hz；漏检记 joint failure。
- 报告 Pd/AUC、C0–C3 与全部 6 个 C3 子组 Pfa、range/Doppler MAE/RMSE/bias、true-range-gated Doppler、range mismatch、任务曲线、aux loss、参数量、updates/数据遍历、训练时间、峰值显存及 OOM 降 microbatch 记录。
- 配对 bootstrap 按 `base_scene_id` 重抽样 2,000 次；T1−T0 和 T2−T0 使用 Bonferroni 同时单侧下界（每个下界取 bootstrap 分布的 2.5 百分位），名义 95% CI 另行报告。区间只表示场景抽样，不表示训练 seed 方差。
- **G0：** 代码/协议/data split/seed 冲突预检正确，CUDA `cuda:0` 可见，磁盘与 VRAM 足够，禁止 CPU fallback。
- **G1：** 四臂 forward/backward、损失和梯度有限；16 对固定小样本 200 updates 的过拟合检查成功；教师输入隔离与 K3/K2-local/T3 target 的 Nyquist 周期边界测试通过。
- **G2/G5 pilot 筛选：** T1/T2 任一臂相对 T0 的 C2+C3 joint gain 至少 `+0.02` 且其 Bonferroni 同时单侧 97.5% bootstrap lower bound 大于 0，此外 overall Pd/joint、C2/C3 RMSE 及 Pfa 保护均不触发既有 R5-B 任务退化门；每臂 C3 H0 Pfa 点估计须 `≤0.10`，同时报告 exact 单侧 95% 上限。
- 任何 arm 都不因 teacher/KD loss 下降而过门。若 T0–T3 都未通过 G1，停止并定位实现/优化问题；若基础学习通过但 T1/T2 未过收益或保护门，停止对应教师内容，不启动教学时机、可靠性权重、正式多 seed 或 blind。
- 只有一个预注册教师臂满足收益与保护门，才另行冻结至少 3 个独立训练 seed 的确认协议。该筛选数据不称最终 blind。

## 协议草案中的操作化落实

精确 split 条件计数、seed block/chunk 公式、两段 scheduler 边界、bootstrap seed 和当前执行禁令已记录在 `experiments/D2-A-R5/R5-C-teaching/protocol.md` 与 `config.yaml`。它们仍是草案，等待 R5-C 独立 ChatGPT 审查；逐场景输出 schema 将随评估器实现固定，不能据此提前生成数据或训练。
