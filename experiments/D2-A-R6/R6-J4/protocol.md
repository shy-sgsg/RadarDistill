# D2-A-R6-J4：固定物理前端与联合任务学生 pilot

**协议版本：** 1.0
**状态：** 已预注册；G0/G1、场景生成和科学训练尚未开始。
**研究总纲：** `docs/RESEARCH_CHARTER.md` v1.3。
**最终盲测：** 未授权。
**依赖：** R6-G、R6-H、R6-I 均已结束；本阶段是新的独立 J4 候选比较，不复用 R6-I 校正式或开发阈值。

本协议检验固定几何 DPCA 与加密频率网格能否给联合检测/参数估计学生提供有用的物理表示，以及学习式读出是否比同一物理频谱上的固定峰值更有效。传统 B3 和 R6-D J1-J 为任务参照。R6-I 的 scale correction 未通过门，不据此调整 J4 分数或阈值。

## 1. 研究问题与可证伪假设

1. 固定、可检查的几何 DPCA + 520 点慢时间频谱表示，能否支持可靠的 null/目标候选选择及连续距离、周期 Doppler 估计？
2. 在同一训练场景与预算下，J4 学习读出相对原始复 IQ 联合学生 J1-J 是否提升可辨识 C2+C3 joint success，并保护 Pd、距离误差、Doppler 误差和 C3 虚警？
3. J4 相对固定 520 点物理峰值的差异能否显示学习读出的作用；520 与 130 点固定峰值的差异能否显示零填充网格密度的作用？

假设 `H-R6J4-001` 为：固定几何 DPCA + 520 点候选表示上的学习读出，在本冻结模拟条件下可比 matched raw-IQ J1-J 提高 C2+C3 joint success，且不越过主要任务和 C3 虚警保护门。它尚无本阶段支持证据。单个 seed 只作候选筛选；失败仅约束本候选与协议。

## 2. 冻结场景和数据划分

保持 D2-A-E001 单目标三通道静止杂波复 AWGN 仿真：PRF=1300 Hz、130 脉冲、32 距离单元及 R6-A 的 12 个条件。条件列表逐项复制自 `experiments/D2-A-R6/R6-A/config.yaml`，训练和验证使用配对 H0/H1 场景；同一 scene ID 的两种假设共享背景与噪声，目标仅在 H1 存在。所有学习臂读取完全相同的 IQ、条件分配和场景顺序。真值只进入训练目标和离线评价。

| Split | 场景数 | 用途 |
|---|---:|---|
| Train | 51,200 paired base scenes（各含 H0/H1） | 两个学习臂使用同一场景多重集和顺序 |
| Validation | 5,000 paired base scenes | 仅按各自冻结的 truth objective 选 checkpoint |
| H0 calibration | 120,000 CPI；每条件 10,000 | 各方法各自估计一个全局 5% CPI 阈值 |
| H0 verification | 120,000 CPI；每条件 10,000 | 独立虚警率与阈值迁移评价 |
| H1 development | 6,144 CPI；每条件 512 | 单次开发评价；绝不视为盲测 |
| G1 micro-set | 16 paired scenes | 物理/梯度/过拟合工程门；不进入科学结果 |

新 seed 基数见 `config.yaml`。在冻结前扫描了 31 个既有 scene manifest、共 1,220,168 个带 seed 记录；观测到的最大历史 seed 为 3,290,001,009。所有 J4 基数均大于 3,400,000,000。此静态核对不替代 G0：G0 必须重建实际训练、验证、校准、验证 H0、H1 development 及 G1 ID，扫描全仓历史 manifest 和所有先前已注册计划，证明 split 内和 split 间均无重复/交叠后，才可生成科学数据。

Train/Validation 暂存 IQ 使用 `[real, imag]` float32 平面，估算空间 11,222,016,000 bytes。G0 要求可用磁盘至少为 15,516,983,296 bytes；不足则停止，不缩小科学 split。正式 evaluation 按冻结 ID 分块生成，不持久化全量 IQ。临时数据只能由 R6-J4 owner manifest 清理；检查点、训练曲线、场景 manifest、预测、metrics、协议和审计永久保留。阶段结束后项目目录（含 `.git`）须低于 20 GiB。

## 3. 模型与基线

| 名称 | 输入/处理 | 作用和解释范围 |
|---|---|---|
| J1-J | 三通道原始复 IQ，以实部/虚部平面输入 `R6J1Student(width=96)` | 当前 raw-IQ 联合学习基线 |
| J4 | 固定几何通道 (0,1) 分数慢时间移位与 DPCA 差分；慢时间 520 点零填充 FFT；`log1p(|Z|²)`；width=48 学习式 2-D 候选/null 与连续参数、bounded feedback | 测试固定物理表示上的联合学习读出；前端不训练，不使用条件标签、SNR/CNR 或真值 |
| 固定 DPCA+130 峰值 | 与 J4 相同几何差分，130 点 FFT，最大功率候选 | 原生频率网格的非学习物理参考 |
| 固定 DPCA+520 峰值 | 与 J4 相同几何差分，520 点零填充 FFT，最大功率候选 | J4 物理前端本身的非学习读出参照 |
| B3 | 冻结 R6-A B3 detector | 检测分数/Pd/Pfa/AUC 参照；无连续参数输出资格 |

固定峰值方法的 range 输出为最大单元；Doppler 在峰值的周期相邻三点 log-power 上做冻结抛物线插值，顶点位移裁剪到 `[-0.5,0.5]` bin，并按 PRF 周期回绕。平坦/非有限邻点时位移置零。130/520 两方法使用完全相同的插值公式。J4 相对固定 520 峰值的收益说明“学习读出在固定前端上的增量”；固定 520 相对固定 130 说明加密网格的增量；J4 相对 J1-J 是完整表示和学习系统比较，不能单独归因为 DPCA 因果效应。

J4 的 `Z[k,r]` 按注册 FFT shift 顺序排列，Doppler 轴周期长度为 1300 Hz。520 点零填充仅加密有限 CPI 响应采样，不增加独立观测信息。前端分数时移和通道差分按 E001 冻结几何配置计算；不从数据学习系数。学习器显式输出 null 与 520×32 目标候选，参数头输出有界 range 偏移和 PRF 回绕 Doppler 残差，参数候选后验与 bounded feedback 保持可微。

J1-J 与 J4 参数量不同，所有实际参数量必须记录。两臂训练场景、数据顺序、H0/H1 配额、optimizer updates、effective batch、优化器、精度、验证和 checkpoint 规则相同；比较的是两个完整候选系统，不做参数量匹配或结构因素的单因果声明。

## 4. 训练目标和预算

只做物理真值监督，不训练教师、不用硬标签或软响应、不蒸馏。每 update 抽取 32 个 H0 与 32 个 H1 CPI，有效 batch=64；H0 只计算 presence 损失，H0 的 range/Doppler/candidate 参数损失必须精确为零。H1 计算 presence BCE、注册 target candidate CE、range smooth-L1（beta=0.1）和 PRF 周期 Doppler loss；主损失权重均为 1。J1-J 使用对应的 R6-D joint truth-only objective。J4 candidate 位置由连续真值映射到冻结 520×32 格点，连续参数项根据 non-null 候选后验池化；禁止 H0 伪造目标位置监督。

每臂固定 25,600 optimizer updates、seed=909、AdamW、`LR 1e-3→1e-5` 单段 cosine、weight decay `1e-4`、clip norm `1.0`、FP32。有效 batch 固定 64；microbatch 从 16 开始，OOM 时仅可按 8、4 递减并用梯度累积保持有效 batch 和 update 数，microbatch=4 仍 OOM 则停止对应臂。两臂按 J1-J、J4 顺序依次驻留 CUDA，不并发。训练曲线每 400 updates 记录；恢复 checkpoint 每 800 updates；不逐 update 写日志。

各臂仅以自己的 validation truth objective 选择 checkpoint；H0 verification/H1 development 不能用于选模型、改阈值、改损失或延长训练。25,600 是冻结比较预算，不是模型学习极限；末段若仍改善，记录“预算内未观察到平台”，不单独追训 J4 或 J1。

## 5. G0/G1 前置门

**G0（冻结与数据/资源）**：须在本协议及实现已提交并推送到 `main` 的 HEAD 上运行；检查 protocol/config/source 提交绑定、干净 tracked tree、实际 scene-ID 不相交、owner/cache 估算、磁盘阈值、AISAR 环境及 NVIDIA RTX 3050 Laptop `cuda:0`、至少 1.5 GiB 可用显存和 FP32。若 GPU 不可见、环境错误、场景 ID 重叠或磁盘不足，停止，不静默回退 CPU。

**G1（实现正确性）**：

1. 固定 CUDA DPCA/520 FFT 与 complex128 NumPy reference 在预注册容限 `rtol=1e-4, atol=1e-6` 内一致；注入周期单音定位、频点顺序、wrap 和距离轴正确。
2. 公共复相位旋转不改变 J4 log-power、presence、候选和估计；记录复数中间响应只按公共相位旋转。`X→aX` 诊断原始物理峰值分数随 `|a|²` 变化，J4 不把它隐式归一化为尺度不变。
3. 检查连续 Doppler 周期边界、range 候选、null/H0 参数 loss 精确为零、候选概率归一化、有限非零双向任务梯度及输出有限。
4. J1-J/J4 同一 16-pair 数据各自过拟合 200 updates；各臂冻结主损失最后 20 步均值相对第 1 步至少下降 10%。这是工程门，不作为模型能力结论。
5. 必须在 AISAR 中实测 `torch.cuda.is_available()`、GPU 型号/设备和训练前后峰值显存；任何隐藏 CPU 张量若导致 CPU 执行则视为失败。

G0 和 G1 全部通过后才能生成正式训练/验证缓存和启动全预算训练。前置失败只修正能定位的实现问题；不能靠复杂损失掩盖固定物理算子或基础学习失败。

## 6. 评价、门槛和停止规则

对 B3、固定 130、固定 520、J1-J、J4 分别用独立 H0 calibration 的 `higher` 95% 分位数设置一个全局阈值，判定为 `score >= threshold`。H0 verification 和 H1 development 只评价。报告整体及 12 条件的 Pfa、Pd、AUC、range/Doppler bias 与 RMSE、门错、all-H1 与 detected-only 参数误差、joint success；分别报告 C2+C3、C3 和 6 个 C3 condition。D-only 按不可辨识性约束解释，不要求恢复唯一来源标签。

主要 endpoint：J4−J1-J 的 C2+C3 paired joint-success difference，按 `base_scene_id` 分层配对 bootstrap 2,000 次、95% CI；候选筛选收益要求点估计至少 `+0.02` 且 CI 下界 `>0`。保护门为：overall Pd 差值 CI 下界 `≥−0.02`；all-H1 range RMSE ratio 和 PRF-wrapped Doppler RMSE ratio 的 CI 上界各 `≤1.05`；C3 六格 Bonferroni exact 单侧 Pfa 上界各 `≤0.10`。此外报告 J4−固定520峰值和固定520−固定130峰值；这两项不替代主端点。

任何保护门未通过，均不得称为可靠的低虚警 GMTI 检测器或申请 formal seeds/final blind。J4 G1 失败则停止科学训练；J4 完成预算后相对 J1-J 和固定峰值均无有价值增益，记录该候选负结果并停止此参数化，不扩展无界架构搜索。开发结果不能用于回调阈值、协议门或继续训练。

## 7. 资源、复现和解释边界

AISAR conda、RTX 3050 Laptop `cuda:0`、FP32、禁止 CPU fallback；J1-J 与 J4 顺序训练。每个 checkpoint 保存模型、optimizer/scheduler/update、seed、sampler、history 游标、protocol/config/source SHA。history 不逐步刷盘。场景 ID、cache owner、训练暴露、参数量、实际显存、墙钟和吞吐须记录。

J4 只测试当前冻结的三通道静止杂波仿真及固定几何 DPCA 假设。没有教师蒸馏、训练 seed 稳定性、blind 泛化、真实数据资格、实时时延或系统时延结论。单一 J4 参数化失败不否定所有物理驱动网络。
