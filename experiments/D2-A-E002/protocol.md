# D2-A-E002：课程学习与教师辅助任务级学生 Pilot

**英文：** *Curriculum Learning and Teacher-Guided Task Learning for Multichannel GMTI Detection and Parameter Estimation*
**协议版本：** 2.1
**冻结状态：** 2026-10-25；先于 E002 数据生成、K2 质量审查和训练。
**执行授权：** 用户在本任务附件中明确批准调整动态路线并实施本轮 A/B/C 有界 pilot。
**研究总纲：** `docs/RESEARCH_CHARTER.md` v1.1；本协议不修改总纲。
**旧版本：** Git 提交 `1e4300c` 中的 v1 truth-only 提案已被本版本取代；原协议审查作为历史记录保留。它未产生数据、训练或结果。

v2.1 在 v2.0 基础上量化 G1 中已要求的 micro-overfit/梯度工程检查，并固定 G6 开发诊断子集为每个课程层按缓存顺序取前 64 对、合计 256 对；A/B/C 处理、正式模型、数据划分、训练预算和科学验收门不变。本修订先于任何 E002 数据生成。

## 1. 研究目的与可证伪假设

本轮以任务级检测、距离和连续 Doppler 为终点，同时比较学生架构的既有适用性、任务难度顺序及传统软响应辅助作用。它直接服务于研究总纲 Q1–Q4、Q6：物理真值与传统知识能否共同帮助学生从多通道复数 IQ 学习最终任务。R3 的固定阈值 G2 失败仍然成立；该门评价固定阈值教师资格，不等同于软响应质量。本实验不授予 T1 教师资格，也不把教师响应当作物理真值。

- **H-LEARN-001（难度/课程）：** 在相同训练样本多重集、样本呈现次数、模型初始化和更新预算下，由易到难再综合的 C0→C1→C2→C3→C4 顺序，相对一开始就随机混合全难度分布，可能改善优化过程或最终任务指标。
- **H-KD-D2-001（教师辅助）：** 在与 B 完全相同的课程和样本顺序上，加入始终由物理真值监督的 K2 软响应损失，可能进一步改善收敛或最终任务。
- **H-ARCH-D2-001（架构）：** 当前普通实值 CNN/TCN 的跨通道相干归纳偏置可能不足；A/B/C 不直接比较架构，此假设只由失败诊断形成下一步依据，不视作本轮已检验。
- **H-OPT-D2-001（多任务优化）：** range、Doppler 与 detection 的优化难度可能不同；训练在真值距离门上监督 Doppler、推理在预测距离门读取 Doppler 的不一致可能放大误差。本轮记录其发生率和误差影响，不事后改变损失。

单 seed 结果仅为筛选证据。A/B/C 的主干、任务头、同一初始权重、训练样本多重集、总样本呈现次数、总更新数、有效 batch、优化器和验证/开发集全部匹配。B 与 C 使用完全相同的课程 batch 顺序；其差别是 C 在 H1 上额外接收 K2 软响应损失。

## 2. 物理模型与范围

沿用 `experiments/D2-A-E001/config.yaml` 的研究模型：3 通道，PRF=1300 Hz，N=130，32 个距离单元，距离压缩复 IQ，静止点杂波和圆对称复 AWGN；三通道理想标定、平地/点散射、共址 Tx 参考假设和既有波形/几何均保持不变。H0/H1 配对共享同一杂波、噪声及场景 nuisance，H1 只增加一目标。使用 A/B/C 可辨识速度条件；D exact-overlap 只作不可辨识边界描述。

课程条件固定如下；目标振幅、初相、距离和方位 nuisance 继续遵循 E001 生成器分布：

| 课程层 | SNR / CNR | 目标条件 | 含义 |
|---|---|---|---|
| C0 | 20 / -5 dB | A separated | 极弱杂波、高 SNR 基础相干学习 |
| C1 | 15 / 0 dB | A、B 等量 | 弱杂波，加入近杂波边缘 |
| C2 | 10 / 10 dB | A、B、C 尽量等量 | 中等杂波和谱边缘/部分重叠 |
| C3 | SNR 0/10 dB 等量，CNR 20 dB | A、B、C 与两个 SNR 尽量等量 | 强杂波、低 SCNR 和部分重叠压力 |
| C4 | C0–C3 各 25% | 继承各层的目标条件 | 综合混合，避免只适应最后困难层 |

每层 CNR/SNR 均处于 E001 配置范围内。A/B/C 只指 E001 冻结的速度/杂波关系；目标距离、方位等连续 nuisance 按原模型采样。除此之外不增加多目标、加速度、通道误差、真实数据或原始 ADC。

## 3. 学生、K2 教师与损失

### 3.1 学生架构

复用 `src/radardistill/students/task_student.py` 的 D1-E001 `TaskStudent`：`[B,3,130,32]` 复 IQ 转为 `[B,6,130,32]` 的 I/Q 平面；width=32；时间 dilation `[1,2,4,8,16,32,64]`；K3 输出为 presence、32 距离 logits、距离单元内偏移及每个 range cell 的单位 Doppler `(cos,sin)`。时间理论感受野 255 脉冲，覆盖 N=130；网络先逐点混合六个实通道，再做实值时间卷积和时间均值池化。该结构能通过训练学习通道交互，但不具备显式复数线性等变、跨通道复乘或几何 DPCA 层；SiLU/GroupNorm 也不保证公共相位旋转等变。

全部 A/B/C 训练模型都实例化相同 K2 辅助 head，并从同一初始化 state 开始；A/B 不计算、不使用其输出，C 只在训练期间使用。所有最终推理 checkpoint 导出时移除辅助 head。学生推理只接收 IQ，不调用 DPCA、FFT、CFAR、硬阈值或教师图生成器。

### 3.2 现有损失与预注册的限制

沿用当前 task loss，不因预期结果增加模块：presence BCE 对 H0/H1；range-cell CE、真值 range cell 的 offset Smooth-L1 (`beta=0.1`) 和同一真值 cell 的圆周 Doppler cosine loss 只对 H1；四项权重均为 1。Doppler 误差按 1300 Hz 周期回绕。正常推理按模型预测的 range argmax 读取 offset/Doppler。故本轮明确诊断真值 range 索引和预测 range 索引不一致率及其 Doppler 误差；除非训练前诊断发现代码实现错误，否则 A/B/C 使用同一冻结损失，不在结果出来后修补。

训练前记录初始小批次各项损失值和单任务梯度范数，训练中记录每项损失、总损失及非有限值。验证 checkpoint 只按共同 validation 的等权物理真值任务损失选择；K2 loss 不参与选模。

### 3.3 K2 定义及信息隔离

K2 是**未阈值化 DPCA 杂波抑制响应图**：输入 IQ → E001 冻结 B3 几何分数时间对齐/通道差分 → N=130 居中慢时间 FFT → 非负 RD power `P[d,r]` → `q=P/(sum(P)+1e-12)`，形状 `[130,32]`。不作 T1/T2 检测阈值，不用真值裁剪、目标位置、场景标签、CNR/SNR sidecar 或教师 hard output。归一化和 map temperature 固定为 power/temperature=1。

K2 质量审查可用物理真值作离线评价，但真实输入教师函数只接受 IQ。训练时 K2 KL 只在 H1 样本上计算，H0 仍参与 presence 真值任务；使用 `truth_presence` 仅作 loss mask，不参与 K2 生成或模型输入。这避免学生被要求复刻 H0 噪声/杂波的任意最高峰。教师/学生软图和物理真值分别保存。

预先冻结的严重误导判据：若 K2 数值/归一化/信息隔离任一检查失败，停止 C 并修复后重新冻结协议；或在独立 K2 audit 的 C3 合并 A/B/C 条件下，配对 target-window saliency mass lift（H1 减去同场景 H0）的 95% scene-bootstrap 区间上界 `≤0`，且 teacher 最大响应的 range `≤1 cell`、圆周 Doppler `≤20 Hz` 命中率 `≤5%`，则取消 C 臂。其余 K2 弱项逐条件报告，不因低速/重叠的预期风险隐瞒；A/B 仍可继续。

## 4. 数据、随机种子与公平性

所有 split 按 `base_scene_id` 隔离。任意 H0/H1 pair 及其教师 map 完整属于同一 split；没有 seed 或 pair ID 作为模型输入。训练数据、teacher audit、validation、H0 calibration、H0 verification、development、D boundary 使用互不重叠的新 seed 区间；均不复用 D1、E001、R3 数据，也不生成 final blind。

| Split | 数量与构成 | 用途 |
|---|---|---|
| Train | 51,200 对 H0/H1，每个 C0–C3 各 12,800 对 | 1600 updates × 32 对/update；每个 base pair 恰呈现一次 |
| Validation | 5,000 对，C0–C3 各 1,250 对；C1 的 A/B 各 625，C2 的 A/B/C 为 417/417/416，C3 的 A/B/C × SNR 0/10 dB 为 209/208/208/209/208/208 | 每 100 updates 共同评价并选择真值任务损失最低 checkpoint |
| K2 audit | 2,048 对，四个课程层各 512 对；C1 的 A/B 各 256，C2 为 171/171/170，C3 的 A/B/C × SNR 0/10 dB 为 86/85/85/86/85/85 | 只审教师 response，不用于训练或调参 |
| H0 calibration | 15,000 scenes | 独立估计各学生/基线各自的全局 5% 阈值 |
| H0 verification | 15,000 scenes | 独立评估固定阈值 Pfa，不调阈值 |
| Development | 5,000 对；采用与 validation 相同的逐单元样本数和独立场景 seed | pilot 开发评价与有界阶段决策，不是盲测 |
| D boundary | 1,024 对 exact-overlap，SNR/CNR=10/10 dB | 仅说明同流形来源归因不可辨识，不纳入 A/B/C 门 |

Train cohort 的 `base_scene_id` 预先固定。每个课程层内的训练 cell 次序由 config 中 `assignment_seed=2026102606` 洗牌；场景生成按各层固定 seed base、chunk 大小 16 和 seed stride 1009 顺序产生。课程批次计划由独立固定 seed `2026102603` 生成：每次 update 取 32 个不同 base scenes，并同时包含各自 H0/H1，构成有效 batch 64。A 使用同一 51,200 对样本的固定随机排列，难度从第一步起是全分布混合。B/C 共享同一计划：C0–C3 各 256 updates（每层 8,192 对）；C4 为 576 updates，以均等数量交错使用四层各自未呈现的 4,608 对。故 A/B/C 使用完全相同的样本多重集、每对恰一次、总更新和有效 batch 完全相同；区别只有呈现顺序及 C 的 K2 loss。Validation、K2 audit 与 development 各使用 config 中单独登记的 cell-assignment seed；所有场景 seed block 依次使用 1009 stride，并经 preflight 检查互不重叠。

H0 calibration 与 verification 每个 scene 独立等概率抽取 C0–C3 nuisance 层；C3 内 SNR 0/10 dB 独立等概率。生成器对目标参考振幅继续从 `[-6,6] dB` 均匀抽样，所以总体混合是预注册的 IID scene mixture。校准/验证各 `n=15,000`。validation 与 development 的单元计数固定于 `config.yaml`；development 使用相同单元计数及新的 seed，逐组记录实际样本数。

Train/validation IQ 与 train K2 可存本机 CPU memmap 缓存，数据不放 GPU、不提交 Git；evaluation split 以流式 batch 或本地 CPU memmap 复用。预估存储：train complex IQ 约 10.2 GB、validation 约 1.0 GB、train K2 约 0.83 GB；开跑前检查实际磁盘余量。每次只把当前 microbatch 搬到 GPU。

## 5. A/B/C 训练预算

三个 pilot 臂各一个训练 seed `101`，相同 `TaskStudent` 初始权重/初始化 seed，均 1,600 次 AdamW optimizer updates、effective batch=64、FP32。每个臂单独顺序训练；GPU 任一时刻只驻留一个学生。

| 臂 | 样本呈现顺序 | 监督 |
|---|---|---|
| A | 全 C0–C3 训练 pair 随机混合 | 物理真值任务损失 |
| B | C0→C1→C2→C3→C4 | 物理真值任务损失 |
| C | 与 B 完全相同的 course batch plan | 物理真值任务损失 + `lambda(u) × K2 KL` |

AdamW 初始 lr `1e-3`，余弦降至 `1e-4`；weight decay `1e-4`；梯度范数裁剪 `1.0`。`lambda(u)` 从第 1 次更新的 `0.05` 线性降至第 1600 次更新的 `0`；任务真值监督在每一步始终存在。K2 map loss 使用 `KL(q_teacher || softmax(student_logits))`，对每个 H1 的 4160 cells 展平后计算，再对 H1 样本取均值。

初始 microbatch=16，OOM 时按 8、4 依次重启该臂并从同一初始权重、固定 batch plan 重新训练；梯度累积相应调整为 4、8、16，保持 effective batch、样本多重集、scheduler 和更新数不变。microbatch=4 仍 OOM 即停止该臂并记录，不改成 CPU/AMP，不减少科学预算。训练过程每步写一条日志；每 100 updates 保存 progress/optimizer checkpoint 并评价共同 validation。报告 microbatch、CUDA 峰值 allocated/reserved memory、实测用时、生成和预处理耗时及学生参数量；不声称完整链速度优势。

## 6. K2、任务及物理评价

### 6.1 K2 质量 audit

在训练前按 2,048 对独立 audit 评估：教师输入仅 IQ；map finite/nonnegative，形状正确，和为 1；H0/H1 均不崩溃；公共复幅度缩放 `X→aX`（`a=0.5,2.0,0.6+1.2j`）后归一化图最大绝对差 `≤1e-6`。离线使用真值计算 target window（range 误差 `≤1 cell` 且 circular Doppler 误差 `≤20 Hz`）概率质量、相对配对 H0 的 lift、教师 top-cell 命中率及配对区间。不得用这些结果调整 K2 定义或参数。

### 6.2 H0 阈值与任务指标

学生分数固定为 presence logit。每个学生和传统分数各自使用相同 H0 calibration scene IDs 的分数，按 `numpy.quantile(q=0.95, method="higher")` 定阈值，推理比较符固定 `score >= threshold`；记录 `>`、平局数与校准经验 Pfa。学生不借用教师阈值。验证/开发不得回调阈值。

传统对照：E0 `max_r sum_(channel,pulse)|IQ|²`；E001 冻结 B0/B1/B2/B3；R3 冻结 T1。T1 只称 R3 research candidate，不是获准教师。各基线独立校准阈值。所有方法共用 H0 verification 和配对 development sample IDs。

至少报告：

- H0 verification 总 Pfa、单侧 95% Clopper–Pearson 上限和分层 Pfa；每个 SNR/CNR 层、四个幅度四分位及 A/B/C 均记录样本数。
- development ROC/AUC；方法自己的校准阈值下 Pd；development ROC 共同经验 FPR=0.05 的 Pd。
- 所有 H1 及命中子集的 range MAE/RMSE（cell 与米）、circular Doppler MAE/RMSE/bias、异常率；漏检作为 `joint_success` 失败。联合成功定义为 detected、range error `≤1 cell`、Doppler error `≤20 Hz`。
- 按课程层、A/B/C、SNR/CNR、SCNR、range/Doppler、幅度、初相分组；训练集确定的 range 中位数和 circular Doppler 均值作为无输入预测常值基线。
- 训练/验证各损失、AUC、range/Doppler RMSE、达到 A 组最佳 validation task loss 的更新数；报告 A–B 和 B–C 的 paired base-scene bootstrap 区间。
- K2 teacher 原始 B3 score 与 T1 ROC/AUC/Pd、target-region mass lift、低速/重叠结果。教师 fixed threshold 仍按 R3 失败记录，不因软图可用而改判。

采用 2,000 次按 `base_scene_id` 配对 bootstrap 计算 development AUC、Pd、RMSE、joint success 和组间差值区间；Pfa verification 主区间为精确单侧二项上限。由于只有一个训练种子，bootstrap 只反映场景抽样不确定性，不代表 seed 不确定性，也不据此声称统计显著蒸馏收益。

### 6.3 物理一致性与数据捷径

在固定 development 子集进行：每个课程层按缓存顺序取前 64 对，合计 256 对；32 个公共复相位旋转；整体幅度乘 `0.5/2.0` 且不重新校准；目标慢时间分量乘 `exp(j2πδn/PRF)`，`δ∈{-25,-10,10,25} Hz`；独立常相位扰乱各通道的相干消融。真值只用于生成受控诊断或事后分组，不输入模型。报告 phase decision agreement/range/Doppler changes/AUC、frequency-shift equivariance residual、scale Pfa/Pd 变化和通道相干消融变化。固定参考门：公共相位决策一致率 `≥99%`、range 变化 95 百分位 `≤0.25 bin`、Doppler 变化 95 百分位 `≤5 Hz`、AUC 绝对变化 `≤0.01`；频移残差 95 百分位 `≤10 Hz`、range 中位变化 `≤0.25 bin`。幅度缩放导致 Pfa `>0.10` 或 Pd 绝对下降 `>0.10` 时标记尺度敏感。通道相干消融报告为诊断，不要求输出不变。所有变换均使用原 calibration 阈值，不另行校准。

## 7. 冻结阶段门与解释

- **G1 工程正确性：** E001 T1–T12、split/seed/pair 检查通过；检测器只读 IQ；物理 truth、teacher map 和模型输入分离；A/B/C 参数/预算/初始化/样本多重集校验通过；模型/损失梯度有限；RTX 3050 CUDA 真机可见。未通过则不启动正式 1600-update pilot。
- **G2 K2 质量：** 第 6.1 节数值/尺度/信息隔离通过，且严重误导规则未触发；未通过时停止 C，但 A/B 仍可继续。
- **G3 H0 检测：** 对每个学生，15,000 个独立 verification H0 的 Pfa 点估计 `≤0.06` 且单侧 exact 95% 上限 `≤0.06`；任一预定 SNR/CNR 或幅度层上限 `>0.10` 记为分层失稳。不得用旧 R3 门“CI 必须覆盖精确 0.05”替代此门。
- **G4 学生任务学习：** 对每一臂，在 pooled identifiable scenario A/B/C 上学生 AUC paired 95% CI 下界 `>0.5`；相对 E0 的 AUC 点增益 `≥0.02` 且差值区间下界 `>0`；共同 ROC FPR=0.05 下 Pd 差值下界 `>−0.05`；range 和 Doppler RMSE 均优于仅由 train truth 计算的常值基线且 paired 95% CI 上界 `<0`；主 A/10/10 dB joint success 点估计 `≥0.50`、单侧 95% bootstrap 下界 `>0.40`。这些指标逐臂报告，不以某臂掩盖另一臂失败。
- **G5 课程/K2 筛选信号：** B−A、C−B 的 pilot 改善信号分别按预注册解释：joint success 增益至少 `0.05` 且 paired 95% 场景 bootstrap 区间下界 `>0`；或至少提前 `20%` 更新达到 A 的最佳 validation task loss，并且最终 joint success 差值区间下界 `>−0.05`。若不满足，记录未见清晰收益。该门只用于决定是否另提正式 3-seed 确认，不是跨 seed 推断。
- **G6 物理诊断：** 第 6.3 节所有必做变换完成并如实报告。公共相位/频移数值门失败时，不声称当前学生已可靠利用相干物理量，也不启动 final blind；幅度/通道干预限定鲁棒性表述。

G1 的固定 micro-overfit 检查从 C0 预注册随机分配后的前 16 对训练场景取 H0/H1，共 32 个 IQ 输入；相同初始化、FP32、AdamW `lr=1e-3`、`weight_decay=1e-4`、梯度裁剪 1.0，重复真值任务监督 100 次。末次 loss / 初始 loss 必须 `≤0.75`。初始真值总损失对 stem、presence、range、range-offset、Doppler 模块的梯度范数都必须有限且非零；训练 K2 缓存存在时，K2 KL 对 spectrum head 的梯度范数也必须有限且非零。该 sanity 拟合不计入 A/B/C 的 1,600 次更新、不用于选模型，也不通过改变结构/损失来满足门槛；失败时先定位实现或数据问题并停止正式 pilot。

结果解释：A 失败、B 成功则支持课程的初步价值；B/C 相近则没有教师增量证据；C 相对 B 达到 G5 才支持继续正式 K2 对照；A/B/C 都失败则先查正确性、range-index mismatch、梯度/损失和结构，再提出有界物理相干表示诊断，不直接扩大模型。C 独自成功时，因三个臂均实例化相同辅助 head 且样本/预算相同，排除了 head 参数量与额外 task updates；仍需在后续确认实验检验随机/非目标辅助 response 等替代解释。没有跨 seed 数据时不做“蒸馏已被证明”结论。

无论 G2–G6 结果如何，R3 G2、E001、D1 和历史负结果保持原状。本轮不启动正式三种子、不生成 final blind、不把 R3 候选改称获准教师。若后续要做正式确认，先另行冻结三个独立训练种子、统计功效、模型选择和全新盲测。

## 8. 复现和输出

训练入口：

```bash
conda run --no-capture-output -n AISAR python scripts/run_d2a_e002.py --config experiments/D2-A-E002/config.yaml --stage preflight
conda run --no-capture-output -n AISAR python scripts/run_d2a_e002.py --config experiments/D2-A-E002/config.yaml --stage generate-data
conda run --no-capture-output -n AISAR python scripts/run_d2a_e002.py --config experiments/D2-A-E002/config.yaml --stage k2-audit
conda run --no-capture-output -n AISAR python scripts/run_d2a_e002.py --config experiments/D2-A-E002/config.yaml --stage pilot --device cuda:0
conda run --no-capture-output -n AISAR python scripts/run_d2a_e002.py --config experiments/D2-A-E002/config.yaml --stage evaluate --device cuda:0
```

必须保存 `config_used.yaml`、`environment.json`（PyTorch/CUDA/driver/GPU、磁盘余量、`nvidia-smi` 前后、代码 SHA）、K2 audit 指标/逐组 CSV、A/B/C 初始/最佳/最终 checkpoint、逐 update history、Pfa calibration/verification 指标、开发 `metrics.json`/`result.csv`、收敛和物理诊断、RMSE/ROC/课程图、`analysis.md` 与 `docs/stage_reviews/D2-A-E002_review.md`。训练 cache、memmap、可重建 H0/H1 IQ 默认不加入 Git。报告必须记录真实运行；未运行项清楚标注，不用计划替代结果。
