# D2-A-R6-F：完整检测与参数估计处理链比较

**协议版本：** 1.0.2

**状态：** 运行前冻结；开发实验，`final_blind_authorized: false`

**对应总纲：** `docs/RESEARCH_CHARTER.md` v1.3

**执行前修订 1.0.1：** 将 G1 合成链路检查专用 seed 固定为 `2,298,000,000`，条件索引固定为 10，样本数为 16 CPI。该 seed 仅用于正确性检查，不进入校准、验证或 development 指标；修订发生在 G0/G1 和正式场景生成之前。

**执行前修订 1.0.2：** 系统 D 固定为 B3 提供检测分数、CE-TCN 输出其原生距离/Doppler 估计，不再把 B3 FFT 峰值门用作 CE 的 offset 基准。CE range head 的训练 offset 相对于 `floor(truth_range)` 监督门定义，而 B3 选择离散功率峰；二者门索引语义不能预先视作相同。D 将同时记录 B3 检测候选与 CE 参数候选的一致率。此项在 G0/G1 和正式数据生成前完成，不改变六链数量、数据划分、阈值、指标或统计门。
**依据：** R6 总体设计第 9 节、R6-A 公共场景定义、R6-E 架构筛选与阶段报告。

## 1. 研究问题与范围

在同一批新的三通道复数 IQ 场景上，比较传统检测与估计、R6-E 学生直接输出，以及神经检测和传统精估串联后，完整链路的虚警、检测、距离/Doppler 参数和计算成本。核心问题是：CE-TCN 的任务收益能否保留到完整处理链；收益来自检测、参数估计还是二者配合；全链重新校准后能否满足 C3 虚警保护。

本阶段不训练网络、不做知识蒸馏、不选择 final blind，也不把单个开发 seed 的结果称为跨 seed 优势。所有方法均只接收可观测 IQ 和固定物理配置，目标真值只在预测完成后用于离线评价。

研究总纲映射：Q1、Q3、Q6、Q7、Q8、Q9、Q10。R6-F 直接检验完整系统收益与传统链路边界，服务于“直接 GMTI 检测和参数估计”主目标；它不是独立的传统算法优化项目。

## 2. 冻结的六条系统链

每条完整链对自己的单 CPI 检测分数独立用 H0 calibration 拟合全局 `Pfa_CPI=0.05` 阈值，固定比较符为 `score >= threshold`。不得沿用网络子模块、R6-A、R6-E 或其他链的阈值。未检测时，最终距离和 Doppler 输出为 null/NaN；不得把候选估计伪装成检测结果。

| 系统 | 检测与最终分数 | 检测通过后的距离/Doppler | 作用 |
|---|---|---|---|
| A | B3 DPCA 最大响应 | B3 距离门上的 4× 零填充 FFT 峰 | 传统检测 + 传统精估主基线 |
| B | CE-TCN `presence_probability` | CE-TCN 原生 range/Doppler 联合输出 | CE-TCN 原生联合学生链 |
| C | CE-TCN `presence_probability` | CE-TCN 距离候选门上的局部 profile-ML Doppler；range 用 CE-TCN 连续输出 | 神经检测 + 传统精估 |
| D | B3 DPCA 最大响应 | CE-TCN 原生 range/Doppler 输出；B3 只决定是否输出，CE 参数候选不受 B3 峰值门强制覆盖 | 传统检测 + 神经精估 |
| E | J1-J `presence_probability` | J1-J 原生 range/Doppler 联合输出 | 与 CE-TCN 配对的共享主干联合 NN 参考 |
| F | `CE_probability × fit_fraction` | CE-TCN 距离候选门上的 profile-ML Doppler；range 用 CE-TCN 连续输出 | CE 检测 + profile-ML + 统计拟合验证 |

`fit_fraction` 是在 CE 候选距离门的 DPCA 慢时间残差上，4× FFT 最大复正弦投影功率除以 `N × 总能量`；理论范围 `[0,1]`。它是可观测的归一化拟合统计量，不宣称是经解析校准的 p 值。F 的唯一最终阈值仍由 F 整链的 H0 calibration 决定；不另外搜索验证阈值、权重或组合式。数值越界仅允许由浮点舍入导致，计算后截在 `[0,1]`。

系统 A 采用 R6-A 已注册的 4× FFT 传统 Doppler 精估，避免用局部搜索给传统主基线不必要地增添算法自由度；局部 profile-ML 仅用于 C/F 与 R6-A 注册的估计诊断。所有 DPCA 候选来自 B3 真实最大响应；神经参数候选来自模型输出。D 的检测候选由 B3 产生，参数候选由 CE-TCN 原生 range logits 产生；两者不一致时仍按冻结的 B3 检测阈值输出 CE 参数，并将候选一致性单独报告。该链不访问真值。

### oracle 估计诊断

在同一 H1 development IQ 上，另用离线真值最近距离门运行 4× FFT、profile-ML 及 CE 逐门参数头，估计“候选已正确”时的参数误差上界。真值不进入任何六条链、分数、候选、阈值或模型选择；oracle 数值单独列出，只作误差来源诊断，不作为系统性能或晋级依据。

## 3. 数据、模型和隔离

物理与场景条件保持 R6-A：三通道，PRF 1300 Hz，130 脉冲，32 距离单元，静止杂波及 12 个冻结 C0–C3 条件。条件表从 `experiments/D2-A-R6/R6-A/config.yaml` 读取，运行前校验与本协议的来源一致。六条链对每个 split 使用完全相同 IQ；H0、H1、calibration、verification、development 的 scene ID 两两不重叠。

| Split | CPI 数 | 每条件 | 用途 | seed base |
|---|---:|---:|---|---:|
| H0 calibration | 120,000 | 10,000 | 每条系统独立校准阈值 | 2,308,000,000 |
| H0 verification | 120,000 | 10,000 | 固定阈值虚警验证 | 2,328,000,000 |
| H1 development | 6,144 | 512 | 同场景系统比较、误差和配对统计 | 2,348,000,000 |

每条件 seed block 为 1,000,000，chunk 大小 16，chunk seed stride 1,009。代码必须在运行前检查 split 内 seed block 不重叠、所有生成 scene ID 唯一，且与仓库已有实验 scene manifests 和 R6-A/R6-E 保留 scene ID 不重叠；冲突时停止，不重编号。场景由冻结仿真器按块重建；不落盘 IQ。

模型固定使用 R6-E seed 606 的 validation-selected checkpoint：`R6-E/runs/seed_606/ce_tcn/best_checkpoint.pt` 与 `j1_joint/best_checkpoint.pt`。载入时验证 owner、checkpoint update、参数量、training record 和模型输出契约；不重训、不改权重。CUDA 必须为 RTX 3050 Laptop `cuda:0`，不得回退 CPU；传统基线按其已验证 CPU 实现运行。模型顺序使用并记录设备/显存。

评价输出只保存每场景分数、阈值、候选、参数预测、真值和 scene ID；不保存全量 IQ、DPCA 残差或教师图。预计输出体积小于 100 MiB；清理仅针对本阶段 owner manifest 登记的重建缓存，并保留逐项清理审计。

## 4. 固定评价与统计

每条链单独用其 H0 calibration 全局混合分数按 `higher` 经验分位数校准 0.05，再在独立 H0 verification 上报告全局及 12 条件 Pfa、单侧精确 Clopper–Pearson 上界、C3 六格 Bonferroni familywise 上界、ties 和阈值迁移。全局 Pfa verification 上界 `<=0.055`（相对0.05目标最多允许10%相对膨胀）且 C3 每格 familywise 上界 `<=0.10`（沿用既有R6低虚警保护）是该链的绝对 H0 资格门。阈值容限和样本量在查看 R6-F 结果前冻结。失败链仍报告完整 H1 指标，但不能宣称合格系统。

H1 报告每条件、C0–C3、C2+C3 及总体 Pd/AUC、检测候选正确率、range 和 PRF 回绕 Doppler 的 MAE/RMSE/bias、检测命中条件误差、all-H1 离线估计误差、检测后系统输出误差与 joint success。joint success 固定为 detection 命中、距离误差 `<=1` cell 且 PRF 回绕 Doppler 误差 `<=20 Hz`；miss、null、错误候选或参数越门均计失败。

主比较为 B–F 相对 A 的 C2+C3 joint-success paired scene bootstrap 差值，按 condition 分层抽样，5,000 次，seed `2026110301`；每个对比使用双侧 99% percentile 区间以覆盖五个预注册候选对比。相对候选筛选要求点差 `>=0.02`（沿用 R6-E 的最小有意义收益）且区间下界 `>0`。任务保护为 Pd 差区间下界 `>=-0.02`、all-H1 range 与 Doppler RMSE ratio 的 97.5% 区间上界各 `<=1.05`。所有点估计、区间、绝对 H0 门分别呈现；满足某一收益门不抵消 H0 或任务保护失败。候选对比是开发筛选，不是最终确认。

记录 B3/CE/J1 的检测距离门、各链所用实际候选、候选正确/错误、oracle-only 结果、F 拟合分数组件和完整 pipeline 成本。成本采用同一 H1 development 中每个条件按 scene ID 顺序取前 32 个（共 384 个）场景、batch=16，包含 IQ 实虚转换、必要 CPU/GPU 传输、完整链分数与候选、检测后估计/验证、同步及最终输出；先做 2 个 warm-up batch，再对 24 个 batch 连续计时 3 轮。报告每 CPI 均值/中位数/P95、GPU 峰值 allocated/total、CPU/GPU 边界和每链硬件。该成本只用于完整链比较，不作实时或产品时延声明。

## 5. 阶段门与停止条件

- **G0（实现与隔离）：** protocol/config/source/checkpoint SHA 记录一致；RTX 3050 可见；seed/scene ID 无冲突；六链分数、候选、null 语义及同场景配对检查通过。
- **G1（小样本链路验证）：** 使用 config 中独立保留的 seed、C3 partial-overlap 条件及 16 对合成 IQ；每条链输出有限分数/参数；DPCA 与神经候选按该链定义正确传播，D 分别保留检测候选和参数候选；H0 不生成真值参数；H1 与 oracle 明确分开；batch=1 和 16 一致到预设数值容差；CUDA 路径无 CPU 回退。
- **G2（H0）：** 各链按自身 H0 calibration 独立定阈值，并只用 verification 判定上述全局/C3 门。任何一链失败时不得转借其他链阈值或以开发 H1 调整分数。
- **G3（H1/完整链）：** 所有系统在相同 H1 scene IDs 上评价；候选级误差和 end-to-end joint 分开；阈值、模型、估计器、统计组合均不得依据 verification/H1 反复调节。
- **G4（阶段判断）：** 只将通过绝对 H0 与任务保护的链称为候选合格系统。若没有链通过，保留相对性能证据但结论不得升级为系统优势；依据 CE/传统检测/参数子项归因，决定 R6-G 的教师内容是否仍有可检验价值，不能靠教师训练掩盖所有完整系统均未过门。

候选实现错误、真值泄漏、分数非有限、seed 冲突或 CUDA 不可用时停止相应阶段；不根据结果追加系统、统计组合、阈值、背景估计或超参数搜索。模型与协议门通过后执行一次开发评价，不生成 final blind。

## 6. 复现与交付

主入口 `scripts/run_d2a_r6f.py`；核心链函数和评价在 `src/radardistill/evaluation/d2a_r6f.py`。配置在 `config.yaml`。G0/G1、运行环境、checkpoint owner、seed/scene manifests、prediction outputs、metrics、cost、图、analysis 和阶段审查均写入本目录/相应 `docs/`。任何未运行结果明确标为待执行。R6-F 完成后保留失败链和全链 H0 记录，只清理由 owner 清单登记的可再生缓存。
