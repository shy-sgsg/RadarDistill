# RadarDistill 动态研究路线图

**更新时间：** 2026-10-31
**权威关系：** 长期使命见 [`RESEARCH_CHARTER.md`](RESEARCH_CHARTER.md)；简要状态见 [`CURRENT_STATUS.md`](CURRENT_STATUS.md)；历史指标和冻结门以对应 protocol、metrics 与阶段审查为准。

## 当前阶段

**D2-A-E001、R3、E002 v2.1 pilot、T001、R4、E003、R5-A/B/C教师审计、R5-C teaching pilot、R6-A/B/C均已完成各自冻结试验。** R6-C在同数据、有效batch64和12,800 updates下发现更宽学生的C2+C3 joint提高，但绝对C3 Pfa保护失败；数据规模对照未达到主要joint增益门，冻结续训条件也未通过。R6-A/B/C均为development证据，不启动当前候选多seed或blind。R6-D仅完成J1/J2架构与正确性设计，尚未实现或训练。所有历史阶段门、负结果和研究总纲v1.3均保留。

R5-A/B学生及T0–T3四臂的C3 H0 Pfa均未通过绝对0.10保护；本pilot T0/T1/T2/T3的C3 Pfa为0.1943/0.2031/0.1824/0.1973。T1−T0、T2−T0的C2+C3 joint差为−0.0104/−0.0138。教师审计与教学pilot均显示C3 partial-overlap / 0 dB教师定位率约4%–5%。R6-A中Medium全局0.05阈值下joint success为0.8306，但condition 10 AUC=0.5956、joint=0.0391；R6-B预注册困难格AUC bootstrap上界0.63295低于0.70，故运行R分支hard-negative训练对照。HardNegative−TruthOnly全局joint差为−0.00212（paired 95% CI `[-0.00651,+0.00195]`），C3与困难格也未见增益；0.05/0.01全局Pfa上界未达标，0.001仅全局过门，分层仍失败。R6-B裁决为 **REFINE**，不晋级hard-negative多seed。用户允许未来新实验在公平比较下增加updates并探索至多两倍batch；本轮冻结设置不变。当前不启动教师时机/可靠性搜索、R5-D完整机制实验或final blind；研究总纲v1.3不变。

**D2-A-T001 v1.2 已完成，教师候选 STOP。** CUDA covariance 与 ACE×AMF 候选使用 RTX 3050 `cuda:0`；4096 H0 covariance、15k H0 calibration、15k H0 verification、12k paired development 全部完成。总体验证 Pfa 为0.05007（exact单侧95%上限0.05309），但 C3/SNR0/CNR20 三个大分组的上限为0.125–0.127；主困难条件 ACE×AMF joint success为0，低于B3的0.01。A/B/C pooled Pd与joint success分别较T1低0.2267和0.2441，95%区间均完全低于0。G1因公共尺度误差4.58e−5超过冻结1e−6失败；G2失败，G3按协议未评估，S2 pilot未运行。运行代码提交 `91dfb81673e45cad3f3330d90adc707a0cdf859d`，全阶段约59分钟。结果见 T001 分析、指标与阶段审查；该负结果不改变 R3、E002 或总纲。

## 已完成实验与证据

### D2-A-R6-A（同场景检测与参数估计基线）

- 完成12,000 H0 calibration、12,000 H0 verification、6,144 H1 development；12格每格1,000/1,000/512。所有方法共享IQ，未训练新模型、未生成final blind。G0为`PASS_WITH_RECORDED_PROTOCOL_ORDER_DEVIATION`：G0前生成的两个注册场景只检查shape与ID，没有查看样本值或预测，详见 `docs/task_alignments/D2-A-R6_alignment.md`。
- 全局Pfa=0.05开发结果中，Medium AUC/Pd/joint/Doppler RMSE为0.9830/0.9334/0.8306/24.68 Hz；global Pfa单侧exact95%上界0.04996，但C3条件Pfa为0.072–0.109。condition 10（C3 partial-overlap、SNR0、CNR20）AUC=0.5956、Pd=0.2207、joint=0.0391。
- Medium预测range gate上的传统FFT peak/interpolation/4x/profile-ML Doppler RMSE约143–150 Hz；Medium直接输出24.68 Hz。nearest-bin oracle离线profile-ML为68.27 Hz，仅用于诊断。R3 T1/T2总体优于B3，但在困难格仍弱。
- R6-B选支脚本只读取R6-A H0 calibration与H1 development，未读H0 verification。C3 pooled AUC `0.93164 [0.92440,0.93871]`，hard-cell AUC `0.60324 [0.57085,0.63295]`，后者上界低于预注册0.70，选中R6-B hard-negative-resampling分支。归档于 `experiments/D2-A-R6/R6-B/branch_selection.json`。
- 完整分析、逐场景预测、scene/seed manifests、环境记录和图保存在 `experiments/D2-A-R6/R6-A/`。结果仍为development证据，不说明学生已通过可靠检测器门。

### D2-A-R6-B（训练期 H0 hard-negative matched pilot）

- seed303 truth-only 与 hard-negative 两臂使用相同 Medium width96、82,887训练参数、初始化、H1计划、任务损失、FP32 CUDA、有效batch64和12,800 optimizer updates；只有训练H0采样不同。训练缓存按stage owner清理，场景ID和审计记录保留。
- 两臂都在update12,000取得最低验证task loss（0.22854/0.23116）；update12,800分别为0.22989/0.23230。由此不能断言本轮预算已逼近极限，也没有显示最后800步仍有验证损失下降。R5-A的长预算证据支持后续继续检验训练充分性，但不是hard-negative收益证据。
- 全局Pfa=0.05下HardNegative−TruthOnly joint差−0.00212（paired CI `[-0.00651,+0.00195]`），C3差−0.00260、困难格差−0.00391，未证明收益；C3/困难格虚警上界也未过0.10。当前分支不进入多seed、KD或blind。
- R6-B自身不再追加训练；R6-C是路线中独立冻结的下一工作包，不以缺少R6-B ChatGPT审查为阻塞。R6-C对比应匹配更新预算、有效batch、数据暴露、验证与checkpoint规则，区分模型规模、训练预算与独立训练数据规模效应；batch对比作为单独的匹配因素。

### D2-A-R6-C（模型规模、训练预算与数据规模；协议v1.2已完成）

- 同一原始复IQ TaskStudent width96/200/376、seed404、E1 truth-only、FP32、有效batch64与12,800 updates。RTX 3050 Laptop `cuda:0` 顺序训练；训练/验证/H0 calibration/H0 verification/H1 development互斥，评价分别每模型120,000/120,000/6,144场景。
- 在global Pfa 0.05下C2+C3 joint success：Medium/Large/XLarge=`0.8164/0.8628/0.8704`。Large−Medium=`+0.04644`（97.5% paired CI `[+0.03646,+0.05643]`），XLarge−Medium=`+0.05404`（`[+0.04449,+0.06402]`）；任务/RMSE相对保护通过，但C3 pooled Pfa单侧95%上界为`0.10162/0.10207/0.10266`，且六个C3格familywise门均失败，故两候选均不晋级正式seed或blind。
- Medium训练数据对照固定12,800 updates、有效batch64和总样本暴露：51,200对vs102,400唯一场景对。C2+C3 joint差`+0.00673`（95% CI `[-0.00066,+0.01455]`），未达到`+0.02`主门；Doppler RMSE比`0.953`（`[0.917,0.993]`）属次级信号，C3 pooled Pfa上界`0.10412`，保护失败。
- Medium与validation筛选出的XLarge均未过冻结的12,800→25,600验证延长门；本轮没有训练到更长预算，不能据此称12,800已达学习极限。用户允许后续新实验预注册更充分的共同预算，须保持比较臂更新数、有效batch、数据暴露、优化计划和checkpoint公平匹配；history仍仅在固定间隔记录。
- 结果/缓存清理记录见 `experiments/D2-A-R6/R6-C/`；只清理owner记录的21,467,222,779 bytes可重建缓存，清理后项目约1.21 GB。阶段内部报告见 `docs/stage_reviews/D2-A-R6C_review.md`。没有final blind。

### D2-A-R6-D（联合检测与参数估计；设计已完成，实施待开始）

- J1/J2设计见 `experiments/D2-A-R6/R6-D/joint_architecture_design.md`。下一步完成模型/损失实现、概率与H0行为、PRF周期边界、双向梯度、显存和16-pair/200-update G1门；门通过后再冻结新development pilot。
- R6-C显示增加容量可提高可辨识C2+C3 joint，但未通过C3绝对Pfa保护。R6-D须报告并保护各臂C3 H0表现。
- 新protocol应根据用户授权预注册更充分的共同训练预算；D-only/P-only/J1/J2对照匹配updates、有效batch、训练样本暴露、优化计划和验证/checkpoint选择。不可把预算差异当作联合建模收益；不逐update追加训练日志。

### D2-A-R5（R5-A/B学生pilot、R5-C教师审计与R5-C teaching pilot均已完成）

- 设计依据、路线和用户授权见 `docs/superpowers/specs/2026-10-27-d2-a-r5-design.md`、执行计划 `docs/superpowers/plans/2026-10-27-d2-a-r5-implementation-plan.md` 与 `docs/task_alignments/D2-A-R5_alignment.md`。
- R5-A 使用 E003 冻结 train/validation IDs，seed101、E1 truth-only、RTX 3050 `cuda:0` FP32。S0/S2各训练12,800 updates（8遍）；两臂均在6,400步满足预注册延长条件。复用的E003开发材料不作盲测。验证任务指标持续改善；H-CAP-001在该单seed范围内得到支持，但尚不能声称训练收敛或跨seed稳定。
- S0原始复IQ学生的开发任务结果优于S2固定相干前端；S2的C2/C3联合成功差为−0.3945（95% CI `[−0.4189,−0.3696]`），C3 H0 Pfa仍为0.1838，故停止当前S2候选。S0的C3 H0 Pfa为0.2041，也未达到0.10保护门。完整结果见 `experiments/D2-A-R5/R5-A/analysis.md` 和 `docs/stage_reviews/D2-A-R5A_review.md`。
- R5-A容量对照中，Medium−Small C2/C3 joint-success差为`+0.13086`（95% CI `[+0.11230,+0.15088]`），range/Doppler RMSE比分别`0.839/0.713`；但C3 H0 Pfa为`0.2041/0.1890`，G3 absolute protection失败，因此不运行Large。结果见容量分析和阶段审查。
- R5-B TFAS以79,638参数、seed101、E1 truth-only在RTX3050 `cuda:0`完成12,800 updates。G0/G1/G2通过，但C2/C3 joint相对Medium差`−0.60449`（95% CI `[-0.62598,-0.58350]`），自身C3 H0 Pfa=`0.15186`，绝对0.10门失败；按协议停止TFAS扩展，不做额外seed、KD、Large或blind。phase/frequency敏感性与错误集中于C2/C3，均来自复用开发ID的描述性诊断。完整结果见 `experiments/D2-A-R5/R5-B/analysis.md`、`metrics.json` 和 `docs/stage_reviews/D2-A-R5B_review.md`。
- R5-C按独立 protocol v1.1 对4,352个新配对base scenes进行B3软响应质量审计，RTX 3050 Laptop `cuda:0` complex128，无CPU教师回退，无IQ/map cache。K1/K3/K2-local的G4均通过；K3−K1可辨识C2+C3定位率差`+0.03125`（95% CI `[+0.02474,+0.03776]`），target-window全图归一化质量无改善；K2-local采用局部归一化且峰位置继承K3，不能按其较大lift宣称优于全图响应。C3 partial-overlap / 0 dB定位率仅0.021/0.041/0.041；K2-local置信AUC=0.783但未经校准。详细结果见 `experiments/D2-A-R5/R5-C/analysis.md`、`metrics.json` 和 `docs/stage_reviews/D2-A-R5C_review.md`。
- R5-C独立审查以教师审计结果为基准返回 **REFINE**，支持一次匹配的T0 truth-only、T1 truth+K3、T2 truth+K2-local、T3 matched truth-assisted四臂pilot。新protocol/config v1.0使用seed303、51,200训练对/5,000验证对、有效batch64与12,800 updates；G0/G1和CUDA训练/评价完成。T1/T2相对T0主joint差为−0.0104/−0.0138，未过+0.02门；四臂C3 H0 Pfa均为0.182–0.203，未过0.10保护。T3描述性joint差+0.0156是truth-assisted结果，不能归因于教师。当前停止T1/T2教师内容分支，不开展教师时机/可靠性搜索、多seed或final blind；R5-A/B与R5-C历史审计结果保持原样，研究总纲v1.3不变。阶段结果见 `experiments/D2-A-R5/R5-C-teaching/analysis.md` 和 `docs/stage_reviews/D2-A-R5C-teaching_pilot_review.md`。
- 当前总纲 v1.3 修订内容仅新增训练/容量/架构/知识吸收原则与 Q10、事实更新 D2 状态；Q1–Q9 和历史结果不变。

- **D0-E001：** 12/12 简化仿真与 APX15 慢时间 FFT 子链正确性检查通过；不覆盖完整生产链。
- **D1-E001：** 三种子 formal、独立校准和 blind 完成；G1/G2 PASS，G3 FAIL，原阶段决定 GO → D2。C/D 的联合指标满足冻结 G2 规则，但没有收敛加速证据；D 的平均改善量受 seed 1 影响明显。
- **D1-R1：** 仅使用开发材料。高 SNR 预测距离门与 oracle 的 Doppler RMSE 接近；池化后线性 probe 比池化前好，但维数/样本数差异使池化因果未决；最终 checkpoint 梯度均值没有持续强冲突证据；teacher-KL 梯度因训练期 spectrum-head checkpoint 不存在而不可用。
- **D1-E002：** S1/S2 各三 seed、1600 更新，独立校准和 40,960 条 blind 完成。S2 高 SNR 等权 RMSE 23.352 Hz、较 S0 改善 10.86%，3/3 seed 同向；但低 SNR Pd 非劣区间未达门槛。S1 未改善。S3 未通过 pilot screen。总决策 **STOP / REDIRECT**。
- **D2-A-E001：** 合成静止杂波、3 通道、N=130、32 距离单元；R0 T1–T12 全通过。B3 几何 DPCA 静止杂波抑制中位数 15.406 dB，目标差分功率比中位数 3.614；分离 A 和边缘 B 有强检测，部分重叠 C 退化，同流形 D 近随机。校准 H0 每条件 512 个，开发每条件 256 个配对场景；主条件 A 的开发 `Pfa_CPI=0.0898` 高于标定经验值 0.0508。条件 Range/Doppler RMSE 区间已修正为 RMSE 统计量并重新生成。未训练学生、未生成 final blind。决定 **CONDITIONAL GO**，先复核阈值迁移，再考虑 E1。
- **D2-A-R3：** 新种子、3 个 4096 场景主校准组、4 个 4096 场景主 H0/H1 配对验证组，并含 CNR=0/20 dB 有界检查。原始 E001 B3 阈值在 16,384 个新主 H0 上得到 `Pfa=0.06464`（95% Wilson CI `[0.06097,0.06850]`）；有限样本可能解释 E001 `23/256` 的一部分，但层级 bootstrap 达到该点估计的比例为 0.0216。潜在目标参考幅度通过噪声/杂波生成尺度影响 H0，导致 B3 固定阈值的幅度箱 Pfa 极差达 0.169–0.216。T1 scene-wide B3 峰值/中位背景统计量与 T2 local median CFAR-like 分数均通过 G3/G4，但 T1 三次主合并验证 Pfa 为 0.0444–0.0462，T2 一次主验证 CI 不覆盖 0.05；因此都未过 G2，无方法完成全部门。T1 是未来候选，未被选为教师。无 E1、KD 或 final blind。阶段决定 **REFINE**。
- **D2-A-E002：** v2.1 A/B/C 已按相同 1,600 updates、有效 batch64、单 seed 101 使用 RTX 3050 CUDA 顺序训练。K2 audit 通过，G4 三臂通过；development AUC 为 A/B/C `0.9846/0.9787/0.9815`，Pd `0.9158/0.8996/0.9000`。学生整体 Pfa 与精确上限门通过，但 C3/SNR0 最差分层上限达 `0.195–0.204`，G3 分层门失败。G5 B−A joint-success gain `−0.1254`（95% CI `[−0.1386,−0.1122]`），C−B `+0.0036`（`[−0.0088,+0.0162]`）；课程和 K2 均无预注册 pilot signal。G6 phase agreement `0.943–0.950`、频移 residual p95 最大 `73.4 Hz`，等变门失败。阶段为 **REFINE**，没有 formal three-seed 或 final blind。
- **D2-A-T001：** v1.2 CUDA 教师比较已完成。总体验证 Pfa 达标，但 C3/SNR0/CNR20 分层 Pfa、主要困难条件 joint-success、A/B/C 参数保护和 G1 公共尺度严格误差均未过门。频移诊断未见目标峰跟随注入频率移动。按预注册规则 **STOP** 本候选，不评估 G3、不运行 S2、不生成 blind；不追加第二个先进教师。CUDA加速是候选内核实测，不代表完整处理链实时性。完整结果见 [`experiments/D2-A-T001/analysis.md`](../experiments/D2-A-T001/analysis.md)、[`metrics.json`](../experiments/D2-A-T001/metrics.json) 和 [`D2-A-T001_review.md`](stage_reviews/D2-A-T001_review.md)。
- **D2-A-R4：** 在新场景上复核复尺度数值误差、H0幅度nuisance、B3/T1/ACE×AMF和冻结学生。正确complex128缩放后R4 `1e-5`诊断门通过；complex64量化对照最大差`2.44e-4`，CPU/CUDA最大绝对差`3.09e-4`但相对差`9.30e-7`、无峰值/判决翻转。T1总体 H0 幅度四分位Pfa稳定在0.046–0.052，但C3 Pfa仍0.178–0.217；学生A/B/C C3 Pfa约0.152–0.195。只运行一个 seed101、1600 update、CUDA S1 E1 truth-only pilot。后验分层敏感性复核发现首256样本全属C0；平衡子集 S1 phase agreement 0.909。S1−S0 paired H1 Doppler RMSE差`−6.29 Hz`（95% CI `[-9.16,-2.84]`），但Pd差`−0.0117`和joint差`−0.0313`区间均低于0，故不继续此增强方向、不开展KD或final blind。R4是探索性证据，门槛未追溯修改；详见 [`analysis.md`](../experiments/D2-A-R4/analysis.md)、[`metrics.json`](../experiments/D2-A-R4/metrics.json)、[`sampling_sensitivity.json`](../experiments/D2-A-R4/sampling_sensitivity.json) 和 [`D2-A-R4_review.md`](stage_reviews/D2-A-R4_review.md)。
- **D2-A-E003：** 单一 28 通道复相干特征前端、场景尺度归一化及 `28→6` 投影接既有 TaskStudent；R1 数值/物理检查与 R2 A/B/C 可观测增量均通过。seed101 的 S0/S2 matched E1 truth-only CUDA pilot 各1600 updates、有效batch64。S0/S2 均通过基础任务学习 G3，但 S2 overall Pd 差`−0.0474`、joint差`−0.2102`，C2/C3 joint差`−0.2593`（95% CI `[-0.2822,-0.2368]`）；C2/C3 Doppler RMSE比`1.835`（`[1.720,1.959]`），G4失败。总体 H0 Pfa 为0.0483/0.0475，C3 H0 Pfa 为0.1814/0.1528，均高于冻结0.10门。S2公共相位输出一致率为1.0，但估计与任务更差。故 **STOP 当前 S2 候选**，不进入多seed/KD/blind；D2-A学生路线整体 **REFINE**。结果见 [`E003分析`](../experiments/D2-A-E003/analysis.md)、[`metrics.json`](../experiments/D2-A-E003/metrics.json)、[`物理一致性`](../experiments/D2-A-E003/physical_consistency.json)、[`CUDA成本`](../experiments/D2-A-E003/compute.json) 和 [`阶段自审`](stage_reviews/D2-A-E003_review.md)。
- **P0/P1 历史先导：** 按各自冻结阶段审查保留；不将旧开发测试重新称为盲测，也不把其结论扩大到 D0/D1。

## 当前支持、否定与未决判断

| 判断 | 状态 | 证据范围 |
|---|---|---|
| D0 简化仿真和 FFT 子链适配正确 | SUPPORTED | D0-E001 冻结 12 项检查 |
| D1 TCN 能从距离压缩 IQ 学习受测任务 | SUPPORTED | D1-E001 简化单目标复 AWGN、三训练种子 |
| D1 蒸馏改善收敛速度 | NOT_SUPPORTED | D1-E001 B/C/D 均未过冻结收敛门 |
| D1 C/D 蒸馏改善最终联合指标 | 有范围支持 | 原 G2 PASS；多策略比较未作家族错误率调整，D 的效果量受 seed 1 影响 |
| D1-E002 lag-one 特征能改善频率估计并保护检测/距离 | NOT_SUPPORTED（完整门） | S2 Doppler、range 门通过，低 SNR Pd 门失败；只限该协议 |
| 时间均值池化是 Doppler 误差主因 | INCONCLUSIVE | R1 probe 不支持简单因果解释；S1 也未改善 |
| 多任务持续梯度冲突是主要瓶颈 | INCONCLUSIVE | 只有最终 checkpoint、小批量诊断 |
| 教师低 SNR 全图响应可靠 | NOT_SUPPORTED（受测区间） | R1 teacher 的低 SNR 检测和 Doppler 质量弱，且 E001 蒸馏未改善低 SNR Pd |
| 学生有完整链速度优势 | INCONCLUSIVE | 输入边界、硬件不一致且无完整链计时 |
| D2-A 固定 5% H0 阈值可稳定跨独立主场景迁移 | NOT_SUPPORTED（当前仿真协议） | R3 冻结 B3 阈值 Pfa=0.06464；三次校准及新验证仍未使 T0/T1/T2 全部通过 G2 |
| 归一化可消除 H0 整体幅度尺度敏感性 | SUPPORTED（当前仿真内） | T1/T2 通过 G3 幅度分层与受控 `X,aX` 尺度检查；G2 未通过，因此不等于可靠固定阈值 |
| 三通道 post-Doppler ACE×AMF 能改善当前困难条件教师表现并保护 A/B/C | NOT_SUPPORTED（T001协议） | G1/G2失败；ACE×AMF在 C3/SNR0/CNR20 joint success=0，且A/B/C pooled Pd、joint success、RMSE显著劣于T1 |
| T001 ACE×AMF 软图值得蒸馏 | INCONCLUSIVE（G2前置门失败，未评估） | 按协议停止软响应质量审查，未产生有效soft-map audit或S2学生训练 |
| T1 可作为后续稳定传统教师 | INCONCLUSIVE | G3/G4 通过且 H1 能力强，但 G2 失败；保留候选，不进入教师选择 |
| D2-A-E002 学生能在可辨识开发场景学习任务 | SUPPORTED（当前单 seed 仿真 pilot） | 三臂 G4 通过，A AUC=0.9846、Pd=0.9158；未验证跨 seed 或盲测 |
| 课程顺序提高 joint success | NOT_SUPPORTED（当前 pilot） | B−A 差 −0.1254，配对 CI `[−0.1386,−0.1122]`；未见收敛提前 |
| K2 软响应在课程之外带来增益 | NOT_SUPPORTED（当前 pilot） | C−B 差 +0.0036，配对 CI `[−0.0088,+0.0162]`；未见收敛提前 |
| 当前 TaskStudent 满足所需公共相位/频移等变性 | NOT_SUPPORTED（当前 pilot） | G6 phase agreement 0.943–0.950；最大 frequency-shift residual p95 73.4 Hz |
| 一次公共相位增强可恢复 TaskStudent 跨条件相位一致性 | NOT_SUPPORTED（单 seed 探索诊断） | 后验分层64例/组：冻结学生 agreement 0.916–0.925，S1 0.909；S1还降低Pd与joint success |
| T1可同时稳定整体幅度尺度与全部条件Pfa | NOT_SUPPORTED（R4 mixture） | H0幅度四分位Pfa 0.046–0.052；C3子组Pfa 0.178–0.217 |
| 固定公共相位不变的28通道相干特征可改善该 TaskStudent 的困难条件任务 | NOT_SUPPORTED（当前 E003 候选） | S2 phase agreement 1.0，但 C2/C3 joint-success 相对 S0 差−0.2593，区间完全低于0；C3 Pfa仍超门 |
| 延迟 `{1,4,16}` 相对 lag-1 可提供额外可学习 Doppler 任务收益 | INCONCLUSIVE | 数学响应与频率敏感性通过；没有 lag-1-only matched student 对照，S2 Doppler RMSE在 C2/C3 恶化 |
| E003 的 C2/C3 任务退化主要来自不可辨识性 | INCONCLUSIVE | R2 在有限固定条件下特征增量非退化，但不能代表完整 C3 mixture 或排除28→6投影/优化影响 |

## 当前主要问题

1. D2-A-R3 的 T1/T2 与 D2-A-T001 的 ACE×AMF 均未通过独立 H0 分层固定阈值门；E003 的 S0/S2 学生 C3 Pfa 也明显超限。
2. E003 S2 证实公共相位不变性可做到接近数值精度，但未保护 C1–C3 估计和联合任务；固定前端、归一化、投影与优化的贡献仍未拆分。
3. 单 seed 学生开发结果不能替代跨训练 seed 或盲测；传统教师的完整任务比较与蒸馏收益也未由 E003 评价。
4. 真实 GMTI、系统级虚警率 `1e-6` 和完整链成本均未验证。

## 当前唯一优先任务

推进D2-A-R6-D实施与正确性门；R6-A/B/C已冻结的protocol、结果、负结果均不改，R6-C不追加训练或晋级多seed。R6-D新protocol可以在公平条件下采用更充分训练：所有D/P/J臂共享预注册updates上限、有效batch、训练数据/样本暴露、优化与checkpoint规则；若有充分理由改变batch，则作为匹配因素明示，不混淆训练预算效应。训练history继续按固定验证间隔记录，不逐update写日志。ChatGPT综合审查只在R6 P0–P7全部授权目标完成后提交；中间审查不作为已授权阶段的默认阻塞。

详细阶段门见 [`D2-A-E001_review.md`](stage_reviews/D2-A-E001_review.md)、[`D2-A-R3_review.md`](stage_reviews/D2-A-R3_review.md)、[`D2-A-T001_review.md`](stage_reviews/D2-A-T001_review.md) 和 [`D2-A-E003_review.md`](stage_reviews/D2-A-E003_review.md)。本结果不触发多目标、通道误差或加速度扩展，也不构成网络优于传统方法的证据。

## 后续依赖和阶段决策

- **D2：** 以单因素协议逐步引入可辨识的杂波/弱目标条件；每次扩展另设 gate。
- **D3：** 依赖可靠下游任务，再比较距离压缩前端与原始三通道回波。
- **D4：** 只有任务质量可靠后才测量结构、算子和计算优化。
- **D5：** 同硬件、同输入边界、完整前后处理链；未满足样本量时不声称 `Pfa=1e-6`。

任何新盲测都在模型、数据生成器、阈值、统计门、预算和候选集冻结后生成。新路线不能覆盖旧结果；旧阶段门也不妨碍提出单独登记的新假设。
