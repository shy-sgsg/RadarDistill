# RadarDistill 动态研究路线图

**更新时间：** 2026-10-28
**权威关系：** 长期使命见 [`RESEARCH_CHARTER.md`](RESEARCH_CHARTER.md)；简要状态见 [`CURRENT_STATUS.md`](CURRENT_STATUS.md)；历史指标和冻结门以对应 protocol、metrics 与阶段审查为准。

## 当前阶段

**D2-A-E001、R3、E002 v2.1 pilot、T001、R4 与 E003 均已完成。** R3/E002/R4 当前为 REFINE；T001 教师候选 STOP。E003 固定相干特征 R1/R2 前置门通过，但单 seed S2 的任务保护和 C2/C3 joint-gain 门失败，故停止该 S2 候选；这不代表充分训练或所有相干架构上限。按用户授权进行的 **R5：学生训练充分性、容量、架构适用性与教师知识迁移机制联合研究** 中，R5-A Phase 1/2 已完成；旧阶段结论保持不变。

R5 顺序为：Phase 0 设计/总纲 v1.3/协议预注册；Phase 1 seed 101 下 S0/S2 6,400 updates并依预设曲线延长到12,800；Phase 2 Small/Medium width-only容量对照；Phase 3 一个原始IQ输入的轻量时频注意力pilot并完成物理展开设计审查；Phase 4 独立审计教师知识质量；Phase 5 仅在知识质量门通过后比较truth-only、完整响应、教师局部知识和匹配真值辅助；Phase 6 只研究有pilot收益的知识内容的时机和物理吸收机制；Phase 7 综合阶段审查。R5-A容量 pilot 显示 Medium 有 C2/C3 joint 与参数估计收益，但两模型 C3 H0 Pfa 均超过0.10保护门，故 Large 停止；Phase 3 在既有授权内继续单个有界候选。任一分支按 G0–G7 失败即停止，不以完成所有架构/蒸馏臂为目标。

**D2-A-T001 v1.2 已完成，教师候选 STOP。** CUDA covariance 与 ACE×AMF 候选使用 RTX 3050 `cuda:0`；4096 H0 covariance、15k H0 calibration、15k H0 verification、12k paired development 全部完成。总体验证 Pfa 为0.05007（exact单侧95%上限0.05309），但 C3/SNR0/CNR20 三个大分组的上限为0.125–0.127；主困难条件 ACE×AMF joint success为0，低于B3的0.01。A/B/C pooled Pd与joint success分别较T1低0.2267和0.2441，95%区间均完全低于0。G1因公共尺度误差4.58e−5超过冻结1e−6失败；G2失败，G3按协议未评估，S2 pilot未运行。运行代码提交 `91dfb81673e45cad3f3330d90adc707a0cdf859d`，全阶段约59分钟。结果见 T001 分析、指标与阶段审查；该负结果不改变 R3、E002 或总纲。

## 已完成实验与证据

### D2-A-R5（进行中）

- 设计依据、路线和用户授权见 `docs/superpowers/specs/2026-10-27-d2-a-r5-design.md`、执行计划 `docs/superpowers/plans/2026-10-27-d2-a-r5-implementation-plan.md` 与 `docs/task_alignments/D2-A-R5_alignment.md`。
- R5-A 使用 E003 冻结 train/validation IDs，seed101、E1 truth-only、RTX 3050 `cuda:0` FP32。S0/S2各训练12,800 updates（8遍）；两臂均在6,400步满足预注册延长条件。复用的E003开发材料不作盲测。验证任务指标持续改善；H-CAP-001在该单seed范围内得到支持，但尚不能声称训练收敛或跨seed稳定。
- S0原始复IQ学生的开发任务结果优于S2固定相干前端；S2的C2/C3联合成功差为−0.3945（95% CI `[−0.4189,−0.3696]`），C3 H0 Pfa仍为0.1838，故停止当前S2候选。S0的C3 H0 Pfa为0.2041，也未达到0.10保护门。完整结果见 `experiments/D2-A-R5/R5-A/analysis.md` 和 `docs/stage_reviews/D2-A-R5A_review.md`。
- R5-A容量对照中，Medium−Small C2/C3 joint-success差为`+0.13086`（95% CI `[+0.11230,+0.15088]`），range/Doppler RMSE比分别`0.839/0.713`；但C3 H0 Pfa为`0.2041/0.1890`，G3 absolute protection失败，因此不运行Large。结果见容量分析和阶段审查。
- 当前唯一优先任务为 R5-B：针对现有 GMTI 条件完成非 CNN 适用性调查，只实现一个原始复 IQ 轻量时频注意力候选。先通过结构/参数/显存、物理正确性及小批可学习性审查，再比较冻结 CNN 参照。深度展开当前仅作物理设计审查；R5-C/D 均受教师质量和教学收益前置门约束。
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

独立审查 D2-A-E003 结果，判断是否值得提出一个不训练的表示归因诊断，以区分固定相干特征、场景归一化与28→6投影对 C1–C3 任务表现的影响。审查前不启动 S2 正式多种子、蒸馏、final blind 或教师搜索。若审查支持新诊断，须重新对齐研究总纲、登记假设并冻结独立协议与停止条件；总纲保持不变。

详细阶段门见 [`D2-A-E001_review.md`](stage_reviews/D2-A-E001_review.md)、[`D2-A-R3_review.md`](stage_reviews/D2-A-R3_review.md)、[`D2-A-T001_review.md`](stage_reviews/D2-A-T001_review.md) 和 [`D2-A-E003_review.md`](stage_reviews/D2-A-E003_review.md)。本结果不触发多目标、通道误差或加速度扩展，也不构成网络优于传统方法的证据。

## 后续依赖和阶段决策

- **D2：** 以单因素协议逐步引入可辨识的杂波/弱目标条件；每次扩展另设 gate。
- **D3：** 依赖可靠下游任务，再比较距离压缩前端与原始三通道回波。
- **D4：** 只有任务质量可靠后才测量结构、算子和计算优化。
- **D5：** 同硬件、同输入边界、完整前后处理链；未满足样本量时不声称 `Pfa=1e-6`。

任何新盲测都在模型、数据生成器、阈值、统计门、预算和候选集冻结后生成。新路线不能覆盖旧结果；旧阶段门也不妨碍提出单独登记的新假设。
