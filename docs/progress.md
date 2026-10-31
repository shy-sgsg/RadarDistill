# 研究进度

# 2026-10-31：D2-A-R6-C 完成，进入 R6-D 实现

- **执行：** 按冻结R6-C v1.2，在RTX 3050 Laptop `cuda:0`、FP32、seed404下顺序训练Medium/Large/XLarge各12,800 updates；有效batch64，history每400 updates记录，不逐update写盘。另完成Medium 51,200 vs102,400唯一训练对的固定12,800-update对照。每模型评价使用H0 calibration/verification各120,000及H1 development6,144场景，未生成final blind。
- **容量结果：** global Pfa 0.05时C2+C3 joint为0.8164/0.8628/0.8704。Large−Medium=`+0.04644`（97.5% CI `[+0.03646,+0.05643]`），XLarge−Medium=`+0.05404`（`[+0.04449,+0.06402]`）；相对任务保护通过，但C3 pooled Pfa单侧95%上界0.10162/0.10207/0.10266、六格familywise门失败，未晋级formal seeds。
- **训练与数据规模：** Medium与validation选出的XLarge未共同通过预注册续训趋势/过拟合门，未延至25,600/51,200；不据此断言12,800为学习极限。数据扩展组相对core的joint差`+0.00673`（95% CI `[-0.00066,+0.01455]`），未通过主增益门；Doppler RMSE有小幅次级改善，C3上界为0.10412。
- **工程/审计：** 首次预测汇总因布尔数组直接相减失败；在两组布尔向量转换为float64后修复，复用12/12已有预测归档，无重复推理；定向回归用例通过，R6-C测试文件12项通过。按owner记录移除21,467,222,779 bytes重建缓存；清理后项目占用1,205,698,287 bytes。审计见`experiments/D2-A-R6/R6-C/attempts/evaluation_aggregation_fix.json`与`cache_cleanup.json`。
- **判断与路线：** R6-C为REFINE，limited support宽度扩展的相对joint收益，但绝对C3虚警保护失败；数据规模主假设未获支持；更长更新的效果仍未检验。用户允许未来新实验注册更充分更新，须匹配D/P/J各组updates、effective batch、样本暴露、schedule和checkpoint规则，且仍按检查点记录、不逐update日志。当前唯一优先工作转为R6-D J1/J2实现与CUDA正确性门；综合ChatGPT审查待R6 P0–P7全部完成。
- **证据：** `experiments/D2-A-R6/R6-C/{protocol.md,config.yaml,metrics.json,analysis.md,analysis_record.json,cache_cleanup.json}`；内部报告`docs/stage_reviews/D2-A-R6C_review.md`。运行/预测生成SHA=`a34feea10822ac4ef4dfbe9b64771826aa1e80df`。

# 2026-10-29：按用户要求将综合ChatGPT审查放在R6路线收尾，冻结R6-C协议

- **审查时机：** 用户要求完成整项R6研究目标后再形成一份ChatGPT审查请求；R6内部每个阶段仍保留协议、结果与阶段报告，但不分别提前提交审查提示词。未收到中间审查不阻塞已经授权且通过冻结前置门的阶段。
- **R6-B边界：** 已冻结hard-negative匹配结果、有效batch64及12,800 updates均保持不变；R6-B不追加训练或晋级多seed。调整的是外部审查请求时机，不是科学结论或历史协议。
- **R6-C协议：** v1.2已在数据生成前冻结，比较TaskStudent width96/200/376（0.07M/0.29M/1.01M活动参数）、seed404、effective batch64、E1真值监督及12,800初始updates；Medium与validation筛选的一个大模型共同通过延长门时，成对逐级延至25,600、最多51,200。条件块seed布局已明确，以保证51,200对严格嵌套于102,400对且不与validation重叠。另有Medium固定12,800-update的数据量对照。训练history每400步记录，恢复状态每800步保存，不逐update写盘。
- **前置状态：** 协议/配置与新假设已登记，未生成R6-C场景、未开始训练、无实验结果。下一步是实现、G0/G1测试及实际GPU显存门；模型规模、有效batch、训练暴露和训练长度按协议匹配或分开比较。
- **研究范围：** 服务总纲v1.3 Q3/Q4/Q6与GMTI任务级学生目标；不修改研究总纲，不将训练预算假设用于解释R6-B hard-negative负结果。协议/配置冻结后，完成前置实现检查，再开展有界CUDA筛选。

# 2026-10-29：D2-A-R6-B hard-negative matched pilot 完成

- **研究主线：** 本阶段检验训练期 H0 hard-negative 重采样是否能帮助任务级 GMTI 学生改善困难格检测与最终联合任务；不是独立 CFAR 扩展，也没有加入教师或蒸馏。研究总纲 v1.3、冻结 R6-B protocol/config 和 R6-A/R5 历史结果均未改。
- **运行版本：** R6-B branch R；训练执行代码 SHA `091e96b9d570bdaa16444c1e696a633145c6655d`，独立评价 SHA `293f1733b417d7fdc65fd9e294c8511cef43613c`。training/evaluation 修正和尝试记录保留于各自 commit/provenance 与 `experiments/D2-A-R6/R6-B/attempts/`；没有改实验预算或数据划分。
- **训练：** seed303 truth-only 与 hard-negative Medium width96（82,887训练参数）各12,800 optimizer updates；FP32、RTX 3050 Laptop `cuda:0`、microbatch32、有效batch64，顺序训练、无CPU fallback与OOM。两臂均选择update12,000验证checkpoint。耗时3176.7/3278.4秒，峰值allocated1.6455/1.6461 GB，reserved1.7134 GB。历史按400步检查点写入，不逐更新落盘。
- **验证趋势：** Truth-only/HardNegative validation task loss在update12,000为0.22854/0.23116，到12,800为0.22989/0.23230；因此不能说本pilot在预算终点仍持续改善，也不能由单seed与有限检查点确定训练上限。R5-A长预算结果说明训练预算值得继续单独研究。用户允许后续新实验公平延长更新并探索至多两倍batch；任何新比较须对所有匹配组统一有效batch、optimizer updates、数据暴露及checkpoint规则。本轮12,800/effective batch64保持冻结。
- **数据与评价：** 独立 H0 calibration、verification各120,000，H1 development 6,144；训练/验证51,200/5,000对。评价流式执行、未落盘IQ。全局Pfa=0.05时TruthOnly/HardNegative Pfa为0.05018/0.05003，单侧exact95%上界0.05123/0.05108；C3上界0.10192/0.10180，困难格上界0.10895/0.10722。故0.05及0.01全局工作点未过资格；0.001仅全局门通过，C3及困难格未通过。
- **任务结果：** H1全局Pd为0.93620/0.93473，joint为0.85042/0.84831。配对 HardNegative−TruthOnly global joint差−0.00212（95% CI `[-0.00651,+0.00195]`），C3差−0.00260（`[-0.01107,+0.00553]`），困难格差−0.00391（`[-0.01563,+0.00586]`）；区间跨0，不构成收益或等价证据。全H1 direct range/Doppler RMSE为2.707/23.448与2.678/23.323，变化很小。当前不将此分支升级为多seed、KD或blind。
- **决定：** 阶段 **REFINE**。R6-A表明预选困难格本身区分能力低；R6-B training-H0重采样未证实能提高H1最终任务，也未解决低Pfa条件保护。hard-negative按本协议停止扩展；先请求ChatGPT独立审查，再决定是否存在独立的训练预算诊断价值。唯一下一项优先任务是该阶段审查。
- **工程核验：** 最新应用相关完整测试记录为 `AISAR pytest -q` 467 passed（代码修正提交955cdf4）；`git diff --check` 在归档准备中执行。训练缓存清理仅涉及本阶段owner路径，共11,233,520,150 bytes；项目由26,524,167,231降至15,290,647,081 bytes（约14.24 GiB），低于20 GiB目标。保留检查点/history、metrics、预测、scene manifests、环境、审计与分析；未触碰未跟踪用户文件。未生成final blind。
- **审查材料：** `experiments/D2-A-R6/R6-B/review_request.md`，含本阶段执行SHA、结果、研究主线影响、阶段审查与下一步问题。

# 2026-10-29：R6-A 结果归档与 R6-B 分支选择完成

- **研究主线：** 完成了总纲v1.3 Q1/Q3/Q6/Q8/Q10所需的同IQ检测与估计基线，分开评估固定阈值、条件可分性、实际候选参数误差和离线oracle。未修改研究总纲、R5历史结果或冻结实验设置。
- **R6-A数据与执行：** 预测运行SHA `e3972ae94430abd3d79eae7e1ee9179f68062648`；12,000 H0 calibration、12,000 H0 verification、6,144 H1 development，共30,144场景和211,008方法行。AISAR/PyTorch 2.8.0+cu128、CUDA 12.8、RTX3050 Laptop `cuda:0`；冻结Medium仅推理、无训练，峰值allocated129,681,408 bytes。G0为`PASS_WITH_RECORDED_PROTOCOL_ORDER_DEVIATION`；详细偏差见任务对齐和R6-A分析。
- **主要结果：** 全局Pfa=0.05时Medium AUC/Pd/joint/Doppler RMSE为0.9830/0.9334/0.8306/24.68 Hz；但C3 partial-overlap/SNR0/CNR20的AUC/Pd/joint为0.5956/0.2207/0.0391，条件Pfa=0.097。全局Pfa单侧exact95%上界0.04996不能代表困难条件可靠性。R3 T1/T2 joint优于B3但总体仍低于Medium。
- **R6-B选支：** 仅读取R6-A H0 calibration与H1 development，未读取H0 verification。C3 pooled AUC=`0.93164`（95%CI `[0.92440,0.93871]`）；困难格AUC=`0.60324`（`[0.57085,0.63295]`），95%上界低于预注册0.70门；C3 calibration cell-Pfa range=`0.033`，低于S分支所需0.05。已冻结选中R分支：matched truth-only vs training-H0 hard-negative-resampled Medium。分支 JSON提交`b3ad8a5`。
- **审计修正：** 修复oracle距离门floor→nearest-bin映射；仅同场景重建offline oracle sidecar，非oracle指标不变。另修正分条件joint图读取错误，数值表无误。预更正指标/图仍保留于审计目录，未覆盖。整仓测试456项通过，选支CLI在AISAR成功运行，代码提交`58ba8bd`。
- **资源及公平性：** 项目当前约15GiB，分区可用磁盘约29GiB；R6-B预估11.22GB训练缓存可在本阶段临时使用，收尾后按本阶段生成清单清理。冻结R6-B有效batch64、12,800 updates、同初始化和训练样本计划保持不变；扩大batch会改变已注册对照。新实验可将更大batch作为 matched factor。
- **决定与下一步：** R6-A为开发基线，不是blind或部署资格；总体路线 **REFINE**。唯一优先任务是执行已预注册R6-B R分支，保持冻结batch64与其他训练条件，运行独立校准/验证/开发并应用任务保护门。不开展final blind或额外分支搜索。

# 2026-10-29：R6-A 运行过程与更正审计（历史记录）

- 已实现同 IQ 的 B0–B3、R3 T1/T2 与冻结 Medium 输出，实际候选和离线 oracle 距离门分开，加入频率估计、独立阈值统计、分层汇总、分块断点恢复和资源记录。AISAR定向测试19项通过；完整测试套件452项通过，runner `--help` 与Python编译通过。当前尚未生成R6-A汇总指标，不能报告科学结论。
- RTX 3050 上 B3/T1/T2 complex128 CUDA 实现与现有 CPU 参考在16个独立工程随机 CPI 上最大相对分数差小于 `1.8e-15`，候选一致；单次运行 CUDA/CPU 分别约 `0.031/0.138 s`，CUDA 峰值分配约126 MB，仅用于工程资源诊断。包含学生、所有检测器与候选估计的16场景工程批次生成/预测分别约 `0.242/1.109 s`，不构成系统时延结论。
- 保留R6-A/B冻结训练batch；R6-A `chunk_size=16` 是评估流式分块，不是训练batch。未来新训练实验可在公平的匹配设计中预注册至多两倍batch比较；若改变有效batch或训练暴露会破坏比较，则保持原值。R6-A bootstrap补充固定seed `2026102901`、2,000次，正式数据生成前完成。
- 前置接口测试曾使用两个预注册种子各生成一个场景并重复执行，只检查形状和 ID，未查看样本数值或方法结果；这偏离 G0 前不生成实验场景的协议顺序。记录与处置见 `docs/task_alignments/D2-A-R6_alignment.md`，测试现使用独立工程种子。
- 首次正式runner调用在生成器调用前因遗漏 `iter_scene_chunks` 导入而停止；核验检查点为0 chunks、0 rows，CSV只有表头，未生成或查看注册场景。故障与原preflight及空输出已归档至 `experiments/D2-A-R6/R6-A/attempts/pre_data_launch_3f7826f/`；修复增加runner导入回归测试，随后G0通过并开始实验。
- 按G0冻结的执行SHA `e3972ae94430abd3d79eae7e1ee9179f68062648` 已生成全部30,144个注册场景预测：H0 calibration 12,000、H0 verification 12,000、H1 development 6,144；共211,008方法行，未保存IQ，final blind未生成。三split CUDA峰值分配均为129,681,408 bytes。H0两split的完整处理时间约350.44/353.37秒；H1含候选/离线估计约216.71秒。所有分块状态完整。
- 首次汇总在全部预测写完后因runner遗漏 `_bootstrap_mean_interval` 导入而停止；尚未完成指标计算、查看或筛选科研结果。回归测试先失败后通过。当前为已冻结预测文件补齐汇总入口，指标将记录预测运行SHA `e3972ae…` 与独立的汇总代码SHA，并按原数据复算，不重新生成场景。
- **后续复核更正（2026-10-29）：** 汇总代码修复后生成并初步查看了预更正统计，发现离线 oracle 将连续距离 `floor` 到整数门；这与仿真器 `sinc(truth_range_cell - range_axis)` 的距离响应不符。预更正指标和图仅供审计，尚未据此选择方法；保留后先按最近可用距离中心规则重算 oracle-only 频率列及其汇总。实际 detector 输出、阈值、H1场景、实际候选估计不受影响。细节及数据边界见 R6-A protocol 第7节与下方任务记录。
- **图表复核更正（2026-10-29）：** 视觉检查发现分条件 joint 图曾重复全局 joint 值，且横轴不能区分相同C3场景的0/10 dB两格。指标表中的分条件数值无误；绘图现改为读取对应条件结果并标注条件编号/SNR，测试并重新生成，旧图保留在同一审计目录。

# 2026-10-28：R5-C T0–T3教学内容pilot完成并停止当前教师内容分支

- **对齐与协议：** R5-C独立审查以教师审计结果提交`7563c822e70afe761897ddd5e91bb64207720f89`为基准，裁决REFINE并支持一次有保护门的T0–T3单种子pilot。冻结协议v1.0和配置未改；本次未修改研究总纲v1.3或历史R5-C G4结论。
- **数据与门：** 代码运行SHA为`a97ad36877d04dff8d2c8499e4bf783e0435714c`。G0/G1通过；训练/验证新配对场景51,200/5,000，pair plan`[12,800,32]`并固定遍历8次；与检查的56,200个R5-A历史ID重叠为0。独立评价为H0校准8,192、总体H0验证8,192、C3 H0验证4,096、H1 development 4,352，场景分割不重叠。
- **训练：** RTX 3050 Laptop `cuda:0`、FP32、seed303、有效batch64、microbatch16，T0/T1/T2/T3各12,800 updates顺序运行，无OOM或CPU回退。四臂训练参数120,717、推理参数70,277；训练耗时分别6,745.77/4,446.82/4,295.96/3,781.71秒，峰值allocated约821 MB。T0为0.527秒/update，较R5-A Medium记录的0.307秒/update慢约1.72倍；已确认T0跳过520-bin头前向，但现有计时不足以确定减速的唯一原因。训练期间每步写日志开销未做独立对照；按用户要求后续新实验默认只在验证/检查点记录。
- **Batch设置：** 本pilot及已预注册R6-A/B保持原batch。用户允许后续全新训练实验在显存允许时评估至多两倍batch；前提是匹配对照不受影响，必要时把batch作为独立匹配因素。
- **主要结果：** T1−T0与T2−T0 C2+C3 joint差为−0.01042（95% CI`[−0.01849,−0.00208]`）与−0.01380（`[−0.02292,−0.00391]`），均未通过+0.02主增益门；T1/T2未通过综合任务保护。四臂C3 H0 Pfa为0.182–0.203，均超0.10绝对门。T3 truth-assisted对照差+0.01563，仅作描述性结果，不能归因于教师。新C3 partial-overlap/SNR0/CNR20子组K3定位正确24/512（4.69%）。完整分层与统计区间见`experiments/D2-A-R5/R5-C-teaching/metrics.json`、`analysis.md`及`docs/stage_reviews/D2-A-R5C-teaching_pilot_review.md`。
- **研究决定：** 按冻结协议停止当前K3/K2-local教学内容分支，不启动其多seed、教师时机/可靠性搜索、完整R5-D机制实验或final blind。H-TEACH-002在本pilot范围内记为NOT_SUPPORTED。总体方向REFINE；下一项唯一优先任务为R6-A同场景检测/参数估计基线。结果提交SHA为`896e7de4a20c8c0f96a8eb28760617423a2efc95`；ChatGPT审查请求见`experiments/D2-A-R5/R5-C-teaching/review_request.md`。
- **资源清理：** 只删除manifest列出的四个本阶段IQ memmap，共11,222,016,512 bytes；保留场景/seed清单、truth metadata、pair plan、预测、训练记录及其他未跟踪用户文件。清理后项目15,160,472,696 bytes（约14.12 GiB），低于20 GiB限制；记录在`cleanup.json`。

# 2026-10-28：批准并冻结D2-A-R6分阶段研究设计

- **主线：** R6服务总纲v1.3的Q1–Q10，重点用同场景证据拆分C3固定阈值迁移、条件内可分性、候选参数误差与学生/教师收益；不把传统检测统计量扩展成独立研究目标。
- **路线：** R5-C T0–T3仍为当前P0且设置不变；随后R6-A同场景基线、R6-B独立低Pfa诊断，再依据证据推进容量/训练/数据规模、联合D/P/J模型、代表性架构、混合处理链和有界教师协作。R6不等待R5-D机制研究，但GPU训练与推理避免和R5-C训练并发。
- **方案：** 新增R6总设计/执行计划、R6-A基线协议、R6-B低Pfa协议、R6-D联合模型与正确性设计，以及`docs/task_alignments/D2-A-R6_alignment.md`。R6-A/B使用独立scene/seed blocks；R6-D只含设计门，具体科学pilot待前置结果和独立配置冻结。
- **总纲与证据边界：** 审阅Charter v1.3后判断当前研究范围已覆盖这些问题，不修改总纲。以上是设计，不是R6运行结果；没有生成R6数据、训练模型或final blind。


# 2026-10-28：R5-C 数据生成与G1通过，四臂CUDA训练开始

- **数据核验：** G0预检记录绑定提交`a97ad36877d04dff8d2c8499e4bf783e0435714c`；新训练/验证对分别为51,200/5,000，ID唯一且互斥，与56,200个R5-A历史train/validation IDs重叠为0。固定pair plan为`[12,800,32]`、8遍同序访问。CPU仿真和缓存生成714.344秒，数据记录为11,253,944,449 bytes；评价场景仍通过独立seed block流式生成，没有生成blind。
- **G1：** RTX 3050 Laptop `cuda:0`，FP32/microbatch16，四臂16-pair/200-update overfit均通过。T0–T3 truth-task loss最终/初始比为0.005499/0.003440/0.006211/0.017269；presence与H1 range-cell accuracy均为1.0。T1–T3辅助头初始梯度范数49.50/49.98/50.05，T0为0；G1不计作科研性能结果。
- **当前运行：** 按冻结顺序开始T0–T3各12,800 updates；模型顺序驻留GPU。现有检查点/恢复状态在各arm目录写入，R6不占GPU。一次早先G1调用因我提前编辑被冻结的CURRENT_STATUS触发runner拒绝，在执行任何模型步骤前退出；恢复我所改的三个冻结动态文件后，正式G1通过。配置、main SHA和R5-C冻结protocol均未改动。

# 2026-10-28：R5-C teaching pilot 获 REFINE 审查并完成预注册实现

- **审查依据：** 独立审查以 R5-C 结果提交 `7563c822e70afe761897ddd5e91bb64207720f89` 和研究总纲 v1.3 为基准，裁决 **REFINE**，有条件放行一次 T0–T3 单种子教学内容pilot。审查保留原R5-C G4 PASS，同时明确 C3 partial-overlap / SNR 0 dB / CNR 20 dB 教师定位率约4.1%不足以支持全域可靠教学。
- **冻结方案：** 新 protocol/config v1.0 比较 T0 truth-only、T1 truth+K3完整响应、T2 truth+K2-local、T3匹配truth-assisted局部响应。四臂共用Medium width=96学生、初始化、全新训练/验证数据与顺序、truth主损失、有效batch64和12,800 updates；T1–T3共享`[520,32]`响应表示及固定辅助损失。T3是辅助监督控制，不是教师。保留教师错误分层与学生C3 H0 Pfa≤0.10晋级门。
- **执行边界：** 新种子和配对数据计划、CUDA-only在线教师路径、G0/G1前置门、训练/评价代码及定向测试已实现；本次冻结提交推送前复跑测试。实验数据尚未生成，G0、G1和训练尚未运行。G0通过才生成新数据，G1全部通过才运行四臂训练；不提前启动可靠性/时机搜索、多seed、final blind或完整R5-D机制实验。
- **主线影响：** 本pilot直接检验R5-C软知识是否在物理真值之外改善任务级学生，回应总纲Q1/Q2/Q10；不会单独证明知识吸收机制或改变研究总纲。R5-A/B/C历史结果与冻结门不变。

# 2026-10-28：D2-A-R5-D Doppler 与通道相干输入变换工具准备

- **缺口与依据：** 频移结果汇总已有预期 `fd→fd+Δf` 映射，但需要目标分量单独调制的输入；通道相干诊断还需跨 H0/H1 和目标/杂波/噪声分量一致的相位扰动与遮蔽。R5-D 要求区分可检验的 Doppler 响应、通道信息依赖与物理上不合法的反事实。
- **实现：** 在 `d2a_physical_interventions.py` 增加目标慢时间相位旋进、逐通道固定相位扰动/逆变换和通道遮蔽；相位扰动/遮蔽同步作用于所有 IQ 分量并重建 H0/H1，Doppler truth 按 PRF 回绕。遮蔽不可从零值数据反恢复，需从未遮蔽原件重新构造。
- **边界：** Doppler 变换不重建轨迹或更新径向速度；通道相位与遮蔽是受控输入干预，不是几何物理场景仿真。未选择正式变换幅度、未运行模型评价、机制实验、训练或蒸馏。R5-C 独立审查与 matched T0–T3 前门保持不变。
- **验证：** 新增频率调制/混叠、真值回绕、跨通道相干乘积、相位逆变换和通道遮蔽测试；R5-D 定向测试 **13 passed**，AISAR 整仓回归 **420 passed**。
- **任务对齐：** `docs/task_alignments/D2-A-R5D-diagnostic-tools.md`；定义和解释边界见 `docs/research/D2-A_knowledge_transfer_mechanisms.md`。

# 2026-10-28：D2-A-R5-D 目标/杂波强度输入干预工具准备

- **缺口与依据：** D2-A 仿真器已保留同一配对场景的目标、杂波和噪声 IQ 分量，但缺少保持 H0/H1 分量关系且可单独改变目标或杂波强度的诊断输入构造器。R5-D 明确要求分别干预目标与杂波并同时检查检测、参数和虚警，不得把功率捷径当作学习机制。
- **实现：** 新增 `scale_d2a_paired_components`，以 dB 表示组件功率变化、以 `10 ** (delta_db / 20)` 缩放 IQ 幅度。目标只改变 H1 目标分量并同步目标幅度真值；杂波在 H0/H1 同步变化；噪声与 scene realization 固定。输出记录每场景分量功率及 SNR/CNR/SCNR。
- **边界：** 这是已有随机实现上的组件级反事实，不重抽场景、不改冻结仿真器、不代表真实硬件幅度变化；没有计算检测器/学生预测、没有实验比较，也未选正式干预幅度。R5-C 独立审查和匹配 T0–T3 pilot 前置门保持不变。
- **验证：** `tests/test_d2a_physical_interventions.py` 7 项通过，覆盖零移位恒等、目标单独变化、杂波单独变化、功率/真值关系、输入恒等式与非法尺度；AISAR 环境下整仓回归 **414 passed**。普通默认 Python 因未安装 PyTorch 在收集阶段报依赖缺失，随后使用项目环境完整通过。
- **任务对齐：** `docs/task_alignments/D2-A-R5D-diagnostic-tools.md`；使用范围与物理解释边界见 `docs/research/D2-A_knowledge_transfer_mechanisms.md`。

# 2026-10-28：D2-A-R5-D 物理干预配对诊断接口准备

- **缺口：** 现有通用干预汇总器用同一真值评价基线和干预输出，不能正确计算 `fd→fd+Δf` 的预测等变残差；它也不应要求 H0 有参数真值。
- **实现：** 增加可选预期 Doppler 移位映射与频率响应残差；相位/幅度/通道/频率配对变化在提供场景ID时均按 `base_scene_id` 做区间估计。H0 参数字段允许 NaN，并仅参与检测统计。没有生成新 IQ、读取 R5 缓存/权重或运行物理干预。
- **状态与边界：** 这是后续 R5-D 使用的评价代码准备，不是实验协议或模型结果。R5-C 独立方向审查和匹配 T0–T3 pilot 门保持不变，正式机制比较仍未启动。
- **任务对齐：** `docs/task_alignments/D2-A-R5D-diagnostic-tools.md`；机制评估用法见 `docs/research/D2-A_knowledge_transfer_mechanisms.md`。

# 2026-10-28：D2-A-R5-D 配对错误诊断工具准备

- **主线与边界：** 服务总纲 v1.3 的 Q1/Q2/Q4/Q9/Q10 与 H-TEACH-002/H-TEACH-004/H-LEARN-002；仅建设后续机制研究的评估工具。R5-C独立方向审查仍待返回；没有运行preflight、训练、KD或新数据实验。
- **实现：** 扩展现有 `physical_consistency.py`，按H0/H1、condition和任务分别计算教师/学生四格正确性、学生纠错/退化比例及场景bootstrap区间。周期Doppler使用PRF回绕；range/Doppler正确性与detection分开，joint则包含三项门。配对H0/H1共享的`base_scene_id`作为bootstrap簇。
- **范围限制：** 当前没有同一批scene上的R5-C教师与T0–T3学生输出，故没有错误矩阵科研结果，也不从错误相关性推断因果知识吸收。若独立审查不支持pilot，不启动学生侧R5-D比较。
- **任务对齐：** `docs/task_alignments/D2-A-R5D-diagnostic-tools.md`；研究设计说明更新于 `docs/research/D2-A_knowledge_transfer_mechanisms.md`。这一准备不改变当前R5-C审查优先级或任何冻结假设状态。

# 2026-10-28：D2-A-R5-C 教师软知识质量审计完成

- **总纲对齐：** 按 `docs/RESEARCH_CHARTER.md` v1.3，检验B3教师响应中的目标定位/Doppler信息与可观测可靠性信号，服务任务级GMTI学生和后续有界知识迁移判断；未修改研究总纲、R5-B冻结门或既有负结果。
- **协议与数据：** 按 `experiments/D2-A-R5/R5-C/protocol.md` / `config.yaml` v1.1，在3通道、PRF=1300 Hz、N=130、32距离单元静止杂波条件下，对4,352个新配对base scenes比较K1原生全RD、K3 4×全RD和K2-local，共13,056条候选记录。H0/H1共享背景噪声；教师仅接收IQ与冻结物理配置；场景按16 CPI流式处理，未缓存IQ或全量教师图。
- **环境/执行：** AISAR Python 3.13.7、PyTorch 2.8.0+cu128、CUDA 12.8、RTX 3050 Laptop `cuda:0`；教师complex128/FP64，无CPU回退。CPU场景生成41.83秒，K1/K3 CUDA阶段1.38/1.58秒，峰值allocated/reserved 23.43/44.04 MB，进程峰值RSS 859 MB；这不是完整端到端/实时成本。项目占用14.054 GiB，低于20 GiB限制。
- **主要结果：** 可辨识C2+C3中K1/K3/K2-local localization分别0.7810/0.8122/0.8122；K3−K1为`+0.03125`（95% CI `[+0.02474,+0.03776]`），达到预注册定位增益门。K3相对K1的全图target-window lift差为`−0.000320`（`[−0.000421,−0.000219]`），细网格没有提高全图归一化软质量。K2-local lift为局部条件归一化，不能与K1/K3直接比较，且峰位置继承K3。
- **困难子组与解释：** C3 partial-overlap / 0 dB定位率K1/K3/K2-local为0.021/0.041/0.041；同组置信AUC为0.404/0.326/0.783，K2-local分数未经校准。按冻结联合严重子组规则，三者均通过G4；这不表示该困难子组可靠。未认证固定阈值检测器、未训练学生、未运行KD或final blind。
- **假设/决定：** R5-C支持教师软响应含有可定位目标信息及K3定位相对K1有冻结幅度的提升；不支持学生已吸收知识或蒸馏有效。H-TEACH-002/H-TEACH-004继续PROPOSED，H-LEARN-002继续INCONCLUSIVE；R5-B TFAS限定候选STOP不变。Codex阶段建议为 CONTINUE 但仅限一轮匹配T0–T3教学内容pilot，须保留Medium学生C3 H0 Pfa=`0.18896`的绝对保护。当前先提交独立ChatGPT阶段审查，再按审查意见确定后续。
- **交付：** `experiments/D2-A-R5/R5-C/` protocol/config、preflight/environment、scene IDs、逐场景指标、metrics、analysis和图；`docs/stage_reviews/D2-A-R5C_review.md`；研究状态/路线/假设/决策登记。运行代码SHA `58e422de7a9de1290d38ebbb206862e23b49ab42`；完整报告提交SHA与ChatGPT请求见 `experiments/D2-A-R5/R5-C/review_request.md`。
- **验证：** 定向测试覆盖R5-C审计、CUDA K2、E002 teacher与D2-A baselines；修复代码提交后为34 passed。报告包入库前重新运行定向测试、编译及差异检查。

# 2026-10-28：D2-A-R5-B 架构调查与 TFAS pilot 预注册

- **主线对齐：** 服务于研究总纲 v1.3 的 Q3/Q4/Q6/Q8/Q10，检验原始三通道复 IQ 的时序/距离注意力能否在任务级检测和连续参数估计中胜过预算匹配 CNN；不涉及教师和蒸馏。
- **证据审查：** R5-A Medium 对 C2/C3 joint 与参数误差有单 seed 开发收益，但 Small/Medium 的 C3 H0 Pfa 均超过0.10。雷达Transformer、复数网络、SSM、深度展开和物理前端文献已按体制/输入/任务匹配程度归纳；各自结果不外推为本GMTI收益。
- **冻结设计：** 只选一个约79,638参数的实值 I/Q temporal-frequency attention student：两层130脉冲注意力、每距离门时间池化、一层32距离注意力，使用完整六平面输入；physics unfolding 仅作数学适配审查。
- **预注册：** `experiments/D2-A-R5/R5-B/protocol.md`、`config.yaml` 与 `architecture_survey.md` v1.0；seed101、E1真值、复用冻结数据和R5-A pair plan、12,800 updates、FP32 RTX3050 `cuda:0`，匹配Small/Medium CNN，C3 Pfa绝对门≤0.10；失败即停止扩架构、KD与blind。
- **当前状态：** 仅完成调查/预注册；未实现、未运行 CUDA preflight/G1/训练、无 R5-B 指标或结果。研究总纲未修改。

# 2026-10-27：D2-A-R5-A 训练充分性 pilot 已完成

- **训练：** 按 protocol/config v1.1，在 RTX 3050 Laptop `cuda:0`、FP32 顺序训练 seed101 的 S0/S2。固定51,200对训练数据，microbatch16、有效batch64；每臂先6,400 updates，两臂均满足预注册延长规则后连续训练至12,800 updates。两臂G1小样本过拟合门通过；未用教师/蒸馏，不生成final blind。
- **预算证据：** 同一长程schedule内，S0验证task loss从update1600的0.519降到12800的0.321，range RMSE从2.844降到2.203 cell，Doppler RMSE从35.49降到24.16 Hz；S2也持续改善。支持H-CAP-001但仅限单seed、冻结数据与复用开发集。
- **任务结果：** 复用E003开发集上S0/S2 Pd为0.9619/0.9336、AUC为0.99374/0.98999、joint为0.8389/0.6055；S2−S0 joint差−0.2334（paired scene bootstrap 95% CI `[-0.2488,-0.2188]`），C2/C3差−0.3945（`[-0.4189,-0.3696]`）。C3 H0 Pfa为0.2041/0.1838，均超过0.10；S2未通过任务保护和主要收益门。
- **物理诊断：** S2公共相位旋转几乎不变，但其频率干预和C2/C3任务表现弱；S0任务表现较好但公共相位敏感。物理干预复用E003开发诊断ID，不作为独立blind证据。
- **代码/复现：** 评估器把继承自E003的`G3_task_learning`映射到R5-A的`G1_basic_learning`，并以`G3_architecture_value`单列架构收益；指标计算与冻结阈值不变。定向测试13 passed。修复后评估命令：`PYTHONPATH=src conda run --no-capture-output -n AISAR python scripts/run_d2a_r5.py --phase evaluate`。
- **运行记录限制：** 旧R5-A记录仅保留扩展调用时长，完整12,800-update累计时长未保存；后续代码已添加训练分段和累计墙钟计时。不得把S0 878秒、S2 1,005秒当作全程用时。
- **阶段判断：** R5-A **REFINE**。停止当前固定相干S2候选；不否定原始IQ非CNN路线。H-CAP-002尚未测试。阶段分析、metrics及图表归档后，唯一下一项是已冻结的Small width32/Medium width96单因素容量对照；无容量收益则不运行Large。无KD、正式多seed或final blind。

## 2026-10-27：D2-A-R5 Phase 1 执行器与预注册门实现

- **协议冻结：** R5-A 更新至 protocol/config v1.1，在任何 R5 数据生成或训练前量化续训保护门；除原有末段 validation loss 条件外，检查各任务损失与 RMSE 恶化不超过5%、AUC下降不超过0.01、range-cell mismatch增加不超过0.01。未查看 R5 结果。
- **执行器：** 增加 CUDA-only R5 runner，包含冻结数据重建/分割审计、16-pair G1 overfit、同一配对场景计划、S0/S2 顺序训练、完整优化器/调度/RNG/数据游标恢复、OOM microbatch回退及复用 E003 的独立评价。checkpoint 绑定配置、protocol版本和源 SHA。
- **输出正确性：** paired H0/H1 scene prediction CSV 分别记录两个假设，避免遗漏 paired H0 预测。
- **验证：** AISAR 环境下 `py_compile` 通过，R5-A 定向测试 **10 passed**，R5/E003配置加载有效。当前尚未运行 CUDA preflight、R5 数据生成、G1 overfit 或训练；不存在 R5 实验结果。

## 2026-10-27：批准 D2-A-R5，Phase 0 设计与 R5-A 预注册

- **研究总纲：** 根据用户明确授权将 `docs/RESEARCH_CHARTER.md` 由 v1.2 修订为 v1.3，限于学生训练充分性、模型容量、非 CNN 架构适用性、教师知识内容/时机/可靠性及吸收机制，并新增 Q10。Q1–Q9、主题、既有研究假设边界和冻结负结果保留。今后总纲继续受用户批准机制约束。
- **主线对齐：** D2-A-R5 直接服务任务级 GMTI 学生；E003 1,600 updates 对应固定训练集一遍，不是充分训练上限。先测固定数据训练预算，再测容量/架构，最后仅对经过质量门的知识开展最小KD对照。
- **设计交付：** 增加 R5 design/implementation plan、task alignment、容量研究、雷达架构调研、教师知识与机制说明。首个非 CNN 候选为保留原始三通道复IQ输入的慢时间/距离轴分解注意力；物理深度展开当前只做适用性审查。
- **R5-A 预注册：** seed101，S0/S2 从头训练，配对复用 E003 的51,200对 train / 5,000对 validation，FP32、RTX3050 `cuda:0`、有效batch64；主预算6,400 updates/4遍，每400 updates验证；只有末段loss趋势满足 protocol 条件时才将两臂同态连续延长至12,800。未运行实验。
- **边界：** R5-A不训练教师/蒸馏、不生成final blind；不覆盖E003/R3/E002/T001/R4历史结果。`docs/CURRENT_STATUS.md`/路线图仅描述计划和活动状态，未把未运行的实验写成结果。

## 2026-10-27：D2-A-E003 固定相干特征学生 pilot 完成

- **总纲对齐：** 依据研究总纲 v1.2，围绕 Q1/Q4 及 Q3/Q8，测试复相干物理表征能否改善任务级 GMTI 学生；本轮没有修改研究主题、路线边界或 `docs/RESEARCH_CHARTER.md`。
- **前置门：** R0 分层稳定 scene-ID 抽样与 split 审计通过；R1 固定 28 通道前端通过 complex64 误差、公共相位、Lag-1 频率映射、lag-4/16 alias、跨通道相位、距离 locality 和极值数值测试。相对最大误差 `9.5371e−8`（门限 `1e−4`），相位误差 `2.9072e−7`，lag-1 映射最大回绕误差 `1.14e−13 Hz`，距离泄漏为0。R2 在 A/B/C 可辨识几何的固定10 dB SNR/CNR控制下，特征 H1−H0 中位RMS为0.1686/0.1433/0.1562，均远高于数值地板的10倍；这不是学生可用信息的充分证明。
- **数据与训练：** 新建 train 51,200 pairs、validation 5,000 pairs、H0 calibration/verification各8,192、专用C3 H0 verification 4,096、H1 development 4,096 pairs、exact-overlap boundary 512 pairs。81,288 scene ID 跨 split 不重叠。S0 raw-IQ TaskStudent 与 S2（固定复相关/功率特征＋28→6投影＋同主干）各用 seed101、E1物理真值、FP32、RTX3050 `cuda:0`、microbatch16/有效batch64、1,600 updates；参数数13,383/13,557，未使用教师/KD/课程。训练约350/494秒，无OOM。
- **训练执行审计：** 首次训练工具会话只完成 S0 60 updates，在任何预设验证点前中断；该部分没有被当作结果。随后从相同初始权重、seed和pair plan完整重跑S0与S2。最终保存的是完整重跑的初始/最佳/末尾 checkpoint、逐更新history与配置。
- **独立评价：** 总体H0 Pfa S0/S2=`0.04834/0.04749`，但C3 H0 Pfa=`0.18140/0.15283`，均超0.10上限。Development AUC=`0.98416/0.97713`，Pd=`0.91089/0.86353`，joint=`0.68652/0.47632`，range RMSE=`3.108/4.130 cells`，circular Doppler RMSE=`46.42/85.70 Hz`。S2−S0总体Pd差`−0.04736`（95%CI `[-0.05713,-0.03759]`）、joint差`−0.21021`（`[-0.22559,-0.19507]`），C2/C3 joint差`−0.25928`（`[-0.28223,-0.23682]`）。G3两臂通过，G4失败，主要增益门失败。
- **物理与计算：** S2在32个公共相位旋转下H0/H1判决一致率为1.0，输出变化约`1e−5`；C1–C3任务质量仍更差。CUDA输入至输出batch1延迟S0/S2=`1.77/4.20 ms`，batch16=`8.21/10.94 ms`，不代表完整处理链实时性。重生成速度几何的频移残差仍大，整体IQ调制/目标分量反事实/几何重生成分开保存。
- **工程错误修复：** 首次完整评价在S2最终CUDA计时处遇到PyTorch inference tensor/autograd冲突；没有最终metrics写出。修复 `_cuda_timing` 的推理上下文并增加回归测试后，用相同检查点与split完整重评；AISAR全套 `303 passed`。保留原始训练前preflight，另记录训练后评价代码重验。
- **决定：** S2当前候选 **STOP**；D2-A任务级学生路线 **REFINE**。不进行正式多seed、E2/E3、final blind或继续S2架构搜索。H-ARCH-D2-002、H-FREQ-D2-001、H-DET-D2-002、H-IDENT-D2-001已登记更新；总纲未改。
- **清理与交付：** 评价完成后只按 `cache_manifest.json` 删除 90 个 E003 可重建缓存文件，共 14,232,309,192 bytes；没有清单外文件或用户未跟踪文件被触及。项目清理后测得 801,766,373 bytes（0.747 GiB），低于 20 GiB 限制。保留协议、配置、训练检查点/history、scene IDs、metrics/result/ROC、物理一致性/成本、4张图、分析、阶段自审、审查请求和 `cleanup.json`。当前下一项唯一优先任务是独立审查E003，再决定是否值得做不训练的表示归因诊断。

## 2026-10-26：D2-A-R4 数值、教师学生物理瓶颈与单 seed S1 pilot

- **总纲对齐：** 按 `docs/RESEARCH_CHARTER.md` v1.2 执行，服务于任务级 GMTI 学生与教师可靠性判断；研究总纲未改。D2-A-R4 保持探索性，T001 G1/G2 历史结论不变。
- **执行：** 在 RTX 3050 Laptop `cuda:0` 完成 B3/T1/ACE×AMF 和 E002 A/B/C 复核。新 H0 calibration/verification 各4096场景，H1为2048对。数值与 CUDA parity 审计记录在 `metrics.json` / `numeric_precision.json`。
- **数值结果：** complex128 中构造复尺度变换后，R4 的 `1e-5` 数学不变性门通过；complex64先乘legacy路径最大差`2.44e-4`，CPU/CUDA最大分数绝对差`3.09e-4`、相对差`9.30e-7`，无峰值/阈值判决翻转。T001冻结`1e-6` G1仍FAIL，历史`4.58e-5`未改。
- **H0与教师：** 同种子替换场景标签的H0 IQ逐元素相同。B3幅度四分位Pfa为`0–0.119`，T1为`0.046–0.052`，但T1在C3仍为`0.178–0.217`；T1与B3峰值估计相同，收益来自归一化分数，不是更佳参数图。T1不取得固定阈值教师资格。
- **取样复核：** 首轮phase/shift前256个H1全是C0，旧结果保留。独立提交`c496057`加入确定性分层敏感性入口；`66c9631`增加同一pilot H1的配对区间。分层敏感性 A/B/C phase agreement为`0.916/0.925/0.916`，S0/S1为`0.905/0.909`；S1的Doppler变化p95仍63.1 Hz。
- **Pilot：** 仅一个 seed101、1600 updates 的 E1 truth-only S1，CUDA用时约302.5秒、峰值分配显存约275MB。S1−S0 paired H1 Doppler RMSE差`−6.29 Hz`（95% CI `[-9.16,-2.84]`），但Pd差`−0.0117`、joint差`−0.0313`且区间均完全低于0；range RMSE差异未定。未运行KD、正式三种子或blind。
- **阶段建议：** **REFINE**。不继续单一公共相位增强或蒸馏，先做独立ChatGPT研究方向审查；若获得新的研究批准，再冻结最小表示比较协议。结果、表和图见 `experiments/D2-A-R4/`，Codex自审见 `docs/stage_reviews/D2-A-R4_review.md`。

## 2026-10-26：完成 D2-A-T001 CUDA 教师比较并按门 STOP

- **总纲映射与授权：** 根据 `RESEARCH_CHARTER.md` v1.2 的 Q1/Q2/Q3/Q4/Q9 与任务级学生目标，用户批准开展一项有界教师增强实验；不修改总纲、历史冻结结果或最终盲测。
- **设计冻结：** 唯一先进候选为模型辅助三通道 post-Doppler ACE×AMF；实现前的代数检查发现纯空间 ACE 会消掉单目标 Doppler-bin 能量，故在数据生成前发布 v1.1 勘误：保留 ACE 空间相干项，以白化能量乘回并除以可观测 CPI 总功率；其余候选数、分组、种子、预算与门槛不变。协方差仅用新的 H0 IQ 拟合。协议固定 4,096 covariance scenes、各 15,000 H0 校准/验证 scenes、12,000 配对开发场景、独立 seed blocks、B3/T1/K2 对照、配对 bootstrap 和教师 G1–G3 门。理论边界及文献审查见 `docs/research/D2-A_teacher_candidate_selection.md`、`D2-A_teacher_literature_review.md` 和 `D2-A_teacher_bottleneck_analysis.md`。
- **停止/分流：** 只有教师 G1/G2/G3 全通过才运行一个固定预算 S2 pilot；否则保留负结果并停止该教师分支，不再追加教师复杂度，也不生成 final blind。
- **v1.2 后端修订：** 按用户要求，covariance 拟合和 ACE×AMF 批量推理改为强制 CUDA `cuda:0`，complex64 GPU 运算前以 double precision 做可观测 CPI 尺度/公共相位规范化；科学公式、样本种子/分组、预算和门槛不变。项目20 GiB只按阶段结束清理后的最终体积核验，中间体积不设硬上限。
- **正式运行：** 代码提交 `91dfb81673e45cad3f3330d90adc707a0cdf859d` 已推送至 `main` 并通过 CUDA preflight；RTX 3050 Laptop，CUDA `cuda:0`，正式CUDA covariance拟合与ACE×AMF评分完成。4096 H0用于covariance、15,000 H0 calibration、15,000 H0 verification、12,000 paired development。运行用时3539.7 s；CUDA covariance拟合47.95 s、85.42 scenes/s；峰值allocated/reserved 1233/1582 MiB。候选复用16 CPI内核的CUDA相对CPU吞吐为batch1 7.40×、batch16 15.63×；此为内核计时，不是端到端实时性能。结果提交前整仓回归 `AISAR pytest -q` 为270 passed。
- **结果/停止：** G1因最大公共复尺度 score/map误差4.58e−5超过1e−6失败；split互斥、finite和covariance正定检查通过。总体H0验证Pfa=0.05007、exact单侧95%上限0.05309，但C3、SNR0、CNR20三大组上限0.125–0.127，G2分层门失败。困难C3 partial-overlap/SNR0/CNR20下ACE×AMF/B3/T1 joint分别为0/0.01/0.004。A/B/C pooled候选相对T1的Pd差−0.2267、joint差−0.2441，区间均完全低于0；Range/Doppler RMSE差+3.353 cells/+93.822 Hz。目标单独频移残差约等于注入频移的相反数；频移峰定位未跟随。G3因G2失败未评估，S2学生和final blind均未运行。T001教师候选按冻结规则 **STOP**。
- **数据与体积：** 保存`metrics.json`、`result.csv`、逐场景calibration/verification/development分数、covariance bank与诊断、物理诊断、环境、图表和分析；不保存完整IQ或模型权重。阶段后项目9,379,091,889 bytes（8.735 GiB），低于最终20 GiB上限；不清理或提交未跟踪的 GMTI 用户目录。阶段分析与审查见 `experiments/D2-A-T001/analysis.md`、`docs/stage_reviews/D2-A-T001_review.md`。
- **研究后续：** T001 negative result 不验证其软图是否可蒸馏，H-KD-D2-002维持INCONCLUSIVE。下一步只做独立研究方向审查；不继续T001教师搜索，不运行S2或蒸馏，除非另有明确科学论证与新协议。

## 2026-10-25：D2-A-E001 静止地杂波 R0/R1 开发验证与阶段审查

### 实施与验证

- 按已冻结的 D2-A 对齐与协议实现逐脉冲双程几何散射、复数相干三通道静止杂波、配对 H0/H1、组件功率指标和 B0–B3 传统基线；未改动生产 GMTI 处理链。
- T1–T12 全通过，包括 D0 零杂波回归、静止散射体双程相位/几何 Doppler、平台速度、波束指向、相位中心、配对残差、功率定义、随机流隔离、真值无泄漏、坐标/公共相位关系和距离徙动。
- 复核时发现早期评估把对平方误差做均值 bootstrap 的区间直接标作 RMSE CI（量纲错误）。统计量已改为每次 cluster bootstrap 样本上先算 `sqrt(mean(squared_error))`；新增回归测试，补充 T2 跨通道解析相位断言及 T7 生成数据的 CNR 标定验证，并完整重跑开发基线（173.3 s）。
- 开发实验包含 5 个 SNR/CNR calibration 条件（每条件 512 H0）和 12 个检测条件（每条件 256 个配对 base scenes）；使用 2000 次按 scene cluster bootstrap。没有提交 IQ 缓存、训练学生或生成 final blind。

### 结果与阶段判断

- B3 DPCA 的分离场景静止杂波抑制中位数为 15.406 dB（95% cluster CI 15.328–15.480 dB），目标差分功率比中位数 3.614。该比值相对未归一化单通道输入功率，DPCA 相干增强可使其大于 1。
- 在 A 主条件（输入 SNR/CNR=10/10 dB），B0–B2 的 ROC AUC 约 0.500；B3 的 Pd/AUC/joint success 为 1.000/1.000/1.000，Doppler RMSE 2.83 Hz，但开发 `Pfa_CPI=0.0898`（95% CI 0.0586–0.125），高于 calibration 经验值 0.0508。20/10 dB 的 A 条件通过冻结的 R1 可检测性门；这不证明主条件的虚警迁移稳定。
- B3 在 B near-edge 10/10 dB 下 Pd=1.000、AUC≈1；C partial-overlap 下 Pd=0.605、AUC=0.864；D exact-overlap 下 AUC=0.501，95% CI `[0.499,0.504]`，符合单 CPI 同流形源归因不可辨识预期。提高 CNR 至 20 dB 时 C 的 B3 Pd 降至 0.230，呈现杂波掩蔽影响。
- 现在 `analysis.md` 和阶段审查同时报告检测命中条件下的 Range/Doppler RMSE 及按 RMSE 统计量计算的 95% CI；漏检在 joint success 中计失败。条件均值方差区间不会再错误地把 MSE 单位标为 RMSE。
- R2 学生协议修正为 D1-E001 已验证 `TaskStudent`（时间残差卷积骨干、原 K3 推理头）；K1/K2 是导出时移除的 training-only soft-map 辅助头。A/B/C/D 预算匹配 D1 为 1,600 updates、有效 batch 64；当前仍无学生训练授权。
- 所有结果来自合成平地独立点散射体，采用 15.8 GHz、PRF 1300 Hz、N=130、3 通道、32 距离单元；Tx 相位中心与平台参考原点共址、二维 Gaussian 方位主瓣均为明确假设。配置保留 D0 16 GHz 零杂波回归适配。
- 阶段决定 **CONDITIONAL GO**：先对固定 H0 阈值的独立迁移和幅度 nuisance 做新的预注册诊断；不得以本开发 H0 回调门限。前门通过后才做 E1 truth-only student pilot；E1 不通过则不运行蒸馏 E2/E3。阶段报告为 `docs/stage_reviews/D2-A-E001_review.md`，完整指标和图在 `experiments/D2-A-E001/`。

## 2026-10-24：批准并启动任务级教师蒸馏新主线

### 审查依据

- 读取了本轮完整研究路线任务文件、当前 P0/P1 阶段材料及未提交的 APX15 算法包。旧 P0-E007 GO、P1-A-E001 STOP/REDIRECT 均保持历史状态；新路线 D0/D1 不再以前一阶段的相位回归为前置。
- 当前机器 `AISAR` 环境为 PyTorch 2.8.0+cu128，CUDA 可见，设备为 NVIDIA GeForce RTX 3050 Laptop GPU；本轮基线测试为 **67 passed**。
- `GMTI_APX15_CoreX_Int16_20261014` 本地约 4.3 GiB，包含六份约 753 MB 正式 BIN。包级 CMake 面向 CoreX，本机没有 CoreX SDK；代码适配先限于 D0 需要的慢时间频谱子链，不声称整条生产链已接入。

### 已完成

- 已记录新研究目标与 D0→D5 路线，保留历史 P0/P1 阶段结论。
- 已编写 `docs/research/task_level_progressive_distillation.md`、`docs/research/D0_D1_design.md` 和 `docs/superpowers/specs/2026-10-24-d0-d1-task-level-distillation-design.md`。
- 已冻结实现计划 `docs/superpowers/plans/2026-10-24-d0-d1-task-level-distillation.md`；D0 和 D1 数值协议均已在各自代码/训练前冻结。
- D0-E001 仿真、APX15 慢时间 FFT 教师及 12 项正式检查全部通过；阶段审查允许进入 D1。结果与范围限制见 `docs/stage_reviews/D0_review.md` 和 `experiments/D0-E001/analysis.md`。
- D1-E001 已冻结轻量 TCN、A/B/C/D 1600-update 匹配预算、三个正式 seed、50,000 条独立 H0 校准样本、40,960 条最终盲测样本及 G1/G2/G3 门槛；配置见 `experiments/D1-E001/`。

### 后续结果

- D1 学生/损失、CUDA pilot 与正式训练、独立 H0 校准、最终盲测和阶段审查均已完成。G1、G2 通过，G3 未通过；按预注册规则 D1 为 GO → D2。结果、指标和限制见本文件末尾的 D1-E001 盲测记录。

## 2026-10-24：D0-E001 数值协议预注册

- 冻结 `experiments/D0-E001/config.yaml` 和 `protocol.md`：PRF=1300 Hz、N=130、3 通道、固定 32 距离单元；复 AWGN、连续距离/Doppler、随机幅度/初相及 APX15 仿真三接收通道子集几何。
- 对齐 APX15 参数后，距离轴明确使用理论带宽分辨单元 `c/(2B)=2.99792458 m`；APX 配置的 5 m 分辨率字段与 DDC 间距 `c/(2Fs)=2.498270483 m` 单独记录，D0 不声称复现完整生产距离处理。
- 冻结 191 点离散 LFM—匹配滤波核对、原生 130-bin 软频谱、520-bin 补零硬估计、噪声/频轴/无噪参数门、各正确性样本量及独立 seeds。教师接口只接收 IQ；CoreX 正式链不在 D0 适配范围。
- 数值门限依据和一组小规模、非持久化设计 sanity check 已在协议中披露，明确排除在正式 D0 指标之外。正式检查见下方运行结果；预注册门槛未按结果更改。

## 2026-10-24：D0 仿真与 APX15 FFT 教师实现

- 实现 `src/radardistill/simulation/d0.py`：固定监视窗复 IQ、连续 range/Doppler、三通道 APX15 仿真相位中心子集、H0/H1 共先验复 AWGN，并分字段输出真值、信号项和噪声项。
- 实现 APX15 `ColFFTTranspose` 的 FFTW3 C++ 适配及 Python 教师：130-bin 软响应、520-bin 连续 Doppler 硬估计、峰值/中位数检测分数；`predict` 只接收 IQ，不读取真值或已知噪声功率。
- 加入流式 D0 正确性运行器 `scripts/run_d0_e001.py`，覆盖冻结样本、边界、频轴、AUC 和 bootstrap 汇总，不保存可再生完整数组。
- 新增 12 个定向测试；`conda run --no-capture-output -n AISAR pytest -q` 为 **79 passed**，`compileall` 与 `git diff --check` 通过。
- 正式 D0 使用配置中冻结的独立种子分批生成数据，记录 12/12 门通过；完整指标、CSV、环境与解释见 `experiments/D0-E001/`。全量测试 **79 passed**，`compileall` 与 `git diff --check` 通过。

## 2026-10-24：D0-E001 正式正确性检查与阶段审查

- 复现命令：`conda run --no-capture-output -n AISAR python scripts/run_d0_e001.py --config experiments/D0-E001/config.yaml`。运行约 39.1 s，CPU 峰值 RSS 约 585 MB；D0 FFTW 教师运行于 CPU，未进行训练。
- 12/12 冻结门通过。191 点 LFM 匹配滤波 PSF 最大幅度误差为 `1.2662e-4`；1,000,000 个复噪声样本功率相对误差 `0.1187%`；256 对 H0/H1 具有逐元素一致的背景噪声。
- 4096 个无噪 H1 的距离 RMSE 为 `7.17e-10` cell，Doppler RMSE 为 `0.02904 Hz`、最大误差 `0.04052 Hz`；130 个原生 FFT 栅格音调全部落在正确频轴。
- 每个 SNR 的 H0/H1 教师诊断样本数为 1024/1024；-10 dB 教师检测 AUC `0.98856`（bootstrap 95% CI `[0.98414, 0.99261]`），达到预注册 `AUC≥0.98` 可用性门。-20/-15 dB AUC 分别为 `0.48675/0.64430`，因此教师在这些低 SNR 区间没有可靠检测判别力，必须保留并在 D1 中报告教师错误。
- 阶段决定：**D0 GO → D1**，仅表示冻结仿真与教师子链通过正确性门，可开展 D1 学生可行性实验；不代表学生已学会任务、蒸馏有效、完整 APX15/CoreX 链已复现或系统虚警率已验证。详细审查见 `docs/stage_reviews/D0_review.md`。

## 2026-10-24：D1-E001 数值协议预注册

- 在实现和 pilot 之前冻结 `experiments/D1-E001/config.yaml` 与 `protocol.md`：TCN 感受野 255 覆盖 130 个脉冲；A/B/C/D 使用相同数据、初始化、更新次数和有效 batch；教师只提供全图软频谱辅助损失，真值单独监督最终任务。
- 固定训练 SNR `[-20,20] dB` 均匀抽样，每个训练 seed 4096 H0+4096 H1；三个正式训练 seed；每策略 1600 个 optimizer updates，batch 64，CUDA FP32 顺序执行。先以独立 pilot seed 检查四组训练及 GPU 资源。
- 固定 50,000 H0 阈值校准样本及每 SNR 4096 H0/1024 H1 的新盲测；工作点为 `Pfa=1e-2/1e-3`。已冻结参数指标、层级配对 bootstrap、传统 FFT/教师/相位差/profile-ML 基线以及 G1/G2/G3 和阶段决策门。
- 数据缓存仅忽略 `experiments/D1-E001/data/cache/*.npy` 和 `*.npz`；模型、汇总报告、配置和训练记录仍可单独审阅。协议冻结时未生成 pilot 或盲测数据。

## 2026-10-24：D1 学生、任务损失与 CUDA pilot 训练器实现

- 实现 `TaskStudent`：32 通道深度可分离 TCN、255 脉冲理论感受野、presence/range/周期 Doppler 头及训练期全图软频谱头；导出推理模型时移除辅助头。
- 实现 H0 屏蔽的距离/Doppler 真值损失、周期 Doppler 编码与解码、全图 teacher-response KL、CUDA fail-fast、A/B/C/D 固定更新计划、CPU memmap 批量搬运及 64/32/16 microbatch OOM 回退。梯度累积分别按有效 batch 样本数和 H1 数归一化，与整批损失一致；训练前保存 update 0 初始 checkpoint。教师硬输出与真值分字段存入本地训练缓存，学生输入仅为复 IQ 实/虚平面。
- 定向 D1 测试 **20 passed**；全仓库 `conda run --no-capture-output -n AISAR pytest -q` 为 **100 passed**；`compileall` 和 `git diff --check` 通过。验证缓存测试发现并修正了多 SNR 样本分配长度错误；学生逐 SNR 指标、教师验证基线和按 H1/SNR 分层的物理一致性汇总已接入。首次直接 CLI 启动暴露 CUDA 设备只检查未选择的问题；`require_cuda` 现显式选择 `cuda:0`，已验证其后显存统计 API 正常。正式 runner 现支持按配置顺序分三 seed 运行，但正式训练尚未开始。
- 训练前代码提交 SHA 为 `bc0d4d8939ddd8e6a1c3d3062ec33856c722b9e8`；实际使用 RTX 3050 Laptop `cuda:0`、PyTorch 2.8.0+cu128、CUDA 12.8、驱动 580.178.04、FP32。四组各 1600 更新，训练 133–140 秒/组，峰值分配显存 1046–1049 MiB，无 OOM。首次直接启动发现 runner 未显式选择 CUDA device；根因修复见提交 `651ece3`。

## 2026-10-24：D1-E001 CUDA pilot 筛选

- 复现命令为 `conda run --no-capture-output -n AISAR python scripts/run_d1_e001.py --phase pilot`；数据缓存留在 `experiments/D1-E001/data/cache/` 并被窄范围忽略。pilot 的初始/最佳/最后 checkpoint、训练历史、配置、环境、metrics 和诊断均保存在 `experiments/D1-E001/pilot/`。
- A 组验证任务损失从 5.238 降至 2.118；在 0/5/10/20 dB 的 AUC 均为 1.0，range RMSE 为 0.044–0.111 cell、Doppler RMSE 为 22.2–31.4 Hz。-10 dB 及以下性能明显下降。APX15 教师在高 SNR 频率估计仍强于学生。
- 一 seed 主指标 J：A=0.01203、B=0.01095、C=0.01358、D=0.01076；B/D 的高 SNR 改善仅是 pilot 线索，不能据此断言蒸馏收益。共同相位旋转/频移结果已按 H1 与 SNR 分层；20 dB H1 Doppler 等变残差 p95 为 2.87 Hz（0.625 Hz 输入频移）和 9.34 Hz（5 Hz 输入频移）。
- **筛选决策：进入正式三种子 A/B/C/D 对比。** A 在 20 dB 的 detection/range/Doppler 数值参考达到冻结 G1 数值，且高 SNR 物理一致性较好；低 SNR 失败、蒸馏效果和传统竞争力留待正式独立测试。此筛选不构成 D1 阶段 GO，也不判定 G1/G2/G3。
- 代码复核修正了 formal 启动门：不再信任旧版 `pilot_gate.passed`，而是按冻结配置从四组原始策略摘要重算筛选门；正式 seed 数量锁定为恰好三组。`conda run --no-capture-output -n AISAR pytest -q` 为 **101 passed**；其后 formal 训练按冻结预算完成，见下一节。

## 2026-10-24：D1-E001 三种子 CUDA formal 训练

- 复现命令：`conda run --no-capture-output -n AISAR python scripts/run_d1_e001.py --phase formal`。A/B/C/D × seed_1/seed_2/seed_3 共 12 组，每组严格 1600 optimizer updates；逐 seed 输出初始/最佳/最后 checkpoint、训练历史、验证指标、CUDA 峰值资源和环境。
- 三组四策略均成功完成，未发生 OOM，实际 microbatch 均为 64。RTX 3050 Laptop `cuda:0`、FP32、PyTorch 2.8.0+cu128/CUDA 12.8；单策略平均约 133 秒，峰值 allocated 最多约 1049 MiB、reserved 最多约 1150 MiB。训练环境记录代码 SHA `df25b79f8cb8d18c44b020ebb77d096c6b8ef265`。
- 20 dB 验证集上 A 的三 seed AUC 均为 1.0，range RMSE 为 0.067–0.076 cell，Doppler RMSE 为 25.4–32.1 Hz。此处仅是冻结验证集诊断，不是校准或盲测验收；传统基线差距、蒸馏配对收益和 G1/G2/G3 暂不判定。
- 正式训练汇总、逐 seed `metrics.json`/`analysis.md`、12 组模型权重和训练历史保存在 `experiments/D1-E001/formal/`；独立 H0 校准及新盲测尚未生成。

## 2026-10-22：P0 实施、复核与阶段判断

### 已完成

- 建立研究规范、方案、P0–P6 阶段计划、统一实验协议、文献综述/矩阵、研究空白和决策记录。文献范围区分汽车 FMCW、SAR-GMTI、机载多通道 GMTI 与 STAP。
- 实现 P0 复单音加复 AWGN 仿真、FFT 峰值/插值、相位差、连续 profile-ML、CRLB、圆周误差、局部教师似然软目标和 NumPy CPU Conv1D。
- 主实验 `P0-E001`：3 个训练种子；每个种子 6000 个训练样本和 1200 个验证样本；8 个独立测试 SNR，每点 1024 个样本。物理真值、教师频率、教师软目标和学生预测分别保存。
- `P0-E002` 在相同训练种子和测试集上把训练延长至 160 轮，用于检查优化不足。
- `P0-E003` 使用位置保留 flatten + Dense(32) 读出，`P0-E004` 使用更低容量 flatten + Dense(8)，用于检验均值池化损失时间位置与大读出头过拟合的可能性。两个诊断各只有一个训练种子，因此不用于种子间结论。
- 代码和行为测试覆盖信号/噪声、种子复现、基线、圆周边界、CRLB、教师标签、模型训练和实验产物；本轮完整测试运行结果为 **13 passed**。

### 主要结果

全部结果来自模拟单目标 AWGN，不是实测 GMTI 数据。

| 实验 | 学生/训练 | 20 dB E1 RMSE | 同一测试上的参考 RMSE |
|---|---|---:|---|
| P0-E001 | 6 通道均值池化，40 轮，3 种子 | 15.00 Hz | FFT 峰值 2.96；FFT 插值 1.23；相位差 0.206；连续 ML 0.0332 Hz |
| P0-E002 | 同一结构，160 轮，1 种子 | 8.55 Hz | 基线与测试输入相同；E1 仍高于 FFT 峰值、插值和相位差 |
| P0-E003 | flatten + Dense(32)，160 轮，1 种子 | 26.18 Hz | 训练损失较低，但验证圆周 MSE 0.241，呈训练/验证差距 |
| P0-E004 | flatten + Dense(8)，160 轮，1 种子 | 36.27 Hz | 容量下降未恢复测试精度 |

E001 的 E1 在 7/8 个测试 SNR 点比零频常数预测 RMSE 低至少 20%，支持“存在基本频率学习”这一有限结论。但在 -5 至 20 dB，E1 未超过 FFT 峰值；高 SNR 下与插值、相位差和连续 ML 仍有明显差距。E002 表明延长训练显著降低 5–20 dB 的 E1 误差，但一次种子仍无法满足基线门。E003 训练损失下降而泛化变差；E004 降低容量仍未解决。

E001 的 20 dB 分组 RMSE 进一步显示：频率五分组为 10.09–23.67 Hz；初相四扇区为 13.93–16.43 Hz；幅度四分位为 12.51–17.35 Hz；FFT 栅格偏移四分位为 14.23–16.01 Hz。E002 单种子对应范围分别为 6.88–11.01、7.56–10.02、7.84–9.10 和 8.19–9.19 Hz。组间误差不完全一致，中心频段在 E001 尤其较差；这些诊断没有消除主结论。

### 阶段门判断

1. **实现门：通过。** 13 项项目测试通过；独立数据划分及可审阅产物已生成。
2. **基本学习门：部分支持。** 与常数预测的差异和随 SNR 的变化表明学生确实利用了频率信息；但高误差、频率分组波动及架构诊断结果不支持稳健性能结论。
3. **继续价值门：未通过。** 学生在 -5 至 20 dB 未超过 FFT 峰值，蒸馏没有稳定的高 SNR 质量收益。E001 在部分 SNR 的 E3−E1 bootstrap 区间低于零，但 20 dB 区间跨零；E002–E004 为单种子诊断，不能估计训练种子不确定性。

因此 **P1 尚未开始**。当前证据不支持三通道或 GMTI 检测性能主张。

### 当时安排的下一步（现已执行）

保持 E001 的独立测试集不变，只安排一个预注册的 P0 诊断：使用小容量、位置敏感读出或显式相位增量特征，检查是否能同时避免全局均值池化的信息损失和 flatten-Dense 的过拟合。沿用传统基线及真值/教师三策略，预先指定训练种子、训练轮数和“至少部分正常 SNR 稳定优于 FFT 峰值”的通过条件。若仍不通过，先完成误差来源分析并重新评估纯 CNN 路线，再决定是否启动 P1；不进行无目标的超参数搜索。

### 环境和复现

- Python 3.13.5、NumPy 2.1.3、pytest；运行环境无 PyTorch/TensorFlow/JAX，也没有可用 NVIDIA 驱动，因此训练为 NumPy CPU，不代表 GPU 性能。
- 主实验配置和完整数据/权重/预测位于 `experiments/P0-E001/`；后续诊断位于 `P0-E002` 至 `P0-E004`。
- 复现入口：`python scripts/run_p0.py --config experiments/P0-E001/config.yaml`；代码检查：`pytest -q`。

### GitHub 同步状态

项目文件已在当前工作区整理完成，但尚未写入 `shy-sgsg/RadarDistill`。GitHub Contents API 和 Git Data blob API 均返回 HTTP 403 `Resource not accessible by integration`；远程仓库仍为空。需要为当前 GitHub 连接授予该仓库内容写权限，或配置一个可用的 git 写入连接，再记录远程提交 SHA。

> 状态更正：以上是 2026-10-22 当日的 GitHub 连接记录。用户之后已确认此前提交已推送；截至 2026-10-23，本轮 E006 与审计改动仍在本地工作区，尚未提交或推送。

## 2026-10-24：P0-E007 三范式实验与阶段验收

### 设计与执行

- 在保留 E006 STOP/REDIRECT 历史结论的基础上，按新批准的预注册协议运行 E007。信号模型仍为 PRF=1300 Hz、N=130 单目标复 AWGN；训练/验证和开发集用于筛选，最终盲测另用新 seed。
- M1 是固定、不可学习的 16N 复 FFT + 可学习循环 Conv1d 频谱读出，输入含 log-power 与相邻谱线复互谱相干；另训练仅 log-power 的 M1-power 消融。M2 是固定 4N FFT 粗峰 + 27→64→32→1 子栅格修正 MLP。M3 是直接处理 130 个 IQ token 的两层 d=32、4 头小型 Transformer。
- 三种架构均先以一个 CUDA seed 筛选；M1 通过后正式训练三个种子。M1-power 作为同架构 E1 消融也运行三个种子；只有具备基础学习能力的 M1 进入 E2-H、E2-S、E3-S 蒸馏对比。
- 所有训练使用 AISAR、PyTorch 2.8.0+cu128、CUDA 12.8、RTX 3050 Laptop GPU（4 GiB）、FP32、batch 64，顺序运行；没有将全部数据缓存到 GPU。最终盲测每 SNR 1024 例，共 8192 例，只生成/推理一次。模型、种子、checkpoint、训练历史、教师字段、预测和资源指标均保存在 `experiments/P0-E007/`。

### 筛选结果

| 模型 | 验证损失首轮→最佳 | 20 dB 开发 RMSE | 结论 |
|---|---:|---:|---|
| M1 | 1.688→1.273 | 0.03862 Hz | 通过，进入正式 E1 |
| M1-power | 1.671→1.275 | 0.03567 Hz | 仅作为三种子 E1 消融 |
| M2 | 0.03295→0.03158（下降 4.2%） | 0.09585 Hz | 未达到验证下降门；比同局部 4N 抛物线高 0.06179 Hz，比连续局部 ML 高 0.06270 Hz，停止扩展 |
| M3 | 6.997→6.997 | 371.30 Hz | 验证未改善且五个主 SNR 点均未胜 FFT 峰；训练 loss 下降而验证恶化，停止三种子训练 |

M2 在 20 dB 开发集比 N 点 FFT 插值低 1.1370 Hz，但匹配局部抛物线及连续 ML 明显更好；该混合模型没有达到进入正式训练的标准。M3 只有一个筛选 seed，结果支持该配置泛化失败，不足以否定全部 Transformer 结构。

### 最终盲测结果

- M1 E1 的 20 dB 三种子 RMSE 为 0.03722、0.04369、0.03607 Hz，均值 0.03899±0.00411 Hz。当地 CRLB 标准差为 0.03419 Hz（M1 约 1.14× CRLB）；连续 profile-ML 为 0.03487 Hz（约 1.02×）。
- 在 0、5、10、15、20 dB 五个主 SNR，M1 均以配对 95% bootstrap 区间显著优于 N 点 FFT 峰、N 点 FFT 插值、相位差估计和 16N FFT 峰。20 dB M1−16N FFT 峰为 -0.14751 Hz [-0.15336,-0.14179]。
- M1 未胜过 4N/16N 局部抛物线和连续 profile-ML。20 dB M1−local profile-ML 为 +0.00412 Hz [+0.00120,+0.00816]，不支持替代高精度传统估计器。
- 相干输入未显示收益：M1−M1-power 在 20 dB 为 +0.00222 Hz [-0.00057,+0.00660]，五个主 SNR 区间均未证实 M1 更好；M1-power 的点 RMSE略低。
- E2-H 与 E1 在 20 dB 无显著差异（-0.00205 Hz [-0.00599,+0.00096]）；E2-S 明显更差（+0.07889 Hz [+0.07084,+0.09075]）；E3-S 更差（+0.00884 Hz [+0.00151,+0.01515]）。
- M1 公共相位旋转预测变化最大 0.01221 Hz，复数全局尺度变化最大约 0.01782 Hz；0.625 Hz 频移等变误差均值约 0.00021 Hz、最大约 0.01245 Hz。无噪声全频和 Nyquist 边界集合的 G1 指标三个种子均通过；带噪盲测 Nyquist 邻域每 SNR 约 13 个样本，分组估计不确定性较高。
- 分组结果、SNR 指标和 CRLB 对照在 `result.csv`；种子配对 CI 在 `paired_comparisons.json`；曲线在 `rmse_snr.png`；全部屏幕/正式训练记录在 `training_history.csv`、各阶段 JSON 与 `models/`。

### 阶段门及下一步

- G1 学习可行性：**PASS**，M1 无噪声 MAE/P95、Nyquist 子集、20 dB 相对零频预测和数据隔离条件均为 3/3 种子通过。
- G2 传统竞争力：**PASS**，M1 在五个主 SNR 点均优于 N 点 FFT 峰值，并在五点均优于匹配的 16N FFT 峰值。
- G3 蒸馏价值：**FAIL**，E3-S 没有任何通过点，更无两个相邻且种子一致的收益。
- 按阶段门，P0-E007 决策为 **GO**，仅允许进入 P1-A 方案设计/预注册；三通道实验尚未开始。单通道结果不能证明三通道相位、角度或 GMTI 检测能力。

完整设计、协议、代码、可复现配置、训练和测试产物、图表及阶段分析见 `docs/superpowers/specs/2026-10-23-p0-e007-design.md`、`experiments/P0-E007/` 和 `docs/stage_reviews/P0-E007_review.md`。规范指标入口为 `experiments/P0-E007/metrics.json`。验证命令 `conda run --live-stream -n AISAR pytest -q` 结果为 **58 passed**；`python -m compileall -q src scripts tests` 与 `git diff --check` 通过。重建汇总命令：`conda run --no-capture-output -n AISAR python scripts/run_p0_e007.py --phase summarize`。

## 2026-10-22：P0-E005 执行与误差来源复核

### 实验结果

- 按预注册协议运行三种子、E002 的 160 轮上限。学生输入为 129 点单位相位增量；FFT、插值、相位差和连续 ML 基线继续使用原始 IQ。
- E005 保存的训练、验证和测试 NPZ 与 E001 对应文件逐数组完全相同，因此本次模型对比沿用同一独立数据划分。
- 主门未通过。0、5、10、15、20 dB 的 E1−FFT 峰值 RMSE 差及 95% 配对 bootstrap 区间分别为：

| SNR (dB) | E1 RMSE (Hz) | FFT 峰值 RMSE (Hz) | E1−FFT 差值及 95% 区间 (Hz) |
|---:|---:|---:|---:|
| 0 | 31.66 | 2.86 | 29.31 [22.97, 35.80] |
| 5 | 14.35 | 2.92 | 11.64 [8.14, 13.89] |
| 10 | 11.57 | 2.86 | 9.15 [4.51, 11.70] |
| 15 | 14.22 | 2.86 | 11.48 [9.12, 13.30] |
| 20 | 15.42 | 2.96 | 12.59 [10.25, 14.91] |

所有差值区间均高于零，E1 在五个主 SNR 点均未胜过 FFT 峰值。20 dB 下 E1/E2/E3 RMSE 分别为 15.42/15.77/14.92 Hz；E3−E1 的区间为 [-1.90, 0.37] Hz，蒸馏优势不稳定。相位差基线在 20 dB 为 0.206 Hz，连续 ML 为 0.0332 Hz。

### 误差来源复核

以 20 dB 的同一测试样本为例，E1 三种子的 RMSE 为 13.05、15.26、17.95 Hz，显示有训练种子差异。合并三种子预测后，预测频率与真值相关系数为 0.935，线性斜率为 0.930；预测和真值标准差分别为 373.76 与 376.01 Hz。这说明模型覆盖了频率范围，但精度仍有系统残差。

20 dB 分组 RMSE：初相扇区 15.33–16.16 Hz、幅度四分位 15.15–15.92 Hz、FFT 栅格偏移四分位 15.22–15.86 Hz；频率五分组为 10.77–18.02 Hz，组内平均有符号误差从 -5.65 到 +7.82 Hz。当前证据更指向跨频率区间的校准/逼近误差和训练种子波动；没有显示误差主要由初相、幅度或 FFT 栅格偏移驱动。E1 训练损失与验证真值 MSE 数值接近，也没有明显的训练/验证鸿沟证据。该复核是诊断线索，不能单独证明误差的因果来源。

### 阶段判断与后续

P0-E005 已执行，继续价值门仍未通过，**P1 不启动**。显式相位增量输入没有让小型 CNN 接近 FFT 峰值、插值、相位差或连续 ML 基线。先据此重新评估纯 CNN 路线的研究价值；若仍继续，需要先提出一个针对频率相关残差或种子波动的有界、可证伪方案，再决定是否开展另一项 P0 诊断。E005 代码、协议、指标和完整逐样本产物见 `experiments/P0-E005/`；误差复核见 `experiments/P0-E005/error_source_review.md`。

## 2026-10-23：R0 方法审计与 P0-E006 预注册

### 审计与修复

- E2 早停目标现在显式区分教师验证、真值验证及历史兼容模式；E006 的 E2-A/E2-B 分别走教师验证和真值验证。历史 E001–E005 结果保留为原协议结果，不追溯重算。
- `estimate_mle` 的底层方法准确标注为 FFT 初始化局部 profile-ML，并保留旧名称兼容；新增全局网格 profile-ML 和多延迟相干基线。固定低 SNR 例显示局部搜索会受 FFT 错峰影响。
- 分组逻辑已从固定 N=130 改为信号真实样本数。补充 PRF/N、Nyquist、空批次/错误形状、仿真物理变换和多参数有限差分梯度检查。
- NumPy CNN 将损失/梯度计算公开为可检查接口；卷积权重/偏置、隐藏层、输出层权重/偏置均与有限差分比较，梯度测试包括联合蒸馏损失。
- E006 的 G3 现在同时要求配对区间在两个相邻主 SNR 低于零，并要求每个点至少 2/3 个训练种子的 RMSE 差方向为负。
- E001–E005 共有 79 个实验文件、109,764,138 字节。逐字节 hash 审计发现 7 组重复（test、共享 train/validation 文件），理论可回收 76,867,174 字节；为保留历史实验完整性，本次没有删除文件。

### E006 预注册状态

- 方案与配置见 `experiments/P0-E006/protocol.md`、`config.yaml`。S0 冻结 E005 CNN，S1 为归一化多延迟复相关 MLP，S2 为 4N 频谱输入的物理引导混合 MLP；共 33 个 S1/S2 训练实例（11 个协议×3 个训练种子）。
- 原始预注册使用 E006 final seed 909090、每个固定 SNR 1024 个样本；该种子后来因评估器缺陷作废，已在下一条更正并归档。E001–E005 测试输入只作为开发基准。
- 用户明确要求 GPU 训练并允许当前工作区访问设备。AISAR 报告 PyTorch 2.8.0+cu128、CUDA 可用、RTX 3050 Laptop；runner 在无 CUDA 时停止，不使用 CPU 训练。
- 首轮按冻结训练协议完成 33 个 CUDA 训练实例。产物审计发现经典估计器的一维批量向量被误读为逐样本训练种子，造成指标/配对比较失真。首轮数据、权重、日志和无效评估已保留于 `experiments/P0-E006/attempts/attempt_1_invalid_evaluation/`；错误 CSV 无损压缩，seed 909090 不复用。
- 统一预测重复解析和结果序列化修复后新增回归测试；最终完整测试 **35 passed**，compileall 与 `git diff --check` 通过。修正版改用独立 seed 919191，模型、训练种子、预算和门槛不变。

### P0-E006 修正版结果与阶段门

- 在 919191 final test 上，每个 SNR 1024 例。20 dB 时 S0/E1、S1/E1、S2/E1 RMSE 分别 15.273、2.947、27.238 Hz；传统 FFT 峰 2.884、FFT 插值 1.210、相位差 0.207、局部/全局网格 ML 约 0.0327、多延迟相干基线 0.0625 Hz。
- S1 在 20 dB 显著优于零频预测，但其三个 noiseless 种子 MAE=2.44–3.01 Hz、P95=5.75–6.40 Hz，G1 未通过。S1−FFT 只在 15 dB 单点的配对 95% 区间完全低于零；它在全部主点均未胜过同相关特征非学习基线，G2 未通过。S1 E3/T2 在 0、20 dB 有孤立收益但不相邻，E3/T3 变差，G3 未通过。
- 阶段决策为 **STOP/REDIRECT**：停止当前单音学生网络扩模，不启动 P1-A 训练。正式总结见 `docs/stage_reviews/P0_final_review.md`；仅保留未注册的 P1-A 理论草案 `docs/stage_reviews/P1_A_plan.md`。
- 修正版 artifacts、seed、设备、代码提交/dirty 状态和 SHA-256 清单见 `experiments/P0-E006/metrics.json` 与 `data_manifest.json`；所有数据为仿真，不含真实 GMTI 或部署时延结论。

## 2026-10-24：P1-A-E001 理想三通道相对相位实验与阶段验收

### 设计与执行

- 在 P0-E007 GO 后，按 `experiments/P1-A-E001/protocol.md` 冻结理想共享单音、三通道独立同方差复 AWGN、相对相位标签、解析基线、CRLB、数据种子、训练预算和验收门。学生为归一化复互相关物理前端加 `4→64→64→4` MLP，仅使用 E1 真值监督；不含教师/蒸馏。
- 使用三个独立训练/验证数据种子、2048 条新无噪声诊断样本和全新盲测 seed `401999` 起按 SNR 每次偏移 1000；测试为 8 个 SNR、每点 1024 条，共 8192 条。盲测只在全部 checkpoint 冻结后生成。
- 在 AISAR、PyTorch 2.8.0+cu128、CUDA build 12.8、RTX 3050 Laptop GPU（4 GiB）顺序训练三个模型，FP32、batch 64、最多 160 epochs、patience 40。每种子训练 14.52–14.90 s，峰值已分配显存 17.6 MiB。

### 结果与阶段门

- 预注册 G1 为 **FAIL，0/3 种子通过**。三种子都通过 20 dB 零相位基线改善（RMSE 降约 98.6%）和公共相位旋转/复尺度/频移不变性（最大输出差 `4.77e-7 rad`）；但无噪声两相位 MAE 为 `0.0177–0.0206 rad`，未达 `<0.01 rad` 门槛，部分 P95 也超过 `0.05 rad`。
- 20 dB 学生 RMSE 为 `0.02315–0.02603 rad`（Δφ21）和 `0.02329–0.02584 rad`（Δφ31）。B1 互相关、B2 逐样本圆均值、B3 秩一 profile-ML 均约 `0.0097–0.0100 rad`，条件 CRLB RMS 标准差约 `0.00995 rad`。学生与 B1/B3 的配对区间在三种子和两相位均为正，显示学生明显更差。
- 训练与验证向量 MSE 均在约 0.10 平台化，未见明显过拟合鸿沟；结果更符合当前小型读出欠拟合/优化受限，但不能由本轮数据确定因果。
- 按冻结门槛，P1-A-E001 决定 **STOP / REDIRECT**：不进入 P1-B、不开展知识蒸馏，不根据盲测更换模型或降低门槛。结论仅适用于本轮模型及理想单音 AWGN 仿真，不否定所有学习架构，也不代表实测 GMTI 能力。

代码、协议、数据、checkpoint、epoch 记录、逐样本预测、分组指标、配对 bootstrap 和 RMSE-SNR 图见 `experiments/P1-A-E001/`；完整阶段审查见 `docs/stage_reviews/P1-A-E001_review.md`。本轮验证命令 `/home/shy/anaconda3/envs/AISAR/bin/python -m pytest -q` 结果为 **67 passed**。

## 2026-10-24：D1-E001 独立校准、正式盲测与阶段审查

- 按冻结协议完成独立 50,000 H0 校准和 40,960 条新盲测（每 SNR 4096 H0 + 1024 H1；8 个 SNR）。正式 A/B/C/D 三策略种子训练、校准和学生盲测推理使用 RTX 3050 Laptop GPU；盲测结果复算使用保存的预测数组。
- **G1 PASS，3/3 seed：** A 真值监督在 20 dB 三 seed 的 AUC/Pd（校准 `Pfa=1e-2`）均为 1.0；全 H1 range RMSE 0.0592–0.0682 cell，Doppler RMSE 25.95–35.13 Hz，均显著优于常值预测。
- **G2 PASS：** C 联合监督 ΔJ=-0.00013472，层级配对 bootstrap 95% CI `[-0.00025944,-0.00000810]`；D 渐进蒸馏 ΔJ=-0.00086207，CI `[-0.00214046,-0.00004611]`；二者均 3/3 seed 方向改善。B 未通过；B/C/D 均无预注册收敛加速证据。
- **G3 FAIL：** A/B/C/D 均未通过任一预注册相邻高 SNR 的传统竞争触发条件。20 dB profile 子集上，D range/Doppler RMSE 为 0.0689 cell/25.42 Hz，APX15 为 0.0415 cell/0.0360 Hz，profile-ML 为 0.01994 cell/0.01906 Hz。
- 盲测预测已先于指标汇总生成并保存。评估器发现 G2 AUC 非劣界限的配置路径错误后，修复读取位置，并仅从保存的预测重放指标；样本、阈值、模型、指标定义与门槛均未改变，审计见 `blind/metrics.json.analysis_replay`。首次盲测输出和分析器失败记录没有被覆盖。
- 阶段决策 **GO → D2**，但仅按阶段规范一次增加一个复杂度。低 SNR 检测与参数估计、传统竞争力均有限；`Pfa=1e-6`、真实 GMTI、多目标、杂波、失配和完整 CoreX 链均未验证。完整审查见 `docs/stage_reviews/D1-E001_review.md`。
- 补充了已距离压缩 IQ 输入边界的初步成本测量：D 学生 CUDA 单样本平均/P95 为 1.53/1.91 ms，batch 128 为 59.75/62.11 ms，CUDA 峰值分配显存约 359 MB；APX15 FFT 子链 CPU 单样本为 0.83 ms。设备不同、处理链边界有限，不据此宣称速度优势。数据和计时命令见 `experiments/D1-E001/benchmark/inference_cost.json`、`scripts/benchmark_d1_e001.py`。
- 完成 D0/D1 综合报告，回答了检测、参数估计、FFT、教师预训练、联合蒸馏、去教师化、传统算法差距及精度/计算优先级八项问题。下一项建议是精度优先的 Doppler 表示对照；它尚未预注册或启动。见 `docs/stage_reviews/D0-D1_overall_report.md`。

## 2026-10-24：D1-R1 Doppler 误差来源诊断

- **范围：** 使用 D1 的 pilot_validation、三个 seed 的 train/validation 缓存和 A/D formal inference checkpoints；不读取最终 blind 或 calibration。推理在 AISAR/PyTorch 2.8.0+cu128、RTX 3050 Laptop CUDA 上完成，batch 32，峰值分配显存约 548 MiB。复现入口为 scripts/diagnose_d1_r1.py。
- **发现：** A 的 0/5/10/20 dB 预测距离门与 true-range oracle Doppler RMSE 跨 seed 均几乎一致；-10 dB oracle 从约 311 Hz 降至 186 Hz，表明距离选择会额外影响低 SNR，但不是高 SNR 主要误差源。高 SNR Doppler 仍约 20–32 Hz。
- **表征/任务：** 固定 ridge 下池化后 probe 比池化前好，但池化前维度 4160 对比池化后 32、训练样本仅 1024，不能作池化因果结论。final checkpoint 上各任务梯度余弦均值接近 0，样本/阶段范围有限，不能排除训练中的瞬态冲突。仅有 inference checkpoint，KL 共享主干梯度无法重算。
- **教师：** teacher 在 -20/-15 dB 检测 AUC 约 0.51/0.67且 Doppler RMSE 约 370/332 Hz；-10 dB 检测 AUC约0.982但 Doppler RMSE约287 Hz。检测知识和精频率知识不能视为同等可靠。
- **决定：** H-ARCH-001 与 H-OPT-001 仍为 INCONCLUSIVE。D1-E002 优先 pilot 低成本 lag-one 复相干统计作为 S2；S1 只做单 seed 时间读出反证；S3 固定 FFT 混合结果单独归因。D1-E001 G1/G2/G3 保持原冻结结论。
- **产物：** experiments/D1-R1/protocol.md、config.yaml、metrics.json、result.csv、analysis.md；测试 tests/test_d1_r1_diagnostics.py。

## 2026-10-24：冻结 D1-E002 受控 Doppler 读出协议

- **问题与证据：** D1-R1 显示高 SNR 残差在真值距离门下仍存在；池化前 probe 没有支持单纯扩大时间特征的说法；任务梯度诊断不支持直接改损失权重。S2 归一化 lag-one 相干统计因此作为优先纯 IQ 候选，S1 仅一次单 seed 反证，S3 作为 FFT 混合方法单独归因。
- **冻结设计：** `experiments/D1-E002/protocol.md` 与 `config.yaml` 固定 S1/S2/S3 前向、参数量、E1 真值监督、1600 更新、有效 batch 64、train/validation/pilot/calibration/blind 种子、检测 `Pfa=1e-2` guardrail、高 SNR Doppler 主指标、range 非劣界、paired hierarchical bootstrap 与停止规则。protocol/config 测试展开 313 个数据流种子，确认无内部重复且不与登记 D0/D1 种子相撞。
- **状态：** protocol/config 与实现已提交到 `main`；CUDA pilot 已完成，训练/验证仿真数据只存本地可重建缓存，尚未生成 E002 calibration 或 blind。

## 2026-10-24：D1-E002 单 seed CUDA pilot 与筛选器更正

- **执行：** RTX 3050 Laptop GPU（4 GiB）上按 S1→S2→S3 顺序运行，FP32、有效 batch 64，每模型 1600 更新；三者均完整完成。训练耗时分别 160.0、147.0、155.0 秒，峰值已分配显存约 1.09、1.10、1.16 GiB。原始 metrics、训练曲线和初始/最佳检查点已提交；训练与验证缓存未提交。
- **原始 pilot 验证：** S0 高 SNR 平均 Doppler RMSE 为 27.129 Hz。S1 为 24.403 Hz（改善 10.05%），低 SNR 平均 AUC 0.658，range RMSE 0.0690 cell；S2 为 22.811 Hz（改善 15.92%），低 SNR 平均 AUC 0.652，range RMSE 0.0734 cell。二者通过原预注册 pilot screen。S3 为 68.110 Hz；匹配三点 FFT 为 67.963 Hz，未见混合学生增益；S3 另未通过 range guard 与 20 dB 回归门。
- **审计修正：** 初始 `selection.json` 因选择器从顶层读取更新次数，把实际存于 `training.completed_updates` 的 `1600` 次误判为未完成。原 `selection.json` 和 `metrics.json` 保留不改；独立入口只从已保存 pilot 验证指标重算，门槛、数据、模型和预算均不变，也不把 pilot 重标为 blind。测试已增加嵌套字段回归覆盖；`selection_corrected.json` 与原因记录已生成并单独保存。
- **当前门：** 对已保存验证指标重算后，S1、S2 满足原定三 seed formal 候选筛选门；S3 停止。formal 已完成；新 calibration 与 final blind 均未生成。pilot 只用于候选筛选，不作为正式效应证据。

## 2026-10-25：D1-E002 S1/S2 三 seed formal 训练

- **执行：** S1、S2 各按 seed_1/2/3 顺序完成 1600 次更新，共六个 CUDA 训练实例。AISAR、RTX 3050 Laptop GPU（4 GiB）、FP32、有效 batch 64；无 OOM、无预算缩减。S1 每 seed 约 148 秒、峰值已分配显存约 1080 MiB；S2 每 seed 约 138 秒、峰值约 1046 MiB。
- **formal validation 诊断：** 每个 checkpoint 仅按协议选自对应验证集。三 seed 的高 SNR 平均 Doppler RMSE：S0 为 29.09/22.98/25.60 Hz；S1 为 26.22/28.92/26.74 Hz（仅 1/3 seed 点估计改善）；S2 为 21.30/26.27/24.32 Hz（2/3 seed 点估计改善）。这些验证数据用于 checkpoint 和训练诊断，不能替代独立盲测或正式区间结论。
- **状态与下一步：** formal metrics 记为 `COMPLETED`，所有实例都满足更新预算；`formal/model_manifest.json` 已冻结 12 个 checkpoint（S0/S0_D/S1/S2 各 seed）及对应 Git blob，六个新训练实例的逐更新/验证记录保存在 `formal/training_history.csv`。新 calibration 已使用 50,000 个 H0；12 个模型 checkpoint、APX15 和共享能量分数共 14 个校准分数，经验 `Pfa` 均为 0.00998。该段记录为 blind 生成前的历史状态；最终盲测和阶段判断见下方更新。

## 2026-10-25：D1-E002 独立盲测与阶段审查

- **执行与审计：** 按冻结阈值运行一次新盲测：8 个 SNR，每点 4096 H0 + 1024 H1，共 32,768 H0、8,192 H1；未用于模型选择。首次评估已保存固定 blind cache 及除 profile-ML 外的全部预测，但因从 D1 而非 D0 读取 `waveform` 配置而中断。修复提交 `e7404f6` 仅更正评估器范围区间读取并增加重放入口；profile-ML 子集与所有指标均从原 blind IDs 计算，模型、校准阈值和门槛未变。原失败 run manifest 与 partial metrics 均保留，最终清单标记 `COMPLETED_AFTER_REPLAY`。
- **阶段门：** S1 高 SNR 主 RMSE 为 28.30 Hz，对比 S0 26.20 Hz，改善 −8.02%，配对 hierarchical bootstrap 95% CI 为 `[0.378, 4.622]` Hz；Doppler 门未通过。S2 为 23.35 Hz，改善 10.86%，95% CI `[-6.338, -0.173]` Hz，三个 seed 均优于配对 S0；主 Doppler 门通过。S1/S2 高 SNR range RMSE 增量分别为 `0.0104/0.0155` cell，95% CI 上界均低于 `+0.05`，距离门通过。低 SNR Pd 差值区间下界 S1=`−0.0554`、S2=`−0.0939`，均未高于冻结的 `−0.05` 非劣界，因此检测保护门均失败；S2 在 seed_2、−5 dB 的 Pd 为 0.471，另两 seed 为 0.837/0.864，显示跨 seed 波动。
- **传统基线：** 在 0/5/10/20 dB 等权平均的 Doppler RMSE（Hz）为：S0 26.20、S1 28.30、S2 23.35；原生 FFT peak 2.86、原生三点插值 1.49、四倍 FFT peak 0.738、四倍 FFT 三点插值 0.104、lag-one 非学习估计 3.74、APX15 0.122。profile-ML 在每个 SNR 预设的 128 个 H1 样本上为 0.100 Hz。高 SNR 条件 CRLB 平均约 0.098 Hz；S2 误差约为其 238 倍。S2 虽超过 S0 主指标门槛，但没有超过传统精估计器。
- **物理一致性与误差：** 三 seed S2 对统一公共相位旋转 `2.83 rad` 的平均 Doppler 输出变化为 `11.17–14.62 Hz`。对 `±0.625 Hz` 频移的等变残差均值为 `2.94–4.18 Hz`；对 `±5 Hz` 为 `9.79–14.73 Hz`。这些结果显示 lag-one 输入没有令网络自动获得相位不变性或频移等变性。20 dB S2 各 seed RMSE 为 `22.65–28.58 Hz`，而 10 dB 为 `16.67–22.75 Hz`，高 SNR 表现非单调。完整 Doppler/距离四分位、正负频率、Nyquist 邻域、初相、幅度与 SNR 分组见 `blind/metrics.json`。
- **决定：** 按冻结规则作 **STOP / REDIRECT**：S1 未通过主 Doppler 门；S2 虽通过 Doppler 和 range 门，但检测保护门失败，故没有纯 IQ 候选满足合取验收条件。不得据此声称 E002 纯 IQ 结构成功或推翻 D1-E001 结论。D1-E002 候选仅用 E1 仿真真值监督，没有执行新的 E2/E3 蒸馏对照；既有 S0_D 只是历史模型锚点，不能作为本实验新增蒸馏收益证据。
- **环境与产物：** Formal 训练沿用 RTX 3050 Laptop GPU、CUDA FP32、batch 64 和固定 1600 update 预算；重放记录 RTX 3050 Laptop GPU、PyTorch 2.8.0+cu128、CUDA 12.8、batch 64、峰值分配显存约 169 MiB。指标表、全部保存预测、ROC/RMSE-SNR 图、初次失败及重放 provenance 均在 `experiments/D1-E002/blind/`；缓存数据未纳入 Git。审查见 `docs/stage_reviews/D1-E002_review.md`。

## 2026-10-25：D0/D1 E002 后科研主线审查

- **D1-R1 审查补齐：** 新建 `docs/stage_reviews/D1_R1_doppler_diagnosis.md`，记录工程边界、预测门/oracle 误差、probe/梯度证据和 E002 后判断。R1 未证明池化、梯度冲突或教师 KL 是 Doppler 瓶颈的因果来源。
- **E001 统计复核：** 保留 D1-E001 原 G2 PASS。C 三个 seed 的联合指标方向稳定；D 的绝对改善量约 84.6% 来自 seed 1。B/C/D 多策略区间没有家族错误率调整；低 SNR Pd 不在原 G2 保护门内。因此主张范围限定为该协议结果，不称为普遍蒸馏优势。
- **路线决定：** E002 STOP / REDIRECT 只停止本轮 AWGN Doppler 结构搜索，不覆盖 E001 的 GO → D2。推荐的唯一下一项是准备 D2-A 静止杂波单因素协议和强传统基线/可辨识性验证；尚不训练、不生成新 blind。
- **同步文档：** 更新总纲、路线图、当前状态、假设登记、AGENTS/README、详细计划和历史审查索引；旧负结果、冻结门和既有 blind 产物保持原样。

## 2026-10-25：完成 D2-A-R3 阈值迁移与尺度 nuisance 诊断

- **范围与执行：** 按预注册协议先运行冻结 B0–B3 原始分数，再运行两个固定候选 T1/T2。使用新场景种子、20,480 个 H0 校准场景、32,768 个配对验证 H0/H1 场景和 128 个受控尺度场景；CPU 流式处理，未缓存 IQ。候选组完整复放了 T0 分数。R3 是合成开发诊断，不是 final blind。
- **主要证据：** E001 冻结 B3 阈值在 R3 新 16,384 个主 H0 上 `Pfa_CPI=0.064636`（95% Wilson CI `[0.060972,0.068504]`）。E001 的 `23/256=0.089844` 在新率下验证抽样精确尾概率为 0.07019；分层校准/验证 bootstrap 达到该点估计的比例为 0.0216。有限样本可能有贡献，但 H0 的噪声和杂波尺度明确随潜在目标参考幅度变化，故不能把原始偏差只归结为抽样。
- **候选比较：** T1/T2 均通过 G3 幅度与 `X,aX` 尺度检查及 G4 可辨识 H1 保护；两者均未通过 G2 固定阈值迁移。T1 在部分重叠 C 的 Pd/joint 达 0.982/0.982，T2 为 0.926/0.921，B3 为 0.530/0.530。T2 的局部背景易被目标污染，C 条件 Doppler RMSE 约 23 Hz。没有方法完成全部 G1–G4。
- **实现审计：** 发现原候选后处理的 G3 汇总漏掉预注册的受控尺度子门，且 complex64 乘复数因子会带来量化误差。`f81093e` 将相同场景提升到 complex128 进行尺度控制并纳入完整 G3 门；没有放宽容限或改变实验数据。修正后 T1/T2 受控 map 最大差分别为 `3.20×10⁻¹³` 和 `9.06×10⁻¹⁴`。修正后相关测试 45 项通过。
- **决定：** 阶段决定 **REFINE**。冻结 R2 E1 前置条件未通过，未运行学生训练、E2/E3 蒸馏或 final blind。建议下一步独立提交一份 truth-only 学生 pilot 协议提案，预注册学生自己的 H0 校准/验证、任务与物理一致性门；需独立审查批准后执行。总纲、D2-A-E001 冻结结果与协议未修改。
- **产物：** `experiments/D2-A-R3/` 下的协议、配置、metrics、逐场景 H0/H1 记录、指标表、图表和分析；阶段审查见 `docs/stage_reviews/D2-A-R3_review.md`。更新当前状态、动态路线图和假设登记表。

## 2026-10-25：提出 D2-A-E002 独立 truth-only TaskStudent pilot 协议

- **触发依据：** D2-A-R3 的 T0/T1/T2 均未通过固定阈值 G2，当前没有正式获准教师；但这不检验学生自己的真值监督分数。任务级学生仍是研究总纲的核心方向。
- **完成内容：** 新增 E002 任务对齐、H-LEARN-D2-001、协议与配置草案、协议阶段自审。方案沿用 E001 三通道静止杂波和 D1-E001 `TaskStudent`，设置 50k train pairs、5k validation pairs、15k 独立 H0 calibration、15k H0 verification、5k A/B/C development pairs及 1,024 个 D 边界诊断 pairs；预注册一 seed 101、1,600 updates、有效 batch 64、FP32、RTX 3050 CUDA。
- **虚警设计：** 以学生 presence logit 建立独立全局阈值，目标 Pfa=0.05、容许上限0.06；15,000 个独立验证 H0 使用单侧 exact 95% Clopper–Pearson 上限。样本分析说明不要求置信区间机械包含精确 0.05，也不加一个会使真实 5% 时通过率约减半的点估计硬门。
- **状态与边界：** E002 自审为 **CONDITIONAL GO（协议提案）**，不是实验 GO。未实现 runner、未生成数据、未训练、未蒸馏、未生成 final blind；没有 E002 指标或科学结论。R3 的 REFINE 和所有 E001/R3 历史结论保持不变。
- **下一步：** 将本提案送 ChatGPT 审查；用户明确批准后才实现 preflight 和运行一个 truth-only seed。不得自动启动三 seed、E2/E3 或 final blind。

## 2026-10-25：用户批准并冻结 D2-A-E002 v2 课程学习 + K2 A/B/C pilot

- **路线与授权：** 用户明确批准 v2 有界 A/B/C pilot，替代从未执行的 v1 truth-only 提案；研究总纲 v1.1、D1/D2-A/R3 冻结结果均未修改。R3 的 T1/T2 仍未通过固定阈值 G2；本实验将 K2 作为 IQ-only 非阈值软响应，不能据此授予固定阈值教师资格。
- **协议与设计：** 冻结 A 直接真值全难度混合、B 同样本多重集上的 C0→C4 课程真值、C 同课程加 H1-only K2 响应；三个臂共用初始化、模型、训练样本多重集、1600 updates 和有效 batch64。加入新的独立 split、K2 质量门、学生 Pfa 校准/验证、检测/参数/联合任务门、课程与教师筛选信号及物理一致性诊断。
- **执行状态：** 已完成 v2 protocol/config、总纲映射、架构/损失静态审查和路线/假设状态更新。**本记录时尚未实现 runner、生成 E002 数据、运行测试、K2 审计或训练；没有 pilot 结果。** 实验缓存路径已加入 `.gitignore`。
- **提交：** 本预注册与状态更新将作为独立小提交推送到 `main`；代码、测试和实测结果分开提交，外部 GMTI 用户文件与现有实验缓存不纳入。
- **下一步：** 按 TDD 实现 seed/split/配对数据、IQ-only K2、A/B/C matched sampler、CUDA trainer 和评估器；完成物理回归、边界/标签/梯度/小样本过拟合及 K2 audit，再按门控顺序运行。

## 2026-10-25：D2-A-E002 v2.1 单 seed CUDA pilot 完成与缓存归档

- **执行范围：** 按冻结 v2.1 协议运行 A（直接真值）、B（课程真值）、C（课程 + IQ-only K2），各 1600 updates、有效 batch64、seed 101；在 NVIDIA GeForce RTX 3050 Laptop GPU 上顺序训练，FP32，无 OOM。生成的独立 train/validation、teacher audit、H0 calibration/verification、development 和 exact-overlap boundary split IDs 不重叠。K2 审计通过，G1 micro-overfit loss 从 7.1634 降至 0.02834（100 updates），梯度检查通过。
- **任务证据：** development AUC A/B/C=`0.9846/0.9787/0.9815`，Pd=`0.9158/0.8996/0.9000`。三臂均通过 G4 并优于 E0/AUC=0.6103；A 的主 A/10/10 dB joint success=`0.8489`（单侧下限 `0.8201`）。T1 AUC=`0.9863`、Pd=`0.9206`、joint success=`0.9010`，仍不是获准的固定阈值教师。
- **虚警与方法比较：** 学生总体 H0 verification Pfa A/B/C=`0.04573/0.04953/0.04693`，精确单侧 95% 上限=`0.04864/0.05255/0.04987`；总体子门通过，但 C3/SNR0 分层上限达 `0.1953/0.2045/0.2022`，G3 失败。G5 B−A joint-success 差=`−0.1254`（paired 95% CI `[−0.1386,−0.1122]`）；C−B=`+0.0036`（`[−0.0088,+0.0162]`）；B/C 均未达到 A 的最佳 validation loss，无课程或 K2 pilot signal。
- **物理一致性：** 公共相位决策一致率 A/B/C=`0.950/0.947/0.943`；Doppler 变化 p95=`61.7/71.9/62.4 Hz`；目标频移等变残差 p95 最大=`59.6/73.2/73.4 Hz`，G6 失败。独立通道相位扰乱的 AUC 下降约 `0.176–0.208`，说明模型利用了通道相干，但输出的公共相位与频移变换不稳定。整体幅度 0.5/2.0 诊断未见尺度敏感。
- **决定：** 阶段结论 **REFINE**。保留所有冻结门；不做正式三 seed、知识蒸馏确认或 final blind。下一优先项为预注册一个有界诊断协议，定位 C3 虚警失稳与相位/频移等变失败；未经新协议审查不启动后续训练。
- **复现与审计：** 运行/模型代码 SHA=`1f822470b906c733706c1a10f031e7a627e140a8`；原始数据首次生成耗时 `1195.4 s`，完整缓存复用验证 `0.2 s`，评价 `973.3 s`。G5 数值从保留的 `development_per_scene.csv` 按冻结 base-scene bootstrap seed 和 2000 次重采样重算；没有重新训练或重新生成实验数据。完整指标、CSV、checkpoint、逐 update history 和图见 `experiments/D2-A-E002/`，阶段审查见 `docs/stage_reviews/D2-A-E002_review.md`。
- **缓存清理：** 评价完成后删除 D1-E001/D1-E002 可再生成训练/校准缓存及 D2-A-E002 IQ 缓存，共释放 `30,657,060,864` 字节分配空间；24 份缓存 manifest 已归档。D1-E002 独立 blind cache 和本地 GMTI 源码/验收包未变。项目清理前约 `40.03 GB`，清理后约 `8.73 GiB`；审计见 `docs/maintenance/CACHE_CLEANUP_2026-10-25.md`。
- **验证：** `conda run --no-capture-output -n AISAR python -m pytest -q`：**250 passed**。定向 E002 runner 测试：2 passed。
- **补充架构/损失审查：** validation 中预测/真实 range-cell 错配率 A/B/C=`10.30%/11.14%/10.96%`；用同模型真实 range cell 读取 Doppler 后 RMSE=`40.95/56.03/59.55 Hz`，预测门为 `46.66/58.55/60.03 Hz`。Development 错配率为 `10.04%/11.02%/11.30%`。错门会增加部分误差，但真值门下误差仍高，尚不支持其为主要 Doppler 瓶颈；H-OPT-D2-001 保持 INCONCLUSIVE，冻结损失不改。K2 audit 的 pooled C3 允许 C，但 partial-overlap/SNR0 子组 localization 仅 4.7%。详见 E002 架构/损失审查和阶段分析。

## 2026-10-28：D2-A-R5-A Small/Medium 容量 pilot 完成

- **执行：** R5-A capacity runner 与配置提交 `1678a468d5c150ff734d80ad40cf75407a49651f`。在 RTX 3050 Laptop `cuda:0`、AISAR / PyTorch 2.8.0+cu128 上，Small width32 复用 seed101 R5-A S0 checkpoint；Medium width96 独立初始化。两者共用冻结 E003 train/validation IDs、R5 pair plan、E1 truth-only、FP32、有效 batch64和12,800 updates。Medium microbatch16，无 OOM，训练用时3,951.3秒，CUDA peak allocated 820,737,024 bytes。
- **可学习性：** Medium 的16对 scene、200-update过拟合检查通过，task loss `4.1465→0.00238`，presence/range-cell准确率均1.0；该检查只验证工程学习路径。
- **任务收益：** Medium−Small overall joint-success差`+0.07178`（95% CI `[+0.06201,+0.08204]`），可辨识C2/C3 joint差`+0.13086`（`[+0.11230,+0.15088]`）。C2/C3 range/Doppler RMSE比`0.839/0.713`；primary-gain门通过。
- **保护与判定：** 普通H0 verification Pfa=`0.05017/0.04797`，但C3 H0 Pfa=`0.20410/0.18896`，两模型均高于0.10限制；相对C3 Pfa差值区间虽未显示Medium恶化，绝对保护门失败。容量阶段判 **REFINE**，H-CAP-002为有限支持；不运行Large、KD或final blind。数据是E003既用开发材料，非blind，单seed不估计seed方差。
- **交付：** `experiments/D2-A-R5/R5-A/capacity/` 含config、preflight、G1、training/metrics、per-scene输出、图和analysis；`docs/stage_reviews/D2-A-R5A-capacity_review.md` 提供Codex阶段自审。当前总纲未修改。下一优先阶段为R5-B一个轻量原始IQ时频注意力候选的架构调查和有界pilot设计。

## 2026-10-28：D2-A-R5-B TFAS 单seed CUDA pilot 完成

- **协议与运行：** 按 R5-B v1.0，seed101、E1物理真值监督、12,800 updates、有效batch64、FP32、固定训练集8遍，在 RTX 3050 Laptop `cuda:0` 完成；模型79,638参数，无 CPU 回退或OOM。代码/runner SHA=`55e7c7810e6cbb0277f79b1b6b4c401052b559dc`。模型和runner/训练测试共26项通过；compileall与配置检查通过。
- **训练与G2：** 16-pair G1 loss `5.57683→0.000867`，下降99.984%。development AUC=`0.96139`（95% CI `[0.95760,0.96512]`），range RMSE=`4.595 cells`、Doppler RMSE=`82.00 Hz`，优于训练真值常值基线，G2通过。共同验证最低task loss checkpoint为update12400，`1.71301`；末段已近平台但仅一个seed。
- **G3失败：** TFAS总体Pd/joint=`0.8020/0.5054`；C2/C3 joint相对Medium差`−0.60449`（95% paired-scene CI `[-0.62598,-0.58350]`）；range/Doppler RMSE比`2.385/4.599`。C3 H0 Pfa=`0.15186`（单侧exact95%上限`0.16138`），不满足0.10绝对门。虽然Pfa低于Small/Medium基线，但任务退化显著且三者均未过绝对门。
- **物理诊断及边界：** 公共相位32旋转与频移诊断复用已见development/verification IDs；phase下H1判决一致率中位数0.9258，Doppler变化p95最高122.18 Hz；frequency shift未呈现稳定等变。仅为描述性诊断。训练、校准、验证与开发数据来自已用E003/R5-A ID；单seed、非blind，不估计seed方差。
- **决定：** G0/G1/G2通过、G3失败。停止本TFAS候选，不进行额外seed、KD、Large、调参或final blind；不外推否定所有非CNN/attention，也不声称CNN或任何教师已通过全门。H-ARCH-003对当前TFAS为NOT_SUPPORTED；H-LEARN-002保持INCONCLUSIVE。阶段建议 **REFINE**，下一唯一任务为独立ChatGPT阶段审查；审查前不启动R5-C/D或新架构。
- **交付与复核：** R5-B protocol/config、训练history、模型checkpoint、逐场景输出、metrics、物理诊断、曲线和 `docs/stage_reviews/D2-A-R5B_review.md` 已归档。可重建IQ缓存（manifest 90 files，14,232,309,192 bytes）未纳入Git；项目目录当前约15 GiB，低于20 GiB最终上限。ChatGPT审查请求将在结果提交后记录准确结果SHA。
