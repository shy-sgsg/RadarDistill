# D2-A-R6-B：低虚警诊断与首个受控改进

**状态：** 协议 v1.0 已预注册；R5-C已完成，但R6-B须等待R6-A及其资源门通过后执行。R6-B H0/H1数据与R5-C/R6-A完全独立。未达到样本量/资源门的工作点标为未验证。

## 1. 研究问题与决策分支

R5-A Medium在已用开发材料上的 C3 H0 Pfa=0.18896。R6-A先在同场景下判断它是否属于条件内不可分，或固定混合阈值在条件间迁移失效。R6-B复现该现象，并按 R6-A 的**development-only**证据选择一个预注册改进分支；verification数据不可用于分支选择。

- **分支 S（可分但迁移失效）：**若C3 pooled与partial-overlap/0 dB/CNR20硬格条件ROC满足下述可分性门，而全局阈值Pfa随条件/可观测输入尺度明显变化，则比较冻结Medium原始IQ与一个预定义的尺度归一化输入。每CPI尺度固定为`s(X)=sqrt(median(|X[c,n,r]|²))`，候选输入为`X_norm=X/max(s(X),1e-12)`；不输入真值SNR/CNR/目标标签。不得追加按条件阈值、窗口搜索或第二种归一化。单调标量校准只改变分数刻度而保持排序，不能单独修复全局阈值的条件迁移，因此不作为改进臂。
- **分支 R（条件ROC差）：**若困难C3条件ROC本身低，则比较相同初始化/数据/update预算的truth-only基线与仅在训练split上进行H0 hard-negative重采样的学生。困难样本排序分数仅来自训练数据；不从calibration/verification中挖掘，不更改网络结构。
- **分支 U（证据不足）：**若R6-A区间无法区分上述两种来源，先完成H0分数/尺度分布的确认性诊断；不启动训练改进。

R6-A分支选择只允许使用H0 calibration的条件分数/可观测IQ尺度关系与H1 development条件ROC；不得读取、挑选或依据R6-A H0 verification结果。分支触发规则为：若C3 pooled AUC点估计≥0.75且95% bootstrap下界>0.65，partial-overlap/0 dB/CNR20硬格AUC点估计≥0.70且下界>0.55，并且以0.05全局阈值计算的C3 H0 calibration各格Pfa极差≥0.05、raw score与`s(X)`的Spearman `|rho|≥0.30`且95%区间不含0，则选S；若C3 pooled AUC的95%上界<0.75或困难硬格AUC上界<0.70，则选R；其余情况选U。所选分支、计算和配置须在查看R6-B H0 verification数据前记录并冻结。另一个分支留待独立协议，不在本阶段展开。

若选R，使用新R6-B training/validation split训练两个Medium width=96模型：matched truth-only基线和训练期H0 hard-negative重采样模型。每个update固定32个H1、32个H0；两个臂的H1 scene-ID顺序及条件配额完全相同。H0槽位从同条件H0训练池独立抽取，不要求与该H1共享base_scene_id；本训练目标没有成对H0/H1损失。基线按条件在H0训练池内均匀抽样；干预臂对每个条件的H0槽位，50%从该条件最高分四分位困难池抽样、50%在该条件内均匀抽样。困难分位由训练split内冻结的R5-A Medium checkpoint分数计算，固定采样seed；不从validation、calibration、verification或development挖掘。两模型从相同seed-303初始化、使用相同优化器、有效batch64和12,800 updates；仅H0采样改变。checkpoint均按相同validation truth-task loss规则选择。若选S或U，不运行学生训练，training/validation seed block保持保留但不生成。

## 2. 低虚警工作点和样本量

首轮工作点为 `Pfa_CPI=0.05、0.01、0.001`；0.0001只在下方样本扩展可行时作探索。单位是完整CPI最高候选分数造成的一次检测/虚警事件，不是每个RD CUT。所有方法用同一H0 scene ID。

R6-B的主H0 calibration和独立H0 verification各 `120,000` CPI，12个注册条件格每格 `10,000`。H1 development为每格512，共6,144。使用分块流式仿真和评分，只保存紧凑scene/seed manifest和scores/predictions，不保存H0/H1评价IQ或教师RD图。仅分支R生成训练缓存，预估paired train/validation IQ约11.22 GB，训练后按清单清理。

单个预注册总体在零误警时，单侧95% exact binomial上界≤目标Pfa所需样本量分别约为：0.05:59；0.01:299；0.001:2,995；0.0001:29,956；1e-6:2,995,731。该表是独立验证的零事件界，不是足够精确估计极端分位数的通用数据量。R6-B首轮的120k校准约提供120个0.001尾样本，120k验证约对应120次误警；C3六格各有10k H0。报告精确区间，不预设一定能通过；若单侧上界未达到目标，该工作点明确记为“未获资格”，不能用点估计冒充资格。

0.0001需要约1M级别独立校准/验证才能获得约100个尾部样本；只在CPU仿真吞吐、磁盘余量和正式预算允许并预先记录补充配置时执行。不得将1e-4 exploratory结果外推为各条件资格。1e-6不属于本阶段。

## 3. H0/H1生成、校准与评价

12格定义固定为R6-A同一物理条件表。R6-B各方法/arm共用一套H0 calibration、H0 verification、H1 development样本；训练/验证/校准/验证/开发scene ID互斥。H0/H1生成器、组件功率定义和几何配置均沿用E001冻结物理配置。

每方法对整个CPI输出一个presence分数和候选。阈值仅由H0 calibration的全局等条件混合估计，各目标点用冻结的 `higher`顺序分位数，规则`score >= threshold`；记阈值、`>`/`>=`率与ties。验证中绝不重调阈值或选择模型。

按目标点报告：

- H0 pooled、C3 pooled、C3六格Pfa和单侧 exact Clopper–Pearson 95%上界；C3六格主门使用Bonferroni family-wise α=0.05/6，其他格按冻结补充指标报告；
- H1 pooled、C3、每格Pd/ROC/AUC；负样本AUC来自独立同条件H0 verification score；
- 全H1与检测命中/正确候选的range、PRF回绕Doppler MAE/RMSE/bias和joint success；
- H0原始分数随条件、噪声/杂波组件功率代理及可观测尺度估计的关系；真值功率组件仅作离线诊断；
- R6-B改进与匹配Medium比较的配对base_scene_id bootstrap差值和置信区间。

阈值校准不使用真实场景SNR/CNR、H0/H1标签门或目标参数作为运行时输入。按条件分别校准的ROC曲线只用于判断可分性；部署式的主Pfa始终由一个全局阈值报告。

## 4. 低虚警资格与任务保护

工作点资格必须分别报告点Pfa与精确单侧95%上界；若预注册门的上界未达标则该工作点失败或证据不足，不以ROC插值、校准集Pfa或CUT级虚警替代。主要难度保护为C3 pooled和partial-overlap / SNR0 / CNR20 hard cell。

任何改进都必须同时报告Pd、全H1与正确候选参数误差和joint success。降低Pfa但使Pd或joint success明显退化，判为失败。只在条件AUC仍高、分数尺度偏移时考虑背景归一化/校准；若条件AUC差，先改善representation或H0覆盖。统计估计不确定时判REFINE，不靠在同一verification反复搜索。

## 5. 资源、门与交付

- 继承R6-A种子扫描要求并使用独立1.3B–1.5B reserved seed blocks；执行前扫描所有已知记录，不自动移位。
- RTX 3050 Laptop GPU训练时单模型驻留、FP32、有效batch64、按microbatch梯度累积。R6-B的H0校准与验证共240,000 CPI，按R5-C实测CPU仿真吞吐约113 CPI/s估算，生成约35分钟；加H1 development按约40分钟级CPU预算预估，不含GPU评分。若分支R训练，两个12,800-update模型顺序训练；目前没有可信训练时长估值，先用16-pair/200-update G1测显存和吞吐。所有GPU工作等待R5-C正式学生训练释放GPU。
- **G0：** 独立seed/ID、文件空间、R5-C GPU释放、配置和方法实现通过。
- **G1：** H0/H1信息隔离、全局阈值来源、exact CI、固定C3分层、对照配对均通过。
- **G2：** 仅当H0 verification上界和H1任务保护共同通过，方法才可在对应工作点晋级；不通过时保留失败和效应区间。
- 保存协议/config、seed/scene manifests、分支选择记录、紧凑预测/score、训练历史（若分支R）、资源记录、metrics和analysis。数据不得与R5-C混为一组。

## 6. 停止条件

若模型条件ROC良好而固定全局Pfa受尺度/背景影响，优先推进一个可观测尺度/校准候选；若条件ROC在C3本身差，优先改善可观测特征与训练H0覆盖；若候选只降低Pfa而损害Pd/joint，则停止该候选；若结果区间不足以支持任何归因，则保留 `INCONCLUSIVE` 并只提出一次有明确样本增益的后续协议。不得在R6-B启动teacher timing、可靠性权重或多架构网格。
