# D2-A-R6-D：联合检测与连续参数估计学生 pilot

**协议版本：** 1.0  
**状态：** 预注册；场景生成和科学训练尚未开始。  
**研究总纲：** `docs/RESEARCH_CHARTER.md` v1.3。  
**最终盲测：** 未授权生成。

本阶段检验共享多任务学习和一次软候选双向交互能否改善物理真值监督学生的最终检测与参数估计。R6-C在12,800步的延长门未通过；那只表示没有达到该阶段的成对延长资格，既不表示网络已收敛，也不提供更长预算的收益估计。R6-D采用固定25,600步，让所有臂在完全匹配的训练预算下筛选。25,600步是本阶段比较预算，不是能力上限。

## 1. 研究问题与假设

1. 在同一训练场景和预算下，共享检测/参数表征（J1）是否优于只训练检测或只训练参数的专用目标？
2. 在J1基础上，软候选池化和单轮参数到检测反馈（J2）是否增加可辨识C2+C3场景的最终joint success？
3. 若训练曲线在25,600步仍改善，是否说明当前模型在该预算下尚未观察到验证平台？该证据不外推为无限训练仍有收益或已达到/未达到通用极限。

本阶段只有一个训练seed，是模型与优化路线筛选；不估计训练seed方差，不进行多seed晋级或final blind。蒸馏、教师图、教师可靠性权重、外部先验和参数搜索均不进入本阶段。

## 2. 冻结仿真和场景划分

保持D2-A-E001三通道静止杂波单目标复AWGN模型：PRF=1300 Hz、130脉冲、32距离单元、R6-A的12个条件。每个paired base scene生成独立H0/H1 IQ；H1 truth仅供训练loss和离线评价。训练和验证为H0/H1配对scene；校准/验证H0与H1开发集分别独立。所有臂使用完全相同的scene ID及IQ。

| Split | 样本 | 用途 |
|---|---:|---|
| Train | 51,200 paired base scenes（各含H0/H1 CPI） | 四臂统一训练数据 |
| Validation | 5,000 paired base scenes | 训练历史和各臂自身checkpoint选择 |
| H0 calibration | 120,000 CPI，每条件10,000 | 仅校准各检测器的global Pfa=0.05阈值 |
| H0 verification | 120,000 CPI，每条件10,000 | 独立Pfa迁移和分条件保护评价 |
| H1 development | 6,144 CPI，每条件512 | Pd、参数误差和joint success；只作development |

Train各course配额相同；course内按条件均匀分配16-scene块，余数由冻结assignment seed决定。Train和Validation使用四个相互间隔5,000,000的course seed block；条件内seed offset为850,000，chunk offset为1,009，正式preflight须证明区间和生成的scene ID均无冲突。Evaluation沿用R6-A的确定性condition/chunk分块，使用本协议独立seed base。G1保留seed `2200000000`及其16-scene小样本ID；该seed不与科学划分交叠。

R6-D ID不得与D2-A-R5、R6-A、R6-B、R6-C或G1测试ID重合。正式IQ只物化Train和Validation，evaluation按冻结ID分块确定性生成并流式计算，避免保存H0全量IQ。Train/Validation IQ估算占用11,222,016,000 bytes；开始生成前须至少有该值加4 GiB可用空间。所有临时文件只允许由R6-D owner manifest标记和清理。

## 3. 比较臂和公平性

| Arm | Student | 优化目标 | 可声明的评价 |
|---|---|---|---|
| J1-D | width96 TaskStudent共享主干 | H0/H1 presence BCE | Pd/Pfa/AUC；参数质量不作模型结论 |
| J1-P | 同width96 TaskStudent | 仅H1 range分类、range offset、周期Doppler | H1条件参数误差；无检测或Pfa资格 |
| J1-J | 与J1-D/P相同模型与初始化 | presence + range + Doppler | 完整检测/参数/joint |
| J2-J | 相同宽度的TaskStudent-compatible主干 + learned 130→520候选投影、null候选、一次有界参数反馈 | presence + 候选分类 + 软候选参数池化 | 完整检测/参数/joint |

J1-D、J1-P、J1-J由seed505建立完全相同的TaskStudent初始化。J2-J复用相同初始化seed和相同结构主干；额外候选层导致参数量不同，必须报告参数、显存和计算差异，不声称参数匹配。四臂共享训练scene/顺序、每update的H0/H1配额、effective batch=64、optimizer updates=25,600、AdamW、学习率计划、FP32、gradient clip、validation/恢复间隔和seed。P-only每update仍抽取相同H0/H1 batch，但H0不产生参数loss；其梯度样本数较少，作为任务专用参考臂解释，不与joint臂声称计算量完全匹配。

**Training budget：** 每臂固定25,600 optimizer updates；每update 32 H0 + 32 H1，故每个presence各曝光819,200 CPI，合计1,638,400 CPI slots；Train中每个H0/H1 scene平均约16次曝光。Optimizer为AdamW，初始LR=1e-3、weight decay=1e-4、gradient clip norm=1.0，LR在全部25,600步上按一段cosine从1e-3降到1e-5。microbatch从16开始，只有OOM可降到8或4，并以梯度累积保持effective batch64。禁止AMP或CPU回退。

history每400步记录一次；恢复checkpoint每800步；不逐update写日志。对各arm，只用对应validation objective选择checkpoint：J1-D最小presence BCE；J1-P最小H1参数loss和；J1-J/J2-J最小各自预注册truth-task总loss。不得读H0 verification/H1 development选择checkpoint、结构、阈值、loss或训练预算。报告训练、验证曲线及25,600步末的趋势；末端仍下降时写明“预算内未观察到平台”，不推断网络极限。

## 4. 损失定义与输出边界

- J1-D：`L_presence`。
- J1-P：`L_range_cell + L_range_offset + L_circular_doppler`，仅对H1计算；H0三个参数项均精确为零。
- J1-J：`L_presence + L_range_cell + L_range_offset + L_circular_doppler`。
- J2-J：`L_presence + L_candidate + L_range_offset + L_circular_doppler`。candidate loss在H1对连续truth映射到最近520×32单元；range与Doppler连续loss基于条件non-null soft candidate posterior池化结果；H0不计算参数loss。

所有主项权重为1，range smooth-L1 beta=0.1；不使用teacher loss。Doppler按PRF=1300 Hz周期误差；±PRF/2视为同一端点。模型forward只接收observable IQ。J2 feedback末层std≈1e-3小幅非零初始化，以保留首步检测loss到参数分支的梯度；参数loss通过软候选posterior回传至candidate score。具体数值、null handling、H0 masks和两侧detach探针在G1中冻结并测试。

## 5. 主要指标、阈值和验收门

各检测器分别用同一新H0 calibration流中`higher`经验95%分位数设global阈值，比较符为`>=`。仅检测臂和joint臂有检测阈值；J1-P不校准检测阈值。主要endpoint为J2-J相对J1-J在C2+C3 H1上的paired joint-success差，按base_scene_id做2,000次bootstrap，报告双侧95% CI；pilot有实质正收益的筛选标准为点估计至少+0.02且CI下界>0。它只是单seed开发筛选证据，不触发正式seed/blind自动晋级。

保护和次级指标：

1. H0 verification总体Pfa及one-sided exact95% upper；每条件和C3 pooled Pfa；C3六个condition cell用Bonferroni familywise exact upper，要求各≤0.10才可称通过C3虚警保护。
2. H1整体、C2+C3、C3及六格Pd；AUC；距离/Doppler RMSE、bias、门错率；检测命中、正确候选和joint success。
3. J2相对J1-J总体Pd退化不得超过0.02；range和Doppler RMSE比率的95% CI upper不得超过1.05，作为候选保护门。
4. D-only和P-only的专门指标如上表；不为其未训练任务填充“成功”指标。
5. 报告四臂逐scene预测、差值bootstrap、训练seed、独立scene数、训练暴露、updates、模型参数、GPU显存峰值、墙钟时间和验证曲线。

threshold、model selection、candidate、C3 analysis均不得用verification/dev结果反向调整。本阶段没有final blind；重复使用R6-A/B/C或R5开发ID一律违规。

## 6. 阶段门、停止与结论边界

- **G0：** main/远端提交、protocol/config哈希、seed与ID disjointness、磁盘空间、RTX 3050 `cuda:0`及FP32可用性均通过；正式场景生成前记录preflight。
- **G1：** CPU contract tests、CUDA输出/梯度/显存检查、每种架构的16-pair/200-update微过拟合检查通过；有效batch保持64。失败时只修复已诊断实现问题，或停止对应结构。
- **G2：** 四臂完成匹配单seed开发pilot；按前述定义完成checkpoint冻结后再运行独立calibration、verification和H1 development。
- **停止：** 任一模型出现NaN/梯度断路/无学习、OOM在microbatch4仍无法保留有效batch、协议/ID错配、H0/H1监督泄漏则暂停该臂并调查；不通过G1不进入完整训练。训练末端未平台不允许单臂追加updates。

若所有学习臂均未优于简单检测/参数基线，记录负结果并评估R6路线，不无限扩架构。即使J1/J2在开发集改善，若C3虚警保护失败也不能宣称具备可靠GMTI检测资格或启动formal seeds/blind。R6-D结果只解释当前三通道静止杂波仿真及注册条件。

## 7. 资源和复现

GPU：NVIDIA GeForce RTX 3050 Laptop GPU，`cuda:0`，4 GiB；FP32，模型顺序驻留。J2训练从microbatch16开始；记录实际peak allocated/reserved、运行时间和吞吐。所有训练checkpoint包含模型、optimizer、scheduler/update、seed、历史、sampler状态、protocol/config哈希及git SHA，可确定恢复。评估输入分块生成，不缓存所有H0 IQ。完整项目最终占用须低于20 GiB；删除只能依据R6-D owner manifest，保留代码、scene manifests、checkpoint、history、metrics、审计和高价值预测。
