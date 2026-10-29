# D2-A-R6-C：模型规模、训练预算与训练数据规模

**协议版本：** 1.2
**状态：** 预注册；实现和G0/G1尚未完成，未生成R6-C实验数据。
**研究总纲：** `docs/RESEARCH_CHARTER.md` v1.3。
**最终盲测：** 未授权生成。

**版本记录：** v1.1于2026-10-29在任何R6-C数据生成前，将续训设计明确为Medium与一个validation筛选的大模型组成匹配延长对；两臂共同满足门时一起延长，否则一起停止。v1.2于2026-10-29在任何数据生成前，修订确定性分块seed布局，使Train core可由Train extended按同一条件分块精确嵌套，且所有课程/条件seed区间互不重叠。matched-extension要求落实R6执行计划；v1.0和v1.1均保留在Git历史中。

## 1. 研究问题与假设

本阶段服务于RadarDistill通过物理仿真真值训练任务级GMTI学生的总体目标，重点对应Q3/Q4/Q6/Q10：在相同可辨识信号模型下区分模型容量、训练充分性和独立训练数据多样性对检测、距离和Doppler任务的影响。R6-C不是R6-B hard-negative结果的补救实验；R6-B冻结训练不追加。

检验以下范围有限、可证伪的假设：

- **H-CAP-001-R6C（训练预算）：** 对至少一个受测模型，在相同固定训练场景上从12,800增加到25,600或51,200 updates仍能改善独立验证任务指标，且没有明显扩大训练-验证差距。若平台或过拟合门触发，停止该模型续训。
- **H-CAP-003（模型规模）：** 在同一TaskStudent拓扑、相同训练数据、effective batch、updates、损失和学习率计划下，约0.29M或约1.01M活动参数的候选可比Medium（约0.07M活动参数）改善验证及独立开发任务。扩大容量也可能无增益、增加优化困难或加重过拟合。
- **H-CAP-004（训练数据规模）：** 固定Medium初始化、更新数、effective batch、损失和计划时，使用102,400而非51,200个唯一训练场景对可改善独立验证/开发指标。若未达到预注册实质增益门，则不追加曝光匹配搜索。

12,800 updates不预设为充分预算或性能上限。历史R5-A支持特定模型/数据下长训练仍可改善；R6-B在12,000时验证task loss最低、12,800略回升，说明趋势依赖模型和训练条件。两项历史证据均不预判R6-C结果。

## 2. 冻结仿真和数据划分

沿用D2-A-E001的三通道静止杂波单目标复数IQ模型：PRF=1300 Hz、130脉冲、32距离单元、原条件表及H0/H1标签规则。所有模型对同一评价split使用完全相同的IQ场景。训练/验证/校准/验证集/开发集使用新scene ID，不复用R5-C、R6-A或R6-B；数据生成后运行独立ID与seed block审计。

| Split | 样本数 | 用途 |
|---|---:|---|
| Train core | 51,200个H0/H1训练场景对 | 初始模型规模比较；按条件/课程配额均衡 |
| Train extended | 102,400个训练场景对，包含同一注册Train core及51,200个全新场景对 | 仅用于固定更新数的Medium数据规模对照 |
| Validation | 5,000个独立H0/H1场景对 | 训练趋势、checkpoint选择与延长门 |
| H0 calibration | 120,000个CPI，每个注册条件10,000 | 为每个模型单独校准同一全局Pfa阈值 |
| H0 verification | 120,000个CPI，每个注册条件10,000 | 独立阈值迁移/条件虚警评价；不得参与选择 |
| H1 development | 6,144个场景，每格512 | Pd、距离、Doppler及joint开发评价；不得称为盲测 |

Train core为Train extended的确定性嵌套子集。训练分配按课程/条件分层到16场景配对块，训练块种子按“该split课程seed base + 课程内条件序号×850,000 + 条件内块序号×1,009”生成。课程seed base间隔5,000,000；Train core与Train extended使用相同训练seed base和条件配额规则，core的每个块与extended中同条件、同seed、同16场景块逐一相同。按已注册条件数和最大扩展块数，课程内各条件的seed区间不重叠，最大总偏移为4,518,394，小于5,000,000课程间隔；train/validation使用不相交seed blocks。数据生成后再独立核验scene ID、条件和seed集合。Train/validation按H0与H1平衡采样；每个有效batch含32个H0、32个H1 CPI。H0仅使用presence真值；range/Doppler损失只对H1计算。训练代码不得使用场景条件以外的真值生成困难样本或改变采样权重。

训练IQ可由stage-owned memmap暂存。仅按R6-C owner manifest在阶段结束删除可重建训练缓存；保留scene/seed manifest、配置、审计、checkpoint、history、预测和结果。数据容量不足时不得删除用户文件，先停止数据扩展并记录；最终整项目占用须低于20 GiB。

Train extended与validation cache按R6-B相同的scene张量布局估计为21,445,632,000 bytes。数据生成前要求文件系统至少保留4 GiB余量；实际G0以实时可用空间和预分配大小复核。若不满足，不生成102,400对缓存或清理其他数据。

## 3. 模型与训练设计

使用同一个无物理前端的原始复IQ实虚平面TaskStudent拓扑，只改变宽度；所有候选不启用训练中未参与E1真值损失的辅助spectrum head，以避免计入未训练参数。输出均包含presence、range分类/连续偏移和周期Doppler。参数量均从实际模型实例计算并写入运行记录。

| 候选 | 宽度 | 活动训练参数 | 推理参数 |
|---|---:|---:|---:|
| Medium | 96 | 70,277 | 70,277 |
| Large | 200 | 292,005 | 292,005 |
| XLarge | 376 | 1,012,197 | 1,012,197 |

旧R6-B Medium `82,887`计数含未被E1损失调用的spectrum head；R6-C另报告该存储/声明计数以供历史对照，不把未获梯度的参数算入活动训练规模。R6-C参数比较只使用上表同口径的活动训练和推理计数。

### 3.1 初始matched模型规模pilot

先顺序训练Medium、Large、XLarge，各一个初始化seed=404，物理仿真真值监督E1。三臂共享Train core、validation、H0/H1采样计划、有效batch=64、初始12,800 optimizer updates、损失权重、AdamW优化器、weight decay、gradient clipping、FP32、验证与checkpoint规则；唯一计划内因素是模型宽度。模型顺序驻留GPU，不同时占用显存。effective batch保持64；在本阶段内不把扩大batch与模型规模/训练预算效应混合。可依据实际显存降低microbatch并用梯度累积保留effective batch。

起始学习率和schedule沿用R6-B Medium truth-only设置：AdamW，初始LR 1e-3、weight decay 1e-4、gradient clip norm 1.0；首段6,400步cosine至1e-4，次段6,400步cosine至1e-5。模型初始化均由seed=404和候选width确定；同seed不表示不同张量形状具有逐元素相同初始化。

### 3.2 预注册训练延长门

三个模型均完成12,800步初始screen后，仅用共同Validation task loss选择最优的一个Large/XLarge候选，与Medium组成后续延长对；不依据H0 calibration、H0 verification或H1 development选该对。固定验证点为11,200、11,600、12,000、12,400、12,800。只有Medium与所选大模型**两臂各自同时**满足以下条件，才将延长对一起训练至25,600；另一大模型保留12,800步结果：

1. 11,200至12,800的validation task loss相对下降至少1%；
2. 四个相邻间隔中至少三个后点的task loss不高于前点0.5%以上；
3. 11,200至12,800间，presence/range/Doppler验证损失和range、Doppler RMSE均未恶化超过5%，AUC下降不超过0.01、预测距离门错配率增加不超过0.01；
4. `max(0, validation_task_loss - interval_train_task_loss)`在末点相对11,200最多增加0.01。

若延长对进入25,600阶段，LR从1e-5以cosine降至1e-6，固定评估点每400步记录。是否延至51,200使用对应的24,000、24,400、24,800、25,200、25,600验证点及完全相同的数值条件；须由Medium与所选大模型两臂同时通过，才将二者一起延长至51,200，否则两臂均停在25,600。初始12,800筛选中未入选的候选不进入扩展阶段。训练loss、验证指标和资源记录每400步保存，不逐update追加日志；resume checkpoint最多每800步更新。

训练轨迹是筛选证据，不等于收敛证明。12,800/25,600/51,200步的模型规模比较只在共同训练至该update的Medium/所选大模型之间进行，使用相同数据、effective batch、优化与验证规则；最佳checkpoint只由validation task loss选择。训练-验证差距、唯一scene暴露、固定数据遍历次数和实际optimizer updates都要报告。

### 3.3 独立训练数据规模对照

初始规模pilot完成后，Medium以相同seed=404初始化，训练12,800 updates，分别使用嵌套Train core 51,200对和Train extended 102,400对。有效batch64、每步H0/H1平衡、learning-rate schedule、validation、loss和checkpoint选择完全一致。两臂看到的总训练examples相同；51,200臂重复遍历更多次，102,400臂接触更多唯一场景。该比较识别固定updates与batch下的数据多样性/重复暴露差异，不将其解释为与训练step无关的一般数据定律。除非此预注册对比显示明确任务收益且无保护退化，不启动额外曝光匹配搜索。

## 4. G0/G1前置工程门

- **G0（环境/数据）：** 确认main提交、干净的tracked工作区、配置版本、CUDA设备与可用显存、磁盘余量、scene/seed ID独立性、仿真和训练代码SHA。训练必须在NVIDIA RTX 3050 Laptop `cuda:0`执行，不允许静默CPU回退。用户未跟踪数据不得删除或覆盖。
- **G1（模型/梯度/资源）：** 三种宽度严格输出期望shape且有限；E1任务损失所有活动模块得到有限梯度；16个固定H0/H1训练样本过拟合200 updates时loss下降；在目标GPU记录microbatch候选显存/吞吐，CUDA OOM只通过降低microbatch处理并保持effective batch64。H0/H1须在每个microbatch内平衡，故候选microbatch最小为2；若microbatch=2仍不满足显存门，停止该候选。CPU仅用于小型结构/数值单元测试，不作为正式训练设备。
- **G2（pilot完整性）：** split/scene ID互斥、相同训练计划、验证checkpoint规则、参数统计与训练资源记录均通过；否则不得生成H0/H1开发评价结果。

## 5. 指标、选择及阶段门

主要模型规模比较在12,800共同update点进行。训练趋势与所有阶段best validation checkpoint均报告；每个候选的正式开发预测使用各自validation选择的checkpoint，但不得读取H0 verification挑选checkpoint或模型。阈值只由共同H0 calibration分别校准，使用全局等条件混合分布、`higher`分位数及`score >= threshold`规则。

报告总体与12条件的validation task/subtask loss、AUC、Pd、Pfa、range RMSE/bias、PRF回绕Doppler RMSE/bias、hit/correct-candidate误差、joint success、训练-验证gap、训练种子/scene不确定性、更新数、唯一样本暴露、固定数据遍历数、活动训练/推理参数量、FP32 CUDA峰值显存及墙钟。H0 verification报告 pooled/C3/六个C3格Pfa及单侧exact95%上界；H1 development报告同场景配对bootstrap区间。0.05为主要虚警点，0.01为次级，0.001作条件允许的开发诊断；0.0001及1e-6不作R6-C资格主张。

按固定验证task loss筛checkpoint后，仅在H0 calibration上对每个模型校准阈值；H0 verification只能作独立报告。候选优劣不得仅按validation loss判断，须同时看Pd、参数估计、joint和C3虚警保护。所有开发数据均参与阶段筛选，不能重命名为final blind。

本阶段pilot只用seed404，不能估计训练seed方差。若一个或多个候选在共同update点显出预注册任务收益且未明显恶化C3虚警/其他任务，最多选两个候选进入后续至少三个独立训练seed确认；若没有清晰收益或保护失败，保留负结果并按R6路线评估是否停止增加容量，不自动开展蒸馏或final blind。完整final seed和blind生成仅在R6后续路线中模型、数据与阈值全部冻结后另行执行。

主要模型规模端点为global `Pfa_CPI=0.05`工作点的可辨识C2+C3 joint success。Large/XLarge相对Medium的有意义增益预设为至少`+0.02`，其配对scene bootstrap 97.5%区间下界须高于0（两项候选对比Bonferroni控制family-wise alpha=0.05）。其他保护包括总体Pd配对区间下界不低于`-0.02`，range与Doppler RMSE比值的95%上界不超过1.05；同时报告总体及各条件差异。C3 pooled与六格Pfa采用独立H0 verification和冻结精确区间。C3 pooled单侧95%上界不高于0.10，且C3六格Bonferroni单侧上界均不高于0.10，才可作为后续正式seed候选；否则即使相对任务收益有研究价值，也不得晋级blind或宣称检测资格。Medium本身作为matched参照，不因已有旧条件Pfa失败而剔除。

## 6. 停止条件和交付

- RTX 3050资源门、scene/seed隔离、任务loss梯度、200步过拟合任一失败，先定位具体实现错误；未修复前停止正式训练。
- 某候选OOM时只逐级降低microbatch，不改变有效batch；若microbatch=2仍不满足显存门，停止该候选并记录失败，不换CPU、不改模型问题。
- 某模型在预注册延长门不通过则在当前预算停止；不根据H0 verification或H1开发结果临时延长。
- 若扩展容量无法在共同update预算下带来有保护的任务改善，停止当前更大候选，不外推否定其他架构。
- 若Medium在延长阶段持续改善或出现可泛化任务增益，继续至冻结上限51,200；仍未见平台时只报告“预算内未观察平台”，不宣称达到极限。
- 数据扩展的总新增存储/磁盘不足G0时，不删除用户文件，不运行该对照；保留已完成模型规模pilot。
- 任何结果都保留模型负结果、曲线、配置、seed和scene manifests；R6-C完成后清理owner manifest列出的可重建IQ缓存，项目最终占用保持<20 GiB。
