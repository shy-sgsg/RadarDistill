# D2-A-R6-E：复数等变卷积与轴向Transformer架构筛选

**协议版本：** 1.0
**状态：** 预注册；冻结提交及 G0/G1 通过前不生成正式 train/validation IQ。
**研究总纲：** `docs/RESEARCH_CHARTER.md` v1.3。
**路线位置：** R6 P4；单训练seed的代表性架构筛选。
**最终盲测：** 未授权生成。

本阶段检验两种机制不同的原始复IQ学生，是否能相对当前任务级卷积基线改善可辨识场景中的最终检测与连续参数估计：显式公共相位等变复数TCN，以及保留完整慢时间—距离网格的轻量轴向Transformer。该比较服务研究总纲Q3、Q4、Q6、Q10，并为后续混合链和教师协同提供有界架构证据；本阶段不做蒸馏、部署时延或盲测。

R6-D显示 J1-J 的单seed validation task loss 在约14,400步达到最低，而若干不同训练目标到20,400–22,800步仍有改善。用户允许新实验在公平前提下增加更新数。因此本阶段把所有三臂统一设为51,200 updates（常见12,800预算的4倍、R6-D预算的2倍），并匹配场景暴露、有效batch、学习率计划和checkpoint规则。51,200是本筛选预算，不是学习极限；本阶段末端仍改善时，报告“预算内未观察到验证平台”，不得单臂追补或声称达到收敛。

## 1. 预注册假设与比较臂

| 臂 | 定义 | 明确的结构贡献 | 不作出的保证 |
|---|---|---|---|
| J1-J | R6-D `R6J1Student(width=96)`，直接接收三通道六平面I/Q | 已测任务级卷积基线；无FFT或固定物理前端 | 不保证公共相位不变或频移等变 |
| CE-TCN | 从六平面输入还原三通道复数序列；复数`(1,3)`range stem；7个残差block，慢时间depthwise kernel `(3,1)`扩张率`1,2,4,8,16,32,64`，随后复数pointwise通道混合；复Group RMSNorm仅以模长均方根缩放，径向modReLU，所有复卷积无偏置；末端按模长、沿时间均值后读出实值任务头 | 复数权重与径向算子使潜在特征对公共相位旋转等变；模长任务读出使输出理论上不变 | 不保证Doppler频移等变；不使用固定FFT/谱层；不将数学不变性当作任务质量证明 |
| RAX-Transformer | 原始I/Q六平面经无位置项的`1×1`实卷积嵌入48维；两个block各按“时间轴注意力→距离轴注意力→token-wise FFN”顺序更新完整130×32网格；4头、FFN维度96、GELU、pre-LayerNorm、dropout=0；每轴使用每head可学习的精确相对偏移bias（时间lag−129…129，距离offset−31…31）；完成两个block后才沿慢时间均值池化并生成与J1相同的任务输出 | 全网格轴向信息交互；完全可学习输入变换 | 不保证公共相位不变或Doppler频移等变；不使用预训练权重或固定频谱前端 |

CE-TCN初始宽度64；复权重的每个复系数按两个实自由参数计。RAX初始维度48、4头、两个轴向block。每候选推理参数量目标为J1-J的±20%（基线70,277，即56,222至84,332）；G0报告实际可训练/推理参数。若初始容量不满足，仅允许一次、且只能依据参数计数的宽度/维度调整；不得查看任何新场景训练或评价结果后调整容量。

CE-TCN的复Group RMSNorm固定8组，统计每样本/每组所有通道和时空位置的`|z|²`均值，加`eps=1e-8`开方后归一化，仅乘可学习实数通道尺度、不加偏移；径向modReLU对每通道学习实阈值`b`，按`z·ReLU(|z|+b)/max(|z|,1e-8)`缩放。RAX的每个轴向注意力均为非因果全双向softmax attention，时间/距离相对bias独立学习；无attention mask。

三种学生都从相同原始IQ学习presence、range cell、range offset、周期Doppler向量，不包含教师、FFT响应、真值输入或训练期辅助谱头。CE-TCN与RAX均沿慢时间均值池化后在每个range位置产生range logit、sigmoid offset和归一化二维Doppler向量；presence logit按J1的`logsumexp(range_presence)−log(32)`聚合。三个模型都输出`presence_logit` `[B]`、`range_logits` `[B,32]`、`range_offset` `[B,32]`和`doppler_vector` `[B,32,2]`，使用与R6-D J1-J相同的`r6d_loss(arm="j1_joint")`和`decode_task_outputs`。

## 2. 仿真、划分和随机性

保持D2-A-E001三通道静止杂波、单目标复AWGN模型：PRF=1300 Hz、N=130、32距离单元、R6-A 12个条件格。仿真物理参数和条件仅从`D2-A-E001/config.yaml`与`R6-A/config.yaml`读取；不得修改场景分布或目标标签定义。

| Split | 数量 | 用途 |
|---|---:|---|
| Train | 51,200个配对base scenes（每个含H0/H1） | 三臂共同E1真值训练 |
| Validation | 5,000个配对base scenes | 固定验证/checkpoint选择 |
| H0 calibration | 120,000 CPI，每格10,000 | 每臂独立选取全局0.05阈值 |
| H0 verification | 120,000 CPI，每格10,000 | 独立虚警迁移评价 |
| H1 development | 6,144 CPI，每格512 | 独立检测、参数和joint评价；仅开发集 |

训练和验证每个base scene的H0/H1作为配对输入数据，H0/H1标签及物理参数仅用于对应truth loss和离线评价，前向模型只接收IQ。三臂共享同一split scene IDs及I/Q，使用同一冻结采样顺序。初始化seed=606；采样/排序seed=607；同一训练seed在各臂独立初始化，不宣称异构架构权重可逐元素匹配。

固定seed block（场景ID格式沿用D2-A生成器）：

| Split | seed base / 分配 |
|---|---|
| Train | 每course `[2008000000, 2013000000, 2018000000, 2023000000]`；assignment seed `2026110101` |
| Validation | 每course `[2028000000, 2033000000, 2038000000, 2043000000]`；assignment seed `2026110102` |
| H0 calibration | `2108000000` |
| H0 verification | `2128000000` |
| H1 development | `2148000000` |
| G1工程小样本 | `2300000000`，16个scene |

所有chunk按已验证的850,000 condition seed offset、1,009 chunk stride和16-scene训练chunk规则生成。正式G0须重新遍历全仓既有`scene_manifest*.csv`与`selected_scene_ids.csv`，精确检查计划scene ID的历史重用、split内重复及split间重叠；任何非零重叠都停止。冻结前只读预演已检查22个清单、884,072个ID；以上计划与历史及相互间的重叠均为0。新清单、G1 ID和数据生成后结果以G0实际审计为准。

Train/Validation临时缓存预计11,222,016,000 bytes；开始生成时须至少有该值加4 GiB的可用磁盘空间。evaluation按固定scene ID分块生成，不保存全量IQ。临时训练缓存仅可由R6-E owner manifest清理。正式blind不生成。

## 3. 固定训练预算与公平性

- 三臂按J1-J、CE-TCN、RAX顺序单模型训练，RTX 3050 Laptop `cuda:0`，FP32；CUDA不可用或设备名不符则停止，禁止CPU静默回退。
- 每臂51,200 optimizer updates；每update 32 H0 + 32 H1，有效batch=64；同一AdamW、初始LR=1e-3、最终LR=1e-5、单cosine覆盖完整51,200 updates、weight decay=1e-4、gradient clip=1.0。用户允许新研究改变batch，但本阶段固定batch64以只比较架构；batch128留给独立匹配设计。
- 每种presence训练样本暴露量为1,638,400 CPI slots，对51,200唯一train scene的期望为32次/scene/presence；样本重访与新scene必须分开报告。三臂的scene和batch顺序相同，optimizer updates、总暴露、learning-rate schedule、validation间隔、checkpoint及选择规则相同。
- 初始microbatch=16；发生OOM时先降低至8或4并用梯度累积保持有效batch64；若microbatch4仍无法运行则停止该候选，不改变训练问题或有效batch。禁止AMP及CPU回退。
- 每400 updates记录一次history并验证；每800 updates保存可恢复checkpoint；不逐optimizer update写日志。checkpoint包含模型/optimizer/scheduler状态、update、初始化/采样seed、sampler游标、protocol/config SHA及运行git SHA。
- 每臂按自身validation joint truth loss选择最低checkpoint。不得用H0 verification、H1 development或物理干预评价选择模型、阈值、架构、loss、训练长度或参数。不得因为单臂validation仍下降而只延长该臂。

训练日志须报告每臂unique train scene数、总optimizer updates、每presence数据暴露、固定训练集遍历次数、LR计划、完整train/validation task loss及参数子项、最佳checkpoint、模型参数量和实际CUDA峰值显存/墙钟时间。51,200 updates不是收敛声明；若学习曲线末端持续下降，后续更长训练须作为新匹配实验预注册。

## 4. G0/G1 前置门

**G0：** 预注册文件及代码已提交并推送`main`；protocol/config SHA固定；全仓scene ID无重叠；磁盘容量满足缓存与4 GiB余量；实际GPU为NVIDIA GeForce RTX 3050 Laptop且`cuda:0`可用、显存满足一次模型的FP32 probe。记录Python、PyTorch、CUDA、驱动/GPU、OS、git SHA、磁盘和显存。

**G1：** 三臂分别执行contract、有限输出和有限非零梯度检查；CE-TCN在复潜特征公共相位旋转下满足等变、模长任务输出不变；RAX验证完整time/range轴交互及周期Doppler输出接口；参数范围满足56,222–84,332。随后每臂在相同16个paired scenes上运行200个updates，主task loss最后20步均值相对初始下降至少10%。小样本只作为工程学习门，不作为科学性能结果。失败候选先排除实现错误；确认为该实现无法学习时停止并保存完整负结果。

所有前置门通过后才生成正式Train/Validation。G1通过不代表候选任务有效，也不授权blind。

## 5. 指标与冻结判定

每臂仅用其H0 calibration scores以`higher`经验95%分位数设置全局`Pfa_CPI=0.05`阈值，决策比较符为`>=`；在完全独立H0 verification与H1 development评价。阈值、验证集和开发集不可互换。

**主端点：** C2+C3 H1 paired joint success。分别计算CE-TCN−J1-J、RAX−J1-J，按base_scene_id进行2,000次paired bootstrap。由于有两个架构对比，两个差值报告双侧97.5%区间（Bonferroni family-wise 95%）；candidate的筛选增益须点估计至少`+0.02`且97.5%区间下界大于0。两候选之间只作描述性比较，不据此做未校正的优胜声明。

**任务保护：** 相对J1-J的overall H1 Pd差、all-H1 range RMSE ratio、PRF-wrapped Doppler RMSE ratio使用相同base_scene_id配对bootstrap及97.5%区间。保护标准分别为Pd差区间下界`≥−0.02`，range/Doppler RMSE ratio区间上界`≤1.05`。此外报告overall/C2+C3/C3及12条件的Pd、Pfa、AUC、range/Doppler bias与RMSE、门错率、joint success；为每臂给出必要的exact binomial区间。

**绝对虚警保护：** 报告H0 verification overall和每个条件的Pfa及one-sided exact95%上界。C3六格使用Bonferroni familywise exact上界，每格均`≤0.10`才可称通过C3低虚警资格。此绝对门与相对架构收益分开；相对收益通过不等于低虚警检测合格，绝对门失败也不抹去受限架构机制证据。

**物理一致性诊断：** 在冻结H1 development输入上评估共同相位旋转`16`个均匀相位点的决策一致率、presence分数和参数输出变化；CE-TCN报告理论/实测不变误差，J1/RAX仅报告实测结果。频移干预使用全通道IQ乘以`exp(j 2π Δf n/PRF)`，预设`Δf∈{−130,+130}` Hz，并同步周期平移离线Doppler真值；报告wrapped估计残差和range/detection变化。该干预是受控反事实，不等同于完整目标运动重仿真。物理诊断为次级结果，不改变checkpoint或训练参数选择。

所有单seed结果均为架构筛选开发证据，不估计训练seed方差、不声称跨seed泛化。若任何模型在51,200末仍持续改善，只能报告预算内未观察到平台。候选满足相对门时仅可进入后续新协议和至少三seed正式比较；C3绝对门不通过时不得称为合格低虚警检测器或生成final blind。

## 6. 停止与路线影响

1. G0的提交、独立scene ID、磁盘、CUDA设备或协议hash检查失败：不生成正式数据。
2. G1暴露模型接口、相位性质、显存或overfit实现问题：先按证据修复并保留历史；排除实现错误后仍失败的候选停止，不以蒸馏损失掩盖基础映射失败。
3. OOM优先降低microbatch且保持有效batch/update数；microbatch4失败则停止该候选。
4. 若两个新候选都没有达到主增益门，或任务保护出现不可接受退化：停止本轮新候选，不自动扩展SSM、CNN+序列混合、蒸馏或超参数搜索；根据证据确定后续路线。
5. 若候选满足相对主门但C3绝对Pfa失败，保留架构研究信号并限制结论，不启动blind或系统级资格声明。

保留R5-C及R6-A至R6-D全部历史协议和负结果；不修改`docs/RESEARCH_CHARTER.md` v1.3。该阶段只使用E1物理真值监督；R6-G知识协同路线需由后续证据另行筛选，不由R6-E自动放行。
