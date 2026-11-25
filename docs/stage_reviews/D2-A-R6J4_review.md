# D2-A-R6-J4 阶段审查

**结论：REFINE（当前J4参数化停止，不进入确认阶段）**

**结果基准提交：** `e6b17e1e8d01b66457d9889db6d9672efb02ae7f`

**研究总纲：** `docs/RESEARCH_CHARTER.md` v1.3；本阶段未修改。

**最终盲测：** 未生成。

## 研究问题与范围

本阶段检验固定几何DPCA与520点零填充频谱候选上的学习式null/候选和连续参数读出，能否在相同新场景及训练预算下改善matched raw-IQ联合学生J1-J。实验属于R6物理驱动联合候选的单训练种子开发筛选，不含教师蒸馏，不是跨seed确认或盲测，也不把学习前端的收益等同于新增观测信息。

冻结协议为 [`experiments/D2-A-R6/R6-J4/protocol.md`](../../experiments/D2-A-R6/R6-J4/protocol.md)，配置为 [`config.yaml`](../../experiments/D2-A-R6/R6-J4/config.yaml)。G0/G1分别记录于`preflight.json`和`g1.json`。两臂使用seed909、51,200个配对训练场景、5,000个配对验证场景、有效batch64、FP32及25,600 optimizer updates；训练集按每个presence累计819,200次CPI暴露。J1-J为70,277参数、最佳update16,000；J4为10,494参数、最佳update24,000。AISAR环境RTX 3050 Laptop `cuda:0`顺序训练，无CPU回退、无OOM。训练与显存资源细节见 [`analysis.md`](../../experiments/D2-A-R6/R6-J4/analysis.md) 和 [`metrics.json`](../../experiments/D2-A-R6/R6-J4/metrics.json)。

独立评价包括H0校准120,000、H0验证120,000和H1 development 6,144场景，覆盖冻结的12个条件。两个学生和固定DPCA+130/520峰值、B3检测分数均以各自H0校准数据设全局5%阈值。指标及配对区间均保存在`metrics.json`。

## 结果与冻结门

J4相对J1-J的C2+C3 joint-success差为`+0.03255`，配对95% bootstrap CI为`[+0.02408,+0.04080]`，超过预设2个百分点收益门。总体Pd差CI为`[-0.01660,-0.00635]`，通过−0.02非劣门；全H1 Range RMSE比CI为`[0.768,0.896]`，通过1.05保护门。

但Doppler RMSE比CI为`[1.723,2.085]`，明确超过1.05保护上限。C3六格familywise虚警门也失败：J4的C3 pooled Pfa为9.65%，但condition 6/8/10分别为19.24%/18.68%/18.31%；最难condition 10的Pfa为18.31%，Pd为23.05%，joint为2.15%。匹配J1-J在该格Pfa为9.77%、Pd为27.73%、joint为10.35%。因此不能用总体或pooled C3结果掩盖该困难格退化。

相对固定DPCA+520峰值，J4学习读出在C2+C3 joint上增加`+0.56098`（95% CI `[+0.54709,+0.57465]`），说明该学习读出在这批开发场景中学到有用的候选信息。固定520点相对固定130点的C2+C3 joint增量为`+0.03299`（95% CI `[+0.02648,+0.03950]`），符合更密频率网格改善峰值定位的解释；零填充本身不增加独立观测信息。这些比较不能消除J4对J1-J的Doppler及困难格代价。

| 冻结门 | 结果 |
|---|---|
| C2+C3 joint提升至少2个百分点，95% CI下界大于0 | PASS |
| 总体Pd非劣，差值CI下界不低于−0.02 | PASS |
| 全H1 Range RMSE比CI上界不超过1.05 | PASS |
| 全H1 Doppler RMSE比CI上界不超过1.05 | FAIL |
| C3六格familywise单侧Pfa上界均不超过0.10 | FAIL |

## 工程和数据审查

G0/G1的物理/数值/梯度检查、scene ID独立性、匹配训练及评价均按冻结记录完成。教师知识不参与本阶段。B3只输出检测分数；其检测分数与DPCA+130峰值相同，所以总表将两行分别标为检测器与完整参数估计器，不能当成两份独立检测证据。

完成训练/评价后，按J4 `owner_manifest.json` 中明确列出的路径清理了12个可重建训练/验证数组，共释放11,233,313,736 bytes（10.46 GiB）。owner/scene manifest、训练记录与checkpoints、预测和指标均保留；清理审计见[`cleanup.json`](../../experiments/D2-A-R6/R6-J4/cleanup.json)。

首次分析命令因绘图未处理B3“无参数输出”而中断。之后只修正了绘图映射并扩展Pfa图的纵轴范围；`metrics.json`、预测和冻结评价未变化，评价没有重跑。该分析绘图修正发生在G0绑定提交之后，详见源文件 [`d2a_r6j4.py`](../../src/radardistill/evaluation/d2a_r6j4.py) 的当前差异。该差异只影响展示，不构成新的指标证据；指标仍绑定结果基准提交SHA。

## 对研究路线的影响

本阶段支持有限结论：物理前端加学习读出能相对固定频谱峰值改善联合任务，但尚未证明它优于匹配raw-IQ学生，也未达到困难条件下稳定检测和连续Doppler估计要求。J4的较小参数量没有自动带来更快训练或更好频率精度；本阶段不作实时性结论。单seed、合成静止杂波和H1 development的边界不支持跨seed、真实系统或盲测泛化主张。

R5-C及R6-G此前均未显示已测K3/K2-local知识在truth-only之外带来可靠任务增益。J4也没有蒸馏，因此本阶段不改变教师路线的结论。所有历史负结果保持不变。

## 阶段决策

按冻结保护门，停止当前J4参数化；不启动J4多种子确认、蒸馏、final blind或进一步开发集搜索。该决定不外推否定所有物理前端/联合网络。整合路线的R5/R6模型、教师与传统基线比较见[综合性能报告与总表](../research/D2-A_model_performance_matrix.md)。接下来只完成结果归档、性能总表和一次整条路线ChatGPT审查请求；若未来要继续物理驱动联合路线，应提出独立新假设和新协议，并先解释如何保护连续Doppler与C3逐格虚警。
