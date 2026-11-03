# D2-A-R6-E 阶段审查

**结果提交/训练代码版本：** `2eaf7838c07f0b2b9292343fa9ef56f19260db01`
**研究总纲：** `docs/RESEARCH_CHARTER.md` v1.3；本阶段未修改。
**路线位置：** R6 P4，单seed代表性架构筛选。
**决定：REFINE。**

## 审查摘要

R6-E对J1-J、复数公共相位等变CE-TCN和全网格轴向RAX-Transformer，按冻结协议使用新独立scene IDs、E1物理真值监督、相同seed606、有效batch64和51,200 updates。三臂均在RTX 3050 Laptop `cuda:0`、FP32顺序训练；未生成final blind。

最重要的正结果是CE-TCN相对J1-J的可辨识C2+C3 joint success增加`0.02713`（97.5%配对bootstrap CI `[0.01736,0.03668]`），Pd/range/Doppler任务保护通过，达到协议的候选筛选门。CE-TCN也在16个公共相位旋转上保持1.0决策一致率，输出变化接近数值级别。这说明显式复数/相位结构在该实现中可行，并有单seed开发集的相对任务收益。

关键限制是三个模型都没有通过C3六格familywise Pfa上界≤0.10门。CE-TCN在困难C3部分重叠、0 dB格的Pd为0.381、joint为0.150、AUC为0.699，仍不具备合格检测器证据；它也没有表现出频移等变性。RAX的C3经验Pfa在0 dB约0.17–0.18、10 dB约0.02，且Doppler RMSE高，当前RAX配置停止。

## 工程与实验审查

- G0/G1均通过；历史与split scene ID交叉审计无重叠。数据划分和样本数符合冻结protocol。
- 三臂共享scene顺序、E1真值目标、有效batch64、FP32、AdamW/cosine预算和checkpoint规则，参数量在冻结区间。CE-TCN遇到OOM后按protocol降microbatch至4，并保留有效batch64、optimizer/sampler状态和总更新数。
- 各臂仅用本臂H0 calibration阈值，并在新H0 verification/H1 development评价；开发数据不是blind，未用于训练checkpoint选择。
- 数值证据见[experiment metrics](../../experiments/D2-A-R6/R6-E/metrics.json)、[training records](../../experiments/D2-A-R6/R6-E/runs/seed_606/)与[stage analysis](../../experiments/D2-A-R6/R6-E/analysis.md)。按条件图和训练曲线见experiment `figures/`。
- R6-E是单训练seed；不同网络的架构特征并非单因素控制，因此不能把所有收益因果归于复数结构或外推至整类Transformer。51,200是本轮预算，不是学习上限。未运行传统算法对照、teacher distillation或final blind。

## 对研究主线的影响

R6-E推进了研究总纲Q3/Q4/Q6/Q10：相位等变网络可以通过结构而非隐式学习实现稳定公共相位行为；这轮相对任务改善与该结构共现，但尚未完成跨seed确认或机制归因。学生架构的比较不等同于教师教学能力验证。R5-C T0–T3的教师蒸馏未取得正收益，故必须在后续工作中用匹配truth-only与truth-assisted对照重新证明教师知识能否改善学生任务结果。

RAX负结果只适用于本次48维、两block、相对偏置、固定训练预算配置；不能概括否定全部Transformer。J1-J仍是重要基线，但其C3各格置信上界也未全部达标。

## 后续建议

1. 进入R6-F前先冻结端到端混合链协议；将CE-TCN作为优先NN候选，J1-J作为matched student reference，同时保留传统/传统基线。每条完整链独立校准H0，不能继承子模块阈值。
2. R6-G仅选择一个有证据的教师知识形式和有限学生组；与truth-only及必要truth-assisted控制匹配训练预算，单独保护困难C3子组，不用教师错误取代物理真值。
3. 直到R6-F/G候选冻结后，再按P7决定formal multi-seed与独立blind；不得仅凭CE-TCN的单seed开发增益宣称正式胜出。
4. 研究总纲v1.3和R6-E冻结protocol、历史负结果保持不变；本阶段不创建ChatGPT审查请求，待用户批准的R6 P0–P7整条路线完成后统一提交。
