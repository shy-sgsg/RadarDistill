# D2-A-R6-G 教师教学内容 pilot 设计

## 意图

检验 R5-C T0–T3 的负蒸馏结果能否推广到 R6 中任务表现更强的 CE-TCN 学生：在完整物理真值监督之外，K3 的 IQ-only 全图 RD 软响应能否提高最终检测与参数联合任务，或让学生更快学会同一任务。它直接服务于总纲 Q1、Q2、Q6、Q9、Q10；不把蒸馏损失下降单独当成研究成功。

## 当前依据与边界

- R5-C 的 K3（固定 4× 零填充完整 RD 响应）在可辨识 C2+C3 的教师峰定位率为 0.812，优于 K1 的 0.781；但 R5-C T1−T0 C2+C3 joint 差为 `−0.01042`，区间完全低于零。最困难的 C3 partial-overlap / 0 dB / CNR20 子组中 K3 定位率只有 0.041。教师整体有任务相关信息，不代表困难场景可靠。
- R6-E 的 CE-TCN seed606 有比 J1-J 更强的 R6-F 完整链开发任务表现；它的推理模型有 63,429 个参数。此处选 CE-TCN 是依据开发任务表现和复相位结构，不声称它参数更多或已通过 C3 低虚警门。
- R6-F 中 CE-TCN 原生链在 C2+C3 的 joint 为 0.85807，但所有完整链的 C3 六格 familywise Pfa 门均失败。R6-G要区分“教学增益”和“合格低虚警系统”两种结论。

## 考虑的方案

1. **重复 R5-C 的 Medium TaskStudent 四臂实验。** 可直接比较，但已经重复测试过 K3/K2-local 的教学效果，不回答新学生结构是否更能吸收知识。
2. **在 CE-TCN 上测试 K3，并保留 truth-only 与 truth-assisted 控制。** 复用已筛选的任务学生与 K3 全图知识，隔离额外辅助头正则化和教师响应贡献；范围有界，直接对应 R6-G。**采用此方案。**
3. **同时比较多种学生、K1/K2/K3、可靠性权重和教学时机。** 研究因素相互混杂且 R5-C 教师在 C3 有严重错误，不适合作为首轮 R6-G 试验。

## 目标与非目标

目标：

- 在新独立训练、验证、校准、验证及 H1 development 场景上，比较 T0/T1/T3。
- 测量 K3 相对 truth-only 的最终任务收益，并由 truth-assisted T3 判断一般辅助响应训练能解释多少收益。
- 记录验证任务曲线，查看 K3 是否改善固定预算内的学习进程。
- 逐条件检查教师正确/错误分层，尤其是 C3 partial-overlap、SNR0、CNR20。

非目标：

- 不重复 T2 K2-local；不搜索教师种类、可靠性门、蒸馏温度、权重或时机。
- 不改 R5-C 冻结协议、R6-E/F 代码与结果，不做多 seed 确认或 final blind。
- 不宣称 R6-G 一次单 seed pilot 证明普遍教师优势。

## 实验设计

三臂使用同一个 R6G-CE-TCN 结构和 seed707 从头随机初始化的 bit-identical 初始权重；不从 R6-E seed606 checkpoint warm-start。这样检验的是教师在学生从头学习时的增量贡献，而不是预训练后微调收益。三臂使用相同 51,200 个配对训练场景、相同样本顺序、相同验证数据、任务损失、有效 batch64、FP32 和 51,200 optimizer updates。所有臂的 checkpoint 统一按 validation truth-task loss 最低选择；不按含各臂辅助项的 total loss 选择，避免 T1/T3 使用不同选择标准。

| 臂 | 监督 | 解释 |
|---|---|---|
| T0 | 物理真值任务损失 | 当前学生独立学习基线 |
| T1 | 物理真值任务损失 + K3 完整 RD 软响应 KL | IQ-only 传统教师内容 |
| T3 | 物理真值任务损失 + 匹配的真值辅助 Dirichlet RD 响应 KL | 控制辅助头/响应监督本身的作用；不是教师 |

CE-TCN 增加只在训练时计算的 `520×32` response head；三臂结构与初始化完全匹配，推理导出不包含该辅助头。训练真值任务损失沿用 R6-E 联合损失。T1 的 K3 map 只能由当前 H1 IQ 和冻结物理配置用 CUDA 教师生成；不把真实参数、场景 SNR/CNR、教师正确性标签输入教师或学生。H0 不产生教师响应目标。T3 的解析响应由真实 range/Doppler 生成，仅用于控制臂的训练辅助标签。

教师辅助损失固定权重 `0.05`，H1-only，不按结果调参。训练预算 51,200 更新，匹配 R6-E 的充分筛选预算；每个 update 有32个 H0 和32个 H1 CPI。验证每400步，检查点每800步；不逐 update 写日志。microbatch 固定4，梯度累积保留 effective batch64。

数据使用 R6-A 的 12 个冻结条件及三通道、PRF1300 Hz、N=130、32距离单元静止杂波仿真。训练/验证计划各51,200/5,000个配对基础场景；H0 calibration/verification各120,000；H1 development 6,144（每条件512）。所有 ID 使用新 seed bases，并在 G0 检查与历史清单和 split 互斥。H0/H1开发场景不落原始 IQ；训练/验证 IQ 与辅助响应缓存由 owner 清单管理，阶段结束后清理。

## 指标与预注册判断

- 首要结果：T1−T0 的 C2+C3 joint-success 配对场景差；使用按条件分层的 5,000 次 bootstrap 95% 区间，seed `2026110401`。
- 教学收益筛选门：joint 差点估计至少 `+0.02` 且 95% 区间下界大于0；Pd差下界不低于`−0.02`；全H1 range/Doppler RMSE ratio 的95%上界不高于1.05。
- T1−T3 和 T3−T0 作预注册对照，解释收益是否超过一般 truth-assisted response 学习；不以蒸馏 KL 本身作为门。
- H0 指标：各臂独立校准 `Pfa_CPI=0.05`，在新 H0 verification 报告全局及12条件 exact 区间，并对 C3 六格做 familywise 上界。C3六格上界≤0.10 才算绝对低虚警资格。pilot即使未达该绝对门，仍可报告教学机制结果，但不能宣称系统合格。
- 收敛指标：预定 checkpoint 的 validation truth-task loss、Pd、joint 与 RMSE 曲线；比较固定训练预算下的曲线面积和关键更新点，不进行多点显著性搜索。只有最终任务门也通过，才把曲线改善解释为有用的收敛增益。
- 物理教师错误诊断：K3 peak 与真值的正确性仅在训练外/离线分析中计算。独立报告全体 C3、C3 partial-overlap/SNR0/CNR20 的教师正确率、学生任务指标及负迁移；教师正确性不参与模型选择或训练门控。

## 阶段门与停止

- G0：配置/代码冻结；新 scene IDs、训练臂初始化和数据顺序匹配；GPU/磁盘预算可行。
- G1：CE-TCN响应头形状和梯度正确；T1 teacher map 为 finite、非负且归一化；T0 不请求教师图，T3 不读教师图；真值只进入任务损失及T3响应目标；CUDA-only teacher/student路径通过小批次检查。
- 仅 G0/G1 PASS 后运行三个 matched seed707 arms 一次。OOM先降低 microbatch，不改变有效batch或科学问题。
- 若 T1 未达到收益门、任务保护失败或在预注册 C3 困难组出现明显负迁移，停止当前知识内容分支；不转而搜索温度/权重/可靠性门。
- 即使 pilot 正向，也只进入独立多 seed确认提案；不得复用 development 为 blind 或直接宣称低虚警资格。

## 资源与交付

RTX 3050 Laptop GPU 4 GiB，全部三臂顺序训练。数据按 batch 搬到 CUDA；teacher map 在 CUDA 计算后以 CPU 磁盘缓存紧凑半精度表示；估算训练/验证 IQ 约11.2 GB、T1/T3软响应上界约3.4 GB，生成前要求至少估算占用再加4 GiB可用空间。实际使用由 owner 清单记录，阶段收尾只清理本阶段 owner 缓存，保留协议、配置、训练曲线、检查点、metrics、分析和阶段报告；项目完成后低于20 GiB。

`docs/RESEARCH_CHARTER.md` v1.3 保持不变。综合 ChatGPT 审查请求只在用户授权的 R6 P0–P7 路线全部结束后提交。
