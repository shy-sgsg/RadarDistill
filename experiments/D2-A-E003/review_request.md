请首先读取本次Git提交对应版本的 `docs/RESEARCH_CHARTER.md`，理解RadarDistill的整体研究主题、总体目标、主要研究内容、核心科学问题和长期技术路线，然后结合本阶段的物理表征、学生结构、训练、代码、数据和研究结论进行独立审查。

## 审查基准

- **审查目标 Git 提交 SHA：** 完整实验结果/状态提交，即最终审查请求提交的直接父提交；其完整 SHA 将在后续请求提交中明确列出。
- **审查请求文件所在后续提交：** 上述审查目标提交的直接后续提交；该后续提交只补充此审查请求及其精确目标 SHA，总纲内容与审查目标提交相同。
- **研究总纲路径/版本：** `docs/RESEARCH_CHARTER.md`，v1.2（本阶段未修改）
- **阶段：** D2-A-E003，单 seed E1 truth-only S0/S2 pilot
- **工作区状态：** 提交目标及后续审查包仅包含已列明 E003/治理文件；保留的用户本地未跟踪 GMTI 文件和既有未跟踪进度材料未纳入提交，也未修改
- **总体研究目标：** 基于物理真值和经验证的传统知识训练直接完成 GMTI 检测、距离与连续参数估计的学生，并在任务质量约束下评价物理前端与学习器的贡献和完整成本
- **研究问题映射：** Q1、Q4 为主；关联 Q3、Q8
- **本阶段假设：** H-ARCH-D2-002、H-FREQ-D2-001、H-DET-D2-002、H-IDENT-D2-001，详见 `docs/research/ASSUMPTION_REGISTER.md`

## 本阶段实验与证据

- **协议/config：** `experiments/D2-A-E003/protocol.md`、`config.yaml`、`feature_design.md`
- **代码入口：** `scripts/run_d2a_e003.py --stage preflight|r0-r1|r2|generate-data|train|evaluate|analysis|cleanup`
- **核心实现：** `src/radardistill/students/physical_coherence.py`、`physics_guided_student.py`、`src/radardistill/evaluation/physical_consistency.py`、`src/radardistill/training/d2a_e003.py`、`d2a_e003_data.py`
- **模型：** S0 冻结的六平面 raw-IQ TaskStudent；S2 为固定 28 通道多延迟/跨通道复相关与归一化功率前端，加可学习 `28→6` 投影及相同 TaskStudent；无 raw-IQ bypass。
- **数据：** 三通道、PRF 1300 Hz、N=130、32 range cells、静止杂波、配对 H0/H1、复 AWGN。训练 51,200 对；validation 5,000 对；H0 calibration/verification 各 8,192；独立 C3 H0 verification 4,096；H1 development 4,096 对。81,288 个 scene ID 跨 split 不重叠。最终盲测未生成。
- **训练：** seed 101；S0/S2 各 1,600 updates；FP32、RTX 3050 Laptop `cuda:0`、microbatch 16、有效 batch 64；同初始化、同 pair plan；只有物理真值 E1，无教师/KD/课程。
- **代码版本：** 训练记录所列 SHA `bd18df1f4fa818684bd28874c3ba96d23a1e7dde`；评价 inference-mode 计时修复 SHA `af0f59690ae29f0b3d4638067efe663145c1d670`；最终审查目标 SHA 在上方登记。
- **测试：** 修复后 AISAR 完整测试套件 `303 passed`。首次评价只在最终计时环节失败，之后用相同检查点和 splits 完整重跑；训练和门槛未改变。
- **主要结果：** G1/R1 PASS、G2/R2 PASS、S0/S2 基础学习 G3 PASS；G4 task protection FAIL，S2 C2/C3 joint-success 差 −0.2593（scene-bootstrap 95% CI `[−0.2822,−0.2368]`），C2/C3 Doppler RMSE 比 1.835 `[1.720,1.959]`。总体 H0 Pfa S0/S2 为 0.0483/0.0475，但 C3 H0 Pfa 为 0.181/0.153，均超过 0.10 冻结限值。
- **物理一致性：** S2 公共相位 32 旋转诊断的 H0/H1 判决一致率均为 1.0，Presence 分数变化 p95 最大约 `5.72e-5`；S0 一致率中位数约 0.949/0.957。S2 频移和重新生成几何频率变化的误差仍大。
- **资源/成本：** 每臂训练约 350/494 秒，CUDA 峰值 allocated 约 275/286 MB。CUDA batch 1 输入至输出均值 S0/S2 1.77/4.20 ms，batch 16 为 8.21/10.94 ms；这些不是完整 GMTI 链实时性数据。
- **收尾清理：** 严格按 E003 cache manifest 删除 90 个文件（14,232,309,192 bytes）；清理后项目为 801,766,373 bytes（0.747 GiB）。清单外文件及用户未跟踪文件未改动。逐文件审计见 `experiments/D2-A-E003/cleanup.json`。
- **主要产物：** `metrics.json`、`result.csv`、`roc.csv`、`training.json`、训练 history/checkpoints、`physical_consistency.json`、`compute.json`、`cleanup.json`、`analysis.md`、RMSE/ROC/相位图及 `docs/stage_reviews/D2-A-E003_review.md`。

## Codex 阶段判断

- **当前 S2 表示候选：STOP。** 公共相位不变性成立，但任务保护和主要联合收益门失败。
- **D2-A 学生路线：REFINE。** S0 在可辨识开发任务有学习能力，但 C3 虚警仍显著超限；S2 负结果不能外推为所有物理相干表示失败。
- **长期研究总纲：** v1.2 继续适用；当前证据不足以建议修订总纲。
- **蒸馏与教师：** E003 未进行 E2/E3；教师仅在 R2 小样本诊断作描述性参考，没有新增教师资格结论。
- **下一项唯一优先任务：** 独立审查本阶段，再决定是否批准不训练、不扩充模型的表示归因诊断，定位前端、归一化或 `28→6` 投影与 C1–C3 任务退化的关系。

## 请独立审查

请按 `docs/governance/CHATGPT_REVIEW_PROTOCOL.md`，独立完成以下判断，而不是复述 Codex 自审：

1. **工程正确性：** 物理前端、complex64 容差、数据生成/隔离、标签、阈值、周期误差、CUDA 路径、失败重跑和计算边界是否正确？
2. **实验科学性：** S0/S2 是否足够匹配？单 seed 与按 scene bootstrap 的不确定性是否被准确限定？C3 H0 样本量与置信上限支持什么结论？是否有隐藏选择、多重比较或阈值/盲测泄漏？
3. **总体目标与研究主线：** 本阶段实际推进了总纲中的哪项目标/科学问题？公共相位不变性改善但任务恶化应如何解释？当前候选是否停止合理？
4. **物理信息和创新边界：** 延迟相关、跨通道相干、功率归一化分别保留/丢失什么？R1/R2 是否足以支持“信息保持”？固定前端和投影贡献能否分离？
5. **传统方法与蒸馏：** E003 没有完整任务级传统基线比较或 KD 对照，哪些结论因此不能作出？R2 描述性教师结果可否支持后续教师选择？
6. **战略方向：** 当前最有价值的下一步是表示归因、改进教师、学生/物理前端联合设计，还是其他最小替代方案？请比较其信息价值、成本和停止条件。
7. 假设哪些得到支持、否定或仍未决？是否有证据要求调整长期研究思路？若认为研究总纲要修改，只提出具体问题、证据、拟议文本和影响，等待用户明确批准。

请最终区分 **S2 候选判断、D2-A 学生路线判断、RadarDistill 长期方向判断**，并只选择一个主阶段结论：**CONTINUE / REFINE / REDIRECT / STOP**。给出唯一下一项优先任务、最小可证伪方案、改变判断所需证据和停止条件。不要把未运行的多 seed、蒸馏或 blind 写成结果。
