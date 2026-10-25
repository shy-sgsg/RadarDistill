# D2-A-E002 协议阶段审查

**请首先读取本次Git提交对应版本的 `docs/RESEARCH_CHARTER.md`，理解RadarDistill的整体研究主题、研究目标、研究内容、科学问题和技术路线，然后结合这些内容审查本阶段的实验协议与后续研究方案。**

## 阶段与审查范围

- **总纲：** `docs/RESEARCH_CHARTER.md` v1.1；本次未修改。
- **当前主线状态：** D2-A-R3 为 REFINE；T0/T1/T2 均未通过固定阈值 G2，当前没有正式批准的传统教师；无 D2-A 学生训练、KD 或 final blind 结果。
- **本阶段产物：** 新的独立 truth-only E002 任务对齐、可审查协议和配置草案。仅审查研究设计，不是实验结果审查。
- **总纲映射：** Q1/Q3；在尝试教师转移 Q2 前，检验现有学生能否从可辨识三通道 IQ 和物理真值中完成检测、range、Doppler。该任务回到任务级学生主线，不扩展 CFAR 研究。

## 决策

**阶段决定：CONDITIONAL GO（仅协议提案）。** 该问题由 R3 证据触发、与总纲一致，且 truth-only 设计能把学生可学习性与教师可靠性解耦。允许将方案提交独立 ChatGPT 审查；**此决定不授权代码实现或训练**。本阶段没有生成场景、运行模型或取得 E002 指标。

## 关键设计审查

1. **独立研究假设：** `H-LEARN-D2-001` 认为冻结 `TaskStudent` 可从三通道 IQ 学习 H0/H1、距离和连续 Doppler，并胜过简单基线。它可被单 seed 失败结果否证，不借助蒸馏损失掩盖基础问题。
2. **历史门保护：** R3 G2 仍失败，T1 仍非获准教师；E002 不重算、不覆盖 R3，也不声称绕过它。学生使用自己的全局 H0 threshold。
3. **统计方案：** 15,000 H0 calibration + 15,000 独立 H0 verification；目标 5%，允许上限 6%，用单侧精确 95% Clopper–Pearson 上界。样本量计算显示真实 Pfa=5% 时 UCB≤6% 的理论通过率约 99.99%；若再硬性要求点估计≤5%，通过率约 50.97%，所以不采用该不稳健合取门。此推算依赖 IID 混合抽样，执行器必须通过 preflight 证明实现一致。
4. **任务指标：** 学生 AUC、固定阈值 Pfa/Pd、range 与圆周 Doppler、把漏检计作失败的 joint success；与 E0 常值/能量、E001 B0/B1/B3 和 R3 T1 同场景比较。按 base scene 配对 bootstrap。
5. **物理审计：** H0/H1 共享背景噪声；metadata 与 IQ 输入隔离；公共相位、幅度、目标 Doppler 平移与通道相干消融分别作为有物理解释的等变/不变性检查或受控分布干预。
6. **资源：** 单 seed 101、1600 updates、有效 batch64、FP32、RTX 3050 Laptop CUDA；串行执行、CPU 流式数据、无 CPU 回退。当前未实现 runner，执行命令明确标为草案。

## 主要风险与限制

- 单 seed 只能作可行性诊断，不能说明训练稳定性。
- 场景和 noise model 仍是合成理想通道静止杂波，不等于真实 GMTI。
- IID 混合分布是明确但人为规定的工作点；不同真实 SNR/CNR/幅度构成可能改变全局阈值。
- 参数和相干诊断的数值门尚需独立审查；G2/G4 通过也不证明优于完整传统链或可部署。
- E002 不运行 final blind，不产生可供正式论文结论的盲测证据。

## 执行前必须完成

- ChatGPT 按 `docs/governance/CHATGPT_REVIEW_PROTOCOL.md` 审查总纲、科学问题、统计门和长期研究路线。
- 用户明确批准运行单 seed truth-only pilot；此前不得运行训练或生成 E002 数据。
- 若获批，先实现最小 runner、split/标签/配对/CUDA preflight 和对应测试，再执行；任何协议修订须先冻结并记录。
- pilot 通过后另提正式多 seed E1 与教师知识评估方案；不得自动开启 E2/E3 或 final blind。

## 对整体方向的影响与下一项任务

E002 若通过，只能说明任务级学生有基础可学习性，支持继续评估真值监督学生，再独立审查教师软知识是否增加收益。若失败且工程与可辨识性检查有效，应暂停蒸馏，回到物理表示和标签问题；若所有候选长期落后传统链，审查应重新评估“直接学生”是否值得继续，而不是扩大网络或继续搜索 CFAR。

**下一项唯一优先任务：** 将本协议提交 ChatGPT 独立审查，并取得用户对执行单 seed E1 pilot 的明确批准；批准前不实现训练运行路径、不训练、不生成 blind。

## 需要 ChatGPT 审查的文件

- `docs/RESEARCH_CHARTER.md`
- `docs/CURRENT_STATUS.md`
- `docs/RESEARCH_ROADMAP.md`
- `docs/research/ASSUMPTION_REGISTER.md`
- `docs/task_alignments/D2-A-E002_alignment.md`
- `experiments/D2-A-E002/protocol.md`
- `experiments/D2-A-E002/config.yaml`
- `docs/stage_reviews/D2-A-E002_protocol_review.md`
- `docs/stage_reviews/D2-A-R3_review.md` 与 `experiments/D2-A-R3/{protocol.md,analysis.md,metrics.json}`

本提交 SHA 和工作区状态由提交记录及 Codex 汇报提供；协议中不伪造运行代码、metrics 或结果。
