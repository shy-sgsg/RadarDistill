# D2-A-R4 阶段审查材料与 Codex 自审

**状态：** Codex 自审完成，待独立 ChatGPT 审查；本文不是 ChatGPT 审查意见。

**建议判定：REFINE。**

**研究总纲：** `docs/RESEARCH_CHARTER.md` v1.2，冻结提交 `289b812e8bc93fbdabca0cfb3722ad24f045351b`。总纲正文未改。
**证据提交：** 主诊断 `6038d83d9464ea476ba873596fe430a9bdf6b6fd`；pilot `971dd3704053ebfa462388d4cf34cb3e60efcc48`；分层复核与配对结果 `66c9631f6102ff7477ff600b0263685782be5211`。最终归档提交 SHA 由推送后的 Git HEAD 提供。

## 1. 工程正确性

R4 维持三通道、130 脉冲、PRF 1300 Hz 和 32 距离单元。H0/H1 生成在同一随机种子下替换场景标签时逐样本完全一致，未见标签依赖。噪声功率遵照 `A² × 10^(-SNR/10)`，杂波功率遵照噪声功率乘 `10^(CNR/10)`；A 及 SNR/CNR 控制整体尺度，是明确数据生成机制。教师 B3/T1 CUDA 输出与冻结 CPU 公式完成 parity 比较；学生 S1 只运行一次、seed101、1600 updates、RTX3050 CUDA，峰值显存和训练时间已记录。

首轮 256 个 H1 诊断场景意外只覆盖 C0。原始结果保留；提交 `c496057` 增加固定种子的 64 场景/条件分层复核，提交 `66c9631` 增加同一 2048 个 pilot 场景的配对任务区间。后验结果另存，不覆盖原指标或改阶段门。分层复核明确显示首 256 样本 C0 计数 256、其他组为0。

数值审计中，正确 complex128 变换后输入 CUDA complex64 的复尺度不变性满足 R4 `1e-5` 门；legacy complex64 先乘路径差值 `2.44e-4`，CPU/CUDA ACE×AMF 分数绝对差 `3.09e-4`、相对差 `9.30e-7`，且 64 个样本无峰值或判决翻转。T001 冻结 `1e-6` G1 及 4.58e-5 原值没有修改。

## 2. 实验科学性

R4 使用新 H0 calibration/verification 和新 H1 development 场景；每个 H0 split 为4096个 CPI，H1 为2048个配对 CPI。固定学生沿用 E002 阈值；传统教师以 R4 H0 calibration 校准。整体 Pfa 接近 0.05，但 C3 分层偏高：T1 两个 C3 SNR 组 Pfa 为0.217/0.178；冻结学生 A/B/C 约0.152–0.195。T1 与 B3 的峰图和参数估计完全相同，因此 T1 整体收益源自检测分数归一化而非更好的参数前端。ACE×AMF 在困难部分重叠条件下参数和联合任务失效。

R4 的新 `1e-5` 数值门只用于本阶段诊断，不追溯改变 T001。后验分层抽样使用每条件64例，样本量不足以作为独立验收或模型选择；其作用是暴露原始取样覆盖限制。单 seed S1 的配对 bootstrap 只描述固定训练种子下的场景不确定性，不代表训练种子泛化。S1−S0 的 Doppler RMSE 差为−6.29 Hz（95% CI `[-9.16,-2.84]`），Pd 差−0.0117、joint success 差−0.0313，两个区间均完全低于0；range RMSE 差区间跨0。指标不支持将 S1 作为当前合格学生方案。

## 3. 总纲与研究主线

R4 检查学生能否在复杂静止杂波中使用通道相干，并探查普通实值 TaskStudent 的公共相位/频移关系，直接服务于总纲 Q1/Q3/Q4/Q9。T1 幅度归一化和 ACE×AMF 仅作为教师可靠性/基线诊断，没有把项目转为独立 CFAR 研究。

证据进一步支持“任务有可学信号，但普通 TaskStudent 没有自动得到跨条件公共相位不变性”。然而，R4 没有比较显式复数、等变、互相关或物理前端模型；所以它没有识别结构原因，也不能断定哪种替代方向最好。T001 软图仍未通过教师门并未作为 KD 输入；H-KD-D2-002 保持 INCONCLUSIVE。D2-A-R3、E002 与 T001 原阶段结论均保持冻结。

## 4. 战略建议

先由 ChatGPT 对照本阶段提交版本的研究总纲审查 R4 是否值得继续。若审查支持继续，下一实验应是一个新的、单因素、可证伪的物理表示诊断：固定同一 TaskStudent 任务、训练样本和预算，只比较普通实值输入与一个事先选定的显式复数/相位等变归纳偏置；独立验证 phase agreement、目标 Doppler equivariance、C3 分层 Pfa、Pd、RMSE 与 joint success。该比较需要先写新协议、冻结候选与门槛、明确对旧 E002/T001/R3 的关系，并由用户批准后实施。教师蒸馏只能在教师独立可靠且 truth-only 学生达到保护门后讨论。

若复表示模型仍不能提升跨条件 phase consistency 或任务联合成功，则停止扩大架构搜索，优先重审标签和可辨识/输入边界；不得以调高 tolerance、额外蒸馏损失或增加候选数遮盖负结果。

**当前建议：REFINE；唯一优先下一步为独立 ChatGPT 阶段审查。** 在审查和用户批准新协议前，不运行正式三种子、E2/E3 蒸馏或 final blind。

## 5. 审查所需材料

- 总纲：[`docs/RESEARCH_CHARTER.md`](../RESEARCH_CHARTER.md)
- 状态与路线：[`docs/CURRENT_STATUS.md`](../CURRENT_STATUS.md)、[`docs/RESEARCH_ROADMAP.md`](../RESEARCH_ROADMAP.md)、[`docs/research/ASSUMPTION_REGISTER.md`](../research/ASSUMPTION_REGISTER.md)
- 协议与结果：[`experiments/D2-A-R4/protocol.md`](../../experiments/D2-A-R4/protocol.md)、[`config.yaml`](../../experiments/D2-A-R4/config.yaml)、[`analysis.md`](../../experiments/D2-A-R4/analysis.md)、[`metrics.json`](../../experiments/D2-A-R4/metrics.json)、[`sampling_sensitivity.json`](../../experiments/D2-A-R4/sampling_sensitivity.json)、[`pilot.json`](../../experiments/D2-A-R4/pilot.json)、[`result.csv`](../../experiments/D2-A-R4/result.csv)
- 代码：[`scripts/run_d2a_r4.py`](../../scripts/run_d2a_r4.py)、[`scripts/summarize_d2a_r4.py`](../../scripts/summarize_d2a_r4.py)、[`tests/test_d2a_r4.py`](../../tests/test_d2a_r4.py)
- 相关历史：[`D2-A-R3_review.md`](D2-A-R3_review.md)、[`D2-A-T001_review.md`](D2-A-T001_review.md)、[`D2-A-E002_review.md`](D2-A-E002_review.md)
