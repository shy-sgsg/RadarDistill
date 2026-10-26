# D2-A-T001 教师增强与知识迁移任务对齐

**日期：** 2026-10-26
**状态：** 已完成。运行代码 SHA `91dfb81673e45cad3f3330d90adc707a0cdf859d`；协议 v1.2 的科学设计、样本和门槛未变。整仓 270 项测试通过，CUDA 教师阶段完成；G1/G2未通过、G3未评估，按冻结门 STOP 本候选，不运行 S2。
**研究总纲：** `docs/RESEARCH_CHARTER.md` v1.2，提交 `289b812e8bc93fbdabca0cfb3722ad24f045351b`。本任务未修改总纲。
**对应协议：** `experiments/D2-A-T001/protocol.md`；配置 `config.yaml`。

## 总纲映射

- **总体目标：** 用物理仿真真值和可信传统教师知识训练直接从三通道复数 IQ 完成 GMTI 检测、距离及连续 Doppler 估计的学生，并逐步评价精度、鲁棒性和后续完整处理成本。
- **科学问题：** Q1 学生是否能从相干多通道数据学习物理参数；Q2 教师提升是否转移为有效学生收益；Q3 物理真值与传统算法如何组合；Q4 三通道观测能否支持弱目标检测和参数估计；Q9 教师任务表现、输出可靠性与学生可蒸馏性是否可区分。
- **主要内容：** 对当前 B3/T1/K2 的失败分层；基于 post-Doppler ACE 做一个有界三通道教师改进；分别评价检测估计与软响应；只有教师性能和知识质量都通过预注册门，才运行一个固定学生 S0/S1/S2 pilot。
- **路线关系：** R3 的 T1/T2 固定阈值 G2 失败历史不变。T001 是新冻结实验，不覆盖 R3，也不将 T1 自动升级为获批教师。候选方法是模型辅助传统教师，不是纯端到端学生成功证据。
- **创新性边界：** ACE、post-Doppler adaptive detection、协方差估计均有既有文献。本项目不声称这些算法为原创，也不把别的阵列/杂波/训练条件下的论文数值外推；可研究贡献是本项目明确物理和任务下的教师可靠性、困难层软知识及学生迁移证据。

## 研究设计与状态

| 阶段 | 冻结设计 | 当前实况 |
|---|---|---|
| 文献与瓶颈 | 参见教师文献、瓶颈与候选选择三个研究文件 | 文献/旧结果审查完成；按唯一候选和停止门执行完一轮新数据比较 |
| 候选 | 唯一先进候选为3维 post-Doppler ACE×AMF；训练 covariance只用独立H0 IQ | 完成CUDA covariance与评分；没有增加备选或调参 |
| 协议 | 4096 H0 covariance；15k calibration +15k verification；12k paired development；困难主条件 C3/C/SNR0/CNR20、n=500；固定B3/T1/K2比较 | v1.2种子、分组、公式、阈值、bootstrap数和阶段门按冻结设置运行 |
| 实现/正确性 | G1：公式、几何、尺度/相位不变、配对、split、基线逐样本一致 | split、有限性、协方差正定和公共相位诊断通过；公共复尺度最大差4.58e−5超过1e−6，故G1 FAIL；运行代码 CUDA阶段正常完成 |
| Teacher G2/G3 | 独立H0 Pfa/分层、困难条件任务增益/保护；只在G2通过后审查soft-map | G2 FAIL：整体Pfa=0.05007但C3/SNR0/CNR20大组上限0.125–0.127；主困难joint为0，A/B/C性能明显劣于T1；G3 NOT EVALUATED |
| 学生 | 只有Teacher G1–G3全通过才运行单S2 seed | 前置门未通过，未训练S2，也未运行蒸馏 |
| Blind/部署 | 不生成final blind；GPU teacher成本仅作实测记录，不作实时主张 | 未生成blind；候选内核CUDA batch1约7.40×、batch16约15.63×于CPU，全阶段约3539.7秒；仅报告计算路径，不主张实时优势 |

## 保留的既有证据

- R3：T1/T2 尺度 G3 与 H1 G4 通过，固定阈值 G2 未通过；T1 比 T2 更有前景但无正式教师资格。
- E002：A/B/C 单 seed 任务 G4 通过但分层 Pfa 与相位/频移门失败；B−A joint gain `−0.1254`，C−B=`+0.0036`；课程和 K2 无预注册 pilot 增益。
- E002 K2 audit：pooled C3 mass lift `0.03623`、top localization `0.7227`；C3/C/SNR0 mass lift `0.000673`，top localization `0.0471`。这组难例启发 T001 主条件，但旧样本不进入新分析。
- T001 不改变 D2-A-E001/R3/E002 frozen protocols、指标或历史结论，也不修改 `RESEARCH_CHARTER.md`。

## 阶段结论与下一步

总体ACE×AMF verification Pfa达到总量门，但3个大H0子组失败；C3 partial-overlap/SNR0/CNR20困难组joint success为0，对B3的差值为−0.010（95% CI `[−0.018,−0.002]`）。A/B/C pooled Pd、joint success和参数RMSE均明显劣于T1。尺度一致性超过冻结门，目标单独频移诊断也未表现出峰值跟随。结果支持停止这一候选，而不是增加新的教师或放宽门槛。阶段决定 **STOP（限T001教师候选分支）**。

下一步只进行独立 ChatGPT 阶段审查，判断后续研究是否应转向新的学生/物理一致性问题；未经新协议和批准，不运行 S2、正式蒸馏或 blind。CPU covariance试跑历史已在协议中披露，其临时bank由正式CUDA拟合覆盖，不作为科学结果。
