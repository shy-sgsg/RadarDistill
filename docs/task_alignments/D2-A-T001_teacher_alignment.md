# D2-A-T001 教师增强与知识迁移任务对齐

**日期：** 2026-10-26
**状态：** 用户明确批准；协议 v1.2 已记录 CUDA 执行后端修订，科学设计和门槛不变。CUDA 教师代码及整仓测试通过；一次 CPU 试跑仅完成 covariance 拟合后中止，没有形成评分结果。CUDA 版本尚待提交推送和重新 preflight。
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
| 文献与瓶颈 | 参见教师文献、瓶颈与候选选择三个研究文件 | 已完成文献/旧结果核验；未运行 T001 新数据 |
| 候选 | 唯一先进候选为 3维 post-Doppler ACE×AMF；训练 covariance 只用独立 H0 IQ | 代码已实现；ACE/AMF、评价和运行器定向测试通过 |
| 协议 | 4096 H0 covariance scenes；15k calibration + 15k verification；12k paired development；困难主条件 C3/C/SNR0/CNR20、n=500；固定 B3/T1/K2 比较 | v1.2 只把 covariance 与候选评分设为 CUDA `cuda:0`；样本、公式、阈值和门槛保持冻结 |
| 实现/正确性 | G1：公式、几何、尺度/相位不变、配对、split、基线逐样本一致 | CUDA covariance/ACE×AMF、逐场景指标和运行器已实现；`AISAR` 整仓测试 270 项通过；运行时 G1 仍待正式 T001 数据验证 |
| Teacher G2/G3 | Pfa/任务增益/保护；target-window mass、localization 和错误知识 | 尚未有结果 |
| 学生 | 只有 Teacher G1–G3 全通过才运行单 S2 seed；S0=E002 B、S1=E002 C、S2=新 ACE/AMF，同课程、init、数据、1600 updates | 仅作为冻结条件分支，未开训 |
| Blind/部署 | 不生成 final blind；GPU teacher成本仅作实测记录，不作实时主张 | 未生成/未测；单批实现核验显示 CUDA 快于 CPU，但不是阶段正式性能结论 |

## 保留的既有证据

- R3：T1/T2 尺度 G3 与 H1 G4 通过，固定阈值 G2 未通过；T1 比 T2 更有前景但无正式教师资格。
- E002：A/B/C 单 seed 任务 G4 通过但分层 Pfa 与相位/频移门失败；B−A joint gain `−0.1254`，C−B=`+0.0036`；课程和 K2 无预注册 pilot 增益。
- E002 K2 audit：pooled C3 mass lift `0.03623`、top localization `0.7227`；C3/C/SNR0 mass lift `0.000673`，top localization `0.0471`。这组难例启发 T001 主条件，但旧样本不进入新分析。
- T001 不改变 D2-A-E001/R3/E002 frozen protocols、指标或历史结论，也不修改 `RESEARCH_CHARTER.md`。

## 当前唯一下一步

下一步为推送 v1.2 代码后执行 CUDA preflight，并从冻结 covariance seed block 在 GPU 重拟合，再开始 H0 评分。先前 CPU 尝试生成过 covariance H0 场景并拟合临时 bank，但在 calibration/verification/development 评分前已中止；该 bank 将被覆盖，不作为结果。G3 仅在 G2 通过后裁定；若 G1–G3 不满足，则停止教师/学生该分支并保留失败结果，不追加第二个 advanced teacher。尚未训练学生或生成 final blind。
