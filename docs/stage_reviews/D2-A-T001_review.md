请首先读取本次Git提交对应版本的 `docs/RESEARCH_CHARTER.md`，理解RadarDistill的整体研究主题、研究目标、研究内容、科学问题和长期技术路线，然后结合这些内容审查本阶段的教师算法、学生实验、代码、数据及研究结论。

# D2-A-T001 阶段审查请求：post-Doppler ACE×AMF CUDA 教师比较

## 提交与研究总纲

- **数据运行代码提交：** `91dfb81673e45cad3f3330d90adc707a0cdf859d`（`main`，已推送）。本阶段结果与审查文件由后续结果记录提交发布。
- **研究总纲：** `docs/RESEARCH_CHARTER.md`，v1.2；T001未修改总纲。该版本已存在于代码运行提交中。总纲提交来源：`289b812e8bc93fbdabca0cfb3722ad24f045351b`。
- **本阶段路线关系：** 本工作只检验一个传统教师候选能否可靠支持三通道 GMTI 学生检测与参数估计；不构成独立 CFAR研究，不改变D2-A-E001、R3、E002或D1历史结论。

## 总体研究目标与科学问题

本阶段服务于项目总体目标：通过物理仿真真值和可靠传统算法知识蒸馏，训练直接完成GMTI动目标检测与参数估计的学生，最终评价精度与完整处理成本。

对应总纲科学问题为：Q1有限网络能否从复数观测学习可辨识物理量；Q2教师知识是否能转化为学生收益；Q3物理真值与传统算法如何组合；Q4三通道回波能否支持弱目标检测和参数估计；以及T001任务对齐中扩展的教师任务表现和可蒸馏性是否可区分。此次教师比较是学生训练前的必要性筛选；阶段门失败时应停止该教师分支。

## 冻结实验、代码与数据

- **协议/配置：** `experiments/D2-A-T001/protocol.md` v1.2、`config.yaml`。科学问题、唯一候选、物理模型、样本、阈值、bootstrap与G1–G3门均在看到本轮数据前冻结；v1.2只将H0 covariance拟合与ACE×AMF评分设为强制CUDA。
- **方法：** 三通道 post-Doppler ACE×AMF，covariance只用独立H0训练场景拟合；比较冻结B3/T1/K2。旧基线保持原CPU NumPy实现。
- **样本：** 4,096个H0 covariance scenes；15,000个H0 calibration；15,000个独立H0 verification；12,000个配对H0/H1 development scenes。数据均为仿真，不是实测或final blind。
- **实现文件：** `src/radardistill/teachers/d2a_t001.py`、`scripts/run_d2a_t001.py`；验证包含CPU/CUDA输出对照、尺度/相位诊断、CUDA路由、运行资源记录和G2失败后跳过G3的测试。
- **正式结果：** `experiments/D2-A-T001/metrics.json`、`result.csv`、`calibration_scores.csv`、`verification_scores.csv`、`development_per_scene.csv`、`verification_subgroups.csv`、`covariance_bank.npz`、`covariance_diagnostics.json`、`physical_diagnostics.json`、`environment.json`、`figures/`与`analysis.md`。
- **测试：** 运行代码提交前及结果归档前 `AISAR` 整仓 `pytest` 均为270 passed；CUDA实现的物理/路由测试及阶段报告格式测试通过。

## 冻结门与实际结果

| 阶段门 | 实际结果 | 判定 |
|---|---|---|
| G1 正确性、分组、有限性与物理尺度不变 | covariance finite/positive definite、split完全互斥、H0/H1 scores finite、label-free API及冻结基线复用均通过；公共相位旋转score/map差为0。但公共复尺度下score/map最大差`4.5776e-5`，高于冻结上限`1e-6` | **FAIL** |
| G2 独立虚警、困难条件收益与A/B/C保护 | 总体verification Pfa=`0.05007`、单侧exact 95%上限=`0.05309`；但C3/SNR=0/CNR=20三个大分组上限分别约`0.125/0.127/0.125`。困难条件ACE×AMF/B3/T1 joint success=`0/0.01/0.004`；ACE×AMF减B3为`−0.010`（95% CI `[−0.018,−0.002]`）。A/B/C pooled相对T1：Pd差`−0.2267`（`[−0.2345,−0.2188]`）、joint差`−0.2441`（`[−0.2522,−0.2361]`）；Range RMSE差`+3.353 cells`，Doppler RMSE差`+93.822 Hz` | **FAIL** |
| G3 教师软图质量 | G2失败后按照冻结协议停止审查；未生成有效soft-map审计，也未训练S2 | **NOT EVALUATED** |

整体开发指标：B3/T1/ACE×AMF验证Pfa分别为`0.04913/0.05160/0.05007`；Pd为`0.5635/0.9189/0.6923`；AUC为`0.9400/0.9862/0.9128`；Range RMSE为`3.0956/3.0956/6.4485` cells；Doppler RMSE为`45.271/45.271/139.093 Hz`；joint success为`0.5407/0.8970/0.6529`。这些都是当前合成开发分布的结果。

物理频移诊断在固定困难子集上对注入`−25/−10/+10/+25 Hz`均得到约等于`−Δf`的估计残差，5 Hz内比例均为0；观测峰值未跟随目标分量频移。此项不是单独的冻结G1数值门，但它与较差的困难条件估计性能一致，需作为模型/统计假设问题审查。公共相位旋转差为0；独立通道相位扰动使候选score均值约增至原值9.53倍，表明方法依赖通道相干，但这本身不证明其估计正确。

## CUDA与资源记录

- 设备：NVIDIA GeForce RTX 3050 Laptop GPU，`cuda:0`，总显存约4 GiB；preflight时空闲约2.60 GiB。
- CUDA covariance拟合：47.95秒，约85.42 scenes/s。
- 候选内核复用16 CPI的benchmark：batch=1 CPU `0.016932 s/CPI`、CUDA `0.002288 s/CPI`（7.40×）；batch=16 CPU `0.019008 s/CPI`、CUDA `0.001217 s/CPI`（15.63×）。这是候选内核计时，不是完整处理链实时性能结论。
- 教师全阶段wall time：`3539.676 s`（约59分钟）；CPU累计时间`3883.503 s`。该全流程包括CPU场景生成、B3/T1/K2 CPU基线、2,000次分组bootstrap等，CUDA仅用于协方差与ACE×AMF候选计算。
- CUDA峰值allocated/reserved约`1233/1582 MiB`；未发生OOM。未保存完整IQ或大型模型文件。
- 阶段后整个项目`9,379,091,889 bytes`，即`8.735 GiB`，低于阶段收尾最终`20 GiB`上限。中间态不受此限制；未跟踪GMTI用户目录保持原样。

## 阶段结论和研究影响

- **本候选决策：STOP。** G1与G2均未通过；G3不评估。按照预注册规则不运行S2、不做正式蒸馏、不生成final blind、不增加第二个ACE/post-Doppler候选，也不因结果不理想而修改门槛。
- **支持/否定的假设：** H-TEACH-D2-001在本冻结仿真与协议下为NOT_SUPPORTED；H-KD-D2-002因教师G2前置门失败而保持INCONCLUSIVE。软响应是否值得蒸馏没有得到证据，不能从本次结果作否定或肯定。
- **整体研究方向：** 结果不否定“真值与教师共同支持任务级学生”这一总纲目标；但明确否定当前唯一ACE×AMF教师升级在该协议下具有所需检测/参数优势。全局Pfa合格不足以掩盖困难子组和估计失败。候选内核CUDA加速也不等于完整流程更快。
- **待审查的方向问题：** 是否应停止近期的传统教师增强搜索，转向对学生的真值学习、相干特征/频移一致性作单独的有界研究；该路线需要ChatGPT依据总纲和全部D1/D2证据判断，不在此报告中更改总纲或直接授权新实验。
- **下一项唯一优先任务：** 独立审查本阶段代码、分组结果、物理诊断和总体路线，再决定是否提出新协议。当前不开展新的教师变体或学生训练。

## ChatGPT审查要求

请结合本阶段证据主动评价：T001是否推进了总体学生目标；该科学问题是否仍值得研究；ACE×AMF候选与当前技术路线是否合理；G1/G2结果是否证伪了已登记假设；是否存在比继续教师搜索更有价值的替代方向；以及阶段 **STOP（候选分支）** 是否合理。请区分T001候选失败、软知识未评估和项目整体目标，不能将阶段门失败自动等同于总研究方向失败。
