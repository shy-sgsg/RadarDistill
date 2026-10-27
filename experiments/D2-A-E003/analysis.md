# D2-A-E003 阶段分析与结果

## 阶段结论

**S2 当前相干表示候选：STOP。D2-A 任务级学生路线：REFINE。** 前端按数学构造实现了公共相位不变性，且 R1/R2 前置门通过；然而配对 E1 pilot 中 S2 的整体检测、C2/C3 参数估计和 joint success 均低于匹配 S0，未通过冻结 G4 与主要收益门。按协议，不进行正式多种子确认、知识蒸馏或 final blind。这个单候选结果不等于否定所有物理相干表示或 RadarDistill 总体方向。

实验问题对应总纲 Q1、Q4，并关联 Q3/Q8：显式复相干特征是否能提高任务级学生对可辨识目标的利用，同时保护检测、参数估计和条件虚警。本轮仅使用合成三通道静止杂波、PRF 1300 Hz、130 脉冲、32 距离单元。

| 阶段门 | 结果 | 证据 |
|---|---|---|
| G0 工程与数据 | PASS | 81,288 个场景 ID 跨 split 不重叠；配对 H0/H1 保持同组；训练计划、检查点、种子与环境留档；无 final blind |
| G1 前端正确性 | PASS | complex64 相对最大误差 `9.5371e-8`；公共相位相对误差 `2.9072e-7`；Lag-1 频率映射误差 `1.14e-13 Hz`；跨通道相位误差 `1.57e-16`；距离泄漏为 0 |
| G2 可观测信息 | PASS | A/B/C 的 H1−H0 特征增量中位 RMS 为 0.1686/0.1433/0.1562，均高于预注册数值地板的 10 倍；仅证明诊断条件下有非退化增量，不代表学生可利用它 |
| G3 基础任务学习 | S0、S2 均 PASS | 两者均高于 AUC 0.5，且 range/Doppler RMSE 优于训练真值常值基线；bootstrap 只反映场景抽样，不是 seed 不确定性 |
| G4 任务保护与增益 | FAIL | 总体 Pd 和 joint success 下降，C2/C3 RMSE 恶化，C3 两臂绝对 Pfa 均超过 0.10；主要 joint-success 增益门未通过 |
| G5 物理一致性 | 分项报告 | S2 公共相位不变性显著改善；频移与真实几何速度变化诊断仍显示较大估计残差。物理一致性不替代 G4 |
| G6 计算 | 已测 | 同一 RTX 3050 GPU 上 S2 输入到输出较慢；仅是模型级 batch 延迟，不是完整 GMTI 链实时测试 |

## 设计与物理含义

S0 是冻结的实值 `TaskStudent`，输入六个 IQ 平面。S2 将三通道 IQ 映射为 28 个实值特征：延迟 `{1,4,16}` 的同通道复相关、三组零延迟跨通道复相关、归一化瞬时功率和 `log(q(X))`；再经可学习 `1×1` 投影 `28→6`，接同一主干和任务头。无原始 IQ 旁路、教师输入或额外时序 CNN。S2 总参数 13,557，S0 为 13,383，差 174（1.30%）。

- 对 `X′=X exp(jφ)`，同通道和跨通道乘积、功率、`q(X)` 均不变，因此固定特征前端严格具有公共复相位不变结构；complex64 只带来已测的有限精度误差。
- 同通道单音相关的相位为 `2πfτ/PRF`。Lag-1 在主 PRF 区间单值；lag-4/16 各自有更短的频率周期，不能单独消除混叠。R1 证明映射正确，R2 证明预定干预会改变特征；没有 lag-1-only 学生对照，因此未证明高延迟相对 lag-1 的独立任务贡献。
- 跨通道乘积保留三组接收通道相对相位，可携带阵列相位中心/传播几何相关信息；本任务没有角度标签或输出，不能由此声称已估角。全通道公共相位被消去，但通道间相对相位没有被独立归一化。
- 特征逐距离门计算，没有跨距离混合，range 索引仍在张量中；不过 `28→6` 可学习投影可能压缩/丢弃信息。S2 结果不能区分固定前端、场景归一化、可学习投影及其优化共同造成的影响。
- `X=C+S+N` 的二阶乘积含背景、目标以及交叉项。该前端不是杂波抵消器。前 27 个通道按可观测 `q(X)` 归一化，`log(q)` 仅提供场景尺度提示；这既不等于绝对幅度信息完整保留，也不证明跨条件虚警稳定。
- 有限延迟复相关并不保证对目标/杂波重叠、任意复混合、通道误差或 exact-overlap 可辨识；D 仅作边界参考。

## 训练与独立评价

S0/S2 均使用 seed 101、相同初始 TaskStudent 参数、相同配对场景顺序、物理真值 E1 监督、FP32、RTX 3050 `cuda:0`、microbatch 16、有效 batch 64、每臂 1,600 次更新；无教师损失、课程、K1/K2 或辅助头。S0/S2 训练耗时约 350/494 秒，峰值 CUDA allocated 约 275/286 MB；没有 OOM。按共同验证 truth-task loss 选 checkpoint：S0 update 1500，S2 update 1600。

独立 split 为 H0 calibration 8,192、H0 verification 8,192、独立 C3 H0 verification 4,096、H1 development 4,096 对（每 C0–C3 1,024）、exact-overlap boundary 512 对。阈值分别由本模型 H0 calibration 的 95% `higher` 分位数设定，决策使用 `score >= threshold`。无 final blind。bootstrap 单位是 `base_scene_id`，只表达本训练 seed 下的场景抽样不确定性。

### 总体任务与条件结果

| 指标 | S0 原始 IQ | S2 相干特征 | S2−S0 / 95% 配对区间 |
|---|---:|---:|---:|
| 总体 H0 verification Pfa，n=8192 | 0.04834（exact 单侧上限 0.05242） | 0.04749（上限 0.05153） | — |
| C3 H0 Pfa，n=4096 | 0.18140（上限 0.19159） | 0.15283（上限 0.16238） | −0.02856 `[-0.04444,-0.01245]` |
| Development AUC（n=4096） | 0.98416 `[0.98209,0.98611]` | 0.97713 `[0.97446,0.97973]` | — |
| Pd | 0.91089 | 0.86353 | −0.04736 `[-0.05713,-0.03759]` |
| Joint success | 0.68652 | 0.47632 | −0.21021 `[-0.22559,-0.19507]` |
| Range RMSE（cell） | 3.108 | 4.130 | — |
| Circular Doppler RMSE（Hz） | 46.42 | 85.70 | — |
| **C2/C3 joint success** | — | — | **−0.25928 `[-0.28223,-0.23682]`；主要收益门需 ≥+0.02，失败** |
| C2/C3 range RMSE 比 S2/S0 | — | 1.329 | `[1.221,1.456]`；冻结上限 1.10，失败 |
| C2/C3 Doppler RMSE 比 S2/S0 | — | 1.835 | `[1.720,1.959]`；冻结上限 1.10，失败 |

| 难度 | S0：Pd / joint / range RMSE / Doppler RMSE | S2：Pd / joint / range RMSE / Doppler RMSE |
|---|---|---|
| C0 | 1.000 / 0.959 / 0.051 / 9.23 Hz | 1.000 / 0.922 / 0.053 / 11.69 Hz |
| C1 | 1.000 / 0.932 / 0.065 / 10.92 Hz | 1.000 / 0.646 / 0.084 / 30.09 Hz |
| C2 | 0.937 / 0.629 / 1.586 / 34.49 Hz | 0.936 / 0.284 / 1.752 / 61.05 Hz |
| C3 | 0.707 / 0.227 / 6.009 / 85.00 Hz | 0.519 / 0.053 / 8.071 / 156.88 Hz |

S2 的 C3 Pfa 比 S0 低约 2.86 个百分点，但两臂都远超冻结的绝对上限 0.10；同时 S2 C3 Pd 低 18.85 个百分点，joint success 仅 0.053。故不能把 Pfa 差值改善解读为整体检测稳健性通过。C3 三种场景子组的 Pfa 在 S0 为 0.175–0.187、S2 为 0.147–0.163，均未达到目标。

### 公共相位、频移与计算

公共相位诊断对每臂使用同一批 H0/H1 样本，每类 256（C0–C3 各 64），施加 32 个固定旋转，并复用 calibration 阈值。

| 结果 | S0 | S2 |
|---|---:|---:|
| H0/H1 判决一致率中位数 | 0.949 / 0.957 | 1.000 / 1.000 |
| Presence 分数变化 p95 最大值（H0/H1） | 1.384 / 2.380 | 3.73e-5 / 5.72e-5 |
| H1 range 变化 p95 最大值 | 9.384 cells | 7.63e-6 cells |
| H1 wrapped Doppler 变化 p95 最大值 | 78.64 Hz | 0.00662 Hz |

这证实 S2 的输出符合公共相位不变结构；不证实其频率等变或任务更优。频率实验分开实施：整体 IQ 调制同时移动背景和目标，仅作数学响应诊断；目标分量单独调制是仿真反事实，不代表可部署输入；重新生成目标速度几何时背景与噪声按 scene ID 配对。按场景和干预类型汇总的 shift error p95 中位数如下（每种每场景 6 个干预值）：

| 模型/场景 | 整体 IQ 调制 | 目标分量反事实调制 | 重新生成目标几何 |
|---|---:|---:|---:|
| S0 separated | 35.46 Hz | 25.33 Hz | 167.10 Hz |
| S0 near clutter edge | 46.80 Hz | 25.66 Hz | 164.25 Hz |
| S0 partial overlap | 91.27 Hz | 61.42 Hz | 141.99 Hz |
| S2 separated | 77.33 Hz | 31.10 Hz | 212.30 Hz |
| S2 near clutter edge | 110.90 Hz | 68.23 Hz | 152.78 Hz |
| S2 partial overlap | 76.18 Hz | 61.05 Hz | 165.57 Hz |

这些小样本诊断显示两模型都没有稳定恢复重新生成几何中的目标频移；它们不是独立频率估计精度结论，也不应和整体 IQ 调制混为一谈。

CUDA batch 1 的完整 IQ→输出时间：S0 1.77 ms、S2 4.20 ms；batch 16：S0 8.21 ms、S2 10.94 ms。S2 固定前端单独约 1.66/2.33 ms；S0/S2 峰值显存 batch 16 约 45/56 MB。此处只计模型输入到任务输出，不含完整 GMTI 距离压缩或系统链；本轮不作实时性结论。S2 在本边界较慢，也没有精度补偿。

## 失败原因判断与假设状态

结果支持“显式公式可提供公共相位不变性”，但不支持“该 28 通道特征 + 28→6 投影能改善任务级学生”。失败集中在 C1–C3 的 joint、距离门定位和 Doppler 估计；参数差异只有 1.30%，因此单纯参数量增加不是合理解释。仍无法从这两个臂分离场景归一化、相关交叉项、28→6 投影信息瓶颈、初始化/优化难度及训练 seed 的作用。R2 仅在 SNR/CNR=10 dB 的少量固定几何场景证明可观测增量远大于数值地板；不等于完整 C3 mixture 可辨识，也不说明网络容易读出。

- **H-ARCH-D2-002：** 当前候选未通过任务收益/保护门，**NOT_SUPPORTED（限本候选）**。公共相位不变性单独通过。
- **H-FREQ-D2-001：** 前端数学上保留周期频率相位，但没有 lag-1-only matched student 对照；S2 Doppler 变差，**INCONCLUSIVE（增量任务价值未隔离）**。
- **H-DET-D2-002：** 当前 S2 未保护 C2/C3 检测与联合参数任务，**NOT_SUPPORTED（限本候选）**；Pfa 相对 S0 改善不足以弥补任务损害。
- **H-IDENT-D2-001：** **INCONCLUSIVE**。R2 非退化特征增量与 C3 任务退化并存，尚不能归因于物理信息不可辨识或学习投影/优化瓶颈。
- 教师能力、教师蒸馏收益、正式跨训练种子稳定性、真实回波迁移均未测试；E003 无 E2/E3、三 seed 或盲测。

最有根据的下一步是先做独立阶段审查，再决定是否批准一个有界的**表示归因诊断**：在冻结的同一批独立开发 scene 上，分别检查固定 28 通道前端与已训练 28→6 投影后的任务相关信息，重点定位 C1–C3 的 range/Doppler 失败；不增加模型容量、不调门槛、不蒸馏、不生成 blind。若现有保存的输入/预测不足以完成诊断，须先另行提出新协议并使用新开发场景。当前不建议修改研究总纲，也不应直接扩展三种子训练。

## 运行复核与文件

完整 AISAR 测试套件在评价修复后为 **303 passed**。模型训练使用预注册代码源 `bd18df1f4fa818684bd28874c3ba96d23a1e7dde`；评价计时处曾因在默认梯度模式下消费 inference tensor 而失败。`af0f596` 加入推理计时上下文和回归测试，保存了原始训练前 preflight，并于训练后重验；相同 checkpoints、split 和阈值重新评价成功。该运行错误不产生或替代任何科学结果。

## 缓存清理审计

分析和审查材料完成后，清理脚本只删除 `cache_manifest.json` 登记在 E003 `data/cache/` 下的 90 个普通文件，共 14,232,309,192 bytes。manifest 列表外文件未发现，预先存在的用户文件和未跟踪 GMTI 目录均未触及。清理后项目占用 801,766,373 bytes（0.747 GiB），低于项目最终 20 GiB 限制；final blind 未生成。逐文件清单与容量计算保存在 [`cleanup.json`](cleanup.json)。

- [`protocol.md`](protocol.md)、[`config.yaml`](config.yaml)、[`feature_design.md`](feature_design.md)
- [`metrics.json`](metrics.json)、[`result.csv`](result.csv)、[`roc.csv`](roc.csv)、[`physical_consistency.json`](physical_consistency.json)、[`compute.json`](compute.json)
- [`training.json`](training.json)、[`runs/seed_101/`](runs/seed_101/)、[`scene_manifest.csv`](scene_manifest.csv)、[`selected_scene_ids.csv`](selected_scene_ids.csv)
- [`controlled_r2.json`](controlled_r2.json)、[`r0_r1_metrics.json`](r0_r1_metrics.json)
- [`figures/rmse_by_course.png`](figures/rmse_by_course.png)、[`figures/c3_rmse_by_snr.png`](figures/c3_rmse_by_snr.png)、[`figures/development_roc.png`](figures/development_roc.png)、[`figures/phase_decision_agreement.png`](figures/phase_decision_agreement.png)
