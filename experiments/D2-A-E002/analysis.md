# D2-A-E002 阶段结果

- 协议：v2.1；代码提交：`1f822470b906c733706c1a10f031e7a627e140a8`。
- 训练臂：A, B, C；每臂 1 个 seed；final blind 未生成。
- 数据：独立 H0 calibration/verification 各 15,000 场景，development 5,000 对；全为仿真数据。
- 完整数据缓存首次生成耗时：1195.4 秒；随后完整缓存复用校验耗时：0.2 秒；评价耗时：973.3 秒。

## 总体结果

- 按 development AUC 最高的方法为 **T1**（AUC=0.9863）。
- 学生臂中 AUC 最高的是 **A**（AUC=0.9846）。
- E0 AUC=0.6103；E0 在自身校准阈值下 Pd=0.0538。
- 训练 seed 101 的各臂 AUC、Pd、RMSE 与联合成功率见 [`result.csv`](result.csv) 和 [`metrics.json`](metrics.json)。
- K2 audit 是否允许 C 臂：**True**。该结果只描述软响应，不改变 R3 固定阈值门状态。

## 任务与课程 / K2 对比

| 方法 | H0 verification Pfa | Pd | AUC | Range RMSE (cell) | Doppler RMSE (Hz) | joint success |
|---|---:|---:|---:|---:|---:|---:|
| B3 | 0.0495 | 0.5916 | 0.9397 | 2.941 | 46.80 | 0.5658 |
| T1（未获固定阈值教师资格） | 0.0504 | 0.9206 | 0.9863 | 2.941 | 46.80 | 0.9010 |
| A | 0.0457 | 0.9158 | 0.9846 | 3.166 | 45.50 | 0.6934 |
| B | 0.0495 | 0.8996 | 0.9787 | 3.284 | 57.95 | 0.5680 |
| C | 0.0469 | 0.9000 | 0.9815 | 3.175 | 61.59 | 0.5716 |

- 三个学生总体 H0 Pfa 的精确单侧 95% 上限为 0.0486–0.0525；但预注册分层门未通过，最差 SNR/CNR 子组上限 A/B/C 分别为 0.1953/0.2045/0.2022。
- B−A 的 development joint-success 差为 **−0.1254**（配对场景 bootstrap 95% CI `[−0.1386, −0.1122]`）；C−B 为 **+0.0036**（`[−0.0088, +0.0162]`）。两组都没有达到 G5；B/C 均未在 1600 updates 内达到 A 的最佳 validation loss。
- G4 逐臂通过：三臂均明显优于常值参数估计与 E0 检测。A 的主 A/10/10 dB joint success 为 0.8489（单侧 bootstrap 下限 0.8201）；B 为 0.5060，C 为 0.6211。单 seed development 结果不能证明跨种子优势。
- G6 的公共相位与目标频移等变性门全部失败：相位决策一致率 0.943–0.950，相位下 Doppler 改变量 p95 为 61.7–71.9 Hz，频移等变残差 p95 最大为 59.6–73.4 Hz。独立通道相位扰乱使 AUC 下降约 0.176–0.208，说明这些模型的检测会利用通道相干，但没有学到所要求的稳定等变映射。

## 预注册门

- G1 工程 sanity：True。
- A G3 H0 检测门：False。
- A G4 任务学习门：True。
- B G3 H0 检测门：False。
- B G4 任务学习门：True。
- C G3 H0 检测门：False。
- C G4 任务学习门：True。
- G5 B_minus_A pilot signal：False。
- G5 C_minus_B pilot signal：False。
- 阶段决策：**REFINE**。
- exact-overlap D boundary AUC（仅描述）：0.5030。
- G3 的总体学生 Pfa 通过全局点估计/上限子门，但分层虚警门失败；G6 公共相位与频移一致性门失败。综合阶段决定 **REFINE**，不开展正式三种子或 final blind。

## 解释边界

本轮是单训练 seed 的开发 pilot。场景 bootstrap 反映场景抽样不确定性，不代表跨 seed 不确定性；结果不能证明课程学习或蒸馏的总体收益。模型选择未使用 final blind，且没有生成 final blind。R3 的独立固定阈值结果、P0/D1 结果和历史负结果保持原样。

方法分组、阈值平局、固定阈值 H0 verification、相位/频移/尺度/通道物理诊断及逐场景预测分别见 CSV、JSON 和 `figures/`。

数据生成 IQ/memmap 缓存已在评价完成后按项目容量要求清理；所有 split manifests、生成记录、训练记录、checkpoint、逐场景评价、图表和审计结果均保留。缓存可按冻结配置与种子重建，清理明细见 [`CACHE_CLEANUP_2026-10-25.md`](../../docs/maintenance/CACHE_CLEANUP_2026-10-25.md)。

## 训练—推理 range 门与 Doppler 损失诊断

TaskStudent 的训练损失在真实 range cell 监督 Doppler，而实际推理按 range argmax 选取 Doppler 向量。该错配已在每个 validation checkpoint 记录，并在选定 checkpoint 的 H1 上比较两种读门：预测/真实 cell 不一致率 A/B/C 为 `10.30%/11.14%/10.96%`；正常预测门 Doppler RMSE 为 `46.66/58.55/60.03 Hz`，同模型改读真实 cell 后为 `40.95/56.03/59.55 Hz`。因此 range 门错误会增加部分 Doppler 误差，尤其 A，但真实门下仍保留约 `41–60 Hz` RMSE，不能把它视为主要瓶颈。

在 development 的 `development_per_scene.csv` 中以 `floor(range_estimate_cell)` 还原 argmax 门，并与 `floor(truth_range_cell)` 比较，错门数为 A `502/5000`、B `551/5000`、C `565/5000`。这些逐场景数据只用于当前 pilot 诊断；没有多训练 seed，且本次未为 true-cell 反事实差值另外计算场景区间。该结果不改变协议、模型选择、训练损失或阶段门。架构/损失细节见 [`architecture_and_loss_review.md`](architecture_and_loss_review.md)。
