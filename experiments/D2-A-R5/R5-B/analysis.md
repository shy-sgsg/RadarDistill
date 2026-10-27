# D2-A-R5-B TFAS 单种子 pilot 分析

**状态：** 已完成 seed 101、E1 truth-only、12,800 updates 和冻结开发评价。
**协议：** `protocol.md` / `config.yaml` v1.0；未作事后改门或调参。
**总纲：** `docs/RESEARCH_CHARTER.md` v1.3，正文未修改。
**训练代码/模型提交：** `55e7c7810e6cbb0277f79b1b6b4c401052b559dc`。
**范围：** 单训练 seed；复用 E003/R5-A 数据和已用开发集；非盲测，不估计 seed 方差。

## 1. 研究问题与总体结论

本阶段检验：对三通道原始复 IQ 显式建模慢时间和距离关系的轻量非 CNN 学生，能否在匹配训练样本、监督和 update 预算下改善 Small/Medium CNN 的可辨识 C2/C3 检测与连续参数联合任务，并满足 C3 H0 虚警绝对保护。

**TFAS 候选未通过预注册 G3，停止此结构的扩展。整体阶段建议为 REFINE。** G0、G1、G2 通过：代码、缓存分组、设备与参数量正确；小批可以过拟合；开发 AUC 和参数误差优于真值常值基线。G3 明确失败：TFAS 相比 Medium 的 C2/C3 joint success 差为 `−0.60449`（95% paired-scene bootstrap CI `[-0.62598,-0.58350]`），候选自身 C3 H0 `Pfa_CPI=0.15186`（单侧 exact 95% 上限 `0.16138`，高于 `0.10`）。任务保护和主增益门均未通过。

这否定的是本次预注册的 TFAS 候选在当前静止杂波仿真、单 seed 和受测预算下的晋级依据，不否定所有注意力、非 CNN 或物理归纳偏置。也不能据此推断任何未测试教师、知识蒸馏或 blind 结果。

## 2. 预注册、工程与训练证据

- 模型为实值 TFAS：显式保留六个 I/Q 平面，`6→56` embedding；每个距离门独立两层慢时间 Transformer，注意力池化后接一层距离 Transformer；不做 FFT、幅度化、固定相干投影或教师输入。
- 实际参数数 `79,638`，比 Medium CNN 的 `82,887` 少 `3.92%`，在预注册 ±20% 范围内。
- G0 使用 `NVIDIA GeForce RTX 3050 Laptop GPU`、`cuda:0`、FP32、PyTorch `2.8.0+cu128` / CUDA `12.8`；预检当时 CUDA 可用显存约 `3.15 GB`。缓存清单 90 个文件、`14,232,309,192` bytes；七个 split 的 base-scene IDs 两两互斥，训练 pair plan 为 `[12800,32]`、完整遍历八次。校准、验证、开发和 exact-overlap split 均与 E003/R5-A 冻结材料一致。
- 实现测试：模型及 runner 定向测试，并加 `tests/test_d2a_r5_training.py` 共 **26 passed**；`compileall`、配置校验和 `git diff --check` 通过。
- G1 固定 16 对 H0/H1 共 200 updates：task loss `5.57683→0.000867`，下降 `99.984%`；presence 和 H1 range-cell accuracy 均 `1.0`；四个任务头梯度有限且非零。使用 `cuda:0`、microbatch 16，无 OOM。
- 正式训练仅用物理真值 E1；seed 101、12,800 optimizer updates、每更新 32 个配对基础场景（64 CPI）、有效 batch 64、训练集遍历 8 次、FP32、AdamW、预注册两段余弦 LR。没有教师硬标签、软响应、truth-assisted window、KD 或 final blind。
- 全程 microbatch 16、gradient accumulation 4；无 OOM fallback。记录训练时间 `7,505.99 s`（约 2.09 小时），包含训练阶段每 400 步 validation；整个 train 命令 wall time `7,532.98 s`。峰值 CUDA allocated/reserved 为 `1,014,349,312 / 1,283,452,928 bytes`。参照 R5-A Medium 记录的 `3,951 s` 训练时间约为 TFAS 的一半；这是同一机器和协议下的单次执行记录，不是实时或跨机器基准。
- 最低 common validation truth-task loss checkpoint 选在 update `12,400`，task loss `1.71301`；update 12,800 为 `1.71355`。验证 loss 从初始 `5.46615` 下降，但在末段只小幅改善并进入平台附近；不能把此单 seed 结果当作架构上限或跨 seed 稳定性证据。validation AUC 从初始 `0.4773` 到选中点 `0.9619`，range RMSE 从 `11.62` 到 `4.62` cells，Doppler RMSE 从 `330.25` 到 `81.71 Hz`。

## 3. G2：基础任务学习

在 development 的 4,096 个配对 H1 场景上，TFAS AUC 为 `0.96139`，paired-scene bootstrap 95% CI `[0.95760,0.96512]`，下界大于 0.5。TFAS 的全 H1 range RMSE 为 `4.595 cells`、周期 Doppler RMSE 为 `82.00 Hz`；训练真值常值基线分别为 `6.917 cells`、`179.64 Hz`。配对误差差为 `−2.323 cells`（95% CI `[-2.576,-2.096]`）和 `−97.64 Hz`（`[-101.45,-93.91]`）。因此 G2 按预注册定义通过。

G2 只说明模型学到了一部分超出常值预测的任务映射。它没有通过检测/Pfa 与 C2/C3 联合性能保护；不能把 G2 通过解释为参数估计已达工程可用水平。

## 4. H0 校准、verification 与开发任务

学生独立用 8,192 个 frozen H0 calibration scenes 校准 `Pfa_CPI=0.05` 的 higher quantile；阈值 `0.0543661`，使用 `score>=threshold`。校准 `Pfa_ge=0.05005`、1 个平局。独立 H0 verification 为 `436/8192=0.05322`，单侧 exact 95% 上限 `0.05748`；总体 H0 门表现正常。专用 C3 H0 verification 为 `622/4096=0.15186`，单侧 exact 95% 上限 `0.16138`，**超过预注册绝对上限**。

| 模型 | 参数数 | 总体开发 AUC | Pd | Joint success | Range RMSE (cells) | Doppler RMSE (Hz) | C3 H0 Pfa |
|---|---:|---:|---:|---:|---:|---:|---:|
| Small CNN | 13,383 | 0.9937 | 0.9619 | 0.8389 | 2.239 | 25.23 | 0.20410 |
| Medium CNN | 82,887 | 0.9958 | 0.9746 | 0.9106 | 1.879 | 17.86 | 0.18896 |
| TFAS | 79,638 | 0.9614 | 0.8020 | 0.5054 | 4.595 | 82.00 | 0.15186 |

均为同一 frozen development ID 上的单 seed结果；Small/Medium 数字来自已完成的 R5-A capacity metrics，没有重训。

| 课程 | TFAS AUC | TFAS Pd | TFAS joint | TFAS range RMSE | TFAS Doppler RMSE |
|---|---:|---:|---:|---:|---:|
| C0 | 1.0000 | 1.0000 | 0.8643 | 0.895 | 12.54 Hz |
| C1 | 1.0000 | 1.0000 | 0.7207 | 1.804 | 19.05 Hz |
| C2 | 0.9274 | 0.7920 | 0.3516 | 3.653 | 48.75 Hz |
| C3 | 0.6994 | 0.4160 | 0.0850 | 8.188 | 154.91 Hz |

性能随难度明显下降；主要退化集中在 C2/C3。C3 的较低 Pfa 与严重 Pd、参数定位失败同时出现，不能视作合格的虚警修复。

## 5. G3：与 Small/Medium 的配对比较

所有区间均为 2,000 次 paired base-scene bootstrap；它表达场景抽样不确定性，不表达 seed 方差。

| 比较（TFAS−基线） | 总体 Pd 差 | 总体 joint 差 | C2/C3 joint 差（主端点） | C2/C3 range RMSE ratio | C2/C3 Doppler RMSE ratio | C3 H0 Pfa 差 |
|---|---:|---:|---:|---:|---:|---:|
| vs Medium | `−0.17261` `[-0.18433,-0.16088]` | `−0.40527` `[-0.42042,-0.39037]` | `−0.60449` `[-0.62598,-0.58350]` | `2.385` `[2.085,2.814]` | `4.599` `[4.224,5.032]` | `−0.03711` `[-0.05396,-0.02148]` |
| vs Small | `−0.15991` `[-0.17163,-0.14819]` | `−0.33350` `[-0.34863,-0.31836]` | `−0.47363` `[-0.49707,-0.45067]` | `2.002` `[1.796,2.269]` | `3.277` `[3.051,3.534]` | `−0.05225` `[-0.06860,-0.03711]` |

TFAS 相对两个 CNN 的 C3 Pfa 差值虽为负，但候选自身 `0.15186` 和两个基线的 `0.18896/0.20410` 都未通过绝对 `0.10` 限制。相对差值保护通过不能覆盖绝对门。

对 Medium，G3 主增益要求至少 `+0.02` 且 CI 下界大于0；实际差值为强负。overall Pd、overall joint、C2/C3 joint CI、range/Doppler RMSE ratio 以及 TFAS 自身 C3 Pfa 保护均失败。TFAS 不保留为候选；不训练额外 seed，不做 attention 调参、KD、Large 或 final blind。

## 6. 物理一致性诊断

公共相位旋转复用 E003 选出的 256 H0 + 256 H1 scenes，32 个等间隔旋转只作描述。TFAS 在零旋转时为基线；跨旋转 H0 FPR 的样本率 min/median/max 为 `0.02734/0.05859/0.07813`。H1 detection agreement min/median 为 `0.9141/0.9258`；score-change p95 最大 `2.572`，wrapped Doppler output-change p95 最大 `122.18 Hz`，range output-change p95 最大 `14.98 cells`。公共相位不是输入真值变化，应保留检测/参数输出；观察到敏感性，符合实值 attention 没有严格群不变保证的结构限制，但仍不能排除模型学习/优化因素。

整体 IQ Doppler 频移也只作描述性诊断。每个已用场景族 64 个 paired scenes、6 种预设频移。Separated 族估计平均 shift 与真值平均差的绝对值约 `29.51 Hz`，最差每场景 shift-error p95 `362.56 Hz`；near-clutter-edge 对应均值约 `5.39 Hz`，但每场景 p95 最大 `93.85 Hz`；partial-overlap 均值约 `11.82 Hz`、平均检测一致率 `0.695`、range-change p95 最大 `15.02 cells`。重生成目标几何干预的频移误差亦高度依场景变化。TFAS 没有呈现可靠的频移等变行为，尤其在重叠场景。

这些诊断复用了 development/verification IDs，不能称为盲测，也不证明破坏物理一致性是 G3 失败的唯一原因。

## 7. 研究假设、局限与阶段判定

- **H-ARCH-003：** 对本轮 TFAS 候选为 **NOT SUPPORTED**：在匹配预算下明显劣于 Small/Medium，主端点区间完全为负，且 C3 绝对 Pfa 失败。不能外推否定所有非 CNN。
- **H-LEARN-002：** 仍 **INCONCLUSIVE**。TFAS 优于真值常值 baseline，显示存在一定任务映射；但公共相位/频移诊断不稳定，教师知识吸收没有实验。
- **H-CAP-002：** R5-A 的单 seed width 收益保持“有限支持”历史状态；R5-B 不重新测试或覆盖它。
- 教师硬标签、软响应、知识蒸馏、truth-assisted 对照、正式多 seed、新独立 blind 和生产数据均未测试。R5-B 未重新运行 B3、T1 或传统基线，不能声称超过传统方法或验证教师资格。
- 总纲 v1.3 不需修改。本阶段对长期研究目标的证据只适用于已复用的 D2-A 静止杂波仿真与开发材料。

**阶段建议：REFINE；TFAS 分支按冻结 G3 停止。** 下一项唯一优先工作是将本报告及其证据提交独立 ChatGPT 阶段审查。在审查前不启动 R5-C/D、KD 或新的学生结构；审查时应判断是否还值得进行独立 teacher-quality audit，而不是从这一负结果自动扩成另一轮架构搜索。

## 8. 可复核材料

- 预注册：`protocol.md`、`config.yaml`、`architecture_survey.md`
- 实现：`src/radardistill/students/temporal_frequency_attention.py`、`scripts/run_d2a_r5_attention.py`
- 测试：`tests/test_temporal_frequency_attention_student.py`、`tests/test_d2a_r5_attention_runner.py`
- G0/G1：`preflight.json`、`g1_overfit_sanity.json`
- 训练：`training.json`、`runs/seed_101/TFAS/training_history.csv`、`runs/seed_101/TFAS/best_checkpoint.pt`
- 指标与逐场景：`metrics.json`、`result.csv`、`roc.csv`、`scene_predictions.csv`
- 物理诊断/图：`physical_diagnostics.json`、`figures/rmse_by_snr.png`
- 运行入口（AISAR 环境）：

```bash
PYTHONPATH=src /home/shy/anaconda3/envs/AISAR/bin/python scripts/run_d2a_r5_attention.py preflight
PYTHONPATH=src /home/shy/anaconda3/envs/AISAR/bin/python scripts/run_d2a_r5_attention.py g1
PYTHONPATH=src /home/shy/anaconda3/envs/AISAR/bin/python scripts/run_d2a_r5_attention.py train
PYTHONPATH=src /home/shy/anaconda3/envs/AISAR/bin/python scripts/run_d2a_r5_attention.py evaluate
```
