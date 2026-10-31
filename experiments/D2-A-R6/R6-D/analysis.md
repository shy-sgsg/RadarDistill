# D2-A-R6-D：联合检测与连续参数估计 pilot 分析

**阶段结论：REFINE。** J2-J 在单 seed development 的 C2+C3 joint success 和距离估计上优于 J1-J，但 Doppler 估计明显恶化，且 C3 六格 familywise 虚警保护未通过；不晋级正式多 seed 或 final blind。

**研究总纲：** `docs/RESEARCH_CHARTER.md` v1.3。
**冻结协议：** `experiments/D2-A-R6/R6-D/protocol.md` v1.2；SHA-256 `c5b30ff2f395617558484ad48821b9239da09278922331c326745f84dc8dddb5`。
**配置：** `experiments/D2-A-R6/R6-D/config.yaml`；SHA-256 `e01cb65244338fc3957b39b0db233c022a9befa467f130bda37ecc667abe3ee9`。
**训练与评价代码提交：** `9a631f4fae445d9fd3793faeba94cdcbc09cc062`。
**结果边界：** 单训练 seed 505；所有 H1 均为 development；未生成 final blind；未训练教师或蒸馏模型。

## 1. 总纲目标与问题

R6-D 对应总体目标中的任务级学生学习、检测/参数联合建模和困难场景泛化（主要关联 Q1、Q3、Q4、Q6、Q10）。问题不是 J2 是否降低一个辅助 loss，而是软候选交互能否在相同物理真值监督和训练暴露下提高检测、距离、Doppler 和 joint success，并同时保护独立 H0 虚警。

J1 使用同一 width-96 TaskStudent 主干：D-only、P-only、联合 J1-J。J2-J 在相同主干上加入可学习 130→520 时间投影、null 加候选后验、候选连续残差及一轮参数到检测反馈。没有固定 FFT 前端；J2 的增加参数和交互均属于学习部分，不能把收益解释成物理频谱前端贡献。所有臂仅用 E1 物理真值监督，未用教师硬标签或软响应。

## 2. 运行与正确性

- G0/G1 均通过。注册设备为 RTX 3050 Laptop GPU `cuda:0`，PyTorch `2.8.0+cu128`、CUDA 12.8、Python 3.13.7、NumPy 2.3.4；全程 FP32、单模型顺序训练，无 CPU 回退、无 OOM。
- 四臂均使用 seed 505、有效 batch 64、microbatch 16、51,200 个配对训练场景、25,600 次 optimizer updates、相同 H0/H1 exposure、AdamW/cosine schedule、验证与 checkpoint 规则。每个 presence 类别访问训练场景约 16 次，即每类 819,200 CPI 暴露。
- G1 的 16-pair/200-update 过拟合门均通过：D-only 主 loss 下降 53.6%；P-only、J1-J、J2-J 分别下降 99.84%、99.77%、99.94%。G1 是实现路径检查，不是泛化能力证据。
- R6-D evaluation 使用独立 H0 calibration 120,000、H0 verification 120,000、H1 development 6,144 场景。R6-D ID 与此前登记场景无重合。训练/验证 cache 的生成器 SHA 为 `76815bb2c2068b3c43643340e640fc008dfebff4`；所有训练记录保留该生成器来源，runner/训练/evaluation SHA 为 `9a631f4...`。本阶段没有重用旧开发 IQ 作为新盲测。

## 3. 训练预算与曲线

| Arm | 参数 | 最佳验证步 | 最佳验证目标 | 25,600 步末验证目标 | 训练用时 | 峰值 CUDA allocated |
|---|---:|---:|---:|---:|---:|---:|
| J1-D | 70,277 | 20,400 | 0.05695 | 0.05707 | 6,758 s | 0.821 GB |
| J1-P | 70,277 | 14,400 | 0.17620 | 0.19052 | 6,390 s | 0.821 GB |
| J1-J | 70,277 | 14,400 | 0.22635 | 0.23747 | 6,216 s | 0.821 GB |
| J2-J | 143,294 | 22,800 | 1.09603 | 1.09859 | 9,492 s | 1.336 GB |

J2-J 最佳验证目标高于 J1-J 的数值尺度，因损失组成不同，不能跨架构直接比较绝对 loss。相对各臂自己的 12,800 步验证值，J1-D 从 0.07023 降至 25,600 步的 0.05707，J2-J 从 1.24713 降至最佳 1.09603；这直接说明本任务中 12,800 不是普遍训练上限。J1-P 与 J1-J 则在 14,400 步达到最佳，之后验证目标回升、训练目标仍更低，呈现训练/验证差距，继续单臂增加 updates 不会公平，也没有本轮证据支持。

因此，用户提出的后续延长训练授权应落实为**新实验中预注册所有比较臂共同预算**，按相关验证曲线选择是否长训；不回溯改写本轮 25,600-step 冻结协议，也不把更长训练自动等同于更好泛化。训练 history 按每 400 步记录，没有逐 update 写日志。

训练曲线见 [`figures/training_curves.png`](figures/training_curves.png)。

## 4. 独立 H0 评价

阈值由各检测臂自己的 H0 calibration 120,000 CPI 的 `higher` 经验 95% 分位数确定，校准经验 `Pfa_GE=0.05`；判决为 `score >= threshold`。verification 不参与选阈值。P-only 没有检测头，不定义阈值、Pfa 或 Pd。

| Arm | 阈值 | H0 verification Pfa（单侧 exact 95% 上界） | C3 pooled Pfa（上界） | C3 六格 familywise ≤0.10 |
|---|---:|---:|---:|---|
| J1-D | 0.19817 | 0.05128（0.05233） | 0.10193（0.10399） | FAIL |
| J1-J | 0.22208 | 0.04969（0.05074） | 0.09865（0.10068） | FAIL |
| J2-J | 0.16325 | 0.04892（0.04995） | 0.09578（0.09778） | FAIL |

J2-J 的 pooled C3 Pfa 上界低于 0.10，但预注册要求六格 Bonferroni familywise exact 上界**每格**均不超过 0.10；该门失败。J2-J 六格 familywise 上界分别为 0.10079、0.10534、0.09862、0.10915、0.09997、0.10441，仅两格通过。不能用 pooled 结果掩盖条件虚警。独立 verification 逐条件 Pfa 与区间保存在 `metrics.json`。

## 5. H1 development 与主对比

| Arm | Overall Pd | AUC vs H0 verification | Overall joint | C2+C3 Pd / joint | C3 Pd / joint | All-H1 range RMSE | All-H1 Doppler RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| J1-D | 0.9401 | 0.9844 | 不适用 | 仅检测；见分条件记录 | 仅检测；见分条件记录 | 不适用 | 不适用 |
| J1-P | 不适用 | 不适用 | 不适用 | 不适用 | 不适用 | 2.801 cells | 20.53 Hz |
| J1-J | 0.9380 | 0.9839 | 0.8670 | 0.9173 / 0.8227 | 0.8760 / 0.7484 | 2.811 cells | 20.67 Hz |
| J2-J | 0.9393 | 0.9836 | 0.9004 | 0.9191 / 0.8674 | 0.8789 / 0.8066 | 2.520 cells | 27.01 Hz |

预注册主对比为 4,608 个可辨识 C2+C3 scene 上 J2-J−J1-J joint-success：

- 差值 `+0.04470`，配对 scene bootstrap 95% CI `[+0.03602,+0.05360]`；超过 +0.02 且区间下界大于 0，主增益筛选通过。
- Overall Pd 差 `+0.00130`，95% CI `[-0.00212,+0.00456]`，通过 `−0.02` 非劣保护。
- All-H1 range RMSE ratio `0.8966`，95% CI `[0.8248,0.9720]`，通过上界 1.05。
- All-H1 Doppler RMSE ratio `1.3066`，95% CI `[1.2015,1.4148]`，未通过上界 1.05；误差上升约 30.7%。

joint 增益主要伴随距离改善，没有伴随检测概率提升；同时 Doppler 质量显著退化。J2-J 不是在所有任务量上都更优的学生，也不能据 joint 单项结果晋级。

### 困难 C3 子组

在 C3 partial-overlap、SNR=0 dB、CNR=20 dB（condition 10），J1-J/J2-J 的 Pd 为 0.2715/0.2871，joint success 为 0.1035/0.1445；all-H1 range RMSE 为 9.655/8.706 cells，而 Doppler RMSE 为 61.81/76.75 Hz。该格中 J2-J 的 joint 和距离更好、Doppler 更差。J2-J 的 Doppler 退化也存在于预测候选正确子集：C2+C3 detected-correct-candidate Doppler RMSE 为 18.55 Hz，J1-J 为 13.32 Hz；因此不能仅将问题归咎于错选距离候选。该分层是机制提示，不是已验证因果解释。

逐条件 H0、Pd、joint、range/Doppler RMSE 见 `metrics.json` 与 [`figures/development_metrics.png`](figures/development_metrics.png)。

## 6. 阶段门与研究解释

| 门/问题 | 结果 | 解释 |
|---|---|---|
| G0：代码、scene ID、设备、资源 | PASS | split 独立；RTX 3050 `cuda:0`、FP32 可用 |
| G1：数值/梯度/资源/小样本学习 | PASS | 四臂 finite gradients、CUDA micro-overfit 和显存限制通过 |
| G2：四臂 matched pilot 与独立评价 | 完成 | 4/4 臂同 seed、同预算和数据计划；输出仅为 development |
| J2-J C2+C3 joint 主筛选 | PASS | `+0.04470`，95% CI 全高于 0 |
| Overall Pd 与 range 保护 | PASS | paired CI 满足冻结限值 |
| Doppler RMSE 保护 | FAIL | ratio 1.3066，95% CI 完全高于 1.05 |
| C3 六格 familywise Pfa | FAIL | pooled Pfa 合格不足以通过逐格门 |

**对研究问题的回答：**

1. **谁最能学频率/参数？** J1-P 的 all-H1 Doppler RMSE 最低（20.53 Hz），但它没有检测能力；检测与参数联合臂中 J1-J 的 Doppler RMSE（20.67 Hz）优于 J2-J（27.01 Hz）。J2-J 的 joint success 和距离 RMSE 最好。没有一个臂在全部任务上占优。
2. **共享任务有无收益？** J1-P 与 J1-J 的参数 RMSE接近，J1-J 提供可校准检测与 joint 输出；本 pilot 未给出 J1-P/J1-J 的预注册配对区间，差别只作描述，不宣称因果优越。
3. **候选交互贡献？** 相对 J1-J，J2-J 的配对 C2+C3 joint 和 range 有增益，但 Doppler 明显退化。J2 比 J1 参数约多一倍，wall time 也较长；该单 seed 结果不能把收益归结为“纯物理归纳偏置”，因为它没有固定物理频谱层，130→520 投影是可学习的。
4. **传统方法比较？** R6-D 没有重跑匹配 FFT、FFT 插值、相位差或 ML 基线，故不能据本阶段宣称网络超过这些方法。R6-A 传统基线属于先前独立 development 实验，不等价于本阶段同条件对照。
5. **蒸馏是否有增益？** 未测试。R6-D 没有教师或蒸馏损失；J2 的监督全部来自物理真值。R5-C teaching pilot 的负结果保持原样。
6. **相位/物理一致性启示？** 本阶段没有运行公共相位旋转或频率平移干预；不能声称模型具备复数相位不变性或频率等变性。J2 的距离收益与 Doppler 退化说明候选定位和连续频率读出需要分开诊断。

## 7. 限制、决策与下一步

结果只覆盖三通道、PRF=1300 Hz、N=130、32 距离单元的冻结静止杂波仿真和一个训练 seed。开发集不能称为 blind；没有 seed 方差、实测数据、系统级时延、CRLB 对比或正式多目标证据。H1 的参数 RMSE 是全场景估计结果，不应与检测命中条件 RMSE 混用；后者也完整记录在 `metrics.json`。

**决策为 REFINE。** 保留 J1-J 作为后续可辨识任务的联合 baseline；J2-J 作为有 joint/range 信号但 Doppler 保护失败的研究候选，不进入 formal seeds/blind。下一唯一优先任务按 R6 P4 路线冻结 R6-E 的有界架构比较：以 J1-J 为匹配 baseline，基于有界 matched pilot 比较少量有理论依据的复数/序列结构；显式把 Doppler、C3 familywise Pfa 和配对 joint 设为保护。训练预算在 R6-E protocol 中统一预注册，允许根据本轮曲线设为高于 25,600 的共同预算，但不追溯延长 R6-D，也不单独续训 J2。

研究总纲、R6-D 冻结协议、R6-A/B/C 结果和所有旧负结果均未修改。R6 P0–P7 尚未全部完成，因此本报告是内部阶段材料，不是整项 ChatGPT 审查请求。

## 8. 数据清理与项目占用

报告、指标、预测归档和检查点完成后，按 R6-D owner manifest 删除仅由本阶段生成的 Train/Validation 可重建缓存，共 `11,233,315,805` bytes。保留协议、配置、scene manifests、训练history、best/last checkpoint、训练记录、预测归档、指标、图表与审查材料。清理后项目树分配空间测得 `1,264,234,496` bytes（`du -x -sB1 .`，包含 `.git` 及项目根目录内本地文件），低于 20 GiB 限制。`cleanup.json` 中的 `workspace_used_bytes_after_cleanup` 是整个挂载盘用量，不应用作项目大小；项目树测量单独记录在 `project_tree_used_bytes_after_cleanup`。
