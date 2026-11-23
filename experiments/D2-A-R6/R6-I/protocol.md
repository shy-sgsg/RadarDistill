# D2-A-R6-I：可观测背景尺度下的 C3 虚警稳健性

**协议版本：** 1.0（执行前冻结）
**状态：** preregistered；在冻结提交上 G0/G1 通过后执行一次。
**研究总纲：** `docs/RESEARCH_CHARTER.md` v1.3。
**路线位置：** R6-P1 低虚警诊断的有界补充；用于判断 R6-H 候选是否能经可观测分数校正满足 P7 准入。
**最终盲测：** 未授权生成。

## 1. 研究问题

R6-H 中 CNN-SSM 对 CE-TCN 的 C2+C3 joint-success 提升达到筛选主端点，但未通过 Range RMSE 保护，三臂 C3 六格 familywise Pfa 门均失败。R6-I 检验：从每 CPI 复 IQ 可计算的稳健背景功率，能否解释 CNN-SSM H0 高分尾部的尺度变化，并在新场景上降低 C3 虚警，同时保护 Pd 与 joint success。

本实验只改变**检测分数**，不改模型权重、输入、参数估计头、仿真或检测阈值校准规则。它不重新训练学生，也不测试教师或知识蒸馏。

## 2. 固定假设与候选

**H-R6I-001：** CNN-SSM H0 高分尾部与可观测 CPI 背景功率存在跨场景尺度关系；用仅由 H0 拟合的一条稳健右尾斜率校正 presence logit，可使 C3 六格固定阈值 Pfa 上界达到 0.10，同时不使 Pd 或 C2+C3 joint-success 下降超过 0.02。

只设置一个校正候选：

1. 每个输入 CPI 计算
   \[
   p_i=\operatorname{median}_{c,n,r}|X_i[c,n,r]|^2,\qquad x_i=\ln(p_i).
   \]
   中位数对只占少量 IQ 样本的目标响应较稳健，且公共幅度缩放 `aX` 会令 `x` 增加 `2 ln|a|`。
2. 在 `h0_scale_fit` 的 CNN-SSM 原始 presence logits 与 `x` 上，把 `x` 按等样本数分成 20 桶；每桶至少50个场景，每桶计算 `higher` 经验95% logit分位数，并以桶内 `x` 中位数为自变量，对这20个点拟合普通最小二乘斜率 `beta`。
3. `center` 是 `h0_scale_fit` 的 `x` 中位数。校正分数为
   \[
   s_i^{corr}=s_i^{raw}-\beta(x_i-center).
   \]
4. `beta` 和 `center` 在阈值校准及 H1 评价前冻结。若 H0 拟合分母为零、桶样本不足或出现非有限值，G1/评价失败并停止，不另换公式。

候选和基线：

| 系统 | 检测分数 | 参数估计 | 作用 |
|---|---|---|---|
| B3 | 原始 DPCA 最大响应功率 | B3 同一最大峰 | 强传统基线 |
| CE-TCN | R6-H seed808 固定 checkpoint 的原始 presence logit | CE-TCN 输出 | 任务基线 |
| CNN-SSM raw | R6-H seed808 固定 checkpoint 的原始 presence logit | CNN-SSM 输出 | 校正前配对对照 |
| CNN-SSM corrected | 上述唯一 H0 背景尺度校正 logit | 与 CNN-SSM raw 完全相同的距离/Doppler 输出 | 唯一新候选 |

R6-H 检查点不重训、不改动。真值、SNR、CNR、条件索引及仿真潜变量仅用于离线分层评价，不能进入估计 `p_i`、拟合 `beta` 或推理打分。

## 3. 信号、数据划分和场景 ID

沿用三通道静止杂波、PRF=1300 Hz、130 脉冲、32 距离单元及 R6-A 冻结的 12 个条件格。各 split 每格等量，生成时只流式保留一个 16 CPI CPU batch；不缓存 IQ。

| Split | 类型 | 场景数 | 每格数量 | 唯一用途 |
|---|---|---:|---:|---|
| `h0_scale_fit` | H0 | 60,000 | 5,000 | 只拟合 CNN-SSM `beta` 与 `center` |
| `h0_threshold_calibration` | H0 | 60,000 | 5,000 | 拟合全局阈值；不再更新 `beta` |
| `h0_verification` | H0 | 120,000 | 10,000 | 独立验证总体与逐格 Pfa |
| `h1_development` | H1 | 6,144 | 512 | 独立开发 Pd、参数误差、joint 与配对差 |
| `g1_test` | H0/H1 smoke check | 16 | — | 只检查接口/数值，不纳入任何指标 |

固定 split seed bases 分别为 3,200,000,000、3,220,000,000、3,240,000,000、3,260,000,000；G1 seed 为 3,290,000,000。条件 seed stride 为 1,000,000，chunk size 为 16，chunk seed stride 为 1009。执行前必须扫描所有历史 CSV manifests，并重建 R6-D/E/F/G/H 注册计划；任意重复即 G0 失败，不得看过结果后换 seed。

四个评价 split 合计 246,144 CPI，其中 H0 240,000、H1 6,144。H0 calibration-fit、H0 threshold calibration、H0 verification、H1 development 与 G1 scene IDs 两两不重叠。12 格条件只用于生成等配额场景和离线评价。

## 4. 阈值与评价规则

- 每个系统/分数分别仅从 `h0_threshold_calibration` 全 12 格的等配额混合分布，按 `higher` empirical quantile 设置全局目标 Pfa=0.05、0.01、0.001 的阈值；决策规则固定为 `score >= threshold`，并报告严格大于、平局数量及 `>=` Pfa。
- 唯一主工作点为 0.05。0.01 与 0.001 只作次要描述，不据其挑选或再次校正模型；本实验不验证 1e-4 或更低工作点。
- 独立 `h0_verification` 报告 overall、每格及 C3 pooled 的 CPI 级 false-alarm count、Pfa 与单侧 exact Clopper–Pearson 95% 上界。C3 六格各用 `alpha=0.05/6` 的 Bonferroni familywise 上界；六格必须全部 `<=0.10` 才算通过。
- 独立 `h1_development` 报告 overall/C2+C3/C3 与 12 条件的 Pd、AUC、Range/Doppler RMSE、bias、correct-candidate 参数误差和 joint success。Joint 定义沿用 R6-H：检测通过、距离误差 `<=1 cell` 且 PRF-wrapped Doppler 误差 `<=20 Hz`。
- 唯一确认性配对比较为 CNN-SSM corrected − CNN-SSM raw，按 base scene ID、在条件内配对 bootstrap 5,000 次、双侧 95% CI。要求：C3 pooled Pfa 至少下降 0.01，且差值 CI 上界 `<0`；overall Pd 和 C2+C3 joint 差值 CI 下界均 `>=−0.02`；全 H1 Range/Doppler RMSE ratio 上界均 `<=1.05`。另报告阈值变化造成的 detected-only 参数误差与每格联合任务，不能以未检出样本删除 all-H1 joint 分母。
- P7 重新准入必须同时满足：校正 CNN-SSM 的 overall H0 单侧 exact 上界 `<=0.055`、C3 六格 familywise 上界全 `<=0.10`，以及上述校正收益与任务保护门。即便通过，也只允许另行冻结 P7 多种子方案，不自动生成 final blind。

## 5. 公共尺度干预

在 `h1_development` 的 C3 partial-overlap / SNR=0 dB / CNR=20 dB 配对 IQ 上，对同一 H0 与 H1 CPI 分别乘实数 `a=0.5`、`a=2.0`，比较各模型原始与校正分数、固定阈值决策一致率以及模型参数输出变化。该干预不进入拟合、校准或主端点。报告 `p_i(aX)/p_i(X)` 与 `|a|^2` 的数值偏差；不把它解释为真实硬件 AGC 模型。

## 6. 执行门、资源与停止条件

**G0：** 协议、配置、代码与定向测试已提交并推送；冻结 SHA 与协议/config/source/checkpoint 哈希记录； tracked worktree clean；场景 ID 与历史 manifests/注册计划零重叠；AISAR 中 `torch.cuda.is_available()` 为真且 GPU 名称为 RTX 3050 Laptop；`cuda:0` 可用显存至少 1.5 GiB；根磁盘可用空间至少 4 GiB；两个 R6-H checkpoint owner 与完整性有效。

**G1：** 不训练模型。检查四 split 计划数量、12 格等配额、场景 ID 两两不重叠；从G1独立种子生成8个H0与8个H1场景，检查IQ和小批量输出值有限、shape正确；两个固定checkpoint在显式CUDA上输出所有预期张量且logits/参数有限；`p_i(aX)/p_i(X)`与`|a|^2`的相对误差不超过`1e-5`；校正函数只接受logit、log-power与已冻结系数。

学生 inference 固定 FP32、AISAR `cuda:0`，模型 CE-TCN 与 CNN-SSM 顺序驻留；默认 microbatch 4，OOM 时降到 2 或 1，评价场景和总体指标不变。B3 在 CPU 上使用完全相同 IQ。数据按 CPU chunk 即时生成，不写大型 IQ 缓存；只保留 split manifests、压缩预测、指标、日志与分析。当前不做实时性基准。

若 G0/G1 不通过，停止正式评价并保留故障证据。完成一次注册评价后，不根据开发集继续改 `beta`、分桶数、背景统计或分数公式。校正候选未过全部 P7 准入门时，保留负结果并停止本分数校正；不开展 final blind。

## 7. 交付

`protocol.md`、`config.yaml`、`preflight.json`、`g1.json`、四 split `scene_manifest_*.csv`、压缩逐场景预测、`metrics.json`、`analysis.md`、`docs/stage_reviews/D2-A-R6I_review.md`、任务对齐/假设/路线状态，以及必要的失败或恢复审计。最终工作区须小于 20 GiB；不改写 R6-H 历史结果或研究总纲。
