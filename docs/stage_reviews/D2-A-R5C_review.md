# D2-A-R5-C 教师知识质量审计阶段自审

**阶段建议：CONTINUE（仅进入一轮匹配的 T0–T3 教学内容 pilot）**。教师软知识 G4 通过，但这既不是固定阈值检测器资格，也不是学生蒸馏收益证据。

## 审查基准与总纲对齐

- **数据/教师运行 SHA：** `58e422de7a9de1290d38ebbb206862e23b49ab42`；该提交在运行前已推送到 `main`。
- **研究总纲：** `docs/RESEARCH_CHARTER.md` v1.3；正文未修改。
- **实验：** `experiments/D2-A-R5/R5-C/` protocol/config v1.1。
- **运行范围：** 新合成场景上的 IQ-only 固定 B3 DPCA 软响应审计；无学生训练、KD、固定阈值 detector 认证或 final blind。
- **核心映射：** 服务总纲任务级学生 GMTI 目标与 Q1/Q2/Q4/Q6/Q9/Q10；直接检验 H-TEACH-002 的教师知识质量前件和 H-TEACH-004 的可观测置信信号前件。R5-C 未检验学生是否吸收这些信息。

R5-B TFAS 的 G3 停止结论保持不变。R5-C 是原完整 R5 任务中独立预授权的教师质量阶段，不重训 TFAS，也不把 R5-B 的失败门改写为通过。

## 1. 工程正确性

G0 通过。运行环境为 AISAR、Python 3.13.7、PyTorch 2.8.0+cu128 / CUDA 12.8、NVIDIA RTX 3050 Laptop GPU `cuda:0`。运行前可用显存约 3.15 GB / 3.94 GB。preflight 核对 `HEAD == origin/main`、12 个条件格、272 个 batch、4,352 个配对场景；已知 `.npy`、CSV 与 JSON seed/scene manifests 中没有 seed 冲突。每批 ID 都与冻结 seed 精确匹配，分析器又验证三个候选各自都恰好覆盖 4,352 个 scene ID。

教师计算由 CUDA complex128/FP64 执行；无 CPU 教师回退。CPU 只逐批仿真场景。运行完成 K1 原生 130×32、K3 固定 4× 520×32 和 K2-local 三种候选，共 13,056 行。未缓存全量 IQ 或 teacher map。CPU 场景仿真 41.83 秒；K1/K3 CUDA 记录分别 1.38/1.58 秒；CUDA 峰值 allocated/reserved 为 23.4/44.0 MB；进程峰值 RSS 859 MB。项目占用约 14.054 GiB，低于 20 GiB 最终限制。这些耗时是本次离线批处理记录，不是实时性结论。

数值/物理正确性由冻结 NumPy K2 parity、有限/非负/归一化/确定性、全局复尺度不变性、CPU-device 拒绝与 off-bin tone 检查覆盖。R5-C、CUDA K2、E002 teacher 与传统基线相关测试合计 **34 passed**；py_compile 和 `git diff --check` 通过。第一次运行曾在任何场景生成前因未先设置 CUDA current device 而退出；补上显式 `set_device(cuda:0)`、加入回归测试并推送后重新 preflight 和完整运行，未把失败尝试当作实验数据。

## 2. 实验科学性与结果

协议保持三通道、PRF=1300 Hz、N=130、32 距离单元和静止杂波。4,352 个配对 base scenes 使用独立于已知旧数据的新 seed blocks；H0/H1 共用同一背景与噪声，真值只进入离线评价。教师输入只包含 IQ 和冻结物理配置。分层区间按 base_scene_id 做 2,000 次 bootstrap。该数据现在已用于候选选择，属于新合成开发审计，不是 blind。

### 汇总结果

| 候选 | 可辨识 C2+C3 lift (95% CI) | C2+C3 top localization (95% CI) | C2+C3 all-H1 Doppler RMSE | C3 top localization (95% CI) | G4 |
|---|---:|---:|---:|---:|---|
| K1 原生全 RD | 0.05162 [0.04943, 0.05383] | 0.7810 [0.7680, 0.7938] | 76.02 Hz | 0.7272 [0.7116, 0.7428] | 通过 |
| K3 4× 全 RD | 0.05130 [0.04910, 0.05350] | 0.8122 [0.8003, 0.8247] | 69.62 Hz | 0.7656 [0.7497, 0.7809] | 通过 |
| K2-local | 0.40122 [0.39182, 0.41082]* | 0.8122 [0.8000, 0.8242] | 69.62 Hz | 0.7656 [0.7500, 0.7803] | 通过 |

\* K2-local 的每个场景响应只在 IQ-only 教师自行选择的距离门上归一化，其 lift 是条件局部频谱质量，**不能与按全图归一化的 K1/K3 lift 直接比较**。K2-local 的峰位置继承 K3 全图最大值，因此不构成独立峰值估计器。

### K3 相对 K1 的配对差

- 可辨识 C2+C3 localization：`+0.03125`，95% CI `[+0.02474,+0.03776]`。
- C3 localization：`+0.03841`，95% CI `[+0.03027,+0.04655]`。
- C2+C3 target-window lift 差：`−0.000320`，95% CI `[−0.000421,−0.000219]`；C3 为 `−0.000250` `[−0.000333,−0.000167]`。细网格没有提高全图归一化软质量。
- all-H1 Doppler RMSE 差：`−6.404 Hz`，95% CI `[−10.893,−2.169]`。冻结的 2.5 Hz 最小收益规则未由该 RMSE 区间满足；K3 的 meaningful-gain gate 是由定位率提升满足。

all-H1 RMSE 把漏定位和大误差计入；正确定位子集的条件 RMSE 另列于 `metrics.json`，但不同候选使用不同成功子集，不能代替主指标。4× 零填充提高峰值定位，不等于它普遍提高学生可吸收的软知识。

### C3 子组与错误区域

| C3 条件 | K1 localization / confidence AUC | K3 localization / confidence AUC | K2-local localization / confidence AUC |
|---|---:|---:|---:|
| separated, 0 dB | 0.990 / 0.377 | 0.996 / 0.319 | 0.996 / 0.996 |
| separated, 10 dB | 1.000 / 不定义¹ | 1.000 / 不定义¹ | 1.000 / 不定义¹ |
| near-clutter-edge, 0 dB | 0.594 / 0.272 | 0.709 / 0.269 | 0.709 / 0.983 |
| near-clutter-edge, 10 dB | 1.000 / 不定义¹ | 1.000 / 不定义¹ | 1.000 / 不定义¹ |
| partial-overlap, 0 dB | **0.021 / 0.404** | **0.041 / 0.326** | **0.041 / 0.783** |
| partial-overlap, 10 dB | 0.758 / 0.266 | 0.848 / 0.288 | 0.848 / 0.993 |

¹ 所有场景均定位正确，ROC AUC 没有正负两类，因此不定义。

三种表达均满足预注册的 C2+C3 正 lift G4 门，且按冻结的联合 severe-subgroup 条件（lift CI 上界≤0 且 localization≤0.05）没有子组触发拒绝规则。**这不表示每个 C3 条件可靠：** partial-overlap / 0 dB 三者定位率仅 2.1%–4.1%。K1/K3 在若干 C3、0 dB 组的固定置信质量 AUC 低于 0.5；K2-local 的 AUC 较高，但它不是校准置信度，协议没有拟合或选择阈值，也没有证明加权后学生会更好。该硬子组必须在后续学生 pilot 中保留并单独报告。

## 3. 假设状态与研究主线

- **H-TEACH-002：** 教师知识质量前件获支持：K1、K3、K2-local 在 C2+C3 有正的配对 target-window lift；K3 相比 K1 的 localization 达到冻结最小有意义收益。**学生收益仍未测试**，因此假设整体维持 PROPOSED，不能说局部知识已改善学生或优于 truth-assisted control。
- **H-TEACH-004：** 可观测置信分数在部分 C3 条件与 localization 有区分信号，尤其 K2-local partial-overlap / 0 dB AUC=0.783；K1/K3 在该组 AUC 分别 0.404/0.326。可靠性权重的作用、校准和部署可观测性仍未测试，假设维持 PROPOSED。
- **H-LEARN-002：** INCONCLUSIVE。教师输出含有任务相关信息不证明学生吸收了相干或连续频率知识；本阶段没有 student intervention。
- **H-ARCH-003：** R5-B TFAS 的限定 NOT_SUPPORTED 状态不变；本阶段不提供新的学生架构证据。

阶段推动总体目标的方式，是将“冻结传统算法有响应”进一步落实为可分层、可配对检验的教师软知识候选；仍未跨过最关键的教师→学生最终任务收益链路。没有检测 Pd/Pfa 资格、真实回波、盲测或完整链效率证据。创新性目前只在“不同相干知识形式如何转化为学生任务收益”的问题层面，R5-C 本身不是蒸馏成功贡献。

## 4. 决定与下一步

**阶段结论：CONTINUE，范围限定为一次新预注册的 T0–T3 教学内容 pilot。** 只做一轮 matched 对照：T0 truth-only、T1 truth+完整 K3 RD 软响应、T2 truth+教师自选距离门的 K2-local 响应、T3 truth+匹配的真值辅助窗口。必须使用同一 Medium CNN 主干、初始化、场景、更新预算和任务损失；软响应只作为训练监督，真值保持纠错来源。T3 必须明示为 truth-assisted，不得包装成教师知识。

当前 Medium CNN 的 C3 H0 Pfa=`0.18896`，尚未满足 `0.10` 绝对保护；它只能作为有明确保护门的教学机制 pilot 学生，不能当作已合格 detector。下阶段协议应采用独立新 H0/H1 开发/验证 seeds、保留该 C3 绝对 Pfa 与联合任务保护，不将 R5-C 质量审计场景重标成 blind。若 teacher arms 未相对匹配 truth-only 显示任务收益，或以明显任务/虚警退化换取单项误差改善，则停止后续时机、可靠性和机制扩展，不跑 final blind。

**唯一下一项优先任务：** 冻结并执行上述四臂 matched teacher-content pilot；该审计没有授权自动蒸馏训练，需由独立协议固定实现细节、样本与停止条件后再运行。R5-B ChatGPT 独立审查仍待提交/反馈；本轮 R5-C 不改写其候选 STOP。

## 复核材料

- Protocol/config：`experiments/D2-A-R5/R5-C/{protocol.md,config.yaml}`。
- CUDA 教师及 runner：`src/radardistill/teachers/d2a_k2_cuda.py`、`scripts/run_d2a_r5c_teacher_audit.py`。
- 数据与结果：`preflight.json`、`environment.json`、`selected_scene_ids.csv`、`teacher_scene_metrics.csv`、`run_resources.json`、`metrics.json`、`analysis.md`、`figures/teacher_quality_c3.png`。
- 本阶段代码回归：R5-C/CUDA K2/E002 teacher/D2-A baselines 定向套件 **34 passed**。
