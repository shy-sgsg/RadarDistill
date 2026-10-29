# D2-A-R6-A 阶段自审

**审查基准：** 预测运行 `e3972ae94430abd3d79eae7e1ee9179f68062648`；指标分析 `f91da92c455af072accb38cac529e48bce684b3b`；分条件图修正 `3c2ab7919fd8a12d973468472b78af710856f95b`。
**研究总纲：** `docs/RESEARCH_CHARTER.md` v1.3，保持未修改。
**阶段材料：** [`R6-A protocol`](../../experiments/D2-A-R6/R6-A/protocol.md)、[`config`](../../experiments/D2-A-R6/R6-A/config.yaml)、[`analysis`](../../experiments/D2-A-R6/R6-A/analysis.md)、[`metrics`](../../experiments/D2-A-R6/R6-A/metrics.json)。

## 1. 工程正确性

R6-A 完成 30,144 个注册场景、211,008 条方法预测，H0 calibration、H0 verification 和 H1 development 分割无重叠；同一 split 的所有方法使用同 IQ。阈值只从 calibration 计算，真值不进入预测调用，Doppler 误差按 PRF 周期回绕。CUDA Medium 使用冻结 width=96 checkpoint（训练参数82,887、推理参数70,277），未重训；RTX 3050 Laptop `cuda:0` 峰值分配129,681,408 bytes。没有 final blind。

G0 通过但必须保留协议顺序偏差：两例注册seed场景在G0之前由接口测试生成，测试仅查看数组形状和 ID，没有查看 IQ/标签数值、调用方法、保留数据或用于阈值/模型选择。后续测试已经改用独立工程seed。

汇总中发现的 nearest-bin oracle 映射错误只影响离线 oracle 参数估计，已按模拟器 `sinc` 响应中心重建同一批 H1 IDs、增加单独 sidecar，并保留未压缩含义不变的原始审计材料。图表复核又发现 joint-success 图的分条件数据取值错误；数据指标未错，现已修复图、添加回归检查并核看图像。两项修正及旧结果均有审计记录，不覆盖原始负结果或方法预测。

## 2. 实验科学性

在全局 Pfa 0.05 的混合开发分布上，Medium AUC 0.9830、Pd 0.9334、joint success 0.8306、直接 Doppler RMSE 24.68 Hz，是本次受测方案中总体最好者。其全局验证 Pfa 单侧 exact 95% 上界为0.04996，刚好低于0.05；此结果不能证明条件级稳健。

关键弱点位于 C3 / partial-overlap / SNR 0 dB / CNR 20 dB：Medium AUC 0.5956、Pfa 0.097、Pd 0.2207、joint success 0.0391。条件 AUC 表明该格的可分性弱，故单纯整体阈值或分数尺度变化不是充分解释。R6-B 的预注册分支选择只读取 R6-A H0 calibration 和 H1 development：C3 pooled AUC `0.93164`（95% CI `[0.92440,0.93871]`），困难格 AUC `0.60324`（`[0.57085,0.63295]`），其95%上界低于0.70停止门；按协议选择 R 分支，运行 matched truth-only 与训练集 H0 hard-negative-resampled Medium 对照。R6-A H0 verification 未用于选择。

R3 T1/T2 明显优于 B3 的 joint success，但在困难格的检测和联合任务仍差，且都低于 Medium 的总体开发表现。0.01 工作点所有方法的 Pfa 上界均高于0.01，不支持1%虚警率主张。正确候选和 oracle 条件下的参数误差是诊断量，不能冒充实际端到端精度；在 Medium 自身预测 range gate 上传统频率估计器 RMSE 约143–150 Hz，Medium 直接输出为24.68 Hz，但这一配对边界应与传统方法各自实际检测候选分开解释。

## 3. 研究主线

本阶段推进总纲 Q1、Q3、Q6、Q8、Q10：它把全局平均上的高学生性能与 C3 困难格失效分离，并显示“候选是否找对”和“候选内精估”会产生显著不同的误差读数。当前证据支持在真值监督学生基础上继续研究困难 H0 采样，而不支持转向更复杂教师知识内容或立即声明 CNN 已足够可靠。该判断仅针对当前 Medium checkpoint 与注册静止杂波开发分布。

R6-A 不改变研究使命、教师路线的历史 STOP、R5 结论或任何冻结协议；总纲 v1.3 保持不变。R6-B 是信息价值最高的最小后续对照，因为它已预注册且本次条件ROC选支结果满足 hard-negative 分支门。按用户对 batch 的最新约束，R6-B 有效 batch 64 必须冻结，以保证同初始化、optimizer updates 和场景顺序的匹配；未来独立协议可在所有比较臂等条件地探索更大 batch。

## 4. 总判断与后续

**阶段判断：REFINE。** R6-A 基线与评价流程完成；Medium 在总体开发分布具备有价值的学习证据，但困难条件辨识与条件虚警仍限制其任务可靠性。它不是最终资格确认。

**唯一下一项优先任务：** 执行已预注册的 R6-B R 分支：新场景 H0 calibration/verification、H1 development；对照重训 truth-only Medium 与训练 H0 hard-negative-resampled Medium。使用相同模型初始化、数据暴露、12,800 updates、有效 batch 64 和验证/阈值规则；硬负例选择只使用训练集。若 Pfa 改善伴随冻结的 Pd、参数或 joint 保护退化，则停止该干预，不扩展到多seed或 final blind。

## 5. 总纲映射

- **总体目标：** 利用物理真值和可靠传统方法训练直接完成 GMTI 检测/估计的学生，并最终按任务质量评价学习与混合系统。
- **科学问题：** Q1 学生收益，Q3 表示学习，Q6 困难目标，Q8 物理/学习分工，Q10 预算和可辨识结构。
- **假设状态：** H-R6-DET-001 按当前表述“困难条件内有足够可分性、主要由阈值/尺度迁移解释”在困难格不获支持；R6-B 将测试一个有界训练采样干预。H-CAP-002 仅保留 R5-A 限定证据，不由本单一冻结 checkpoint 的 R6-A 推断容量上限。
- **结论边界：** 仿真开发证据、单训练seed checkpoint、未做独立盲测；不声明实测泛化、生产 CFAR、系统时延或1e-6级虚警控制。
