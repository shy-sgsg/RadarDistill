# D2-A-R6-A：同场景检测与参数估计基线分析

**预测运行 SHA：** `e3972ae94430abd3d79eae7e1ee9179f68062648`
**指标分析 SHA：** `f91da92c455af072accb38cac529e48bce684b3b`
**绘图修正 SHA：** `3c2ab7919fd8a12d973468472b78af710856f95b`
**G0：** `PASS_WITH_RECORDED_PROTOCOL_ORDER_DEVIATION`
**数据：** 12,000 H0 calibration、12,000 H0 verification、6,144 H1 development；final blind 未生成。
**模型训练：** 无；冻结 Medium checkpoint 仅用于推理。

## 研究问题与实验边界

R6-A 在三通道、PRF=1300 Hz、N=130、32距离单元、静止杂波仿真中，把 B0–B3、R3 T1/T2 和 Medium width=96 放到相同 IQ 上比较。阈值只从各方法自己的 H0 calibration 估计，H0 verification 用于独立虚警评价；H1 development 用于检测、参数与联合任务评价。12 个注册条件每格分别包含 1,000 个 H0 calibration、1,000 个 H0 verification 和 512 个 H1。数据按 scene/seed block 与 R5-C、R6-B 保留 block 无重叠。每种方法输出一 CPI 级分数和候选；真值只用于事后评价。

R6-A 是开发基线，不做模型选择后的最终确认。RMSE、Pd 和 joint success 均按全 H1 报告；漏检计为 joint 失败。另分开列出实际候选、正确候选及离线 oracle 候选结果，避免把正确门条件精度当作实际端到端表现。Doppler 误差按 PRF 周期回绕。

## 固定全局阈值下的结果

下表使用每种方法在 H0 calibration 上独立校准的全局阈值。Pfa 区间为单侧 exact Clopper–Pearson 95% 上界；joint 区间为配对 scene bootstrap 95% CI。

| 方法 | H1 AUC | Pfa 0.05 点的验证 Pfa（95%上界） | Pd | Joint success（95% CI） | 对应实际估计器的全H1 Doppler RMSE |
|---|---:|---:|---:|---:|---:|
| B0 | 0.5307 | 0.05117（0.05460） | 0.0527 | 0.0021（0.0000–0.0005） | 263.70 Hz |
| B1 | 0.5308 | 0.05108（0.05451） | 0.0526 | 0.0021（0.0000–0.0005） | 263.46 Hz |
| B2 | 0.5308 | 0.05108（0.05451） | 0.0526 | 0.0021（0.0000–0.0005） | 263.46 Hz |
| B3 | 0.8656 | 0.05233（0.05580） | 0.3594 | 0.3345（0.3229–0.3460） | 71.51 Hz |
| R3 T1 | 0.9528 | 0.05075（0.05417） | 0.8034 | 0.7806（0.7700–0.7910） | 71.51 Hz |
| R3 T2 | 0.9350 | 0.05058（0.05400） | 0.7830 | 0.7586（0.7477–0.7689） | 87.36 Hz |
| Medium | 0.9830 | 0.04667（0.04996） | 0.9334 | 0.8306（0.8213–0.8402） | 24.68 Hz，网络直接输出 |

Medium 的全局 0.05 Pfa 上界刚低于 0.05；这不能替代条件级保护。R3 T1/T2 相对 B3 的 joint 差分别为 `+0.44613`（95% CI `[+0.43326,+0.45898]`）和 `+0.42415`（`[+0.41113,+0.43653]`）。相对 Medium，R3 T1/T2 分别低 `0.04997`（`[-0.05957,-0.04004]`）和 `0.07194`（`[-0.08154,-0.06234]`）。这些是同一 H1 scene 上的配对差。

在 0.01 工作点，所有方法的验证 Pfa 单侧 exact 95% 上界仍高于 0.01（范围约 0.01046–0.01136），所以这些样本不支持声称已满足 1% 虚警率。该阶段也没有资格声明更低虚警工作点。

## C3 条件迁移与困难格

Medium 在 C3 六格的验证 Pfa（阈值仍为全局 0.05）为 `0.072/0.101/0.085/0.109/0.097/0.094`；条件 AUC 分别为 `0.9998/1.0000/0.9934/1.0000/0.5956/0.9966`，Pd 为 `1.000/1.000/0.984/1.000/0.221/0.996`。除 partial-overlap、SNR=0 dB、CNR=20 dB 的 condition 10 外，Medium 在这些开发格表现较强；condition 10 的 joint success 只有 `0.0391`，全H1 Doppler RMSE为 `71.57 Hz`。

| 方法 | condition 10 AUC | 验证 Pfa | Pd | Joint success |
|---|---:|---:|---:|---:|
| B3 | 0.4858 | 0.215 | 0.1777 | 0.0039 |
| R3 T1 | 0.4913 | 0.109 | 0.1074 | 0 |
| R3 T2 | 0.5097 | 0.097 | 0.1152 | 0.0020 |
| Medium | 0.5956 | 0.097 | 0.2207 | 0.0391 |

condition 10 的 Medium 全局阈值 H0 calibration Pfa 为 0.100；R6-B 选支脚本只使用 R6-A H0 calibration 与 H1 development，未读取 H0 verification。C3 pooled AUC 为 `0.93164`（95% CI `[0.92440,0.93871]`），困难 condition 10 AUC 为 `0.60324`（`[0.57085,0.63295]`）。C3 六个校准格 Pfa 范围为 `0.033`，未满足预注册 scale-shift 分支的范围条件；困难格 AUC 上界低于 0.70，故按预注册规则选择 **R：matched truth-only 与 training-H0 hard-negative-resampled Medium 对照**。这说明仅靠分数尺度归一化不足以解释或解决当前最差格的可分性问题；没有据此推断硬负例训练必然有效。

## 参数估计：实际候选与离线 oracle

在 Medium 自己预测的 range gate 上，传统 Doppler 精估计仍受错误候选影响：FFT peak、FFT 插值、4×零填充峰、lag-one 相位差和局部 profile-ML 的 all-H1 RMSE 分别为 `149.72/149.70/142.63/145.32/149.67 Hz`；Medium 直接 Doppler 输出为 `24.68 Hz`。在正确候选条件下，Medium 直接 Doppler RMSE 为 `17.24 Hz`。因此本批样本中，网络直接输出优于在它自己的预测门上运行这些传统估计器；这不是纯估计器在正确输入上的比较。

离线 nearest-bin oracle 候选下，Medium IQ 对应的 FFT peak、插值、4×零填充、lag-one 和 profile-ML RMSE 分别为 `68.33/68.31/60.20/79.34/68.27 Hz`。oracle 门利用真值最近可用整数 range center，只用于诊断，不能用于运行时，也不是网络输出的理论下界。正确候选及 oracle 条件结果都不能替代 all-H1 实际任务指标。

## 实现审计、修正和资源

G0 记录了一个次序偏差：正式 G0 前两个注册 seed 各生成过一例，只检查形状与 ID，没有观察 IQ/标签值、方法输出或保留数据；后续工程测试改用独立 seed。G1 数据划分、同IQ方法 fan-out、阈值隔离、PRF 回绕、实际/离线候选分离和 scene bootstrap 均通过。所有 30,144 个场景共写出 211,008 行预测；运行期间不保存全量 IQ。运行环境为 AISAR、Python 3.13.7、PyTorch 2.8.0+cu128、CUDA 12.8、RTX 3050 Laptop `cuda:0`；Medium inference 峰值 allocated 为 129,681,408 bytes。H0 calibration / verification / H1 处理耗时约 `350.44/353.37/216.71 s`。没有训练或修改 batch。

结果审查发现两项离线问题并已留审计记录：

1. Oracle range gate 曾用 `floor`，与模拟器 `sinc(truth_range - range_axis)` 的最近中心语义不符。已按 half-up 最近门规则对同一 6,144 个 H1 scene ID 重建 IQ 并重算 oracle sidecar；只改离线 oracle 参数诊断，不影响任何方法分数、实际候选、阈值、标签或非oracle指标。原始指标和图无损压缩保存在 `attempts/oracle_floor_mapping_correction/pre_correction/`。
2. 初版 joint-success 图曾重复显示全局值，指标 JSON 中的分条件数值本身正确。现已按 condition 读取数据、修复标签，并人工核看生成图；旧图留在 pre-correction 审计目录。

修正后 R6-A 定向测试与整仓测试通过；指标计算用 `f91da92`，绘图更正用 `3c2ab79`。对应完整 CSV、manifests、环境记录、指标和图保存在本目录。`R6-A chunk_size=16` 是流式评价批次，不是训练batch。

## 结论与边界

1. 在全局开发混合分布上，Medium 是本次受测方案中检测、joint success 和连续 Doppler 误差最好的候选，但这一结论限于单个冻结 checkpoint、注册合成开发集和 0.05 工作点。
2. C3 partial-overlap / 0 dB 格的低 AUC 与低 joint success 表明困难不只是给固定分数换个尺度；这正是 R6-B 选中 hard-negative resampling 分支的理由。
3. R3 T1/T2 在 C2/C3 pooled 任务上优于 B3，但在 condition 10 的 AUC/Pd/joint 仍弱，也没有资格覆盖 Medium 的整体结果。
4. 该阶段不证明跨seed泛化、blind 性能、生产级 CFAR、1%或更低虚警、CRLB最优性或整链时延。数据不支持立即进入 final blind 或教师蒸馏确认。
5. 总体研究方向维持 **REFINE**：保持物理真值任务学生主线，下一项只执行预注册的 R6-B hard-negative 对照；其训练有效 batch=64 保持冻结，以维护同初始化、更新数和样本计划的公平比较。后续新协议可将 batch 作为匹配因素探索。

## 复现入口

```bash
conda run --no-capture-output -n AISAR python scripts/run_d2a_r6a.py preflight
conda run --no-capture-output -n AISAR python scripts/run_d2a_r6a.py summarize
conda run --no-capture-output -n AISAR python scripts/select_d2a_r6b_branch.py
```
