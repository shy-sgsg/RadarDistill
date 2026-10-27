# D2-A-R5-C 教师知识质量审计协议

**协议版本：** 1.1
**状态：** preregistered；数据运行前必须提交并推送协议、配置、代码。
**研究总纲：** `docs/RESEARCH_CHARTER.md` v1.3；本实验不修改总纲。
**来源：** 用户授权的 D2-A-R5 完整任务 Phase 4；任务对齐见 `docs/task_alignments/D2-A-R5C_alignment.md`。

## 变更记录

- **v1.1（2026-10-28，数据生成前）：** 冻结 K3 相对 K1 的最小有意义增益：可辨识 C2+C3 localization 提升至少 0.02 且配对 95% CI 下界大于 0；或 circular Doppler RMSE 至少改善 2.5 Hz 且配对 95% CI 下界仍不少于 2.5 Hz。其他设计与数据预算不变。该阈值把“有明确增益”从文字说明变成可证伪规则。

## 1. 问题和范围

本阶段独立回答：冻结 B3 DPCA 的 IQ-only 知识响应能否在新场景中定位可辨识目标；固定 4× Doppler 零填充/教师自身距离门局部化是否提供比原生全 RD 响应更精细的频率知识；响应的可观测尖锐度是否与实际教师正确率相关。

保持 D2-A-E001 三通道、PRF=1300 Hz、N=130、32 个距离单元、静止杂波、噪声/目标参数定义、目标和杂波几何不变。仅使用非盲合成开发诊断。当前阶段**不训练学生、不蒸馏、不评价固定阈值检测器、不重训教师、不搜索窗口/温度/阈值、不生成 final blind**。K1/K3 归一化响应称为相对功率支持，不称为概率后验或校准似然。

## 2. 数据设计

每个场景由冻结 `generate_d2a_paired_batch` 同时产生共享 H0/H1 背景和噪声。每个条件格的 base_scene_id 唯一；H0/H1 按同一 ID 配对。场景按 16 CPI 一批生成，立即转交 CUDA 教师，评价后丢弃 IQ 与响应图；不生成或缓存全量 IQ。

| Course | Scenario | SNR (dB) | CNR (dB) | 配对 scenes | 用途 |
|---|---|---:|---:|---:|---|
| C0 | separated | 20 | -5 | 256 | 易条件参照 |
| C1 | separated | 15 | 0 | 128 | 中间条件参照 |
| C1 | near_clutter_edge | 15 | 0 | 128 | 中间条件参照 |
| C2 | separated | 10 | 10 | 256 | 可辨识中等条件 |
| C2 | near_clutter_edge | 10 | 10 | 256 | 可辨识中等条件 |
| C2 | partial_overlap | 10 | 10 | 256 | 可辨识中等条件 |
| C3 | separated | 0 / 10 | 20 | 各 512 | 困难主条件 |
| C3 | near_clutter_edge | 0 / 10 | 20 | 各 512 | 困难主条件 |
| C3 | partial_overlap | 0 / 10 | 20 | 各 512 | 困难主条件 |

总量为 4,352 个独立配对 base scenes（H0/H1 各 4,352 CPI）。条件顺序固定为表格顺序；每个条件使用从 726,000,000 开始、按条件索引增加 100,000 的不重叠 seed block。每 16 CPI 子批使用 `seed_base + chunk_index × 1009`。preflight 必须扫描仓库内已知 seed manifests 和数据缓存元信息，确认无 ID/seed block 重叠；任何冲突均停止，不临时换 seed。

这些数据用于知识候选筛选，不是盲测，也不用于阈值校准。最终 blind 未授权且不生成。

## 3. 教师知识形式

三个输出都来自同一冻结 `dpca_difference` 公式：有效相位中心延迟对通道 1 做慢时间分数移位，再计算 `channel0 − shifted(channel1)`。

- **K1 完整 RD：** 原生 130 点 centered FFT，功率图按每个 CPI 全图和归一化，形状 `[130,32]`。
- **K3 连续频率响应：** 同一 DPCA 残差做固定 4× 零填充 centered FFT，Doppler 网格间隔 `1300/(130×4)=2.5 Hz`，全图功率按 CPI 归一化，形状 `[520,32]`。它是过采样功率响应，不是最大似然后验。
- **K2-local 局部频率响应：** 先由 IQ-only K3 全图最大值确定教师 range cell，再取该 range cell 上的 520 点 Doppler 功率并归一化；不使用真值窗口或 metadata。若该选择遗漏目标，该错误计入教师质量。

教师计算使用显式 `cuda:0` 和 complex128 FFT；没有 CPU teacher fallback。输入 API 只接受 IQ、冻结 config、设备和固定过采样倍率。

## 4. 冻结指标与统计

主要信息量指标是在真值 range ±1 cell、Doppler 圆周误差 ±20 Hz 窗口中的响应质量。对同一 scene 计算 `H1_mass − paired_H0_mass`。K1/K3 使用对应原生/过采样 Doppler 轴；K2-local 报告其 H1/H0 局部频率质量、教师 range 选择正确率及 joint localization。Doppler 误差按 PRF 周期回绕；range/Doppler RMSE 按全 H1 和正确定位子集分别报告，漏定位计入 joint failure。

固定可观测 confidence score 为 K1/K3 图在教师自身最大值附近 ±1 range cell、±20 Hz 的响应质量；K2-local 使用局部 Doppler 响应在教师自身峰值 ±20 Hz 内的质量。按预注册条件报告 confidence 分层命中率、秩相关和 ROC AUC，仅用于诊断 reliability 是否可观测；不拟合 calibration、阈值、可靠性模型或训练 gate。

所有配对区间以 base_scene_id 为 bootstrap cluster，每条件 2,000 replicate；`bootstrap_seed=2026102817`，不同预注册条件用 `seed + condition_index`。同时报告分层点估计和区间，不将多条件样本合并解释为同分布总体。主要对比为 K3−K1 的 teacher-localization rate 和 circular Doppler RMSE；不按结果事后更换 primary endpoint。

## 5. 正确性门与 G4 资格规则

### G0 工程/数据门

- protocol/config/runner/teacher module 已提交并推送，运行前 `HEAD == origin/main`。
- AISAR PyTorch CUDA 可见 RTX 3050 Laptop GPU `cuda:0`；运行过程中不允许 CPU fallback。
- 固定 seed blocks 与现有 tracked/untracked manifest 无重叠；配对 IDs、条件数和每批大小完全吻合配置。
- 输出保存 4,352 scenes × 3 knowledge forms，逐条件准确计数；无 IQ/checkpoint/full map cache；项目最终总占用低于 20 GiB。

### 数值/物理检查

- CUDA 原生 K1 相对冻结 NumPy `k2_response` 的最大绝对差 `≤1e-6`；映射有限、非负、每 CPI 和为 1，固定输入重复计算确定。
- 每个固定响应图在复尺度 `[0.5, 2.0, 0.6+1.2j]` 下的最大绝对差 `≤1e-6`。
- analytic off-bin tone 的 K3 峰频误差 `≤2.5 Hz`，没有 Doppler 符号/fftshift 错误。
- 教师输入不含 presence、truth、target location、SNR/CNR、scene/course metadata；这些信息只能用于结果评价。

### G4 知识资格

每种形式独立评级。**适合提出一次受限 truth+teacher 学生 pilot** 的最低条件：数值与隔离门全部通过；在可辨识 C2+C3 合并的配对 target-window lift 95% CI 下界 `>0`；C3 所有子组逐一报告。若任何 C3 子组同时满足 lift CI 上界 `≤0` 且 top-cell localization `≤0.05`，该知识不得作为无条件全域监督。此时只有当固定可观测 confidence 对该子组正确定位的 AUC `≥0.65`，才可提出另行预注册的可靠性分层 pilot；本协议本身不实施该 gate。

全域 G4 通过不等于固定阈值 detector 资格或学生收益成立。只有 K3 相对 K1 的可辨识 C2+C3 localization 提升至少 0.02 且配对 95% CI 下界大于 0，或其 circular Doppler RMSE 改善至少 2.5 Hz 且 paired-bootstrap CI 下界至少 2.5 Hz，才称 4× 表达有明确知识质量增益；否则保留更简单的合格形式，不继续搜索倍率/温度。

## 6. 中止条件

CUDA 不可用/错误设备、协议或数据未推送、seed/ID 重叠、批大小/计数错误、非有限响应、parity/尺度不变/解析音调测试失败、输入泄漏、OOM，均在产生该批后续数据前停止。不得通过静默 CPU 回退、改精度/容限、改 seeds、删失败子组或调整指标恢复执行。若所有候选未达到 G4，结束教师知识质量分支的本轮工作，不运行 Phase 5。

## 7. 资源、运行与归档

- 设备：RTX 3050 Laptop GPU 4 GiB，显式 `cuda:0`，FP64/complex128；一次只将一个 batch=16 及其 FFT 暂存驻留 GPU。
- 场景仿真在 CPU 分批生成；教师运算和 GPU 指标用 CUDA；记录 CPU 生成时间与 CUDA teacher 时间，二者不合并成实时性能主张。
- 不保留原始 IQ、teacher full maps 或训练 checkpoints。只归档 protocol/config、代码与测试、preflight/environment、逐场景紧凑指标、汇总 metrics、图、分析、阶段审查和可复现命令。
- 运行入口：`PYTHONPATH=src /home/shy/anaconda3/envs/AISAR/bin/python scripts/run_d2a_r5c_teacher_audit.py preflight|run|analyze`。
