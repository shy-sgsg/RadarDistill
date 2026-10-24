# 研究进度

## 2026-10-24：批准并启动任务级教师蒸馏新主线

### 审查依据

- 读取了本轮完整研究路线任务文件、当前 P0/P1 阶段材料及未提交的 APX15 算法包。旧 P0-E007 GO、P1-A-E001 STOP/REDIRECT 均保持历史状态；新路线 D0/D1 不再以前一阶段的相位回归为前置。
- 当前机器 `AISAR` 环境为 PyTorch 2.8.0+cu128，CUDA 可见，设备为 NVIDIA GeForce RTX 3050 Laptop GPU；本轮基线测试为 **67 passed**。
- `GMTI_APX15_CoreX_Int16_20261014` 本地约 4.3 GiB，包含六份约 753 MB 正式 BIN。包级 CMake 面向 CoreX，本机没有 CoreX SDK；代码适配先限于 D0 需要的慢时间频谱子链，不声称整条生产链已接入。

### 已完成

- 已记录新研究目标与 D0→D5 路线，保留历史 P0/P1 阶段结论。
- 已编写 `docs/research/task_level_progressive_distillation.md`、`docs/research/D0_D1_design.md` 和 `docs/superpowers/specs/2026-10-24-d0-d1-task-level-distillation-design.md`。
- 已冻结实现计划 `docs/superpowers/plans/2026-10-24-d0-d1-task-level-distillation.md`；D0 和 D1 数值协议均已在各自代码/训练前冻结。
- D0-E001 仿真、APX15 慢时间 FFT 教师及 12 项正式检查全部通过；阶段审查允许进入 D1。结果与范围限制见 `docs/stage_reviews/D0_review.md` 和 `experiments/D0-E001/analysis.md`。
- D1-E001 已冻结轻量 TCN、A/B/C/D 1600-update 匹配预算、三个正式 seed、50,000 条独立 H0 校准样本、40,960 条最终盲测样本及 G1/G2/G3 门槛；配置见 `experiments/D1-E001/`。

### 后续结果

- D1 学生/损失、CUDA pilot 与正式训练、独立 H0 校准、最终盲测和阶段审查均已完成。G1、G2 通过，G3 未通过；按预注册规则 D1 为 GO → D2。结果、指标和限制见本文件末尾的 D1-E001 盲测记录。

## 2026-10-24：D0-E001 数值协议预注册

- 冻结 `experiments/D0-E001/config.yaml` 和 `protocol.md`：PRF=1300 Hz、N=130、3 通道、固定 32 距离单元；复 AWGN、连续距离/Doppler、随机幅度/初相及 APX15 仿真三接收通道子集几何。
- 对齐 APX15 参数后，距离轴明确使用理论带宽分辨单元 `c/(2B)=2.99792458 m`；APX 配置的 5 m 分辨率字段与 DDC 间距 `c/(2Fs)=2.498270483 m` 单独记录，D0 不声称复现完整生产距离处理。
- 冻结 191 点离散 LFM—匹配滤波核对、原生 130-bin 软频谱、520-bin 补零硬估计、噪声/频轴/无噪参数门、各正确性样本量及独立 seeds。教师接口只接收 IQ；CoreX 正式链不在 D0 适配范围。
- 数值门限依据和一组小规模、非持久化设计 sanity check 已在协议中披露，明确排除在正式 D0 指标之外。正式检查见下方运行结果；预注册门槛未按结果更改。

## 2026-10-24：D0 仿真与 APX15 FFT 教师实现

- 实现 `src/radardistill/simulation/d0.py`：固定监视窗复 IQ、连续 range/Doppler、三通道 APX15 仿真相位中心子集、H0/H1 共先验复 AWGN，并分字段输出真值、信号项和噪声项。
- 实现 APX15 `ColFFTTranspose` 的 FFTW3 C++ 适配及 Python 教师：130-bin 软响应、520-bin 连续 Doppler 硬估计、峰值/中位数检测分数；`predict` 只接收 IQ，不读取真值或已知噪声功率。
- 加入流式 D0 正确性运行器 `scripts/run_d0_e001.py`，覆盖冻结样本、边界、频轴、AUC 和 bootstrap 汇总，不保存可再生完整数组。
- 新增 12 个定向测试；`conda run --no-capture-output -n AISAR pytest -q` 为 **79 passed**，`compileall` 与 `git diff --check` 通过。
- 正式 D0 使用配置中冻结的独立种子分批生成数据，记录 12/12 门通过；完整指标、CSV、环境与解释见 `experiments/D0-E001/`。全量测试 **79 passed**，`compileall` 与 `git diff --check` 通过。

## 2026-10-24：D0-E001 正式正确性检查与阶段审查

- 复现命令：`conda run --no-capture-output -n AISAR python scripts/run_d0_e001.py --config experiments/D0-E001/config.yaml`。运行约 39.1 s，CPU 峰值 RSS 约 585 MB；D0 FFTW 教师运行于 CPU，未进行训练。
- 12/12 冻结门通过。191 点 LFM 匹配滤波 PSF 最大幅度误差为 `1.2662e-4`；1,000,000 个复噪声样本功率相对误差 `0.1187%`；256 对 H0/H1 具有逐元素一致的背景噪声。
- 4096 个无噪 H1 的距离 RMSE 为 `7.17e-10` cell，Doppler RMSE 为 `0.02904 Hz`、最大误差 `0.04052 Hz`；130 个原生 FFT 栅格音调全部落在正确频轴。
- 每个 SNR 的 H0/H1 教师诊断样本数为 1024/1024；-10 dB 教师检测 AUC `0.98856`（bootstrap 95% CI `[0.98414, 0.99261]`），达到预注册 `AUC≥0.98` 可用性门。-20/-15 dB AUC 分别为 `0.48675/0.64430`，因此教师在这些低 SNR 区间没有可靠检测判别力，必须保留并在 D1 中报告教师错误。
- 阶段决定：**D0 GO → D1**，仅表示冻结仿真与教师子链通过正确性门，可开展 D1 学生可行性实验；不代表学生已学会任务、蒸馏有效、完整 APX15/CoreX 链已复现或系统虚警率已验证。详细审查见 `docs/stage_reviews/D0_review.md`。

## 2026-10-24：D1-E001 数值协议预注册

- 在实现和 pilot 之前冻结 `experiments/D1-E001/config.yaml` 与 `protocol.md`：TCN 感受野 255 覆盖 130 个脉冲；A/B/C/D 使用相同数据、初始化、更新次数和有效 batch；教师只提供全图软频谱辅助损失，真值单独监督最终任务。
- 固定训练 SNR `[-20,20] dB` 均匀抽样，每个训练 seed 4096 H0+4096 H1；三个正式训练 seed；每策略 1600 个 optimizer updates，batch 64，CUDA FP32 顺序执行。先以独立 pilot seed 检查四组训练及 GPU 资源。
- 固定 50,000 H0 阈值校准样本及每 SNR 4096 H0/1024 H1 的新盲测；工作点为 `Pfa=1e-2/1e-3`。已冻结参数指标、层级配对 bootstrap、传统 FFT/教师/相位差/profile-ML 基线以及 G1/G2/G3 和阶段决策门。
- 数据缓存仅忽略 `experiments/D1-E001/data/cache/*.npy` 和 `*.npz`；模型、汇总报告、配置和训练记录仍可单独审阅。协议冻结时未生成 pilot 或盲测数据。

## 2026-10-24：D1 学生、任务损失与 CUDA pilot 训练器实现

- 实现 `TaskStudent`：32 通道深度可分离 TCN、255 脉冲理论感受野、presence/range/周期 Doppler 头及训练期全图软频谱头；导出推理模型时移除辅助头。
- 实现 H0 屏蔽的距离/Doppler 真值损失、周期 Doppler 编码与解码、全图 teacher-response KL、CUDA fail-fast、A/B/C/D 固定更新计划、CPU memmap 批量搬运及 64/32/16 microbatch OOM 回退。梯度累积分别按有效 batch 样本数和 H1 数归一化，与整批损失一致；训练前保存 update 0 初始 checkpoint。教师硬输出与真值分字段存入本地训练缓存，学生输入仅为复 IQ 实/虚平面。
- 定向 D1 测试 **20 passed**；全仓库 `conda run --no-capture-output -n AISAR pytest -q` 为 **100 passed**；`compileall` 和 `git diff --check` 通过。验证缓存测试发现并修正了多 SNR 样本分配长度错误；学生逐 SNR 指标、教师验证基线和按 H1/SNR 分层的物理一致性汇总已接入。首次直接 CLI 启动暴露 CUDA 设备只检查未选择的问题；`require_cuda` 现显式选择 `cuda:0`，已验证其后显存统计 API 正常。正式 runner 现支持按配置顺序分三 seed 运行，但正式训练尚未开始。
- 训练前代码提交 SHA 为 `bc0d4d8939ddd8e6a1c3d3062ec33856c722b9e8`；实际使用 RTX 3050 Laptop `cuda:0`、PyTorch 2.8.0+cu128、CUDA 12.8、驱动 580.178.04、FP32。四组各 1600 更新，训练 133–140 秒/组，峰值分配显存 1046–1049 MiB，无 OOM。首次直接启动发现 runner 未显式选择 CUDA device；根因修复见提交 `651ece3`。

## 2026-10-24：D1-E001 CUDA pilot 筛选

- 复现命令为 `conda run --no-capture-output -n AISAR python scripts/run_d1_e001.py --phase pilot`；数据缓存留在 `experiments/D1-E001/data/cache/` 并被窄范围忽略。pilot 的初始/最佳/最后 checkpoint、训练历史、配置、环境、metrics 和诊断均保存在 `experiments/D1-E001/pilot/`。
- A 组验证任务损失从 5.238 降至 2.118；在 0/5/10/20 dB 的 AUC 均为 1.0，range RMSE 为 0.044–0.111 cell、Doppler RMSE 为 22.2–31.4 Hz。-10 dB 及以下性能明显下降。APX15 教师在高 SNR 频率估计仍强于学生。
- 一 seed 主指标 J：A=0.01203、B=0.01095、C=0.01358、D=0.01076；B/D 的高 SNR 改善仅是 pilot 线索，不能据此断言蒸馏收益。共同相位旋转/频移结果已按 H1 与 SNR 分层；20 dB H1 Doppler 等变残差 p95 为 2.87 Hz（0.625 Hz 输入频移）和 9.34 Hz（5 Hz 输入频移）。
- **筛选决策：进入正式三种子 A/B/C/D 对比。** A 在 20 dB 的 detection/range/Doppler 数值参考达到冻结 G1 数值，且高 SNR 物理一致性较好；低 SNR 失败、蒸馏效果和传统竞争力留待正式独立测试。此筛选不构成 D1 阶段 GO，也不判定 G1/G2/G3。
- 代码复核修正了 formal 启动门：不再信任旧版 `pilot_gate.passed`，而是按冻结配置从四组原始策略摘要重算筛选门；正式 seed 数量锁定为恰好三组。`conda run --no-capture-output -n AISAR pytest -q` 为 **101 passed**；其后 formal 训练按冻结预算完成，见下一节。

## 2026-10-24：D1-E001 三种子 CUDA formal 训练

- 复现命令：`conda run --no-capture-output -n AISAR python scripts/run_d1_e001.py --phase formal`。A/B/C/D × seed_1/seed_2/seed_3 共 12 组，每组严格 1600 optimizer updates；逐 seed 输出初始/最佳/最后 checkpoint、训练历史、验证指标、CUDA 峰值资源和环境。
- 三组四策略均成功完成，未发生 OOM，实际 microbatch 均为 64。RTX 3050 Laptop `cuda:0`、FP32、PyTorch 2.8.0+cu128/CUDA 12.8；单策略平均约 133 秒，峰值 allocated 最多约 1049 MiB、reserved 最多约 1150 MiB。训练环境记录代码 SHA `df25b79f8cb8d18c44b020ebb77d096c6b8ef265`。
- 20 dB 验证集上 A 的三 seed AUC 均为 1.0，range RMSE 为 0.067–0.076 cell，Doppler RMSE 为 25.4–32.1 Hz。此处仅是冻结验证集诊断，不是校准或盲测验收；传统基线差距、蒸馏配对收益和 G1/G2/G3 暂不判定。
- 正式训练汇总、逐 seed `metrics.json`/`analysis.md`、12 组模型权重和训练历史保存在 `experiments/D1-E001/formal/`；独立 H0 校准及新盲测尚未生成。

## 2026-10-22：P0 实施、复核与阶段判断

### 已完成

- 建立研究规范、方案、P0–P6 阶段计划、统一实验协议、文献综述/矩阵、研究空白和决策记录。文献范围区分汽车 FMCW、SAR-GMTI、机载多通道 GMTI 与 STAP。
- 实现 P0 复单音加复 AWGN 仿真、FFT 峰值/插值、相位差、连续 profile-ML、CRLB、圆周误差、局部教师似然软目标和 NumPy CPU Conv1D。
- 主实验 `P0-E001`：3 个训练种子；每个种子 6000 个训练样本和 1200 个验证样本；8 个独立测试 SNR，每点 1024 个样本。物理真值、教师频率、教师软目标和学生预测分别保存。
- `P0-E002` 在相同训练种子和测试集上把训练延长至 160 轮，用于检查优化不足。
- `P0-E003` 使用位置保留 flatten + Dense(32) 读出，`P0-E004` 使用更低容量 flatten + Dense(8)，用于检验均值池化损失时间位置与大读出头过拟合的可能性。两个诊断各只有一个训练种子，因此不用于种子间结论。
- 代码和行为测试覆盖信号/噪声、种子复现、基线、圆周边界、CRLB、教师标签、模型训练和实验产物；本轮完整测试运行结果为 **13 passed**。

### 主要结果

全部结果来自模拟单目标 AWGN，不是实测 GMTI 数据。

| 实验 | 学生/训练 | 20 dB E1 RMSE | 同一测试上的参考 RMSE |
|---|---|---:|---|
| P0-E001 | 6 通道均值池化，40 轮，3 种子 | 15.00 Hz | FFT 峰值 2.96；FFT 插值 1.23；相位差 0.206；连续 ML 0.0332 Hz |
| P0-E002 | 同一结构，160 轮，1 种子 | 8.55 Hz | 基线与测试输入相同；E1 仍高于 FFT 峰值、插值和相位差 |
| P0-E003 | flatten + Dense(32)，160 轮，1 种子 | 26.18 Hz | 训练损失较低，但验证圆周 MSE 0.241，呈训练/验证差距 |
| P0-E004 | flatten + Dense(8)，160 轮，1 种子 | 36.27 Hz | 容量下降未恢复测试精度 |

E001 的 E1 在 7/8 个测试 SNR 点比零频常数预测 RMSE 低至少 20%，支持“存在基本频率学习”这一有限结论。但在 -5 至 20 dB，E1 未超过 FFT 峰值；高 SNR 下与插值、相位差和连续 ML 仍有明显差距。E002 表明延长训练显著降低 5–20 dB 的 E1 误差，但一次种子仍无法满足基线门。E003 训练损失下降而泛化变差；E004 降低容量仍未解决。

E001 的 20 dB 分组 RMSE 进一步显示：频率五分组为 10.09–23.67 Hz；初相四扇区为 13.93–16.43 Hz；幅度四分位为 12.51–17.35 Hz；FFT 栅格偏移四分位为 14.23–16.01 Hz。E002 单种子对应范围分别为 6.88–11.01、7.56–10.02、7.84–9.10 和 8.19–9.19 Hz。组间误差不完全一致，中心频段在 E001 尤其较差；这些诊断没有消除主结论。

### 阶段门判断

1. **实现门：通过。** 13 项项目测试通过；独立数据划分及可审阅产物已生成。
2. **基本学习门：部分支持。** 与常数预测的差异和随 SNR 的变化表明学生确实利用了频率信息；但高误差、频率分组波动及架构诊断结果不支持稳健性能结论。
3. **继续价值门：未通过。** 学生在 -5 至 20 dB 未超过 FFT 峰值，蒸馏没有稳定的高 SNR 质量收益。E001 在部分 SNR 的 E3−E1 bootstrap 区间低于零，但 20 dB 区间跨零；E002–E004 为单种子诊断，不能估计训练种子不确定性。

因此 **P1 尚未开始**。当前证据不支持三通道或 GMTI 检测性能主张。

### 当时安排的下一步（现已执行）

保持 E001 的独立测试集不变，只安排一个预注册的 P0 诊断：使用小容量、位置敏感读出或显式相位增量特征，检查是否能同时避免全局均值池化的信息损失和 flatten-Dense 的过拟合。沿用传统基线及真值/教师三策略，预先指定训练种子、训练轮数和“至少部分正常 SNR 稳定优于 FFT 峰值”的通过条件。若仍不通过，先完成误差来源分析并重新评估纯 CNN 路线，再决定是否启动 P1；不进行无目标的超参数搜索。

### 环境和复现

- Python 3.13.5、NumPy 2.1.3、pytest；运行环境无 PyTorch/TensorFlow/JAX，也没有可用 NVIDIA 驱动，因此训练为 NumPy CPU，不代表 GPU 性能。
- 主实验配置和完整数据/权重/预测位于 `experiments/P0-E001/`；后续诊断位于 `P0-E002` 至 `P0-E004`。
- 复现入口：`python scripts/run_p0.py --config experiments/P0-E001/config.yaml`；代码检查：`pytest -q`。

### GitHub 同步状态

项目文件已在当前工作区整理完成，但尚未写入 `shy-sgsg/RadarDistill`。GitHub Contents API 和 Git Data blob API 均返回 HTTP 403 `Resource not accessible by integration`；远程仓库仍为空。需要为当前 GitHub 连接授予该仓库内容写权限，或配置一个可用的 git 写入连接，再记录远程提交 SHA。

> 状态更正：以上是 2026-10-22 当日的 GitHub 连接记录。用户之后已确认此前提交已推送；截至 2026-10-23，本轮 E006 与审计改动仍在本地工作区，尚未提交或推送。

## 2026-10-24：P0-E007 三范式实验与阶段验收

### 设计与执行

- 在保留 E006 STOP/REDIRECT 历史结论的基础上，按新批准的预注册协议运行 E007。信号模型仍为 PRF=1300 Hz、N=130 单目标复 AWGN；训练/验证和开发集用于筛选，最终盲测另用新 seed。
- M1 是固定、不可学习的 16N 复 FFT + 可学习循环 Conv1d 频谱读出，输入含 log-power 与相邻谱线复互谱相干；另训练仅 log-power 的 M1-power 消融。M2 是固定 4N FFT 粗峰 + 27→64→32→1 子栅格修正 MLP。M3 是直接处理 130 个 IQ token 的两层 d=32、4 头小型 Transformer。
- 三种架构均先以一个 CUDA seed 筛选；M1 通过后正式训练三个种子。M1-power 作为同架构 E1 消融也运行三个种子；只有具备基础学习能力的 M1 进入 E2-H、E2-S、E3-S 蒸馏对比。
- 所有训练使用 AISAR、PyTorch 2.8.0+cu128、CUDA 12.8、RTX 3050 Laptop GPU（4 GiB）、FP32、batch 64，顺序运行；没有将全部数据缓存到 GPU。最终盲测每 SNR 1024 例，共 8192 例，只生成/推理一次。模型、种子、checkpoint、训练历史、教师字段、预测和资源指标均保存在 `experiments/P0-E007/`。

### 筛选结果

| 模型 | 验证损失首轮→最佳 | 20 dB 开发 RMSE | 结论 |
|---|---:|---:|---|
| M1 | 1.688→1.273 | 0.03862 Hz | 通过，进入正式 E1 |
| M1-power | 1.671→1.275 | 0.03567 Hz | 仅作为三种子 E1 消融 |
| M2 | 0.03295→0.03158（下降 4.2%） | 0.09585 Hz | 未达到验证下降门；比同局部 4N 抛物线高 0.06179 Hz，比连续局部 ML 高 0.06270 Hz，停止扩展 |
| M3 | 6.997→6.997 | 371.30 Hz | 验证未改善且五个主 SNR 点均未胜 FFT 峰；训练 loss 下降而验证恶化，停止三种子训练 |

M2 在 20 dB 开发集比 N 点 FFT 插值低 1.1370 Hz，但匹配局部抛物线及连续 ML 明显更好；该混合模型没有达到进入正式训练的标准。M3 只有一个筛选 seed，结果支持该配置泛化失败，不足以否定全部 Transformer 结构。

### 最终盲测结果

- M1 E1 的 20 dB 三种子 RMSE 为 0.03722、0.04369、0.03607 Hz，均值 0.03899±0.00411 Hz。当地 CRLB 标准差为 0.03419 Hz（M1 约 1.14× CRLB）；连续 profile-ML 为 0.03487 Hz（约 1.02×）。
- 在 0、5、10、15、20 dB 五个主 SNR，M1 均以配对 95% bootstrap 区间显著优于 N 点 FFT 峰、N 点 FFT 插值、相位差估计和 16N FFT 峰。20 dB M1−16N FFT 峰为 -0.14751 Hz [-0.15336,-0.14179]。
- M1 未胜过 4N/16N 局部抛物线和连续 profile-ML。20 dB M1−local profile-ML 为 +0.00412 Hz [+0.00120,+0.00816]，不支持替代高精度传统估计器。
- 相干输入未显示收益：M1−M1-power 在 20 dB 为 +0.00222 Hz [-0.00057,+0.00660]，五个主 SNR 区间均未证实 M1 更好；M1-power 的点 RMSE略低。
- E2-H 与 E1 在 20 dB 无显著差异（-0.00205 Hz [-0.00599,+0.00096]）；E2-S 明显更差（+0.07889 Hz [+0.07084,+0.09075]）；E3-S 更差（+0.00884 Hz [+0.00151,+0.01515]）。
- M1 公共相位旋转预测变化最大 0.01221 Hz，复数全局尺度变化最大约 0.01782 Hz；0.625 Hz 频移等变误差均值约 0.00021 Hz、最大约 0.01245 Hz。无噪声全频和 Nyquist 边界集合的 G1 指标三个种子均通过；带噪盲测 Nyquist 邻域每 SNR 约 13 个样本，分组估计不确定性较高。
- 分组结果、SNR 指标和 CRLB 对照在 `result.csv`；种子配对 CI 在 `paired_comparisons.json`；曲线在 `rmse_snr.png`；全部屏幕/正式训练记录在 `training_history.csv`、各阶段 JSON 与 `models/`。

### 阶段门及下一步

- G1 学习可行性：**PASS**，M1 无噪声 MAE/P95、Nyquist 子集、20 dB 相对零频预测和数据隔离条件均为 3/3 种子通过。
- G2 传统竞争力：**PASS**，M1 在五个主 SNR 点均优于 N 点 FFT 峰值，并在五点均优于匹配的 16N FFT 峰值。
- G3 蒸馏价值：**FAIL**，E3-S 没有任何通过点，更无两个相邻且种子一致的收益。
- 按阶段门，P0-E007 决策为 **GO**，仅允许进入 P1-A 方案设计/预注册；三通道实验尚未开始。单通道结果不能证明三通道相位、角度或 GMTI 检测能力。

完整设计、协议、代码、可复现配置、训练和测试产物、图表及阶段分析见 `docs/superpowers/specs/2026-10-23-p0-e007-design.md`、`experiments/P0-E007/` 和 `docs/stage_reviews/P0-E007_review.md`。规范指标入口为 `experiments/P0-E007/metrics.json`。验证命令 `conda run --live-stream -n AISAR pytest -q` 结果为 **58 passed**；`python -m compileall -q src scripts tests` 与 `git diff --check` 通过。重建汇总命令：`conda run --no-capture-output -n AISAR python scripts/run_p0_e007.py --phase summarize`。

## 2026-10-22：P0-E005 执行与误差来源复核

### 实验结果

- 按预注册协议运行三种子、E002 的 160 轮上限。学生输入为 129 点单位相位增量；FFT、插值、相位差和连续 ML 基线继续使用原始 IQ。
- E005 保存的训练、验证和测试 NPZ 与 E001 对应文件逐数组完全相同，因此本次模型对比沿用同一独立数据划分。
- 主门未通过。0、5、10、15、20 dB 的 E1−FFT 峰值 RMSE 差及 95% 配对 bootstrap 区间分别为：

| SNR (dB) | E1 RMSE (Hz) | FFT 峰值 RMSE (Hz) | E1−FFT 差值及 95% 区间 (Hz) |
|---:|---:|---:|---:|
| 0 | 31.66 | 2.86 | 29.31 [22.97, 35.80] |
| 5 | 14.35 | 2.92 | 11.64 [8.14, 13.89] |
| 10 | 11.57 | 2.86 | 9.15 [4.51, 11.70] |
| 15 | 14.22 | 2.86 | 11.48 [9.12, 13.30] |
| 20 | 15.42 | 2.96 | 12.59 [10.25, 14.91] |

所有差值区间均高于零，E1 在五个主 SNR 点均未胜过 FFT 峰值。20 dB 下 E1/E2/E3 RMSE 分别为 15.42/15.77/14.92 Hz；E3−E1 的区间为 [-1.90, 0.37] Hz，蒸馏优势不稳定。相位差基线在 20 dB 为 0.206 Hz，连续 ML 为 0.0332 Hz。

### 误差来源复核

以 20 dB 的同一测试样本为例，E1 三种子的 RMSE 为 13.05、15.26、17.95 Hz，显示有训练种子差异。合并三种子预测后，预测频率与真值相关系数为 0.935，线性斜率为 0.930；预测和真值标准差分别为 373.76 与 376.01 Hz。这说明模型覆盖了频率范围，但精度仍有系统残差。

20 dB 分组 RMSE：初相扇区 15.33–16.16 Hz、幅度四分位 15.15–15.92 Hz、FFT 栅格偏移四分位 15.22–15.86 Hz；频率五分组为 10.77–18.02 Hz，组内平均有符号误差从 -5.65 到 +7.82 Hz。当前证据更指向跨频率区间的校准/逼近误差和训练种子波动；没有显示误差主要由初相、幅度或 FFT 栅格偏移驱动。E1 训练损失与验证真值 MSE 数值接近，也没有明显的训练/验证鸿沟证据。该复核是诊断线索，不能单独证明误差的因果来源。

### 阶段判断与后续

P0-E005 已执行，继续价值门仍未通过，**P1 不启动**。显式相位增量输入没有让小型 CNN 接近 FFT 峰值、插值、相位差或连续 ML 基线。先据此重新评估纯 CNN 路线的研究价值；若仍继续，需要先提出一个针对频率相关残差或种子波动的有界、可证伪方案，再决定是否开展另一项 P0 诊断。E005 代码、协议、指标和完整逐样本产物见 `experiments/P0-E005/`；误差复核见 `experiments/P0-E005/error_source_review.md`。

## 2026-10-23：R0 方法审计与 P0-E006 预注册

### 审计与修复

- E2 早停目标现在显式区分教师验证、真值验证及历史兼容模式；E006 的 E2-A/E2-B 分别走教师验证和真值验证。历史 E001–E005 结果保留为原协议结果，不追溯重算。
- `estimate_mle` 的底层方法准确标注为 FFT 初始化局部 profile-ML，并保留旧名称兼容；新增全局网格 profile-ML 和多延迟相干基线。固定低 SNR 例显示局部搜索会受 FFT 错峰影响。
- 分组逻辑已从固定 N=130 改为信号真实样本数。补充 PRF/N、Nyquist、空批次/错误形状、仿真物理变换和多参数有限差分梯度检查。
- NumPy CNN 将损失/梯度计算公开为可检查接口；卷积权重/偏置、隐藏层、输出层权重/偏置均与有限差分比较，梯度测试包括联合蒸馏损失。
- E006 的 G3 现在同时要求配对区间在两个相邻主 SNR 低于零，并要求每个点至少 2/3 个训练种子的 RMSE 差方向为负。
- E001–E005 共有 79 个实验文件、109,764,138 字节。逐字节 hash 审计发现 7 组重复（test、共享 train/validation 文件），理论可回收 76,867,174 字节；为保留历史实验完整性，本次没有删除文件。

### E006 预注册状态

- 方案与配置见 `experiments/P0-E006/protocol.md`、`config.yaml`。S0 冻结 E005 CNN，S1 为归一化多延迟复相关 MLP，S2 为 4N 频谱输入的物理引导混合 MLP；共 33 个 S1/S2 训练实例（11 个协议×3 个训练种子）。
- 原始预注册使用 E006 final seed 909090、每个固定 SNR 1024 个样本；该种子后来因评估器缺陷作废，已在下一条更正并归档。E001–E005 测试输入只作为开发基准。
- 用户明确要求 GPU 训练并允许当前工作区访问设备。AISAR 报告 PyTorch 2.8.0+cu128、CUDA 可用、RTX 3050 Laptop；runner 在无 CUDA 时停止，不使用 CPU 训练。
- 首轮按冻结训练协议完成 33 个 CUDA 训练实例。产物审计发现经典估计器的一维批量向量被误读为逐样本训练种子，造成指标/配对比较失真。首轮数据、权重、日志和无效评估已保留于 `experiments/P0-E006/attempts/attempt_1_invalid_evaluation/`；错误 CSV 无损压缩，seed 909090 不复用。
- 统一预测重复解析和结果序列化修复后新增回归测试；最终完整测试 **35 passed**，compileall 与 `git diff --check` 通过。修正版改用独立 seed 919191，模型、训练种子、预算和门槛不变。

### P0-E006 修正版结果与阶段门

- 在 919191 final test 上，每个 SNR 1024 例。20 dB 时 S0/E1、S1/E1、S2/E1 RMSE 分别 15.273、2.947、27.238 Hz；传统 FFT 峰 2.884、FFT 插值 1.210、相位差 0.207、局部/全局网格 ML 约 0.0327、多延迟相干基线 0.0625 Hz。
- S1 在 20 dB 显著优于零频预测，但其三个 noiseless 种子 MAE=2.44–3.01 Hz、P95=5.75–6.40 Hz，G1 未通过。S1−FFT 只在 15 dB 单点的配对 95% 区间完全低于零；它在全部主点均未胜过同相关特征非学习基线，G2 未通过。S1 E3/T2 在 0、20 dB 有孤立收益但不相邻，E3/T3 变差，G3 未通过。
- 阶段决策为 **STOP/REDIRECT**：停止当前单音学生网络扩模，不启动 P1-A 训练。正式总结见 `docs/stage_reviews/P0_final_review.md`；仅保留未注册的 P1-A 理论草案 `docs/stage_reviews/P1_A_plan.md`。
- 修正版 artifacts、seed、设备、代码提交/dirty 状态和 SHA-256 清单见 `experiments/P0-E006/metrics.json` 与 `data_manifest.json`；所有数据为仿真，不含真实 GMTI 或部署时延结论。

## 2026-10-24：P1-A-E001 理想三通道相对相位实验与阶段验收

### 设计与执行

- 在 P0-E007 GO 后，按 `experiments/P1-A-E001/protocol.md` 冻结理想共享单音、三通道独立同方差复 AWGN、相对相位标签、解析基线、CRLB、数据种子、训练预算和验收门。学生为归一化复互相关物理前端加 `4→64→64→4` MLP，仅使用 E1 真值监督；不含教师/蒸馏。
- 使用三个独立训练/验证数据种子、2048 条新无噪声诊断样本和全新盲测 seed `401999` 起按 SNR 每次偏移 1000；测试为 8 个 SNR、每点 1024 条，共 8192 条。盲测只在全部 checkpoint 冻结后生成。
- 在 AISAR、PyTorch 2.8.0+cu128、CUDA build 12.8、RTX 3050 Laptop GPU（4 GiB）顺序训练三个模型，FP32、batch 64、最多 160 epochs、patience 40。每种子训练 14.52–14.90 s，峰值已分配显存 17.6 MiB。

### 结果与阶段门

- 预注册 G1 为 **FAIL，0/3 种子通过**。三种子都通过 20 dB 零相位基线改善（RMSE 降约 98.6%）和公共相位旋转/复尺度/频移不变性（最大输出差 `4.77e-7 rad`）；但无噪声两相位 MAE 为 `0.0177–0.0206 rad`，未达 `<0.01 rad` 门槛，部分 P95 也超过 `0.05 rad`。
- 20 dB 学生 RMSE 为 `0.02315–0.02603 rad`（Δφ21）和 `0.02329–0.02584 rad`（Δφ31）。B1 互相关、B2 逐样本圆均值、B3 秩一 profile-ML 均约 `0.0097–0.0100 rad`，条件 CRLB RMS 标准差约 `0.00995 rad`。学生与 B1/B3 的配对区间在三种子和两相位均为正，显示学生明显更差。
- 训练与验证向量 MSE 均在约 0.10 平台化，未见明显过拟合鸿沟；结果更符合当前小型读出欠拟合/优化受限，但不能由本轮数据确定因果。
- 按冻结门槛，P1-A-E001 决定 **STOP / REDIRECT**：不进入 P1-B、不开展知识蒸馏，不根据盲测更换模型或降低门槛。结论仅适用于本轮模型及理想单音 AWGN 仿真，不否定所有学习架构，也不代表实测 GMTI 能力。

代码、协议、数据、checkpoint、epoch 记录、逐样本预测、分组指标、配对 bootstrap 和 RMSE-SNR 图见 `experiments/P1-A-E001/`；完整阶段审查见 `docs/stage_reviews/P1-A-E001_review.md`。本轮验证命令 `/home/shy/anaconda3/envs/AISAR/bin/python -m pytest -q` 结果为 **67 passed**。

## 2026-10-24：D1-E001 独立校准、正式盲测与阶段审查

- 按冻结协议完成独立 50,000 H0 校准和 40,960 条新盲测（每 SNR 4096 H0 + 1024 H1；8 个 SNR）。正式 A/B/C/D 三策略种子训练、校准和学生盲测推理使用 RTX 3050 Laptop GPU；盲测结果复算使用保存的预测数组。
- **G1 PASS，3/3 seed：** A 真值监督在 20 dB 三 seed 的 AUC/Pd（校准 `Pfa=1e-2`）均为 1.0；全 H1 range RMSE 0.0592–0.0682 cell，Doppler RMSE 25.95–35.13 Hz，均显著优于常值预测。
- **G2 PASS：** C 联合监督 ΔJ=-0.00013472，层级配对 bootstrap 95% CI `[-0.00025944,-0.00000810]`；D 渐进蒸馏 ΔJ=-0.00086207，CI `[-0.00214046,-0.00004611]`；二者均 3/3 seed 方向改善。B 未通过；B/C/D 均无预注册收敛加速证据。
- **G3 FAIL：** A/B/C/D 均未通过任一预注册相邻高 SNR 的传统竞争触发条件。20 dB profile 子集上，D range/Doppler RMSE 为 0.0689 cell/25.42 Hz，APX15 为 0.0415 cell/0.0360 Hz，profile-ML 为 0.01994 cell/0.01906 Hz。
- 盲测预测已先于指标汇总生成并保存。评估器发现 G2 AUC 非劣界限的配置路径错误后，修复读取位置，并仅从保存的预测重放指标；样本、阈值、模型、指标定义与门槛均未改变，审计见 `blind/metrics.json.analysis_replay`。首次盲测输出和分析器失败记录没有被覆盖。
- 阶段决策 **GO → D2**，但仅按阶段规范一次增加一个复杂度。低 SNR 检测与参数估计、传统竞争力均有限；`Pfa=1e-6`、真实 GMTI、多目标、杂波、失配和完整 CoreX 链均未验证。完整审查见 `docs/stage_reviews/D1-E001_review.md`。
- 补充了已距离压缩 IQ 输入边界的初步成本测量：D 学生 CUDA 单样本平均/P95 为 1.53/1.91 ms，batch 128 为 59.75/62.11 ms，CUDA 峰值分配显存约 359 MB；APX15 FFT 子链 CPU 单样本为 0.83 ms。设备不同、处理链边界有限，不据此宣称速度优势。数据和计时命令见 `experiments/D1-E001/benchmark/inference_cost.json`、`scripts/benchmark_d1_e001.py`。
- 完成 D0/D1 综合报告，回答了检测、参数估计、FFT、教师预训练、联合蒸馏、去教师化、传统算法差距及精度/计算优先级八项问题。下一项建议是精度优先的 Doppler 表示对照；它尚未预注册或启动。见 `docs/stage_reviews/D0-D1_overall_report.md`。

## 2026-10-24：D1-R1 Doppler 误差来源诊断

- **范围：** 使用 D1 的 pilot_validation、三个 seed 的 train/validation 缓存和 A/D formal inference checkpoints；不读取最终 blind 或 calibration。推理在 AISAR/PyTorch 2.8.0+cu128、RTX 3050 Laptop CUDA 上完成，batch 32，峰值分配显存约 548 MiB。复现入口为 scripts/diagnose_d1_r1.py。
- **发现：** A 的 0/5/10/20 dB 预测距离门与 true-range oracle Doppler RMSE 跨 seed 均几乎一致；-10 dB oracle 从约 311 Hz 降至 186 Hz，表明距离选择会额外影响低 SNR，但不是高 SNR 主要误差源。高 SNR Doppler 仍约 20–32 Hz。
- **表征/任务：** 固定 ridge 下池化后 probe 比池化前好，但池化前维度 4160 对比池化后 32、训练样本仅 1024，不能作池化因果结论。final checkpoint 上各任务梯度余弦均值接近 0，样本/阶段范围有限，不能排除训练中的瞬态冲突。仅有 inference checkpoint，KL 共享主干梯度无法重算。
- **教师：** teacher 在 -20/-15 dB 检测 AUC 约 0.51/0.67且 Doppler RMSE 约 370/332 Hz；-10 dB 检测 AUC约0.982但 Doppler RMSE约287 Hz。检测知识和精频率知识不能视为同等可靠。
- **决定：** H-ARCH-001 与 H-OPT-001 仍为 INCONCLUSIVE。D1-E002 优先 pilot 低成本 lag-one 复相干统计作为 S2；S1 只做单 seed 时间读出反证；S3 固定 FFT 混合结果单独归因。D1-E001 G1/G2/G3 保持原冻结结论。
- **产物：** experiments/D1-R1/protocol.md、config.yaml、metrics.json、result.csv、analysis.md；测试 tests/test_d1_r1_diagnostics.py。

## 2026-10-24：冻结 D1-E002 受控 Doppler 读出协议

- **问题与证据：** D1-R1 显示高 SNR 残差在真值距离门下仍存在；池化前 probe 没有支持单纯扩大时间特征的说法；任务梯度诊断不支持直接改损失权重。S2 归一化 lag-one 相干统计因此作为优先纯 IQ 候选，S1 仅一次单 seed 反证，S3 作为 FFT 混合方法单独归因。
- **冻结设计：** `experiments/D1-E002/protocol.md` 与 `config.yaml` 固定 S1/S2/S3 前向、参数量、E1 真值监督、1600 更新、有效 batch 64、train/validation/pilot/calibration/blind 种子、检测 `Pfa=1e-2` guardrail、高 SNR Doppler 主指标、range 非劣界、paired hierarchical bootstrap 与停止规则。protocol/config 测试展开 313 个数据流种子，确认无内部重复且不与登记 D0/D1 种子相撞。
- **状态：** protocol/config 与实现已提交到 `main`；CUDA pilot 已完成，训练/验证仿真数据只存本地可重建缓存，尚未生成 E002 calibration 或 blind。

## 2026-10-24：D1-E002 单 seed CUDA pilot 与筛选器更正

- **执行：** RTX 3050 Laptop GPU（4 GiB）上按 S1→S2→S3 顺序运行，FP32、有效 batch 64，每模型 1600 更新；三者均完整完成。训练耗时分别 160.0、147.0、155.0 秒，峰值已分配显存约 1.09、1.10、1.16 GiB。原始 metrics、训练曲线和初始/最佳检查点已提交；训练与验证缓存未提交。
- **原始 pilot 验证：** S0 高 SNR 平均 Doppler RMSE 为 27.129 Hz。S1 为 24.403 Hz（改善 10.05%），低 SNR 平均 AUC 0.658，range RMSE 0.0690 cell；S2 为 22.811 Hz（改善 15.92%），低 SNR 平均 AUC 0.652，range RMSE 0.0734 cell。二者通过原预注册 pilot screen。S3 为 68.110 Hz；匹配三点 FFT 为 67.963 Hz，未见混合学生增益；S3 另未通过 range guard 与 20 dB 回归门。
- **审计修正：** 初始 `selection.json` 因选择器从顶层读取更新次数，把实际存于 `training.completed_updates` 的 `1600` 次误判为未完成。原 `selection.json` 和 `metrics.json` 保留不改；修复后的独立入口将只从已保存 pilot 验证指标重算，门槛、数据、模型和预算均不变，也不把 pilot 重标为 blind。测试已增加嵌套字段回归覆盖；更正 artifact 将在代码提交后生成。
- **当前门：** 对已保存验证指标重算后，S1、S2 满足原定三 seed formal 候选筛选门；S3 停止。formal 尚未开始；新 calibration 与 final blind 均未生成。pilot 只用于候选筛选，不作为正式效应证据。
