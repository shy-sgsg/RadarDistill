# D2-A-T001 预注册协议：三通道 post-Doppler ACE/AMF 教师与软知识可迁移性

**协议版本：** 1.2
**状态：** 2026-10-26 v1.2 执行后端修订；研究问题、候选公式、样本种子/分组、训练预算、阈值和阶段门与 v1.1 完全一致。
**用户授权：** 用户已明确批准 D2-A-T001、单个先进教师候选、阶段门和必要学生 pilot；本协议将其预先量化。
**研究总纲：** `docs/RESEARCH_CHARTER.md` v1.2；本任务不修改研究总纲。
**实验性质：** 合成仿真开发筛选；不生成 final blind，不等于实测或生产系统结论。

### 协议修订记录

- **v1.1（数据前数学勘误）：** 纯空间 ACE 会消除单目标 Doppler-bin 的能量差，故唯一候选改为保留空间相干项、并恢复白化 Doppler 能量的 ACE×AMF；其他研究设计不变。
- **v1.2（用户要求的计算后端修订）：** covariance 拟合和 ACE×AMF 批处理改为强制 CUDA `cuda:0`，使用批量不超过16 CPI的 PyTorch complex64 计算；公共尺度/相位规范化在 double precision 完成，目标公式与指标门限不变，CPU fallback 禁止。修订前曾短暂完成一次 4,096 场景 CPU covariance 拟合，但尚未评分 calibration、verification 或 development；该临时 covariance bank 不作为实验结果，重启时以同一冻结 covariance seed block 在 CUDA 上重拟合并覆盖。没有保存原始 IQ 或依据这次尝试筛选方法。
- **体积要求：** 项目占用只在本阶段结束、清理不需要的大型可重建文件后核验小于20 GiB；运行中间态不设20 GiB硬中止。原始 IQ 和全量训练软图仍不落盘。

## 1. 研究目标、主线与问题

本实验服务 RadarDistill“物理真值与可信传统教师共同支持任务级 GMTI 学生检测及参数估计”的长期目标和总纲 Q1/Q2/Q3/Q4/Q9。研究对象不是独立 CFAR 项目，而是当前教师的难例信息损失，以及更可靠教师知识能否转移到固定学生。

主要问题：

1. 三通道 post-Doppler ACE×AMF 分数能否在 C3、partial-overlap、SNR=0 dB、CNR=20 dB 的可辨识困难条件下改善检测/联合任务，并保护 A/B/C 的主要性能？
2. 其增益相对 B3/T1 是否源自完整三通道协方差白化，而非旧 T1 的全局尺度归一化？
3. ACE×AMF 图在困难场景是否含有比 K2 更正确、更可学习的目标位置软知识？
4. 若教师和知识两道门均通过，固定 TaskStudent 在匹配数据和预算下是否实际获益？

T001 只选择一个主要先进教师：**模型辅助 post-Doppler 三通道 ACE×AMF（PD-ACE/AMF）**。v1.0 的纯归一化 ACE 对固定空间 steering 的单目标复正弦会丢失每个 Doppler bin 的能量差，不能直接用于连续频率估计；v1.1 冻结以 CPI 可观测总能量归一化的 ACE×白化能量统计量。它等价于逐 CPI 能量归一化的 AMF 分数，空间 ACE 相干度作为可解释分量保留。候选仍为一个，无数据驱动搜索。文献中其他方法只作选择依据或历史基线。候选、边界和文献见 `docs/research/D2-A_teacher_candidate_selection.md` 与 `D2-A_teacher_literature_review.md`.

## 2. 冻结物理场景与候选方法

保持 D2-A-E001 信号模型：三通道复 IQ，PRF=1300 Hz、N=130、32 range cells，静止地面点散射体、理想通道、平地/匀速平台、复圆对称 AWGN、E001 的 Tx phase center/波形/几何/range PSF。H0/H1 配对共享 clutter/noise，`H1−H0=target_signal_iq`。A/B/C 为可辨识条件；D exact-overlap 只描述信息边界。

### 2.1 PD-ACE/AMF 定义

1. H0 covariance training 输入只用 `iq_h0`。逐场景计算所有通道/脉冲/range 的可观测平均功率 `e_i`；对 IQ 做四倍零填充慢时间 FFT，得到 `Y_i[k,r]∈C^3`。
2. 对每个 Doppler bin，拟合 population covariance 形状：
   `R[k] = mean_i( mean_r( Y_i[k,r] Y_i[k,r]^H ) / e_i )`；跨 scene 等权，去除生成器参考幅度造成的全 CPI 整体尺度权重。IQ 在 CPU 分批生成，协方差频谱及批量累加在 CUDA 上执行；不保存完整 IQ。
3. 仅为数值求逆，Hermitianize 后将最小特征值下限固定为 `1e-8 × trace(R[k])/3`；不搜索收缩系数、rank 或温度。记录每个 bin 的条件数、特征值和地板触发比例。
4. 由 E001 公开几何计算 17 点冻结接收 steering grid `a(θ)`。定义 CPI 可观测平均功率 `e_CPI=mean(|X[c,n,r]|²)`，以免固定阈值分数随公共复尺度变化。对每个 `(k,r,θ)` 计算空间相干度
   `ACE[k,r,θ]=|aᴴ R[k]⁻¹Y[k,r]|² / ((aᴴR[k]⁻¹a)(Y[k,r]ᴴR[k]⁻¹Y[k,r]))`，并计算白化匹配能量 `W[k,r,θ]=|aᴴR[k]⁻¹Y[k,r]|²/(aᴴR[k]⁻¹a)`。分母为零时两者均为0；诊断 ACE 裁剪到 `[0,1]` 仅限浮点容差。
5. 唯一检测和参数估计 map 定义为 `S[k,r]=max_θ(ACE[k,r,θ] × Y[k,r]ᴴR[k]⁻¹Y[k,r] / e_CPI)`，代数上等于 `max_θ W[k,r,θ]/e_CPI`。这保留 ACE 的空间相干解释，同时保留 Doppler 能量剖面；不把纯 ACE 的频率平坦图当作频率估计器。不得在看过新数据后改成另一种 map。
6. Detection score 为四倍频率网格和全部 range cell 的最大 `S`。Doppler 由该 score map 对应 bin 及周期三点抛物插值估计；range 由峰值和三点局部抛物线估计，偏移限于 `[-0.5,+0.5]` cell。
7. 为与 K2 匹配，蒸馏图从 N=130 原生 FFT `S` map 产生；对 `[130,32]` map 全图求和归一化。若和为0则使用全零响应并记录；不按真值裁剪、不读场景标签。该图是 saliency，不解释为后验概率。
8. PD-ACE/AMF 推理函数仅接收 observable IQ、已拟合 `R[k]` 与冻结物理几何配置；不读取 truth、潜在幅度、SNR/CNR、组件功率、scene id 或目标标签。Covariance fitting 只读独立 H0 training IQ。

联合分数和归一化软图对 IQ 的公共复数标量缩放不变；单独记录的 ACE 空间相干度也不变。固定 population covariance 不等于 Kelly/ACE 文献中以在线独立 secondary 估计协方差的统计设置。因此不继承解析 CFAR 保证，全部 Pfa 结论由独立 H0 校准和验证给出。

## 3. 冻结分组、样本量和种子

所有 train/calibration/verification/development 场景 ID 与 seed block 相互独立，并避开 E001、R3、E002 (`18M–27M`) 和 final blind reserve (`9M–10M`)。每组按最多16个 CPI 流式生成/处理，无 full IQ 缓存。

| Split | 场景数 | SNR/CNR 构成 | 用途/seed base |
|---|---:|---|---:|
| `covariance_train` | 4096 H0 | C0–C3 等量；C0=20/−5，C1=15/0，C2=10/10，C3=0/10 dB各半且CNR=20 | 31,000,000；只拟合 R[k] |
| `h0_calibration` | 15,000 H0 | 与 E002 C0–C3 等权 mixture；C3 两个 SNR 等量 | 32,000,000；独立估算每方法 5% CPI阈值 |
| `h0_verification` | 15,000 H0 | 与 calibration 同 mixture、独立 | 33,000,000；只评价冻结阈值 |
| `development` | 12,000 paired H0/H1 | 每课层3000；C0 A=3000；C1 A/B=1500 each；C2 A/B/C=1000 each；C3 A/B/C×SNR0/10=500 each；CNR按 E002固定 | 34,000,000 起分组子块；方法/学生开发评价 |

H0 mixture 中场景合成函数以固定 `separated` scenario 调用，但 detector 只见 H0 IQ；R3 已逐元素检查同 seed A/B/C/D 的 H0 不受 scenario tag影响。H0 nuisance `A` 继续按 E001 `Uniform[-6,6] dB` 生成。Calibration、verification 与发展集的 SNR/CNR分层只用于事后组别汇报，不是 detector输入。

主要统计条件固定为 `C3 / partial_overlap / SNR=0 dB / CNR=20 dB`，n=500 paired scenes。此条件可辨识但困难，且旧 K2 top-cell定位仅4.7%。该旧比例仅作动机，不能代替 T001 新数据。D exact-overlap只在开发集描述性报告，不用于候选选择或门判断。

## 4. 对照、独立阈值及指标

主比较使用同一 T001 H0 calibration、H0 verification、paired development IDs：

- B3：E001 几何 DPCA 原始检测/估计分数。
- T1：R3 固定 global robust background normalization detection statistic；保留 B3 argmax输出。
- K2：B3 IQ-only `[130,32]` normalized RD soft response；不是新固定阈值教师。硬检测分数仍报告B3。
- PD-ACE/AMF：本协议唯一先进教师候选。
- S0/S1/S2：若教师门通过，分别为匹配课程 TaskStudent truth-only、旧K2软监督和新PD-ACE/AMF同形软监督。

B3、T1、PD-ACE/AMF 每个 detector 用 calibration H0 自己的 `numpy.quantile(0.95, method="higher")` 取 CPI score 阈值，比较符 `>=`；阈值不从 verification/development 重估。三方法以相同 verification H0 和 paired development H0/H1评价。S0/S1/S2 如运行，也独立用同一 T001 H0 calibration scene IDs 的各自 presence logit校准，并在 verification固定阈值评价。报告`>`、`>=`、阈值平局数。

必须报告：

1. Detection：H0 Pfa及精确单侧95% Clopper-Pearson上限；Pd、ROC/AUC；全局及 course、SNR、CNR、A/B/C、幅度四分位组别。
2. Estimation：所有H1及detected H1的range MAE/RMSE（cells、m）、周期 Doppler MAE/RMSE/bias；misses-as-failures joint success (`detected`, `|range error|≤1 cell`, `|circular Doppler error|≤20 Hz`)。
3. Soft knowledge：H0/H1 score和map响应分布；target-window probability mass（range≤1 cell 且 Doppler≤20Hz）、配对 H1−H0 mass lift、teacher max cell localization、错误峰类型、map normalized entropy、target/false peak margin；重点报告 C3 A/B/C与主要困难条件。
4. Physical behavior：公共复相位、复尺度`0.5, 2.0, 0.6+1.2j`不变性；对叠加目标单独频移`−25,−10,+10,+25 Hz`的响应移位；独立通道相位破坏诊断；方向/频率边界和参数周期误差。所有转换用原始冻结模型/阈值；不做重校准。
5. Compute：covariance拟合 CUDA时间、样本吞吐、CPU peak RSS及CUDA峰值 allocated/reserved；在相同复用的16个校准 IQ上比较候选的 CPU 与 CUDA 推理时间（batch=1/16、各1000 CPI，仅作实际成本记录，不称实时验证）；报告协方差和模型大小。B3/T1/K2基线保留现有 CPU NumPy实现。若运行 student，则记录CUDA/GPU、训练时间、peak allocated/reserved。

区间使用2,000次`base_scene_id`配对 bootstrap（固定 seed `2026102702`）；H0 Pfa使用精确二项区间。ROC/AUC、RMSE、Pd和joint success的对照使用场景配对bootstrap。D不参与可辨识性能门。

## 5. 教师阶段门（学生训练前冻结）

### G1：数学、实现、物理和数据正确性

- 测试 covariance Hermitian/PSD/有限、逆矩阵有限、每 bin 有 condition/eigen记录；无 clutter/noise/单 steering 极限符合 ACE/AMF；已知单目标复频率必须落在目标 Doppler 峰附近且空间 ACE 峰对齐 steering；无目标所有输出有限。
- 相同 IQ 乘任意预设公共复数因子时空间 ACE、归一化 AMF score/map 最大绝对差`≤1e-6`；所有角度、PRF、shape、Nyquist 环绕与 map 和边界正确。
- H0/H1 paired identity、split/seed 互斥、目标信息/sidecar 不进入 covariance 或 detector、PD-ACE/AMF scores 由相同提交代码可重放。
- 冻结B3/T1/K2基线在任一同IQ上的实现与现有函数一致。
- 通过后才能解释统计结果；失败停止并记录最小复现。

### G2：教师检测/参数任务增益

PD-ACE/AMF 独立 H0 verification的总Pfa点估计`≤0.06`且单侧95% exact上限`≤0.06`；预注册course/SNR/CNR各大组上限`≤0.10`。在主要困难条件 C3/C/SNR0/CNR20 中，相对新数据中最好旧teacher基线（T1、B3）取逐场景配对比较：joint success点增益至少`+0.05`且双侧95% bootstrap差值区间下限`>0`。同时在 pooled identifiable A/B/C 上，候选相对 T1的 Pd/joint-success 区间下限均须高于`−0.05`，range RMSE非劣界`+0.5 cell`、Doppler RMSE非劣界`+10 Hz`；各C0–C3大组AUC相对 T1不得低`>0.02`。G2失败则停止教师分支，不因AUC排序或更复杂soft loss继续调参。

### G3：软知识是否值得教

只在 G2 通过后审查 PD-ACE/AMF soft map：C3/C/SNR0 H1−paired-H0 target-window mass-lift的95% paired bootstrap下限`>0`；teacher top-cell localization至少`0.20`（旧K2为0.047）；pooled C3 A/B/C mass-lift下限`>0`。报告H0最大score/map entropy和困难场景错误，不以 pooled 平均替代困难组。如果G3失败，不运行S2蒸馏。

阶段门依据新数据，不允许在测试真值上选择协方差模型、steering、阈值、温度或样本窗口。失败保留，停止，不再实现第二个先进教师。

## 6. 学生有界 pilot（仅G1–G3全部通过时）

学生阶段固定 D1-E001 `TaskStudent`，同一宽度/任务头、training seed 101、1600 optimizer updates、effective batch64、FP32、CUDA `cuda:0`。三组完全同训练样本多重集、初始化、课程 batch顺序和任务监督：

- **S0 truth-only：** 复用 D2-A-E002 v2.1 B 臂 checkpoint（同一课程计划、直接真值监督）。
- **S1 old teacher：** 复用 E002 v2.1 C 臂 checkpoint（B同课程/初始化/训练数据，加旧K2 H1-only软监督）。
- **S2 new teacher：** 仅新增训练一次；沿用同 E002 训练 split/batch计划/模型/初始化，在H1真值任务损失外使用新PD-ACE/AMF `[130,32]` map；软loss格式与K2完全一致，H1-only，lambda按 E002 `0.05→0` 线性退火；H0仍只有真值presence监督。

S0/S1以冻结 checkpoint 作为相同种子 pilot锚点，不重训练；三组均在新T001 H0 calibration/verification和paired development上评价。S2训练IQ按E002冻结scene seeds/batch plan确定性分批生成，CPU流式处理，不将全训练IQ缓存到GPU/仓库。new-teacher maps只保存必要的训练中间缓存并按既有项目忽略规则处理；阶段收尾清理后项目总量需小于20 GiB，中间体积不设该限制。

评估每臂Pfa/AUC/Pd/range/Doppler/joint success、主要困难子组、配对`S2−S1`与`S2−S0`区间、物理变换和训练/资源成本。**pilot signal**需S2相对S1在困难条件 joint success至少`+0.05`且paired95%区间下限`>0`，并且S2相对S0 joint-success CI下限`>−0.05`，全局verification H0门通过且无超`+10Hz` Doppler RMSE退化。单seed只决定是否值得独立预注册正式多seed，不证明一般性收益。

若G1–G3任一不通过，则不训练S2；保留E002 B/C为旧锚点，不生成final blind，不启动正式三种子。若S2相对S1无明确收益，停止简单teacher upgrade，不无限加教师复杂度；根据结果转向学生物理表征/目标知识形式另立审查提案。

## 7. 资源、复现和交付

- Teacher covariance拟合和ACE×AMF推理强制运行在配置指定的CUDA设备`cuda:0`；CUDA不可用或空闲显存低于配置下限时停止，不回退CPU。输入分批且batch≤16 CPI；double precision完成公共复尺度/相位规范化，频谱和协方差大批运算使用complex64 CUDA；每阶段记录峰值显存和运行时间。不缓存IQ或全量训练输出图。`R[k]` covariance bank、校准/验证分数及逐场景指标可保存。
- 20 GiB是阶段结束清理后的项目最终体积上限，不限制中间运行体积。Preflight和实验阶段只记录当前体积；结束后仅清理可重建且不需审计的大型缓存，再实测项目总占用并确保小于20 GiB。
- 若运行student，必须确认 `AISAR` 的 RTX 3050 Laptop CUDA可见；仅S2新训练、顺序单模型驻留GPU，不CPU回退；OOM遵照 E002 microbatch/梯度累计规则。
- `environment.json` 记录当前git SHA、Python/NumPy/Torch、CUDA、GPU、CPU/内存、disk free、代码版本、配置、各阶段运行时间；teacher阶段不作实时测试主张。
- 复现：
  `conda run --no-capture-output -n AISAR python scripts/run_d2a_t001.py --config experiments/D2-A-T001/config.yaml --stage preflight --device cuda:0`
  `conda run --no-capture-output -n AISAR python scripts/run_d2a_t001.py --config experiments/D2-A-T001/config.yaml --stage teacher --device cuda:0`
  仅G1–G3通过：`conda run --no-capture-output -n AISAR python scripts/run_d2a_t001.py --config experiments/D2-A-T001/config.yaml --stage student --device cuda:0`
- 保留 `protocol.md`、`config.yaml`、`config_used.yaml`、environment、covariance bank、metrics/result CSV、配对逐场景 score/estimates、图表、analysis、软知识 audit 和 stage review；不提交原始 IQ、cache、或本地 GMTI 用户文件。
