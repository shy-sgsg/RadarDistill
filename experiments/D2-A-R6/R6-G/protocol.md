# D2-A-R6-G：CE-TCN上的K3教师教学内容 pilot

**协议版本：** 1.0.0
**状态：** 执行前冻结；单训练seed开发 pilot；`final_blind_authorized: false`
**总纲：** `docs/RESEARCH_CHARTER.md` v1.3
**对应路线：** R6 P6；本协议不修改 R5-C、R6-E/F 冻结协议或结果。

## 1. 科学问题与研究主线

R5-C的T0–T3显示，K3完整RD教师图在R5-C Medium学生上没有改善joint任务；困难的C3 partial-overlap/SNR0/CNR20格中K3定位率只有4.1%。R6-E/F则选择CE-TCN为目前任务表现最强的受测学生，但未通过C3六格familywise低虚警门。

本阶段只回答一个有界问题：**CE-TCN从头学习时，K3全图软响应能否在物理真值任务损失之外改善最终GMTI检测/参数联合任务或固定预算内的学习进程？** T0提供truth-only基线，T3控制一般的辅助响应监督。K3是否教会学生真实有用内容，以最终Pd、Pfa、参数误差和joint success判断，不能以KD损失下降判断。

本研究服务总纲Q1、Q2、Q6、Q9、Q10；传统教师响应仅作为学生训练知识，不构成独立CFAR研究。R6-G单seed结果只用于机制筛选，不是跨seed确认或部署资格。

## 2. 冻结的三组实验

三组均为从头初始化，不从R6-E seed606检查点warm-start。使用同一R6G-CE-TCN训练模型、相同seed707 bit-identical初始权重、训练与验证场景、scene采样顺序、优化器、真值任务损失、有效batch和更新预算。推理模型不包含训练辅助响应头。

| 臂 | 训练目标 | 说明 |
|---|---|---|
| T0 | `L_truth` | 物理真值监督基线；不请求K3图 |
| T1 | `L_truth + 0.05 × L_K3` | K3 IQ-only完整`520×32`全RD响应KL，仅H1样本 |
| T3 | `L_truth + 0.05 × L_truth-response` | 与K3形状匹配的物理真值Dirichlet响应KL，仅H1样本；这是辅助监督控制，不是教师 |

`L_truth`沿用R6-E J1联合损失：目标存在BCE、range-cell CE、range-offset smooth-L1和周期Doppler cosine项。T1使用R5-C固定K3：三通道DPCA、4×慢时间频谱、全图非负响应归一化；map只能由当前H1 IQ与冻结物理配置生成。T3的全图响应由物理真值range/Doppler的有限CPI Dirichlet响应生成。两种辅助KL权重均固定0.05，不搜索温度或权重。T0/T1/T3在模型选择时统一使用validation truth-task loss最低的checkpoint，不使用包含辅助项的arm-specific总loss选择。

教师峰值硬标签仅用于教师正确性诊断，绝不作为训练目标。训练代码不得把真值range/Doppler、SNR/CNR条件、H0/H1标签以外的场景标签或教师正确性送入K3教师推理或学生输入。H0 CPI只使用真值检测任务损失，不产生教师/真值RD辅助响应。

## 3. 场景、样本和隔离

保持R6-A物理条件：三通道复IQ、PRF=1300 Hz、N=130、32距离单元、静止杂波、R6-A配置的12个C0–C3条件。数据生成由冻结`simulation/d2a.py`及R6-A条件表提供。

| Split | 数量 | 用途 | seed base |
|---|---:|---|---:|
| Train | 51,200个配对基础场景 | 3臂共同训练；每步32 H0+32 H1 CPI | 按C0–C3分组 `[2408000000,2413000000,2418000000,2423000000]` |
| Validation | 5,000个配对基础场景 | 每400步评估truth-task loss并共同选checkpoint | 按C0–C3分组 `[2428000000,2433000000,2438000000,2443000000]` |
| H0 calibration | 120,000个H0 CPI | 每臂独立拟合0.05经验阈值 | `2508000000` |
| H0 verification | 120,000个H0 CPI | 固定阈值虚警评价 | `2528000000` |
| H1 development | 6,144个H1 CPI（每条件512） | 配对任务及物理知识评价 | `2548000000` |
| G1 correctness | 16个配对C3 partial-overlap/SNR0/CNR20场景 | 仅链路测试 | `2598000000` |

训练、验证、calibration、verification、development、G1的scene ID必须彼此不重叠，并通过全仓历史scene manifest检查。若存在冲突，停止，不自动重编号。Development不得作为final blind。H0 calibration不用于H1模型选择；H0 verification只在训练/模型选择冻结后评价。

训练/验证IQ由owner清单登记并临时存储；每个H1训练样本对应一个K3和一个T3软响应，先由CUDA生成，再以float16写入CPU文件，训练时按batch读取。每个batch向`cuda:0`搬运，不把全部数据驻留GPU。H0评价、H1 development原始IQ按batch生成和处理，不持久化。结束后仅清除owner列出的本阶段缓存。

软响应目标生成固定采用batch size 16，以控制RTX 3050 Laptop 4 GiB显存占用；该batch仅是目标生成的microbatch，不改变学生训练的effective batch或场景曝光量。

## 4. 训练预算与实现

- 初始化：PyTorch seed707；先生成一份R6G-CE-TCN随机初始state，再精确复制给T0/T1/T3。不得载入R6-E seed606权重。
- 学生：width64复数相位等变CE-TCN；任务推理参数数目沿用R6-E `63,429`。训练增加`520×32`辅助response head，T0/T1/T3结构完全匹配；推理前省略该训练专用头。
- 训练：每臂51,200 optimizer updates，effective batch64 CPI（每步32 H0+32 H1），FP32、AdamW、weight decay `1e-4`、gradient clipping 1.0、初始学习率`1e-3`至`1e-5`单cosine计划。
- microbatch固定4、梯度累积16；如果运行前显存门不通过，或训练OOM，按4→2→1降低microbatch但有效batch仍为64、updates和样本曝光不变。若microbatch1仍无法完成有效batch则停止。
- 三臂顺序驻留RTX 3050 Laptop `cuda:0`训练；CUDA不可见或任何静默CPU回退立即停止。
- Validation每400步；resume checkpoint每800步；history每400步；禁止逐optimizer update写日志。checkpoint只按共同validation truth-task loss选取。
- 所有臂使用同一训练scene plan和样本次序；teacher/T3 target仅以预先固定的数据索引取用。保存模型、优化器、scheduler、随机数状态、sampler位置和owner信息，以便断点精确恢复。

## 5. 指标、统计和解释

每臂独立从H0 calibration估计Pfa_CPI=0.05阈值，固定比较符`score >= threshold`，在独立H0 verification报告总体及12条件Pfa、单侧exact95%上界、C3六格Bonferroni familywise上界和ties。只有C3每格familywise上界均`<=0.10`才获得绝对低虚警资格。教学机制结果与系统资格分开判读。

H1报告总体、C0–C3、C2+C3及12条件的Pd/AUC、range RMSE、周期Doppler RMSE、bias、candidate correctness和joint success。joint success定义为检测命中且range误差`<=1 cell`、周期Doppler误差`<=20 Hz`。miss/null计失败；同时分别报告all-H1与detected条件参数误差。

主对比为T1−T0的C2+C3 joint-success paired scene差，按条件分层bootstrap 5,000次，seed `2026110401`，双侧95%区间。筛选门：点差至少`+0.02`且区间下界`>0`。任务保护：Pd差95%区间下界`>=−0.02`，全H1 range及Doppler RMSE ratio的95%区间上界均`<=1.05`。T1−T3和T3−T0按同样paired场景报告，作为预注册解释对照；不以T3替换主对比。

收敛诊断使用固定预算的validation truth-task loss、Pd、range/Doppler RMSE和joint曲线，在0、12,800、25,600和51,200步报告预定点及validation loss曲线积分。该pilot只用一个训练seed，因此曲线改善是单seed筛选证据；只有最终任务指标也改善才可称为有用的收敛收益，不按KD loss下降定义收敛成功。

教师K3正确性在训练/推理之外计算，正确定义为峰值同时满足range误差`<=1 cell`、周期Doppler误差`<=20 Hz`。离线分层报告教师正确/错误的T1/T0/T3学生表现，重点单列C3及C3 partial-overlap/SNR0/CNR20。正确性标签不进入训练、推理或checkpoint选择。若该预注册困难格显示明显负蒸馏，停止当前K3分支，不追加可靠性网络或阈值搜索。

记录训练时间、validation用时、K3 CUDA目标生成时间、每臂峰值显存和完整评价成本；不作实时性声明。报告训练样本暴露、updates、访问次数、参数量和训练/验证曲线。

## 6. 阶段门、停止和交付

- **G0：** protocol/config/source冻结；新scene ID与历史及split无碰撞；三臂初始化、batch plan、总updates和目标数据owner一致；磁盘估算和CUDA设备门通过。
- **G1：** T0/T1/T3输出接口一致；response头输出`[B,520,32]`、全有限；K3 map非负且归一化；T1 response head梯度有限非零；T3梯度有限非零；T0不读取辅助target；H0不生成response map；CUDA教师和学生路径通过；seed/data order配对检查通过。
- G0/G1均PASS后，按T0→T1→T3顺序进行一次单seed pilot；三个arm都完成固定预算前不看H1 development，也不按阶段H0 verification调参。
- 主要教学收益门未过、任务保护失败、teacher错误困难格有明显负迁移或CUDA/owner/scene隔离错误时，停止当前知识分支；保留全部负结果，不增加温度、权重、教师或学生网格。
- 即使通过也只建议R6 P7提出独立多seed确认；不生成final blind，直到候选和所有门在独立protocol中另行冻结。

交付：代码、配置、测试、训练records/checkpoints、metrics、训练与任务图、教师错误分层、analysis、内部阶段报告和cleanup audit。ChatGPT综合审查请求按`AGENTS.md`只在整体R6 P0–P7路线全部目标完成后提交一次。
