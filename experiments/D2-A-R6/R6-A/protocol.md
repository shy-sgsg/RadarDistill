# D2-A-R6-A：同场景检测与参数估计基线

**状态：** 协议 v1.0 已预注册；R5-C T0–T3 已完成。R6-A实现、测试和G0前置检查仍待完成，全部通过后才生成数据并运行。此为独立实验，不使用R5-C数据、阈值、模型选择样本或统计结果。

## 1. 科学问题

在三通道、PRF=1300 Hz、N=130、32距离单元、静止杂波的同一批新 IQ 场景上，统一比较已实现教师/传统统计量与 R5-A Medium CNN；再分开量化检测候选错误和候选正确时参数精估误差。此阶段建立 R6 公共参照，不训练模型、不调参、不声明最终优势。

关联总纲 Q1、Q3、Q6、Q7、Q8、Q10；本阶段假设：C3 整体性能瓶颈是否主要来自检测/阈值还是候选内 Doppler/距离估计，需由同场景数据判断。

## 2. 输入与候选

全部方法收到相同的有限复数 IQ 数组、条件和物理配置；方法不得读取目标真值。每个 CPI 输出检测分数、实际候选、参数估计和运行成本。真值只在预测冻结后进入离线评价。

### 检测分数

- B0–B3：使用已验证实现；B0–B2作基础/物理诊断，B3为主要传统参考。
- D2-A-R3 T1/T2：保留为方法候选，但显式标注未取得跨场景固定阈值资格。
- Medium width=96 CNN：使用冻结 checkpoint `experiments/D2-A-R5/R5-A/capacity/runs/seed_101/Medium/best_checkpoint.pt`；训练记录为同目录`training_record.json`（seed 101、12,800 updates，训练模型82,887参数）。推理时只按冻结导出函数移除130-bin训练频谱头，保留任务头和主干，推理参数为70,277。不在本阶段重训或以R6-A数据选择checkpoint。若checkpoint缺失或state dict/训练记录/参数量不符，停止该方法，不能替换为其他权重。

每方法对完整CPI产生一个max-candidate检测分数。正式基线至少覆盖B3、R3 T1、Medium；只有代码/正确性审计通过的方法才进入主表。B0–B2可作为相干处理负/弱基线。

### 参数估计

- FFT最大峰、二次插值和固定4×零填充频谱峰；
- lag-one相位差估计；
- 局部profile-ML连续 Doppler 搜索；
- Medium学生直接输出；
- 当检测候选存在时，另评估NN检测+profile-ML传统精估。

每种估计分别报告：全H1误差、检测命中条件误差、正确range/candidate条件误差。所有条件误差均注明选择条件。Profile-ML依其单音/噪声假设解释；残余杂波或协方差失配时，不能称为无条件最优估计或普适CRLB。

R6-A只对已加载并冻结的checkpoint推理。若按训练记录导出推理模型，只允许删除训练期频谱辅助头；presence/range/offset/Doppler任务头和共享主干权重必须保持不变，并记录载入checkpoint路径及参数量。

## 3. 场景与划分

复用 E001 的12个预注册C0–C3条件定义和三通道静止杂波生成器；条件表在R6-A config中自包含，物理模型仅引用E001，不读取或复用R5-C场景、阈值、结果或checkpoint选择样本。全部使用新seed与新base_scene_id。场景间独立；每一CPI只保留运行所需数据，采用分块流式生成。相同split中每种方法读取完全相同的IQ。

| Split | 样本量 | 12格分配 | 用途 |
|---|---:|---:|---|
| H0 calibration | 12,000 | 每格1,000 | 各方法各自校准0.05/0.01阈值 |
| H0 verification | 12,000 | 每格1,000 | 独立Pfa与条件迁移评价 |
| H1 development | 6,144 | 每格512 | Pd、ROC/AUC、参数及joint评价 |

校准、验证和H1 development互不重叠。方法间共享split；方法分数/预测不跨方法混合阈值。按base_scene_id做配对bootstrap，重复抽样固定使用seed `2026102901`、2,000次。所有数据是development，不是final blind。

新seed base分别为：H0 calibration `1,208,000,000`、H0 verification `1,228,000,000`、H1 development `1,248,000,000`。每条件seed block宽1,000,000；分块最大16场景、chunk stride 1009。执行前必须对已知记录与R5-C/R6-B保留block做seed和scene-ID冲突检查；不自动重新编号。

## 4. 阈值与统计分析

每种方法在其独立H0 calibration scores上分别以 `higher`经验分位数校准目标 `Pfa_CPI=0.05` 和 `0.01`；操作符固定为 `score >= threshold`。记录 `>`、`>=` 经验率、阈值和ties。阈值一旦由calibration确定，H0 verification不能改变它。R6-A只做0.05/0.01工作点的统计比较；对0.001及以下只报告ROC插值/描述，不授予低虚警资格。

报告：

- 独立 verification pooled及每条件Pfa、单侧95% exact Clopper–Pearson上界和ties；
- Pd/ROC/AUC，条件AUC与全局混合AUC分开；
- all-H1、检测命中和正确候选条件下的距离/Doppler MAE、RMSE、bias；Doppler用PRF回绕误差；
- joint success：检测命中且range误差≤1 cell且回绕Doppler误差≤20 Hz；漏检与错误候选均计失败；
- 传统估计、学生估计和profile-ML在oracle候选/实际候选下的差值；
- 参数量、完整方法边界、CPU/GPU、峰值内存、每CPI及批次成本。R6-A成本是离线对照，不作系统实时性声明。

固定全局阈值只用预注册混合H0。所有SNR/CNR/场景标签仅作离线分层，不作为推理阈值输入。条件ROC诊断可解释条件内区分能力，不能称作可部署条件门限。

## 5. 阶段门与停止规则

- **G0：** main提交与配置一致；独立seed/scene-ID检查无冲突；方法输出和功率/频率轴数值检查通过；R5-C训练不占用GPU；数据预算和流式I/O通过。
- **G1：** 所有方法在完全相同场景上成功输出；真值未进入方法调用；阈值校准与verification分离；回绕误差、oracle/actual分层、bootstrap重抽样单位正确。
- **G2：** 不把0.05/0.01定点上的发展性比较升级为资格结论。该阶段主要结果是同场景基线与失败归因；若区间跨越有意义差值，结论为不确定。

如果实现/物理检查失败，停止对应方法，不以更换工作点掩盖问题。如果Medium出现高条件ROC但全局Pfa偏差，推荐R6-B检验分布迁移；如果C3条件ROC也低，优先查可观测特征/H0覆盖。不得在R6-A verification上开发或训练下一阶段方案。

## 6. 交付

运行代码、config、`preflight.json`、scene/seed manifests、每场景紧凑预测、`metrics.json`、`analysis.md`、图和资源记录。不得缓存全量IQ；保存scene IDs足以复现。R6-A完成后单独提交SHA及阶段结论，且不改R5-C结果。

## 7. 运行后离线 oracle 映射更正（2026-10-29）

在汇总指标复核时发现，原 oracle helper 将连续真值距离用 `floor` 映射为距离门。仿真器的距离响应为 `sinc(truth_range_cell - range_axis)`，所以离散 oracle 门应选择有限距离轴上最近的整数中心。更正规则为四舍五入、半格恰好时向较大索引取整，并将超出采样轴的值裁到最近边界门。

原 H1 预测 CSV 中的 `oracle_*` 列以及首次预更正 `metrics.json.gz`、`analysis.md.gz`（均为无损压缩）和图表均保存在 `attempts/oracle_floor_mapping_correction/pre_correction/`，仅作审计记录。使用相同的6,144个注册 H1 development scene IDs确定性重建 IQ，并将最近门结果写入独立 `predictions/h1_oracle_nearest_bin.csv` sidecar；规范 `metrics.json` 的 oracle-only 频率诊断从此 sidecar 读取。该更正只影响 oracle 候选频率估计及其离线汇总/图，不改变实际检测器、实际候选、阈值、H1标签、样本划分或阶段门。未依据预更正 oracle 数值选择方法或修改方案。该数据仍是 development，不是 blind。
