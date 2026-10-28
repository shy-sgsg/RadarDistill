# D2-A-R5-C Teaching Content Pilot 阶段审查

**运行代码 SHA（G0 绑定）：** `a97ad36877d04dff8d2c8499e4bf783e0435714c`
**研究总纲：** `docs/RESEARCH_CHARTER.md` v1.3（本阶段未修改）
**协议：** `experiments/D2-A-R5/R5-C-teaching/protocol.md` v1.0
**阶段结论：** 当前教师内容分支 **STOP**；RadarDistill 总体路线 **REFINE**，转入已批准的 R6-A 同场景基线。

## 研究问题与范围

本 pilot 检验 R5-C 中有任务相关性的 K3 全 RD 响应和 K2-local 局部响应，能否在匹配物理真值监督之外改善任务级学生。它对应总纲 Q1、Q2、Q9、Q10。T0 是 truth-only，T1 是 truth+K3，T2 是 truth+K2-local，T3 是使用真实目标参数生成辅助响应的 truth-assisted 对照；T3 不是教师。

独立审查依据结果提交 `7563c822e70afe761897ddd5e91bb64207720f89` 返回 REFINE，并有条件放行这一次 pilot。原 R5-C G4 判定及历史结论均未改写。

## 执行与工程核查

- G0 通过，代码/配置绑定 `a97ad36877d04dff8d2c8499e4bf783e0435714c`；G1 四臂均通过。G1 仅证明小样本拟合和梯度链可运行，不计为科研性能结果。
- 仿真沿用三通道、PRF 1300 Hz、130 脉冲、32 距离单元和静止杂波。训练/验证为 51,200/5,000 个新配对场景，pair plan 为 12,800×32、8 次固定数据遍历；两分割无 ID 重叠，与核对的 56,200 个 R5-A 历史训练/验证 ID 无重叠。
- 四臂均使用 seed 303、TaskStudent width 96、FP32、microbatch 16、有效 batch 64、各 12,800 updates，顺序在 RTX 3050 Laptop `cuda:0` 训练；没有 OOM、CPU 回退或并发 GPU 模型。
- 每臂训练参数 120,717、推理导出参数 70,277。T0 跳过 520-bin 辅助头前向；因此参数增量不能解释 T0 相对 R5-A 的全部耗时差。R5-A Medium 的 12,800 updates 用时 3,932.35 秒；本阶段 T0 为 6,745.77 秒（0.527 秒/update），慢约 1.72 倍。T1/T2/T3 分别为 4,446.82/4,295.96/3,781.71 秒。现有计时记录不能把差异归因于单一因素；本轮逐更新日志和 CPU/设备同步开销是可见实现成本，但所有四臂共用同一记录方式，不能据此解释臂间差异。后续新实验默认按验证/检查点记录训练日志，除非有明确的逐步诊断需求。
- 评价通过独立 `cuda:0` 学生推理和 K3 教师诊断完成，用时 539.16 秒；产生四臂共 16 份逐场景预测 CSV。未保存评价全量 IQ/教师图，未生成 final blind。
- 数据清理只删除 manifest 列出的四个 stage-owned train/validation IQ memmap，共 11,222,016,512 bytes；保留 scene/seed 清单、truth metadata、pair plan、预测与训练记录。清理记录显示项目占用 15,160,472,696 bytes（约 14.12 GiB），低于 20 GiB 上限；其他未跟踪用户文件由清理器保留。

## 预注册评价结果

数据量为 H0 calibration 8,192、总体 H0 verification 8,192、独立 C3 H0 verification 4,096、H1 development 4,352。阈值按每臂自身 calibration 的 higher 95th quantile 选择，H0 verification 从未用于阈值选择。主指标为 3,840 个可辨识 C2+C3 H1 场景上的配对 joint success。

| 臂 | 总体 H0 Pfa | C3 H0 Pfa [单侧95%上界] | H1 Pd | AUC | C2+C3 joint | C2+C3 range RMSE (cell) | C2+C3 Doppler RMSE (Hz) |
|---|---:|---:|---:|---:|---:|---:|---:|
| T0 | 0.0520 | 0.1943 [0.2048] | 0.9200 | 0.9531 | 0.7750 | 3.238 | 30.419 |
| T1 | 0.0531 | 0.2031 [0.2137] | 0.9221 | 0.9524 | 0.7646 | 3.248 | 30.298 |
| T2 | 0.0500 | 0.1824 [0.1926] | 0.9207 | 0.9522 | 0.7612 | 3.302 | 31.533 |
| T3 | 0.0553 | 0.1973 [0.2078] | 0.9228 | 0.9530 | 0.7906 | 3.275 | 28.635 |

主要配对差异：

- T1−T0 的 C2+C3 joint 差为 −0.01042，95% 场景 bootstrap 区间 `[−0.01849, −0.00208]`；预注册的 `+0.02` 主收益门未通过。
- T2−T0 为 −0.01380，区间 `[−0.02292, −0.00391]`；主收益门未通过。
- T1 的相对任务保护除 C3 绝对 Pfa 外均通过；T2 的总体 joint 点差 −0.01218 超出冻结 −0.01 保护，且 C3 绝对 Pfa 失败。两者都未通过综合保护门。
- T3−T0 为描述性 +0.01563，区间 `[+0.00781, +0.02344]`，点增益低于 0.02；这是使用真实参数构造辅助响应的结果，不能归因于教师知识。
- 四臂 C3 H0 Pfa 均为 0.182–0.203，远高于绝对 0.10 晋级门。总体 Pfa 校准接近 0.05，未消除 C3 条件虚警问题。

## 教师错误区域与负迁移风险

本次新 H1 development 中，K3 全体定位正确率为 0.8327；在 C3 / partial-overlap / SNR 0 dB / CNR 20 dB 的 512 个场景中仅为 0.0469（24/512）。该组下 T0/T1/T2/T3 joint success 分别为 0.0586/0.0547/0.0645/0.0625，样本量不足以支持小差异的强结论。按 K3 离线正确/错误分层，T1 未优于 T0；T2 的全体错误教师组 joint 仅略高于 T0，Doppler RMSE 则较差。该标签只用于事后分层，不进入输入、损失权重或模型选择。

因此，G4 只说明 R5-C 审计响应含有限任务相关信息，不能解释为教师全场景可靠，也不能推导学生已吸收教师知识。此 pilot 未证明 T1 或 T2 在物理真值之外产生最终任务收益。

## 阶段决定与后续边界

按冻结协议停止当前 K3/K2-local 教师内容分支；不启动其多 seed 确认、教学时机搜索、可靠性权重搜索、R5-D 完整机制比较或 final blind。H-TEACH-002 在本 pilot 范围内记为 NOT_SUPPORTED。T3 的描述性结果可以作为后续问题线索，但不能替代教师增益证据。

项目总体判断为 **REFINE**：教师内容 pilot 已有明确负结果，且 C3 绝对虚警仍未解决；研究总纲 v1.3 的长期 GMTI 学生目标仍值得继续。下一项唯一优先任务为按已批准协议开展 R6-A 同场景检测/参数估计基线，先定位条件内可分性、固定阈值迁移和实际候选估计误差的来源。R6-A 不等待 R5-D，也不复用本 pilot 的评价 IDs。

## 可复现记录

- 训练与逐臂最佳/恢复检查点：`experiments/D2-A-R5/R5-C-teaching/runs/seed_303/`
- G0/G1、数据与分割审计：`preflight.json`、`g1_overfit_sanity.json`、`data_generation.json`、`scene_manifest.csv`、`seed_manifest.csv`、`split_manifest.json`、`training_pair_plan.npz`
- 逐场景预测：`predictions/{T0,T1,T2,T3}/`
- 完整指标、配对区间与机器生成分析：`metrics.json`、`analysis.md`
- 清理审计：`cleanup.json`；研究总纲保持不变，场景真值、seed/scene 清单、pair plan 和所有结果继续保留。
