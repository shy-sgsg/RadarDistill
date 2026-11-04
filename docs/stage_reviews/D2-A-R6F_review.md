# D2-A-R6-F 阶段报告

**内部阶段结论：REFINE**
**实验协议：** `experiments/D2-A-R6/R6-F/protocol.md` v1.0.2
**实际指标：** [`metrics.json`](../../experiments/D2-A-R6/R6-F/metrics.json)
**分析与图：** [`analysis.md`](../../experiments/D2-A-R6/R6-F/analysis.md)、[`full_chain_comparison.png`](../../experiments/D2-A-R6/R6-F/figures/full_chain_comparison.png)

## 研究主线

本阶段比较相同三通道 IQ 场景上的传统检测/估计、CE-TCN、J1-J，以及神经检测与传统精估混合链。它服务于总纲中直接 GMTI 检测与参数估计、知识迁移和完整链性能目标；不是独立的传统算法优化项目。

## 执行与证据

- G0、G1通过；RTX 3050 Laptop `cuda:0` 执行神经链，模型按链路顺序加载。
- 使用120,000 H0 calibration、120,000独立H0 verification及6,144 H1 development；无final blind。
- CE-TCN原生链 B 的 H1 Pd 为0.94743，C2+C3 joint success为0.85807，Doppler RMSE为18.72 Hz；相对A的joint增益为+0.57813（99%配对区间 `[+0.56141,+0.59397]`）。
- 六条链的全局 H0 Pfa上界均通过0.055；C3六格familywise门六条链全部失败。没有完整链获得绝对低虚警资格。
- CE检测＋profile-ML链 C/F 保持较高Pd，但全H1 Doppler RMSE约149.61 Hz、C2+C3 joint约0.722；其任务保护失败。传统检测＋CE参数链 D 的检测受 B3 限制，未改善joint成功。
- 运行成本在该协议的384场景、batch16基准上测量；CPU/GPU路径不一致，不作实时性声明。

## 判断与停止条件

本次证据支持 CE-TCN 是受测完整链中最强的开发任务候选，但不支持其为合格低虚警检测器，也不支持 C/F 的 profile-ML 估计链。R6-F的单seed开发结果不证明跨seed泛化。停止本轮追加链路、分数和阈值搜索；不启动正式blind。

R6-F没有开展教师蒸馏，不能证明哪种知识形式提高学习或收敛。R5-C T0–T3负结果保留。可按已授权的R6-G范围制定一项独立、匹配 truth-only 与 truth-assisted 的教师知识pilot，并把C3错误传播列为停止门。
