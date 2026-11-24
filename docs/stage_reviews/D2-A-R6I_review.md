# D2-A-R6-I 阶段审查

**阶段判断：** FAIL  
**代码提交：** `c26a24f63e1bbfd874d7e7ea03dd974fceecd9c8`  
**研究总纲：** `docs/RESEARCH_CHARTER.md` v1.3（本阶段未修改）。

## 工程与实验科学性

使用AISAR环境和RTX 3050 Laptop `cuda:0`对冻结R6-H模型作FP32推理；模型顺序驻留，未训练或生成final blind。新H0拟合、阈值校准、H0验证和H1开发scene IDs分离，完整结果和逐格指标见[`R6-I分析`](../../experiments/D2-A-R6/R6-I/analysis.md)及[`metrics`](../../experiments/D2-A-R6/R6-I/metrics.json)。

首次指标汇总遇到拟合split观测字段缺失，恢复过程复用既有四split预测缓存、没有重新生成预测。修正汇总代码后，12项定向测试通过；从缓存复算的13个指标组与已记录指标完全一致。运行恢复范围及耗时口径见 `runtime_recovery.json` 和分析报告。

## 阶段结论与主线影响

冻结P7再准入结果为 **FAIL**。stop this score correction; do not generate final blind。本结果只涉及固定CNN-SSM检测分数，不证明任何知识蒸馏收益，也不把整体尺度诊断推广到真实硬件。R6-I没有生成盲测。

## 下一步

保持研究总纲、R6-H历史结果及R6-I冻结门不变；停止本次未通过的单斜率分数校正，不在同一开发数据上搜索其它变换。R6-J4已有独立设计且属于获批R6路线，后续按其自己的协议、前置门和停止条件开展；R6-I失败不被解释为J4结果。整条R5-C/R6授权路线完成后，再编写唯一的ChatGPT综合审查请求。
