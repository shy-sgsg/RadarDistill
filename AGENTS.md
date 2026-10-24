# RadarDistill 执行规则

本仓库长期使命和科学边界以 docs/RESEARCH_CHARTER.md 为准。当前阶段以 docs/CURRENT_STATUS.md 和 docs/RESEARCH_ROADMAP.md 为准；详细历史与冻结阶段门见对应实验 protocol 和审查报告。不得用本文件取代研究总纲。

## 开始任务前

- 阅读 docs/RESEARCH_CHARTER.md、docs/CURRENT_STATUS.md、docs/RESEARCH_ROADMAP.md、docs/research/ASSUMPTION_REGISTER.md，以及当前任务相关的阶段报告、protocol、配置和 decisions.md。
- 按 docs/governance/CODEX_TASK_PROTOCOL.md 完成研究主线对齐。常规且已批准的小范围实现无需暂停确认；改变核心使命或冻结的科学门槛须先获得用户批准。
- 检查当前分支、提交、工作区修改、未跟踪数据和相关代码；保留用户文件，不以对话记忆代替仓库证据。

## 科研诚信与实验

- 先定义问题、可辨识条件、假设、强传统基线、数据切分、预算、指标和停止条件，再实现模型或生成盲测数据。
- 区分物理真值、教师硬标签、软响应和学生输出；教师错误不能覆盖真值。
- 训练、验证、校准和最终测试应独立。用于选架构的数据不能重标为最终盲测；冻结盲测前固定配置、模型、统计门和阈值。
- 周期频率/相位使用回绕误差。报告样本量、独立种子、置信区间、失败结果和适用边界。
- 不预设 AI 优于传统方法。FFT/CFAR 等混合前端必须与匹配非学习前端比较，并区分前端贡献和学生贡献。
- 训练必须使用用户要求的可见 CUDA，禁止静默 CPU 回退；依序训练、控制显存，OOM 时先降低 microbatch 并维持有效 batch 和更新预算。
- 速度结论必须比较同硬件、同输入边界的完整处理链。未测量的实时性能、未运行的实验不得写成结果。
- 研究代码与既有生产 GMTI 算法隔离；没有必要证据和批准，不改生产处理链。

## 代码、记录与 Git

- 只做用户请求或完成请求所必需的改动；不要覆盖历史负结果或未提交用户数据。
- 实验放在 experiments/<阶段>-E编号/，保存 protocol、config、metrics、result 和 analysis；入口放 scripts/，代码放 src/radardistill/，行为/物理约束测试放 tests/。
- 每完成一个可审查工作包建立小而清晰的提交。仅暂存检查过的路径；大型可重建缓存、外部源码包和无关二进制不得进入提交。提交前检查差异和文件大小。
- 未获当前任务明确授权，不推送或重写共享分支历史。
- 研究判断更新 docs/progress.md；路线或假设变化更新 docs/decisions.md、路线图和假设登记。

## 审查和完成

所有有科研意义的阶段均需通过工程正确性、实验科学性和研究主线审查。按 docs/governance/CHATGPT_REVIEW_PROTOCOL.md 给出 CONTINUE、REFINE、REDIRECT 或 STOP，以及证据、结论边界、停止条件和唯一下一项优先任务。阶段门不因已写入旧计划而免于重新审视。
