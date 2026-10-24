# D0/D1 任务级教师引导蒸馏实施计划

> **For agentic workers:** Required sub-skill: `superpowers:executing-plans`. Steps use checkbox syntax and each task ends with a separate commit.

**Goal:** 建立可信的三通道距离压缩数据与 APX15 频谱教师适配，训练并评估直接输出 H0/H1、距离和连续 Doppler 的任务级学生及 A/B/C/D 蒸馏对照。

**Architecture:** D0 用确定性仿真和 APX15 `ColFFTTranspose` 适配器生成独立真值/教师字段；D1 用感受野覆盖 130 脉冲的轻量二维 TCN 和训练期频谱辅助头。教师只参与训练及基线评估，最终推理模型移除教师辅助头。

**Tech Stack:** Python、NumPy、PyYAML、PyTorch CUDA、C++11、FFTW3、pytest、Matplotlib。

**Spec:** `docs/superpowers/specs/2026-10-24-d0-d1-task-level-distillation-design.md`；用户授权路线见 `/home/shy/.codex/attachments/ba0548b9-1e97-4047-a1fc-eccf13ebd1a0/pasted-text-1.txt`。

## Global Constraints

- 保留 P0/P1 历史阶段报告与失败记录；D0/D1 是新主线，不改写旧结论。
- `PRF=1300 Hz`、`N=130`、三通道和固定 32 距离单元窗口；H0 与 H1 都有复高斯背景。
- 真值、教师硬估计、教师软响应、学生输出分别存储；教师不读真值、不按真值裁剪。
- A/B/C/D 最终推理网络相同，总优化步数和数据相同；正式测试集不用于选模或校准。
- 训练必须用 CUDA，禁止 CPU 回退；按种子和方法顺序运行，OOM 先减 batch size。
- 不把 APX15 正式大 BIN、可再生训练数据提交 Git。
- 不运行或改动 CoreX 正式处理链；D0 仅适配所需慢时间 FFT 子链。

## Review Focus

- H0/H1 的噪声/归一化是否产生标签捷径；用匹配先验统计测试固定在 D0 生成器测试。
- Nyquist 周期边界和连续 Doppler 符号是否一致；用频谱轴与模型解码测试固定在教师/模型测试。
- 固定监视窗是否避免真值距离泄漏；用输入与标签独立性测试固定在数据生成测试。
- 教师对 H0 的错误检测、H1 的估计错误是否保留；用字段独立性和故意困难样本测试固定在教师测试。
- 所有训练组是否同预算且最终推理图不含教师头；用 runner 预算/导出测试固定在训练接口测试。

---

### Task 1: 科研路线设计与文档调整

**Files:**
- Create: `docs/superpowers/specs/2026-10-24-d0-d1-task-level-distillation-design.md`
- Create: `docs/research/task_level_progressive_distillation.md`
- Create: `docs/research/D0_D1_design.md`
- Modify: `AGENTS.md`, `README.md`, `docs/research_proposal.md`, `docs/research_plan.md`, `docs/experiment_protocol.md`, `docs/progress.md`, `docs/decisions.md`

**Interfaces:** 采用本计划 Spec 的 D0/D1 输入、输出、模型、数据划分、教师适配和阶段门；不改动历史实验文件。

- [x] Step 1: 读取本地实验状态、P0/P1 阶段审查、APX15 CMake/FFT 源码及全部用户路线要求。
- [x] Step 2: 写入上述设计文档和分阶段研究主线，明确历史结果与新路线边界。
- [x] Step 3: 核对文档中的路径、阶段决定和实验门槛与现有结果一致。
- [x] Step 4: 运行 `git diff --check`、路径检查及既有完整测试 `conda run -n AISAR pytest -q`（67 passed）。
- [x] Step 5: 仅暂存路线文档并提交 `docs: define task-level distillation research route`。

### Task 2: 冻结 D0 仿真和教师正确性协议

**Files:**
- Create: `experiments/D0-E001/config.yaml`, `experiments/D0-E001/protocol.md`
- Modify: `docs/experiment_protocol.md`, `docs/progress.md`

**Interfaces:** Freeze D0 signal parameters, H0/H1 generation, channel geometry, point-spread model, APX15 FFT adapter behavior, correctness datasets, and acceptance checks before simulator implementation.

- [x] Step 1: Specify waveform/matched-filter verifier, noise and amplitude matching, fixed range window, continuous range/Doppler coverage, teacher fields and independent seeds.
- [x] Step 2: Define numeric D0 correctness samples and expected physical/teacher tolerances from analytic resolution and synthetic checks.
- [x] Step 3: Save config/protocol; document that the supplied package is CoreX-only and D0 adapts only the required FFT subchain.
- [x] Step 4: Review all parameters against the approved design; run `git diff --check`.
- [x] Step 5: Commit `docs: preregister D0 simulation and teacher checks`.

### Task 3: D0 三通道仿真与 APX15 Doppler FFT 适配

**Files:**
- Create: `src/radardistill/simulation/d0.py`
- Create: `src/radardistill/teachers/apx15_doppler.py`
- Create: `scripts/run_d0_e001.py`
- Create: `GMTI_APX15_CoreX_Int16_20261014/tools/radardistill_doppler_teacher.cpp`
- Track required adapter header: `GMTI_APX15_CoreX_Int16_20261014/include/transposeFFT.hpp`
- Test: `tests/test_d0_simulation.py`, `tests/test_apx15_teacher.py`
- Consume: `experiments/D0-E001/config.yaml`, `experiments/D0-E001/protocol.md`

**Interfaces:**
- `generate_d0_batch(count, seed, snr_db, presence_ratio, config) -> D0Batch`
- `APX15DopplerTeacher(config).predict(iq) -> TeacherBatch`; the teacher must not receive truth or the simulator's known noise-power metadata.
- `D0Batch` 保存 IQ、presence/range/Doppler 真值、初相、幅度、噪声和采样元数据。

- [x] Step 1: 写测试钉住 `[B,3,130,32]` 形状、种子复现、H0/H1 共享噪声机制、固定窗口与随机相位/连续距离/Doppler。
- [x] Step 2: 运行 `conda run -n AISAR pytest -q tests/test_d0_simulation.py`，确认因接口缺失失败。
- [x] Step 3: 实现 D0 生成器及理想匹配滤波 PSF。
- [x] Step 4: 写/运行小规模 LFM 发射—匹配滤波对照，检查简化 range PSF 和真值映射。
- [x] Step 5: 写测试钉住生产 FFT 居中/符号、三通道功率响应、教师硬/软字段和 H0 假警保留。
- [x] Step 6: 实现 APX15 FFTW3 适配共享库和 Python teacher wrapper；只添加适配器与其必要头文件，不暂存其他包内容。
- [x] Step 7: 运行 `conda run -n AISAR pytest -q tests/test_d0_simulation.py tests/test_apx15_teacher.py`（12 passed）。
- [x] Step 8: 运行全量 `pytest`（79 passed）、`compileall` 和 `git diff --check`。
- [x] Step 9: 仅暂存 D0 生成器、教师、运行器、必要 APX15 头文件/适配器及测试，提交 `feat: add D0 simulator and APX15 Doppler teacher`。
- [x] Step 10: 运行冻结的 D0 correctness/teacher 数据，保存指标、环境、CSV 和 `analysis.md`；完整数组只流式生成，不持久化；12/12 门通过。
- [ ] Step 11: 以代码提交 SHA 记录正式 D0 结果，单独提交 `research: validate D0 simulation and APX15 teacher`。

### Task 4: 冻结 D1 数据、训练预算和验收协议

**Files:**
- Create: `experiments/D1-E001/config.yaml`, `experiments/D1-E001/protocol.md`
- Modify: `docs/experiment_protocol.md`, `docs/research_plan.md`

**Interfaces:** Freeze the D1 signal model from D0, model structure, A/B/C/D stage budget, seeds, calibration/test splits, metrics, gates, and exact CUDA configuration before training code is run.

- [x] Step 1: Set independent train/validation/pilot/calibration/final-blind seeds and train/test SNR ranges; H0/H1 class balance and dataset counts.
- [x] Step 2: Fix the TCN receptive field/width, loss definitions, optimizer, total epochs/updates, batch size and A/B/C/D stage schedules.
- [x] Step 3: Set teacher, FFT/interpolation and profile-ML baselines; define score calibration and ROC/Pd/Pfa intervals.
- [x] Step 4: Freeze range/Doppler error definitions, unconditional vs conditional reporting, bootstrap and G1/G2/G3 numerical criteria.
- [x] Step 5: Add narrow `.gitignore` rules for regenerable D1 train/validation arrays; retain model weights, reports, metrics and predictions.
- [x] Step 6: Review protocol for test-set leakage and matched-budget fairness; run `git diff --check` and commit `docs: preregister D1 task-level distillation experiment`.

### Task 5: D1 学生、损失、训练器和 pilot

**Files:**
- Create: `src/radardistill/students/task_student.py`
- Create: `src/radardistill/distillation/task_losses.py`
- Create: `src/radardistill/training/d1_runner.py`
- Create: `scripts/run_d1_e001.py`
- Consume: `experiments/D1-E001/config.yaml`, `experiments/D1-E001/protocol.md`
- Test: `tests/test_d1_task_student.py`, `tests/test_d1_runner_contract.py`

**Interfaces:**
- `TaskStudent.forward(iq_ri) -> {presence_logit, range_logits, range_offset, doppler_vector, spectrum_logits}`。
- `task_loss(outputs, truth, config) -> dict[str, Tensor]` 分开返回检测、range、Doppler、谱辅助和总损失。
- `export_inference_model(training_state) -> TaskStudentInference` 移除谱辅助头。
- `run_d1_e001(config, output, device='cuda')` uses only CUDA, executes one named strategy at a time, and records matched optimizer steps and CUDA environment.

- [x] Step 1: 写测试钉住 130 脉冲有效感受野、张量轴/输出形状、H0 参数损失屏蔽、Doppler 圆周损失、有限梯度及无教师头导出。
- [x] Step 2: 写 runner 测试钉住 CUDA fail-fast、匹配更新数、独立 seed 派生、有限样本阈值和多 SNR 验证缓存字段。
- [x] Step 3: 运行定向测试，先因 D1 API 缺失而失败（TDD red）。
- [x] Step 4: 实现最小二维深度可分离 TCN、四类损失、阶段 schedule、CPU memmap 批量装载、验证选模和推理 checkpoint 导出。
- [x] Step 5: 定向测试 20 passed；全量 `conda run --no-capture-output -n AISAR pytest -q` 为 100 passed；`compileall` 与 `git diff --check` 通过。
- [x] Step 6: 仅暂存 D1 模型/损失/runner、入口、测试和对应状态文档，提交 `feat: implement D1 task student and CUDA pilot trainer` (`bc0d4d8`)，使 pilot 使用可追踪代码 SHA。
- [x] Step 7: 运行独立 pilot seed 的 A/B/C/D 四组，顺序使用 RTX 3050；检查学习诊断、收敛、物理一致性和峰值显存。
- [x] Step 8: 保存并单独提交 pilot 记录；保留低 SNR 负结果。筛选仅允许进入正式三种子实验，不代表盲测或 G1/G2/G3 通过。

### Task 6: D1 三种子正式训练、盲测和阶段审查

**Files:**
- Create/update: `experiments/D0-E001/{metrics.json,result.csv,analysis.md}`
- Create/update: `experiments/D1-E001/{metrics.json,result.csv,training_history.csv,analysis.md,rmse_snr.png,roc.png}`
- Create: `docs/stage_reviews/D0_review.md`, `docs/stage_reviews/D1-E001_review.md`
- Modify: `README.md`, `docs/progress.md`, `docs/decisions.md`, `docs/research_plan.md`

**Interfaces:** Consume frozen config and trained inference-only students; use independent blind test and separate H0 threshold-calibration data.

- [ ] Step 1: Freeze all code/config/model selection from development validation before generating/opening final blind predictions.
- [ ] Step 2: Train A/B/C/D for three independent seeds, sequentially on CUDA, same total update budget per run.
- [ ] Step 3: Evaluate the APX15 adapter, FFT/interpolated baselines, continuous profile-ML and all student groups on common independent blind samples.
- [ ] Step 4: Compute detection metrics/intervals, per-SNR range/Doppler errors, unconditional/conditional errors, per-seed stability, paired strategy comparisons and resource records.
- [ ] Step 5: Produce plots, stage reports and one GO/CONDITIONAL GO/STOP decision using frozen gates.
- [ ] Step 6: Run full tests and audit committed paths; confirm no APX15 bulk BIN or regenerable training arrays are staged.
- [ ] Step 7: Commit results and stage review as `research: report D0 and D1 teacher-distillation results`.
