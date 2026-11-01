# D2-A-R6-E 架构筛选实施计划

**状态：** 已授权执行；protocol/config冻结及提交后才生成正式训练数据。
**设计依据：** `docs/superpowers/specs/2026-11-01-d2-a-r6e-architecture-screen-design.md`
**适用范围：** R6 P4 的单seed matched architecture screen；不改变研究总纲、R5-C或R6-A至R6-D冻结结果。

## 冻结目标

在新独立D2-A场景中，以R6-D J1-J为参考，比较三个可学习、无固定物理前端的E1 truth-only学生：J1-J、复数公共相位等变CE-TCN、保留慢时间—距离token并交替轴向交互的RAX-Transformer。每臂共用seed 606、scene ID、H0/H1 batch配额、配对采样顺序、有效batch 64、FP32、AdamW计划、验证/checkpoint间隔和51,200 updates；模型按序驻留CUDA。51,200为筛选预算，不是模型收敛上限；若后续实验需要更长预算，必须预注册匹配扩展。

## 执行步骤

### 1. 协议、资源和研究对齐

- [ ] 建立 `experiments/D2-A-R6/R6-E/{protocol.md,config.yaml}`，冻结三种模型定义、参数匹配目标、场景划分、update/exposure预算、checkpoint规则、统计指标、相对任务门、C3绝对Pfa门和停止条件。
- [ ] 全仓scene manifest审计已预演：22个manifest/selected-ID文件、884,072个既有ID；候选train/validation/calibration/verification/development/G1 ID总计302,360，既有及分割间重叠均为0。冻结配置须保留该种子区间和审计逻辑。
- [ ] 新增R6-E task alignment，更新当前状态/路线图、README与R6阶段执行清单的阶段位置；保留所有未跟踪用户数据并只暂存R6-E准确路径。
- [ ] 提交并推送预注册包到 `main`，不包含训练缓存或ChatGPT审查请求。

### 2. 模型与行为测试（先测试再实现）

- [ ] 建立`tests/test_d2a_r6e_architectures.py`：输入/输出shape、有限值、连续Doppler周期表示、范围头和presence头契约、参数计数规则、前反向梯度。
- [ ] 测试CE-TCN对输入公共相位旋转的中间复特征等变性及读出任务输出不变性；测试无复数偏置、径向激活/尺度归一化保持性质。频移等变不是本模型的结构保证，只做独立评价诊断。
- [ ] 测试RAX-Transformer保留完整`130×32`token至两轴交互结束，位置偏置维度正确；不将其描述为相位不变或Doppler频移等变。
- [ ] 实现两个候选并输出与`r6d_loss`、`decode_task_outputs`兼容的相同任务字典；按推理参数（复数权重按两个实自由度计）验证每候选与J1-J相差不超过±20%。只允许一次协议内容量配平调整。
- [ ] 运行定向CPU行为测试和CUDA前向/反向/显存探测；先通过G0/G1后才生成正式训练/验证场景。

### 3. CUDA训练与独立评价

- [ ] 实现独立R6-E入口及owner manifest；沿用R6-D已验证的paired-plan、truth损失、输出解码、阈值、exact interval和paired bootstrap实现，避免修改R6-D冻结代码。
- [ ] G0重新核验`main`与远端提交、protocol/config SHA、全仓场景ID互斥、可用磁盘、RTX 3050 Laptop `cuda:0`、FP32和实际可用显存；禁止CPU回退。
- [ ] G1对三臂依序检查有限输出、有限非零梯度、显存和同一16-pair/200-update overfit条件；失败候选先定位实现错误并保留证据，不进入长训。
- [ ] 仅在G1通过后生成51,200 train pairs和5,000 validation pairs；train/validation IQ临时缓存估算11,222,016,000 bytes，评价按ID分块生成。三臂使用同一scene顺序和有效batch64；初始microbatch16，OOM时先降至8或4并用累积保持有效batch。
- [ ] 依序训练seed606三臂各51,200 updates；cosine LR贯穿全预算；每400 updates记录history/validation，每800保存恢复checkpoint，不逐optimizer update写日志。验证目标用于本臂checkpoint选择，H0 verification/H1 development不参与选择。
- [ ] 每臂完成后卸载模型并清空CUDA缓存再启动下一臂。记录参数量、数据访问/重复遍历、optimizer updates、完整训练/验证loss、参数估计曲线、墙钟时间与峰值显存。
- [ ] 冻结checkpoint后才运行全新H0 calibration/verification各120,000 CPI、H1 development 6,144 CPI。各臂单独在calibration设全局Pfa=0.05阈值，成对比较相同development scenes；不生成blind。

### 4. 结果、清理和提交

- [ ] 生成三臂逐scene预测、`metrics.json`、训练曲线和SNR/条件图、`analysis.md`、内部阶段审查；按协议报告overall、C2+C3、C3及困难格Pd/Pfa/joint/range/Doppler/AUC、bootstrap区间、低虚警门、物理一致性和训练曲线趋势。
- [ ] 明确区分：可学习架构贡献、CE-TCN的公共相位归纳偏置、RAX轴向序列建模；不把单seed开发筛选称为跨seed确认或最终盲测。
- [ ] 更新`docs/CURRENT_STATUS.md`、`docs/RESEARCH_ROADMAP.md`、`docs/research/ASSUMPTION_REGISTER.md`、`docs/progress.md`、`docs/decisions.md`、README及R6执行清单；不改研究总纲和历史协议/负结果。
- [ ] 仅按R6-E owner manifest清理重建IQ缓存；保留协议、配置、scene manifests、checkpoint、训练history、预测、指标、图和报告；确认项目总占用低于20 GiB且用户未跟踪文件未受影响。
- [ ] 检查差异与目标测试结果，分路径提交并推送`main`。R6 P0–P7未全部完成前不生成ChatGPT审查请求。

## 验收边界

R6-E是单seed代表性架构筛选。通过相对joint筛选门不自动意味着满足C3低虚警绝对门；C3门失败不抹去可辨识任务或机制证据，但不得宣称合格低虚警检测器。任一候选末端仍改善时只报告“51,200预算内未观察到验证平台”，不推断能力上限；本轮不得单臂追加更新。
