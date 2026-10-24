# 当前研究状态

**更新日期：** 2026-10-25
**工作分支：** `main`
**审查基线提交：** `3695e163df2b1c7b145f01301e0fa16a750a7f6e`（2026-10-25 本轮核对时的 `main` HEAD；当前版本 SHA 可用 `git rev-parse HEAD` 查询）
**最新完成工作：** D1-E002 独立盲测与阶段审查；D1-R1 诊断审查已补齐。
**阶段判断：** D1-E001 原 GO → D2 保持；D1-E002 的 AWGN Doppler 结构搜索 STOP / REDIRECT。

## 主要发现

E002 的 S2 lag-one 复相干学生在 0/5/10/20 dB 的等权 Doppler RMSE 为 23.352 Hz，较 S0 改善 10.86%，但低 SNR Pd 非劣门失败；四倍 FFT 抛物插值、APX15 和有限网格 profile-ML 为 0.104、0.122、0.100 Hz。S2 不具备已证明的传统估计竞争力，且未通过公共相位旋转/频移一致性检查。

D1-E001 的 C/D 满足原冻结联合指标 G2；没有蒸馏收敛加速证据，多策略区间未作家族错误率调整，D 的效果量受 seed 1 影响。学生低 SNR 检测与高精度 Doppler 仍是主要问题。

## 唯一优先任务

准备 **D2-A 静止杂波单因素实验协议和传统基线/可辨识性验证**。先冻结 H0/H1 生成、静止杂波模型、目标条件、DPCA/杂波抑制与检测基线、校准 Pfa、预算、种子及独立盲测规则；协议未冻结前不训练、不生成新 blind。下一阶段设计状态：D1-E001 阶段门允许进入 D2，D2-A 具体协议尚未冻结。

## 关键材料

- 总纲/路线：[研究总纲](RESEARCH_CHARTER.md)、[动态路线图](RESEARCH_ROADMAP.md)、[假设登记](research/ASSUMPTION_REGISTER.md)
- 审查：[D0/D1 E002 后总审查](stage_reviews/D0-D1_post_E002_review.md)、[D1-R1 Doppler 诊断](stage_reviews/D1_R1_doppler_diagnosis.md)、[D1-E001](stage_reviews/D1-E001_review.md)、[D1-E002](stage_reviews/D1-E002_review.md)
- 数据与结果：`experiments/D0-E001/`、`experiments/D1-E001/`、`experiments/D1-R1/`、`experiments/D1-E002/`
- 环境：AISAR / PyTorch 2.8.0+cu128 / CUDA 12.8 / RTX 3050 Laptop GPU，4 GiB；正式训练必须使用可见 CUDA。
