# D2-A-R7-B：DPCA教师物理正确性与实现一致性协议

**版本：** 0.1
**状态：** 冻结；运行前
**研究总纲：** `docs/RESEARCH_CHARTER.md` v1.4

## 1. 目标与不做事项

核对D2-A当前固定通道(0,1)分数慢时间补偿及差分代码是否符合其共址Tx/双程路径仿真假设；分开测静止杂波、热噪声、移动目标和混合观测；复算E001记录的15.406 dB指标定义；在AISAR CUDA上核对现有CPU和CUDA归一化教师响应。测试不改写历史教师、不按结果修改阈值或算法，也不据合成场景推断真实硬件表现。

## 2. 冻结配置与样本

- 雷达配置沿用`experiments/D2-A-E001/config.yaml`：3通道、130脉冲、PRF=1300 Hz、32距离单元。
- 对新的分量诊断生成1,280个互相独立的基础场景，分为5个分离目标条件，每格256个：SNR/CNR=(10,10)、(10,0)、(10,20)、(0,10)、(20,10) dB。`scenario_id=separated`；H0/H1来自同一对偶场景，所有组件分别保存于当前batch内并立即汇总，不落盘IQ。
- 新场景种子从87,320,101开始，每个条件使用单独10,000 seed offset；与E001 3–4百万校准/开发seed块及盲测保留范围分离。每次最多生成16场景。
- 另生成解析单散射体与4散射体的无噪声有限CPI用例。所有这些只作算法诊断，不替代随机背景统计。

## 3. 公式与指标

沿用冻结公式：

```text
delay_s = - dot(r_rx[1] - r_rx[0], platform_velocity) / (2 * ||platform_velocity||^2)
shift_samples = delay_s * PRF
aligned_ch1 = ifft(fft(ch1) * exp(+j * 2*pi*fftfreq(N) * shift_samples))
residual = ch0 - aligned_ch1
```

分量功率比定义为`mean(|residual|^2)/mean(|ch0|^2)`；抑制量为`-10*log10(power_ratio)`。分离目标保留量沿用同样分母，只作未缩放差分功率比，允许大于1，不能称为能量百分比。预/补偿后复相干为`sum(ch0*conj(ch1))/sqrt(sum|ch0|² sum|ch1|²)`，同时报告幅值、相位、通道幅值比和有限CPI residual。

对1,280个新独立场景用按场景重采样的bootstrap报告中位数95%区间。解析单散射体无随机样本区间。没有额外设置任意的CCR合格门；如果静止散射体的残差未随几何补偿降低，或方向/通道互换行为不符合公式，则记录为失败并暂停教师资格判断。

## 4. CPU/CUDA实现核对

NumPy `dpca_difference`与CUDA `k2_response_cuda`均以complex128做几何移位/FFT，再将归一化响应输出为float32。AISAR环境要求实际可见`cuda:0`；GPU不可用时脚本失败，不切换CPU伪称CUDA。对同一H0/H1批次分别计算NumPy `k2_response`与CUDA版本响应，记录最大绝对差、平均绝对差、总质量误差及GPU峰值显存。CPU/CUDA同一算法只能验证数值一致性，不能证明模型/物理假设正确。

## 5. 历史15.406 dB复算

历史E001脚本在5个`separated`条件共1,280场景上计算：

```text
clutter_after = dpca_difference(clutter_iq)
per_scene_power_ratio = mean(|clutter_after|²) / mean(|clutter_iq[channel 0]|²)
suppression_db = -10*log10(per_scene_power_ratio)
reported_value = median(suppression_db across 1280 scenes)
```

该数值只用纯`clutter_iq`组件，不含热噪声或目标；目标保留比则只用`target_signal_iq`分量。新场景采用相同公式和条件数量做独立复核；旧值与旧数据不修改。

## 6. 交付和解释边界

保存`config.yaml`、本协议、流式运行脚本、精简`metrics.json`、`analysis.md`和针对方向/解析场景的测试。不得持久化IQ、RD图或大数组。报告分别回答数学公式、数值实现、组件行为、有限CPI、历史CCR复算及当前教师可用范围。结论只适用于冻结研究仿真，不代表生产系统或实测数据。
