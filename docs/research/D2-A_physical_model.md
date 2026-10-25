# D2-A 三通道静止杂波物理模型

**范围：** D2-A-E001 的受控合成开发模型；不是完整 APX15 回波链，也不是实测地杂波模型。

## 几何和信号

采用局部坐标 `x=north, y=east, z=up`。平台起点为 `(0,0,6000) m`，匀速沿 `+x` 方向飞行，速度 `60 m/s`。脉冲时刻 `t_n=n/PRF`，`PRF=1300 Hz`、`N=130`。三个接收相位中心取现有四通道实际场景 XML 中的前三个坐标；研究距离压缩接口为 `3×130×32` complex IQ，距离单元为 `c/(2B)=2.99792458 m`。

对位置为 `q_k` 的静止地面散射点，发射相位中心和第 `m` 个接收相位中心随平台运动：

\[
R_{tx,k}[n]=\|q_k-p[n]-b_{tx}\|,\qquad
R_{rx,m,k}[n]=\|q_k-p[n]-b_m\|,\qquad L_{m,k}[n]=R_{tx,k}[n]+R_{rx,m,k}[n].
\]

逐脉冲信号项为：

\[
c_m[n,r]=\sum_k \alpha_k G(\theta_k[n])\,\mathrm{sinc}(r_{m,k}[n]-r)\exp\!\left(-j\frac{2\pi}{\lambda}L_{m,k}[n]\right),
\quad \lambda=\frac{c}{f_c}.
\]

频率从传播路程变化求得，不手工把静止地物设为零 Doppler：

\[
f_{d,m,k}[n]\simeq-\frac{PRF}{\lambda}\big(L_{m,k}[n+1]-L_{m,k}[n]\big).
\]

回波分量为 `x_m[n,r]=c_m[n,r]+s_m[n,r]+w_m[n,r]`。H0 是 clutter+noise；配对 H1 使用同一组散射点、散射系数和噪声，仅增加目标回波，故 `H1−H0=target_signal_iq`。三个通道共用同一散射点集合和反射系数，通道相干关系只由双程路径和相位中心几何决定。

## 随机量、功率和距离响应

- 第一轮散射系数 `α_k` 为 CPI 内固定的独立 `CN(0,1)` 点散射仿真系数；散射体位置在配置的方位波束和 32 个距离单元内抽样。每个场景只用一个共用地物场景生成三通道。
- 波束采用方位向 Gaussian 主瓣，载频 `15.8 GHz`、带宽 `50 MHz`；距离压缩响应为 D0 兼容矩形带宽 sinc。距离压缩之后才合成回波，不模拟 ADC、匹配滤波窗、旁瓣或真实滤波器复响应。
- 通道噪声为独立圆对称复 AWGN，`E|w|²=σ²`，实部、虚部方差各 `σ²/2`；噪声通道协方差为单位阵。
- 目标输入 SNR 沿用 D0 定义，即参考目标幅度平方除以复噪声功率、在连续距离 PSF 峰值处定义。杂波按完整 CPI 的 `mean(|clutter|²)/mean(|noise|²)` 归一到请求的 CNR。另报告完整 CPI 的组件功率 `SNR=Ps/Pn`、`CNR=Pc/Pn`、`SCNR=Ps/(Pc+Pn)`；目标只占部分距离门，故请求 SNR 与完整 CPI 平均 SNR 不相同。

## 设计参数与来源边界

系统参数取值和文件证据见 [`D2-A_system_parameters.md`](D2-A_system_parameters.md)。`PRF=1300 Hz`、`N=130`、`fc=15.8 GHz`、`B=50 MHz`、平台 `60 m/s` 和接收通道坐标来自本地工程配置或兼容研究场景。`N=130` 保留 D0/D1 对比，当前生产 XML 的处理脉冲数为 128。三通道是四接收通道配置的前三个研究子集，不代表完整生产阵列。

发射相位中心在已检查配置里没有明确坐标，模型暂取平台参考原点共址。平台高度、平地、单束 `2.28°` Gaussian 方位主瓣、独立点散射统计和理想通道校准是研究假设；没有真实地物散射统计、真实噪声协方差、旁瓣、地形遮挡、通道误差或运动补偿残差。这些假设限定结论只适用于受控仿真。

## 物理验证和运行证据

`tests/test_d2a_simulation.py` 中 T1–T12 检查 D0 零杂波退化、无噪声散射点距离/传播相位/慢时间 Doppler/跨通道相位、平台速度和波束变化、显式 Tx/Rx 路程、配对 H0/H1、功率定义、固定种子复现、目标参数不改变 H0 背景、坐标与公共相位对称性、距离迁移和 H0/H1 组件结构。T2 与 T7 额外验证了跨通道相位差解析式和生成场景的 CNR 标定。

完整开发实验保存于 [`experiments/D2-A-E001/`](../../experiments/D2-A-E001/)；复现命令为：

```bash
conda run --no-capture-output -n AISAR python scripts/run_d2a_e001.py \
  --config experiments/D2-A-E001/config.yaml \
  --output-dir experiments/D2-A-E001 --batch-size 16
```

全部指标均来自合成开发样本；不得称为真实 GMTI 验证。
