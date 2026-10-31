# D2-A-R6-C 分析

- 提交：`a34feea10822ac4ef4dfbe9b64771826aa1e80df`；协议v1.2。
- 结果来自单个screening训练seed404，所有评价均为development；不是final blind，也不估计训练seed方差。
- 阈值仅从各模型相同的独立H0 calibration流分别确定；H0 verification未参与模型/checkpoint选择。
- 模型规模的继续训练只使用Medium与validation筛出的一个大模型，且两臂共同过门才同步延长。
- 首次推理完成后，配对汇总遇到NumPy布尔数组相减错误；修复后复用原提交生成的12份预测归档完成汇总，未重跑推理。修复及复用审计见 `attempts/evaluation_aggregation_fix.json`。

## 模型与H0保护

| Arm | Active parameters | Updates | C2+C3 joint @ global Pfa .05 | C3 pooled Pfa upper | All six C3 cell familywise upper ≤.10 |
|---|---:|---:|---:|---:|---|
| Medium | 70,277 | 12,800 | 0.8164 | 0.1016 | False |
| Large | 292,005 | 12,800 | 0.8628 | 0.1021 | False |
| XLarge | 1,012,197 | 12,800 | 0.8704 | 0.1027 | False |

## Matched模型比较

- Large−Medium：C2+C3 joint差+0.0464，97.5%配对CI `[0.036458333333333336, 0.05642632378472223]`；正式seed晋级门：**FAIL**。
- XLarge−Medium：C2+C3 joint差+0.0540，97.5%配对CI `[0.04448513454861112, 0.06401909722222222]`；正式seed晋级门：**FAIL**。

## 配对延长训练

- Medium和validation选出的较大模型未共同通过延长门；不进行延长训练比较。

## 训练数据规模对照

Medium 102,400−51,200唯一训练对：C2+C3 joint差+0.0067，95%配对CI `[-0.0006564670138888793, 0.014545355902777748]`。两臂均12,800 updates、effective batch64、总CPI exposure相同；核心区别是4遍与8遍固定训练集遍历。数据规模对照的core臂直接复用模型规模实验中的Medium 12,800步checkpoint。

## 解释边界

- 单seed开发对照只能筛选候选，不能证明跨初始化稳定性或盲测泛化。
- C3 pooled与各cell虚警保护必须独立满足；joint或RMSE改善不抵消条件虚警失败。
- 若验证曲线在51,200前仍持续改善，只能称为预算内未观察平台，不推断已经收敛。
- 等权SNR曲线对条件指标作描述性平均，不替代按场景配对的主指标。

图：`experiments/D2-A-R6/R6-C/figures/r6c_training_joint_doppler.png`。
