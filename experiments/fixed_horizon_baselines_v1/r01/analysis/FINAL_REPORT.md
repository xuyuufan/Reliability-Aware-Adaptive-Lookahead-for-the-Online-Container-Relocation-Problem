# FINAL_REPORT 草稿：Ridge-R 与固定视野基线

> 状态：M3分析草稿，等待主审独立复算。此文件不是论文稿，也不代表方法优势或最终验收。

## 数据与协议

- 协议：`ridge_fixed_horizon_baselines_v1`；输入完整键数：15,120，布局数：20（G1/G2各10），条件数：21。
- merged raw SHA-256：`eb0ecc87c90a50cd7fb62b2fc0466babdc28ab9342151d7c804f8ca11977b906`；分析器 SHA-256：`fb5fa96e4e0d3f23f025f26273ff91d171b948467e82580a0b7c55cdb332a962`。
- M3正式入口先调用`R.experiments.fixed_horizon_integrity.require_stage_release('M3_ANALYSIS')`，出口调用`verify_stage_unchanged`并再次核对merged与分析器SHA-256。
- 配对forecast布局-条件数：420；source_origin：`{'historical_reuse': 5040, 'fixed_h_bank': 10080}`；source_stage：`{'M0_REUSE': 5040, 'M2_BANK': 10080}`。
- 主指标：完整轨迹 `R_raw` 搬移次数，越低越好；配对改善率为 `100 × (M(FIXED_H) − M(RIDGE)) / M(FIXED_H)`，正值表示 Ridge 搬移更少。
- 聚合先在强度内平均repeat，再在每个噪声族内等权强度，三个噪声族等权，G1/G2等权。CLEAN单独报告。RN=R_raw/N仅作描述性结果。
- 共同布局簇bootstrap：NumPy PCG64，seed `202610155301`，100,000次，batch size `1000`；布局顺序为G1后G2、组内fingerprint字典序。每个连续batch先调用rng.integers生成G1索引，再生成G2索引；所有scope、solver、policy、指标和配对差复用同一索引矩阵。分位数固定为NumPy `method=linear`。主推断使用35项Bonferroni双侧同时区间；同时提供普通95% percentile区间。
- 实际进程elapsed由M3分析runner receipt记录；bootstrap累计上限为1,800秒，本轮主推断族含35项。

## 35项主比较

30个算法内Ridge-vs-H比较中，同时区间下界大于0：24；上界小于0：0；其余或不可定义：6。完整值见 [RIDGE_VS_FIXED_H_MACRO.csv](RIDGE_VS_FIXED_H_MACRO.csv) 与 [OVERALL_EQUAL6_RESULTS.csv](OVERALL_EQUAL6_RESULTS.csv)。

| 算法 | 开发集解释（同时区间） |
|---|---|
| EM | supports Ridge only over horizons 0,2,5,10 |
| GP | supports Ridge only over horizons 0,2,5,10 |
| ERI | supports Ridge only over horizons 0,2 |
| EG | simultaneous evidence favors Ridge over all five registered fixed horizons |
| Chain | supports Ridge only over horizons 0,2,5,10 |
| Minmax | simultaneous evidence favors Ridge over all five registered fixed horizons |

六算法等权的S(H)逐H结果：

| H | Ridge改善率 | 普通95%区间 | 35项同时区间 |
|---:|---:|---:|---:|
| 0 | 16.414% | [12.332, 20.115] | [9.780, 22.122] |
| 2 | 11.962% | [9.403, 14.261] | [7.751, 15.414] |
| 5 | 7.652% | [5.796, 9.567] | [4.760, 10.819] |
| 10 | 4.178% | [2.685, 5.627] | [1.764, 6.445] |
| 20 | 3.221% | [1.763, 4.638] | [0.877, 5.402] |

the equal-six macro score supports Ridge over all five registered fixed horizons.

## 开发集最佳固定视野

H_dev按每个算法的noisy宏平均R_raw最小值确定，精确并列取较小H。该选择使用本开发集，继承的35项区间未校正选择效应，不能解释为独立确认。

| 算法 | H_dev | 固定H均值 | Ridge改善率 | 同时区间（继承） |
|---|---:|---:|---:|---:|
| EM | 20 | 29.885 | 2.101% | [-0.349, 4.530] |
| GP | 20 | 29.963 | 2.522% | [-0.135, 5.186] |
| ERI | 10 | 30.122 | 0.655% | [-3.383, 4.174] |
| EG | 20 | 30.177 | 6.648% | [4.774, 8.561] |
| Chain | 20 | 29.851 | 2.875% | [-0.030, 5.663] |
| Minmax | 20 | 29.849 | 3.159% | [0.024, 6.249] |

## 描述性结果与限制

CLEAN、三个噪声族、21个注册条件、固定H-vs-H20及RN保留全部正负结果，均标为描述性普通95%区间，不按逐格跨零作经校正结论。绝对搬移差以`固定H均值 − Ridge均值`报告。

样本单位是20个已曝光开发布局，不是轨迹、repeat、条件或算法。bootstrap次数只改善数值分位数精度，不增加布局样本。结果只适用于本轮注册的六算法、20个布局及条件，不支持未见布局、OOD或部署效果推断。

实验完整性和科学优势是不同结论；负值、区间跨0或算法间差异均保留。主审应独立复算汇总表、配对区间与图表后再更新最终验收状态。

## 图表与精确绘图数据

- [六算法固定H搬移曲线与Ridge水平参照（PNG）](FIXED_H_RRAW_BY_SOLVER.png) / [PDF](FIXED_H_RRAW_BY_SOLVER.pdf)
- [Ridge-vs-H改善率与35项同时区间（PNG）](RIDGE_VS_FIXED_H_SIMULTANEOUS.png) / [PDF](RIDGE_VS_FIXED_H_SIMULTANEOUS.pdf)
- [噪声族分面改善率（PNG）](FAMILY_RESULTS_FACETS.png) / [PDF](FAMILY_RESULTS_FACETS.pdf)
- [CLEAN改善率（PNG）](CLEAN_RESULTS.png) / [PDF](CLEAN_RESULTS.pdf)
- 精确绘图CSV：`PLOT_DATA_FIXED_H_RRAW.csv`、`PLOT_DATA_RIDGE_VS_FIXED_H.csv`、`PLOT_DATA_FAMILIES.csv`、`PLOT_DATA_CLEAN.csv`。
