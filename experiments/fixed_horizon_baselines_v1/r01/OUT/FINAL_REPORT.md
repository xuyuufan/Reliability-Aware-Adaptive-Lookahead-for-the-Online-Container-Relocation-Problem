# 固定视野基线比较：Ridge-R 结果报告

**协议：** ridge_fixed_horizon_baselines_v1；**run：** r01；**状态：** ROOT 独立统计与图表复核通过（M3 技术 PASS）。本报告记录已注册的开发布局比较及其证据边界；交付索引与最终验收仍由 ROOT 完成。本文件不是论文稿。

## 主要结论

本段主结论限定于 noisy 宏指标：三个噪声族等权，CLEAN 单列。对每个算法 a，先按既定层级聚合得到方法均值 M_a，再计算算法内改善率 I_a(H)=100×(M_a(FIXED_H)−M_a(RIDGE))/M_a(FIXED_H)，并令六算法等权总体 S(H)=mean_a I_a(H)。因此，S(H)先取各算法内的聚合均值之比，再对六个改善率等权平均，不是混合六算法的搬移数后再取比。六算法等权总体 S(H) 的五项比较（属于预注册35项主比较家族）同时区间下界均大于 0；这不表示30项算法内比较都成立。正的改善率表示 Ridge 的完整轨迹搬移次数更少。逐算法结果则更有限：30 个算法×H 对比中，24 个同时区间支持 Ridge，6 个无法区分，0 个支持反方向；只有 EG 和 Minmax 在五个 H 上都得到同时区间支持。因此，六算法等权结论成立，但“每个算法上 Ridge 都胜过每个固定 H”尚未获证据支持。

| H | 六算法等权改善率 | 普通 95% 区间 | 35 项同时区间 | 固定 H − Ridge（搬移次数） | 绝对差普通 95% 区间 |
|---:|---:|---:|---:|---:|---:|
| 0 | 16.414% | [12.332%, 20.115%] | [9.780%, 22.122%] | 6.015 | [4.381, 7.677] |
| 2 | 11.962% | [9.403%, 14.261%] | [7.751%, 15.414%] | 4.004 | [3.155, 4.818] |
| 5 | 7.652% | [5.796%, 9.567%] | [4.760%, 10.819%] | 2.427 | [1.810, 3.100] |
| 10 | 4.178% | [2.685%, 5.627%] | [1.764%, 6.445%] | 1.274 | [0.831, 1.725] |
| 20 | 3.221% | [1.763%, 4.638%] | [0.877%, 5.402%] | 0.968 | [0.541, 1.375] |

逐算法的支持范围如下；未列入支持列的其余比较均为无法区分，没有任何一项同时区间整体落在负值方向。

| 算法 | 同时区间支持 Ridge 的 H | 无法区分的 H |
|---|---|---|
| EM | 0、2、5、10 | 20 |
| GP | 0、2、5、10 | 20 |
| ERI | 0、2 | 5、10、20 |
| EG | 0、2、5、10、20 | — |
| Chain | 0、2、5、10 | 20 |
| Minmax | 0、2、5、10、20 | — |

全部 30 个数值及区间见 [RIDGE_VS_FIXED_H_MACRO.csv](../analysis/RIDGE_VS_FIXED_H_MACRO.csv)，五个等权汇总见 [OVERALL_EQUAL6_RESULTS.csv](../analysis/OVERALL_EQUAL6_RESULTS.csv)。

## 开发集内的最佳固定 H

每个算法的 H_dev 是在同一开发数据上，从五个 H 中选取 noisy 宏平均 R_raw 最小者；精确并列时取较小 H。选出的 H 为 EM=20、GP=20、ERI=10、EG=20、Chain=20、Minmax=20。表中区间是该算法所选 H 对应的注册 35 项同时区间，未把同一批开发布局上的选择解释为独立确认。只有 EG 与 Minmax 的该项同时区间下界高于 0。该结果尚未支持“Ridge 在六个算法各自的开发集最佳固定 H 上都占优”。

| 算法 | H_dev | 固定 H 均值 R_raw | Ridge 均值 R_raw | 固定 H − Ridge | Ridge 改善率 | 对应 35 项同时区间 |
|---|---:|---:|---:|---:|---:|---:|
| EM | 20 | 29.885 | 29.258 | 0.628 | 2.101% | [-0.349%, 4.530%] |
| GP | 20 | 29.963 | 29.208 | 0.756 | 2.522% | [-0.135%, 5.186%] |
| ERI | 10 | 30.122 | 29.924 | 0.197 | 0.655% | [-3.383%, 4.174%] |
| EG | 20 | 30.177 | 28.171 | 2.006 | 6.648% | [4.774%, 8.561%] |
| Chain | 20 | 29.851 | 28.992 | 0.858 | 2.875% | [-0.030%, 5.663%] |
| Minmax | 20 | 29.849 | 28.906 | 0.943 | 3.159% | [0.024%, 6.249%] |

逐项见 [DEVELOPMENT_BEST_FIXED_H.csv](../analysis/DEVELOPMENT_BEST_FIXED_H.csv)。这是 20 个已曝光开发布局上的描述性选择，没有独立确认集；不能据此宣称跨布局泛化或部署收益。

## 噪声族、CLEAN 与固定 H 对 H20

以下分层与 CLEAN 结果均保留正负方向，配普通 95% bootstrap 区间作描述性报告，不使用 35 项 Bonferroni 校正作显著性结论。正的 Ridge 改善率表示 Ridge 搬移更少，负值表示 Ridge 搬移更多。

H20 下噪声族的六算法逐算法点估计如下（百分比）；完整普通 95% 区间、算法值与 R_raw/RN 结果见 [FAMILY_RESULTS.csv](../analysis/FAMILY_RESULTS.csv)。delayed_positions 中 6 个算法有 5 个点估计为负；这保留了 H20 在该噪声族的负向信号。

| 噪声族（H20） | EM | GP | ERI | EG | Chain | Minmax |
|---|---:|---:|---:|---:|---:|---:|
| subset_shuffle | 8.27% | 8.59% | 6.97% | 11.81% | 6.60% | 9.16% |
| delayed_positions | -4.03% | -4.15% | -2.20% | 2.40% | -1.28% | -2.76% |
| time_window_shuffle | 0.68% | 1.70% | 0.58% | 4.77% | 2.62% | 1.75% |

在 CLEAN 条件的 H20 对比中，六个算法的 Ridge 改善率点估计全部为负：Ridge 比 H20 多搬移 5.09%–11.09%。表内绝对差为 H20 − Ridge 搬移次数；区间是普通 95% 区间，仍按描述性结果解释。

| 算法 | H20 均值 R_raw | Ridge 均值 R_raw | H20 − Ridge 搬移次数 | Ridge 改善率 | 普通 95% 区间 |
|---|---:|---:|---:|---:|---:|
| EM | 23.90 | 26.55 | -2.65 | -11.09% | [-15.93%, -6.92%] |
| GP | 23.90 | 26.05 | -2.15 | -9.00% | [-11.91%, -6.20%] |
| ERI | 26.55 | 28.00 | -1.45 | -5.46% | [-9.74%, -1.10%] |
| EG | 24.55 | 25.80 | -1.25 | -5.09% | [-8.33%, -2.29%] |
| Chain | 24.45 | 26.35 | -1.90 | -7.77% | [-13.01%, -2.69%] |
| Minmax | 23.90 | 26.25 | -2.35 | -9.83% | [-14.37%, -6.29%] |

CLEAN 全部 H、六算法、R_raw/RN 见 [CLEAN_RESULTS.csv](../analysis/CLEAN_RESULTS.csv)。21 个注册条件的逐项结果见 [CONDITION_RESULTS.csv](../analysis/CONDITION_RESULTS.csv)。

另对 noisy 聚合下四个较短 H 与 H20 作逐算法配对。下表只列六算法等权 R_raw：EQUAL6 绝对差是六个算法各自 H20 − H_fixed 搬移差的等权平均；EQUAL6 改善率先在每个算法内以 H20 为分母计算，再对六个百分比等权平均，即 mean_a[100×(M_a(H20)−M_a(H_fixed))/M_a(H20)]，不是先混池搬移数再取比。绝对差与改善率均按配对算法结果分别平均。负值表示短 H 的搬移数高于 H20。它们与区间均为描述性普通 95% 结果。完整 CSV 还包括六算法逐算法行以及 RN。

| 短 H | H20 − H_fixed（搬移次数） | 短 H 相对 H20 改善率 | 普通 95% 区间 |
|---:|---:|---:|---:|
| 0 | -5.047 | -16.857% | [-22.102%, -11.869%] |
| 2 | -3.036 | -10.130% | [-12.144%, -8.046%] |
| 5 | -1.459 | -4.869% | [-6.345%, -3.000%] |
| 10 | -0.306 | -1.025% | [-1.528%, -0.508%] |

完整 56 行（R_raw/RN、四个短 H、六算法和 EQUAL6）见 [FIXED_H_VS_H20.csv](../analysis/FIXED_H_VS_H20.csv)。

## 指标、聚合和推断

主指标是每条完整轨迹的原始搬移次数 R_raw，数值越低越好。RN=R_raw/N 单独保留为次级描述结果。对布局内的重复先在 strength 内平均；每个 noisy family 内对 strength 等权平均，再对三个 noisy family 等权；G1 与 G2 等权聚合。CLEAN 单列，不并入 noisy 宏平均。改善率是聚合均值之比，不是逐布局百分比的平均；主比较绝对差定义为 M(FIXED_H) − M(RIDGE)。

统计抽样单位为 20 个语义布局指纹（G1、G2 各 10），不是轨迹、repeat、condition 或算法。共同配对 bootstrap 使用 NumPy PCG64、seed 202610155301、100,000 次抽样、batch size 1,000。布局按 G1 后 G2 排列，几何组内 fingerprint 字典序；每批先生成 G1 索引，再生成 G2 索引。相同的配对索引用于全部比较、scope 和指标。分位数使用 NumPy method="linear"。35 项主族由 30 个算法×H 比较及 5 个 EQUAL6 的 H 比较构成；双侧家族错误率 0.05，逐项 alpha 为 0.05/35。表中“同时区间”用于这 35 项主要推断；标为“普通 95% 区间”的结果仅描述性。

ROOT 未导入生产聚合函数，独立重建了汇总、bootstrap 区间和绘图数据，共核对 56,405 个数值字段，最大绝对差为 6.22×10⁻¹⁴。ROOT 对四张 PNG 的可读性及四份单页、无嵌入位图的矢量 PDF 复核通过。独立统计及图表复核结论见 [ROOT_M3_REVIEW.json](../ROOT_M3_REVIEW.json)。

## 样本、来源与执行完整性

分析矩阵包含 20 个布局 × 21 个条件 × 6 个算法 × 6 个方法，共 15,120 条完整轨迹记录、420 个布局-条件单元、每单元 36 个配对方法槽位。六个方法是 RIDGE 与独立完整轨迹运行的 FIXED_H0、FIXED_H2、FIXED_H5、FIXED_H10、FIXED_H20。

- 10,080 行来自固定 H bank：9,984 条本阶段新轨迹，加 96 条通过 M1 smoke 审计后按原记录复用一次的轨迹。
- 5,040 行来自 M0 映射的历史 Ridge 与 H20 原始结果；ROOT merge review 对路径、SHA、源行、原 key、method、R_raw、N、forecast hash/seed、别名及完整配对数作了逐行核验。
- 24 条正式 bridge 轨迹只用于实现验证，不进入分析矩阵。3 次未通过最终阶段验收的 M1 工程执行共涉及 360 条轨迹，全部排除；其中第 2、3 次 runner 虽 exit 0，但审计未通过。所有失败与诊断记录均保留在日志和账本中。96 条 smoke 仅在固定 H bank 中计数一次。

正式 merged raw 文件 SHA-256 为 eb0ecc87c90a50cd7fb62b2fc0466babdc28ab9342151d7c804f8ca11977b906；分析器 SHA-256 为 fb5fa96e4e0d3f23f025f26273ff91d171b948467e82580a0b7c55cdb332a962。M3 分析入口与出口均经过注册 stage gate 和冻结哈希核对；M3 release 文件 SHA-256 为 f8bceb96a2d953d1dcf55e41acc20f9f2ddb82d3acd0cb238d3fa79acfeb4d85。M3 receipt 文件 SHA-256 为 ef92b696597e156a320b4628b452636392dad7d0e3a164c03c752eee47895e75，其 canonical receipt SHA-256 为 10364883ba4447504e51d02b42ef8fdda8dd8106eb73183fb34c33b2676b86fc。

本轮 ROOT M0–M3/merge 复核和绑定记录保存在 run 目录。M0_REV05 source binding SHA-256：f16f483abae18010e3f1f2b9f32c64f54f198926ddcf92d96015fb81acf5ed97。复核文件及 canonical SHA-256：

- [ROOT_M0_REVIEW.json](../ROOT_M0_REVIEW.json)：7afab30584f826b896c1a1f5409ea8a7138cca7f6564ae4df2ddafac05447078
- [ROOT_M1_REVIEW.json](../ROOT_M1_REVIEW.json)：ce5abef28f0b26c8dc143c256086efdc5619b82c5aeeda977e3b2310c5634231
- [ROOT_M2_REVIEW.json](../ROOT_M2_REVIEW.json)：ea639d3ced81be54b563d4626e065f40560078d33fa0599e571af71fdb21df33
- [ROOT_MERGE_REVIEW.json](../ROOT_MERGE_REVIEW.json)：41f2521f79dd14ea2ed15efa88c8f0ac4947dacbdaf6bcfdb6b0e047752545af
- [ROOT_M3_REVIEW.json](../ROOT_M3_REVIEW.json)：d62ddc8077e0e16b8857042961f1f1f1f431e88b0ddc62a52483a1fa99ce49ca

M2 单次科学运行的嵌套进程实际耗时 1,145.626 秒，M2 单次轨迹审计的嵌套进程实际耗时 332.945 秒；两者均以实际 exit 0 完成。本协议 science 累计耗时为 1,250.503 秒（含 M1 与工程重跑），正式及诊断审计累计耗时为 481.852 秒；这两个累计数是跨执行汇总，不是单个进程耗时。M3 外层实际耗时 27.562 秒，嵌套分析进程 26.058 秒。按保守记账将 M3 与 ROOT 活跃独立复核窗口合计为 414.065 秒，低于 1,800 秒 M3 分析上限；外层分析耗时在该合计中只计一次。失败和诊断执行均保留并按原预算记账。

## 图表与表格

固定 H 搬移曲线使用真实数值 H 横轴（0、2、5、10、20），按六算法分面并显示 Ridge 水平参照；改善率图保留负值与零线。主推断应以配对比较图的 35 项同时区间为准；原始搬移曲线、family 与 CLEAN 图中的普通区间为描述性结果。

- [固定 H 的 R_raw 曲线 PNG](../analysis/FIXED_H_RRAW_BY_SOLVER.png) / [矢量 PDF](../analysis/FIXED_H_RRAW_BY_SOLVER.pdf)
- [Ridge 对固定 H 的 35 项同时区间 PNG](../analysis/RIDGE_VS_FIXED_H_SIMULTANEOUS.png) / [矢量 PDF](../analysis/RIDGE_VS_FIXED_H_SIMULTANEOUS.pdf)
- [噪声族结果 PNG](../analysis/FAMILY_RESULTS_FACETS.png) / [矢量 PDF](../analysis/FAMILY_RESULTS_FACETS.pdf)
- [CLEAN 结果 PNG](../analysis/CLEAN_RESULTS.png) / [矢量 PDF](../analysis/CLEAN_RESULTS.pdf)
- 布局聚合：[LAYOUT_AGGREGATES.csv](../analysis/LAYOUT_AGGREGATES.csv)；精确绘图数据：[固定 H](../analysis/PLOT_DATA_FIXED_H_RRAW.csv)、[主比较](../analysis/PLOT_DATA_RIDGE_VS_FIXED_H.csv)、[噪声族](../analysis/PLOT_DATA_FAMILIES.csv)、[CLEAN](../analysis/PLOT_DATA_CLEAN.csv)。

## 解释边界

这是在 20 个已曝光开发布局、六个注册算法和 21 个注册条件上的完整配对比较。bootstrap 的 100,000 次抽样提高区间数值精度，不增加布局样本数。结果不构成未见布局、OOD、部署效果或后续独立确认的证据；没有训练模型或拟合参数。报告保留负向与无法区分结果，不把技术完整性等同于普遍科学优势。analysis/FINAL_REPORT.md 是 M3 分析器生成的草稿，本交付报告单独成文，不改动该草稿。