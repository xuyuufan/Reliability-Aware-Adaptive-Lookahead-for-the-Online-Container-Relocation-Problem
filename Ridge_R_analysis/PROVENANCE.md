# Ridge-R 数据来源与验证记录

本记录随 Ridge-only 副本生成，不是新的实验结果。

## 输入来源与哈希

1. 家族级统计源：D:\DSAI\containter_new\R\output\r_seven_solvers_five_r_v1\r01\analysis\seven_solver_five_r\FAMILY_ABSOLUTE_RRAW_RN.csv  
   SHA-256: 02f45cc4333443059ada1437f9dcee58b0fa66f8862c981a691dcc7b3bdefd51  
   过滤得到六算法 Ridge/NATIVE20 的三族方法行，再按算法与 family 配对；保留已有 improvement 和 paired 95% CI。

2. 条件级统计源：D:\DSAI\containter_new\R\output\r_seven_solvers_five_r_v1\r01\analysis\seven_solver_five_r\CONDITION_ABSOLUTE_RRAW_RN.csv  
   SHA-256: a8cd491e403423fd20090862406811a4b1035310f863a501a18208c451f811e2  
   过滤得到六算法 Ridge/NATIVE20 的 21 条件方法行，另生成 126 个 paired 条件比较。CLEAN 单独呈现，不混入三个噪声族。

3. 冻结模型源：D:\DSAI\containter_new\R\output\universal_r_utility_v1\r02\M1_FIT_01\RIDGE_MODEL.json  
   SHA-256: 07bf606c2a7ab1ac1bffa7d39449c34fc719ad8e454eba1243761af44fd7effe  
   包内副本含 74 个系数。

统计单位为 R_raw relocation，越低越好。改善率方向为 Ridge 相对 NATIVE20 的搬移减少百分比；图表直接沿用原汇总表中每 solver/family 的成对估计与 95% 布局簇 bootstrap 区间（20 个布局、两种几何各10、10,000次）。本次没有从 marginal interval 反推配对区间，也没有拟合或重新运行 bank。

## Ridge 选择器适配来源

- D:\DSAI\containter_new\R\scrp\r_five_versions_actor.py — SHA-256 b06026e8d4ea81fc67eb6843b76689b08ff4c5a5632b0986d36e4ec6aa2a8439
- D:\DSAI\containter_new\R\scrp\plan052_policies.py — SHA-256 75fd59ec4e74dbd545ada9fba5117cb4d6e22ce7709a06b1d71c36aa18713284
- D:\DSAI\containter_new\R\scrp\r_eri_eg_public.py — SHA-256 56d8581e2c4fa0babd4d74ba9c0c9e7a37232702adde649060c89b02461de2a
- D:\DSAI\containter_new\R\scrp\universal_r_utility.py — SHA-256 fa516b5b586071175e1aa2388de54389556ffc28ad49aca6ec4d87c65c5a1a00

Ridge-only adapter保留原候选去重、H20 anchor、评分/平局规则及 EG sequence commit；对照原 PLAN-051 的 EM/GP/ERI/EG 分支、PLAN-052 的 Chain/Minmax 分支进行了同 proposal、同冻结模型的合成单步检查，输出相同；EG 两步 commit 合成检查通过。该验证不等于完整 bank 复现。

## 公共特征提取与 Ridge module imports

- D:\DSAI\containter_new\R\scrp\h20_r_data.py — SHA-256 c5f6e3e58be2eddb2d07e7ba209585792b7a5c35638a40f9ab16f2c8fc149214
- D:\DSAI\containter_new\R\scrp\h20_r_c2.py — SHA-256 f8d8290f9dda39f11b0b9d431da92bd042ec0664490fa1ec23dc1274fd46ef69

ridge_public_features.py摘取冻结 Ridge 所需 C1/C2 公共特征。universal_r_utility.py仅将其两个共享特征 import 指向该 Ridge-only 文件；适配后模块在本包独立解析并能与 RIDGE_MODEL.json 一起加载。RidgeScorer载入74个系数。

## 输出与限制

- 主图和补充图由 Ridge_R_analysis\figures\gen_fig_ridge_six_solver_three_noise_families.py生成；图文件及所有 filtered/derived CSV 的 SHA-256 见 PAPER_CODE_MANIFEST.csv。
- 主图经主审视觉检查；18 格方向和 CI 与源统计数据一致。
- Leveling 证据源：D:\DSAI\containter_new\R\experiments\r_all_methods_r_pairs_analysis.py 的 ANALYTIC_IDENTITY_ONLY / new_scoring=false / CI=null 结论；不作为实测的第七算法。
- 未包含 RIL 代码、其他 R 版本、完整逐步 sidecar 或七算法历史 runner。PLAN-052 状态 NO_UNIQUE_WINNER_SUPPORTED；本包不能支撑“完整七算法可复跑”或“论文科研成功”的结论。