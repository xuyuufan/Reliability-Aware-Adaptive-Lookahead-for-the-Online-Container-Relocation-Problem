# 当前发布交接

- 仓库：https://github.com/xuyuufan/Reliability-Aware-Adaptive-Lookahead-for-the-Online-Container-Relocation-Problem
- 本地发布目录：D:/DSAI/PAPER_CODE_RIDGE_BUNDLE/github_release
- 当前范围：subset_shuffle、time_window_shuffle；CLEAN 单列。20 布局、15 条件、六算法 × 六策略，共 10,800 条轨迹。
- 算法入口：baselines/ 下可见 EM、GP、ERI、EG、Chain、Minmax、Leveling。实际共享实现位于 baselines/_snapshot/，冻结模型位于 models/。
- 完整命令：py -3 -B tools/reproduce_experiments.py --scope full --workers 4 --output-dir runs/full --analyze
- 已执行验证：独立复制目录中完整重跑 10,800 条；194,400 个确定性字段一致；12 张 CSV 共 204,880 单元格完全一致；345,526 个物理步骤独立回放通过。216 条 smoke、10 项统计 synthetic test、8 项 pytest 通过。
- 结果与 trace：results/two_noise_families/；新 bank、public trace、evaluator physics trace 位于 reproduced_run/。
- 当前清单：provenance/RELEASE_MANIFEST.csv；运行输入绑定：reproduction/INPUT_MANIFEST.json。
- Delay 历史数据与旧结果已移出当前发布文件，本地归档：D:/DSAI/PAPER_CODE_RIDGE_BUNDLE/publication_archives/20261015_before_two_noise_scope。原始本地研究包保留。此前的三族重跑因用户调整范围已停止，其部分文件保留在 validation_tmp/independent_checkout/runs/full-verified/，不作为验收结果。
- 边界：仅复现冻结模型在当前两族与 CLEAN 的开发集实验；没有新训练、模型选择、独立确认或 OOD 测试。CLEAN 负结果及未分辨比较仍保留。
