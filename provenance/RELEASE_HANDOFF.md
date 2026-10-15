# GitHub 发布交接

- 仓库：https://github.com/xuyuufan/Reliability-Aware-Adaptive-Lookahead-for-the-Online-Container-Relocation-Problem
- 本地发布目录：D:/DSAI/PAPER_CODE_RIDGE_BUNDLE/github_release
- 保留：Ridge 适配器与 74 系数模型、六算法实验依赖快照、20 布局、15,120 条合并轨迹、正式 trace、结果表与 PNG/PDF、Leveling 和 840 Caserta 实例、最终验收与必要来源记录。
- 筛除：多轮 preparation/history、重复过程日志、重复算法导出和过期交接页。所有筛除文件仍留在原目录，清单见 EXCLUDED_FILES.csv。
- 验证：10 项统计 self-test、4 项 Leveling 测试通过；12 张 CSV 共 279,760 个单元格重生成对照一致，最大数值误差 0；四组 PNG/PDF 重生成成功。没有新实验或拟合。
- 当前发布哈希入口：RELEASE_MANIFEST.csv；原历史索引可能引用未上传材料，不能替代当前发布清单。
- 复现命令：在发布根目录执行 py -3 -B tools/reproduce_results.py，生成目录为 reproduced/（不提交）。
- 边界：可从已有数据复现统计与图；不提供完整历史 bank 的便携端到端重跑。CLEAN 负结果、无法区分结果及开发布局限制完整保留。
