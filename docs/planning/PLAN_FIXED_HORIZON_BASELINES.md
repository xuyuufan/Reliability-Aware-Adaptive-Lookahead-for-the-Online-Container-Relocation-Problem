# Ridge-R 六算法固定视野基线补充计划

日期：2026-10-15（Asia/Shanghai）  
协议：`ridge_fixed_horizon_baselines_v1`  
状态：`USER_AUTHORIZED / M0_IN_PROGRESS`  
授权范围：用户于2026-10-15明确授权实施本计划，主agent负责规划审核，GPT-6 Luna Max负责执行。当前M0准备中；真实bridge、smoke、bank及性能分析尚未执行。各阶段实际状态以新协议目录中的receipt与主审记录为准。

## 1. 目标与要检验的结论

在相同算法、布局、预测噪声和实现规则下，比较冻结 Ridge-R 与始终采用固定 H 的原生策略，判断现有收益中有多少能由一个固定短视野获得。

本轮回答三个问题：

1. 各算法的固定 H0/H2/H5/H10 是否已经优于自身 H20？
2. Ridge 是否优于各固定 H，特别是本批开发布局上表现最好的固定 H？
3. 这些关系在三个噪声族与 CLEAN 下如何变化，是否存在搬移退化？

核心结论按证据分级：`仅优于H20`、`优于部分固定H`、`在本开发集上优于全部固定H`、`无法区分`、`有退化证据`。所有结论限定到本轮注册的算法、布局和条件，不直接推断未见布局、OOD 或实际部署效果。

## 2. 范围与实验矩阵

算法：`EM / GP / ERI / EG / Chain / Minmax`，与当前 Ridge-only 包保持一致。

| 策略槽 | 定义 | 数据处理 |
|---|---|---|
| `FIXED_H0` | 每次规划仅参考当前目标 | 新运行 |
| `FIXED_H2` | 每次规划参考当前目标及最多2个预测未来目标 | 新运行 |
| `FIXED_H5` | 每次规划参考当前目标及最多5个预测未来目标 | 新运行 |
| `FIXED_H10` | 每次规划参考当前目标及最多10个预测未来目标 | 新运行 |
| `FIXED_H20` | 原生 H20 | 满足绑定条件时复用原 `NATIVE20` |
| `RIDGE` | 原冻结 `R_UTILITY_RIDGE` | 满足绑定条件时复用 |

H 固定指每次生成动作或动作组时采用相同预测长度；forecast 随检索进度更新。未知未来保持 censored 状态。每个策略独立跑完整轨迹，各自的动作导致各自的后续状态。

本轮不加入 H15：它不在冻结 Ridge 的候选集合 `{0,2,5,10,20}` 内；新增网格需要另一版事先登记的计划。本轮也不加入 RIL、Leveling、其他 R、模型训练、q 消融、封存确认或专门的成本 benchmark。

布局继续使用已曝光的20个开发布局，G1/G2各10个。条件完全沿用 PLAN-051/052：

| 条件族 | 强度 | repeats | 唯一条件数 |
|---|---|---|---:|
| `subset_shuffle` | p=0.1/0.3/0.5 | 0/1 | 6 |
| `delayed_positions` | p=0.1/0.3/0.5 | 0/1 | 6 |
| `time_window_shuffle` | B=2/5/10/20 | 0/1 | 8 |
| `CLEAN` | 无扰动 | 单条件 | 1 |

轨迹矩阵：

- 新固定视野：`20 × 21 × 6 × 4 = 10,080` 条。
- 复用 Ridge/H20：`20 × 21 × 6 × 2 = 5,040` 条。
- 最终比较：`20 × 21 × 6 × 6 = 15,120` 个唯一完整键。
- 正式新固定视野 bank 包含下述96条 smoke 的精确复用，因此 smoke 后剩余9,984条。
- 另有24条实现桥接验证轨迹，单独记账，不作为额外统计样本；正常执行总新轨迹数为10,104条。

## 3. 数据入口、冻结资产与复用条件

当前整理包没有完整 bank runner 和原始 trace 链。执行阶段须从原工作区读取并冻结所需依赖，在新目录构建独立执行副本；原工作区只读。不得把包内汇总 CSV 当作可独立复算配对区间的原始数据。

已核对的输入如下；执行前仍须逐项重新核验。

| 资产 | 绝对路径 | SHA-256 |
|---|---|---|
| 20布局 | `D:\DSAI\containter_new\R\output\r_frozen_cross_solver_transfer_v1\r01\data\select20.json` | `62f9e205f2df0197260629f4b9ad54a35a412c6ab16838709b0a26d224ecf426` |
| PLAN-051原始表 | `D:\DSAI\containter_new\R\output\r_five_versions_comparison_v1\r01\bank_01\raw_trajectories.csv` | `3c99ad56a47fe5a8bdbb3f7044cf91190084360e9d72fb6c90243160b61a0d44` |
| PLAN-052新算法原始表 | `D:\DSAI\containter_new\R\output\r_seven_solvers_five_r_v1\r01\m2_bank_01\raw_trajectories.csv` | `b4911b67d65bc05ec854c69f55c623735be957e71af1fb9000bfceee79b10862` |
| 冻结Ridge模型 | `D:\DSAI\PAPER_CODE_RIDGE_BUNDLE\Ridge_R_analysis\data\RIDGE_MODEL.json` | `07bf606c2a7ab1ac1bffa7d39449c34fc719ad8e454eba1243761af44fd7effe` |

EM/GP/ERI/EG的Ridge与H20来自PLAN-051；Chain/Minmax来自PLAN-052。只筛选所需策略，不使用其他 R 的结果进行方法选择。

M0须读取两计划的 `FINAL_ACCEPTANCE.json`、当前有效 source bindings、raw audit PASS receipt、模型/布局/forecast绑定，并登记本轮依赖闭包。过滤后的5,040行须满足：

1. 完整、无重复、六算法每个布局条件恰有Ridge与H20两行。
2. 同一布局条件的语义布局指纹、原始N、forecast seed、F hash一致。
3. 原始策略、模型、proposer、截断规则与本轮执行副本保持一致。
4. 能通过旧原始trace和有效审查链回到物理与public契约证据。

方法别名转换须写 `REUSE_MAP.csv`，保存原 trajectory key、源文件及行号、原 method、新 method、源哈希和验证结果。旧计时保留 `reused` 标签，不能与新时间直接构成公平速度比。

任何复用条件失败即停止执行准备，报告具体资产差异。不得自动改成重跑旧Ridge/H20整表，也不得改写旧 source bindings、报告或科学 STOP。

## 4. 公平性与实现契约

### 4.1 公共预测与视野

- 每个布局条件生成与旧协议同 namespace 的固定forecast，六算法和所有策略共享相同F与seed；缩短H不重新生成预测。
- proposer只使用对应H的公开前缀：保留当前rank 1及rank `2..H+1`，其余occupied ranks置为unknown `22`；empty padding保持 `0`。
- 剩余未来目标不足H时，只使用实际可用前缀，不补造目标。
- p、B、repeat、family、seed、真实未来箱号、扰动mask和evaluator sidecar不得进入决策特征。评估器可保存这些字段用于配对和审计。
- 固定H策略仅调用该H的原生proposer，不调用Ridge，不先算五个候选再挑选。
- forecast、history/q更新、自动检索、容量、合法目的地、crane模型、tie规则均沿用旧协议。每条轨迹初始化新策略实例并reset缓存。

### 4.2 六算法行为

- EM/GP/ERI/Chain/Minmax每次产生一个合法搬移动作；维持原有fallback和未知尾部语义，不为短H额外设计新启发式。
- GP沿用冻结表达式和公共慢adapter；不得使用依赖真实rank的快速kernel。
- Chain沿用原来的public-prefix虚拟检索与Minmax fallback；不能读取H之外的下一blocker。
- EG沿用动作组提交：在group start以固定H生成完整当前blocker组，随后提交该组动作，保留snapshot/cache校验；不在缓存步重新规划。固定EG不能变成逐步重规划策略。
- 原生H20实现可能有自己的actor/cache路径；M0必须明确其与新固定H wrapper的连接方式。复用旧轨迹需要语义等价证据，不能仅依据policy名称相同。
- 当前包内 `ridge_policy.py` 的单步合成等价证据不能自动代替新runner的端到端桥接核验。

## 5. 分阶段执行与审核

分工：实现与执行人员提交源码和实际receipt；主审独立检查并给出 `PASS / REVISE / STOP`。本计划编写完成不等于真实实验已获执行授权。后续获授权后，在本轮范围内按阶段审核推进；这些是内部证据检查，不要求用户逐阶段重复确认。

### M0：独立执行副本、适配与冻结

1. 在本轮新目录创建依赖最小闭包，排除无关checkpoint/结果/训练目录，登记复制及适配来源。
2. 新增固定H policy和隔离runner/config，不修改旧全局registry。
3. 登记原始数据/模型/forecast/solver/prefix/runner/配置的SHA-256，生成 `SOURCE_BINDINGS.json`、`REUSE_RECEIPT.json`。
4. 通过针对性合成检查：五H截断；unknown≠empty；对H外rank和私有字段的决策不变性；合法动作与tie；reset；GP公共路径；Chain fallback；EG多步commit及不匹配fail-closed。
5. 对分析器使用人工可算的小矩阵，验证正负方向、两种几何权重、repeat层级、layout配对和缺键拒绝。

验收：依赖齐全、复用条件通过、源码/配置冻结、内部契约测试PASS。此时真实实验仍为0。

### M1：实现桥接与固定H smoke

桥接验证：G1第一个布局，`CLEAN`与`delayed_positions|p=0.3|r=0`，六算法各运行冻结Ridge和原生H20，共24条。与原已审核同键记录比较动作序列、R_raw、N、F hash、public frame及EG组提交。只忽略计时字段。桥接轨迹属于实现验证，不替换旧统计行、不增加样本量。

固定H smoke：G1/G2各第一个布局，上述两个条件，六算法×四个新增H，共96条。检查完整终止、守恒、合法动作、forecast配对、所有decision的固定H及前缀可见性、EG组提交。

验收：桥接24条一致、smoke96条通过独立物理/public核验、源码与绑定无漂移。smoke数据按原键和哈希精确收入正式bank。

若桥接不一致，不得直接进入bank或把差异解释成性能提升；先识别适配问题，保留失败产物，重新冻结必要修正后重审。

### M2：四固定H正式bank

- 生成剩余9,984条；合并smoke得到10,080个新固定H唯一完整键。
- 逐条记录完整终止、R_raw、N、RN、seed/F hash、H、实现与模型绑定、耗时、public trace和评估器private审计trace。
- 日志区分执行中、完成、超时、异常；检查真实子进程exit code，不以外层wrapper成功替代实验成功。
- checkpoint按完整键追加；只有已完成且hash核对通过的轨迹可以跳过。
- 全量审计新轨迹的物理状态、public prefix、history/reveal、动作规则及终止。确定性proposer重放是实现一致性检查；配合独立截断/物理/规则检查，不声称独立重实现全部算法。
- 经核验后合并复用5,040行，得到15,120行。每个布局条件须恰有36个算法策略槽；420个布局条件均完整。

验收：10,080条新键及15,120条总键完整、无重复、无非法动作、配对正确、完整audit PASS、真实exit 0。

性能负值不触发提前删格或停止bank；必须完成已登记的全部矩阵，避免只保留有利结果。

### M3：统计、图表与交付

只在M2证据通过后分析。主审独立实现聚合和比较，核对所有主表数值、同时区间和结论；审核图中负值、CLEAN、样本量、单位与误差线。

实验完整性通过与科学优势成立分别写结论。结果为负或区间跨0时，完整交付仍可通过；不得为了得到PASS增加样本、改模型或改权重。

## 6. 统计协议

### 6.1 指标和聚合

主指标 `R_raw` 为完整轨迹搬移次数，越低越好；`RN=R_raw/N` 单独作为次要描述指标。

对每个layout/solver/policy，依次先平均同强度repeats，再在族内对强度等权，再对三个噪声族等权；G1/G2均值各占一半。CLEAN独立报告，不进入noisy主分数。

令 `M(s,P)` 为该口径下solver s、policy P的平均搬移数。对每个固定H：

`I(s,H) = 100 × [M(s,FIXED_H) − M(s,RIDGE)] / M(s,FIXED_H)`。

正值表示Ridge搬移更少，负值表示Ridge退化。改善率取聚合均值的比值，不先算每layout百分比再平均；同时报告绝对搬移差，便于理解实际幅度。分母为0时不生成百分比，明确标记并停止相关优势判定，不任意加epsilon。

主表包含六算法×五H，共30个Ridge对比。次表报告固定H相对自身H20的改善率，用于回答缩短视野本身的效果。

六算法总体分数按算法等权：`S(H) = mean_s I(s,H)`，共5个。该分数是六个算法内部改善率的宏平均，不是把所有轨迹搬移数直接混算。

### 6.2 配对区间与多重比较

- 统计抽样单位是20个语义布局，G1/G2各10；轨迹、repeat、条件和算法不是独立样本。
- 每次在两种几何内各有放回抽取10个layout；同一次抽样同步用于所有算法、策略及条件，保持配对。
- 使用NumPy PCG64，seed `202610155301`，100,000次布局簇bootstrap。该次数用于同时区间的尾部分位数，不增加科学样本量。
- 普通95% percentile区间使用 `.025/.975`。
- 预登记主要比较族：30个算法内Ridge-vs-H比较，加5个六算法总体S(H)，合计35个。
- Bonferroni双侧同时区间使用 `alpha=0.05`、分位点 `alpha/(2×35)` 和 `1−alpha/(2×35)`，即约 `.0007142857/.9992857143`。
- CLEAN、各family、各strength、RN及固定H-vs-H20的普通区间均标为描述性；不把其中逐格跨0与否作为经过上述35比较校正的结论。
- 区间按完整配对数据重新计算，不能从旧汇总表的CI端点拼接。

20个layout的区间精度有限；100,000次bootstrap不能解决布局样本少的问题。报告这项限制，不根据区间宽度临时扩样。

### 6.3 开发集最佳固定H与选择偏差

每个算法的开发集最佳H定义为 `H_dev(s)=argmin_H M(s,FIXED_H)`；精确并列取较小H。选择只在三个噪声族的预登记宏均值上完成，不按单layout、family或测试标签事后切换。

报告该H、搬移数、Ridge对它的点差与上述同时区间。它在同一开发集上选择，不能称为独立确认的最佳基线；普通95%区间不足以消除这种选择效应。

额外报告“五个固定H中最小的开发集均值”作为同批开发数据上的固定策略包络，不将逐layout取最优H的oracle当可部署基线。若需要描述oracle，须独立标记，本轮默认不计算。

未来独立确认阶段可冻结本轮选出的H_dev再评估；本计划不打开该阶段。

## 7. 验收、结论与停止规则

技术验收：M0/M1/M2/M3证据完整，键数、绑定、物理/public审计和统计复算通过。技术PASS不表示Ridge更优。

科学结论按同时区间逐算法给出：

| 证据 | 允许的结论 |
|---|---|
| 某算法对全部五H的同时区间下界均>0 | 本开发集noisy宏指标上，Ridge优于该算法的全部注册固定H |
| 仅对H20下界>0，其他H不能区分 | 支持优于H20，未证实超过所有固定短视野 |
| 对部分H下界>0 | 仅列出已支持的H，不推断其他H |
| 某H同时区间上界<0 | 在该算法noisy宏指标上，Ridge劣于该固定H |
| 区间跨0 | 证据无法区分，不解释为等价 |

六算法总体对全部五H的S(H)同时区间下界均>0，才支持“六算法等权平均上超过全部五个共同固定H”。它不等于每个算法均更好，也不等于超过每个算法各自最好的H。六算法逐项均超过全部固定H，需要30个算法内同时区间下界全部>0。

不预设一个为了继续研究而必须达到的正收益百分比；本轮是补齐比较。报告点差、区间及退化，用结果决定是否值得另立改进或确认计划。

以下条件立即fail-stop：输入/模型/源码hash漂移、forecast namespace变化、私有信息泄漏、非法搬移/容量/守恒/终止错误、键重复缺失、桥接不一致、嵌套进程失败未解释、任何预算耗尽。

执行完整则停止；性能不佳也完整保留结果。禁止为了改善结论重新拟合、调整H集合或阈值、重选布局、扩样、增加repeat、删除CLEAN或改宏权重。必要工程修正须保留失败日志/旧绑定并重审；新轨迹始终计入预算，受影响正式数据不得混用不同实现版本。

## 8. 预算与恢复

下表是建议执行上限，执行授权时固定；不是预计耗时，也不是当前已使用时间。

| 项目 | 上限 |
|---|---:|
| M0合成测试累计运行 | 1,800秒 |
| M1/M2科学运行累计 | 7,200秒 |
| 单轨迹 | 30秒 |
| 真实trace审计累计 | 7,200秒 |
| M3分析、独立数值复算与渲染累计 | 1,800秒 |

科学运行计入桥接、smoke、bank、失败和重试；采用累计子进程elapsed-time记账，若并发则求和，不用并行缩短的外层wall time规避上限。审计、分析和等待另记，不混成性能测量。

达到上限时终止任务并输出完整/未完成键清单；禁止用不完整矩阵给主排名。工程中断可在原冻结配置和剩余预算内从完整键恢复，不为负结果重试；预算扩展或科学协议变更须新授权/新版本。

## 9. 输出与后续执行入口

计划文件：`D:\DSAI\PAPER_CODE_RIDGE_BUNDLE\docs\planning\PLAN_FIXED_HORIZON_BASELINES.md`。

未来所有新资产输出到：`D:\DSAI\PAPER_CODE_RIDGE_BUNDLE\experiments\fixed_horizon_baselines_v1\r01\`。其中 `workspace/` 放执行副本；不得由该目录的import意外回落到原工作区未冻结源码。

建议交付：

- `preparation/`：依赖清单、源码/配置绑定、复用映射、合成测试receipt、主审记录。
- `bridge/`、`smoke/`、`bank/`：真实进程receipt、完整原始轨迹、public/private trace、完整性清单及失败日志。
- `audit/`：全量物理/public/策略核验与原始数据绑定。
- `analysis/MERGED_RAW_TRAJECTORIES.csv`：15,120行，逐行标记new/reused及来源。
- `analysis/LAYOUT_AGGREGATES.csv`、`RIDGE_VS_FIXED_H_MACRO.csv`、`FIXED_H_VS_H20.csv`、`FAMILY_RESULTS.csv`、`CONDITION_RESULTS.csv`、`CLEAN_RESULTS.csv`、`DEVELOPMENT_BEST_FIXED_H.csv`、`OVERALL_EQUAL6_RESULTS.csv`。
- 图表：六算法固定H搬移曲线及Ridge水平参照、Ridge-vs-H改善率与同时区间、family分面图、CLEAN单列图；提供PNG/PDF和精确绘图数据。
- `FINAL_REPORT.md`、`HANDOFF.md`、`ARTIFACT_HASH_INDEX.json`、`FINAL_ACCEPTANCE.json`：分别记录科学结论、已完成/未运行阶段、哈希与技术验收；保留负值和无法区分结果。

原整理包的历史 `PAPER_CODE_MANIFEST.csv` 继续表示打包时快照，不追加伪造源条目。本计划及未来新增文件由新协议的清单记录。

后续执行交接语句：按本计划准备隔离执行副本，先核验5,040行旧Ridge/H20复用与来源绑定，再提交M0源码和合成证据给主审；后续依次桥接、smoke、bank、审计、统计。只补六算法四个新固定H，保留冻结Ridge及全部负结果，科学工作到本轮开发比较交付为止。
