# WTR Fast-Track 实施计划（当前生效）

本计划按用户2026-09-15最新授权执行，版本v3。Fast-Track原文保存在 inputs/FASTTRACK_USER_DESIGN.txt；本轮批准与优化原文完整保存在 inputs/FASTTRACK_ACCEPTANCE_REVIEW.txt；先前双线修订原文保存在 inputs/BATCH1_USER_DESIGN.txt。旧 MASTER_PLAN/FIRST_BATCH 是历史设计依据；本文件覆盖其串行优先级及“仅建议、未授权实施”的状态。实际完成情况只以代码提交、进程/Slurm、checkpoint和评测回执为准。

## 目标与并行关系

用户最新执行偏好（2026-09-15）：边训边测，避免拆分和汇报大量独立测试任务。每个课程在同一allocation中完成启动验收后进入训练，10/20/40/60/80评测内联执行，产物保存后继续训练。仅真正独立的科学旁路按资源分配；不为每个指标、epoch或验收项另起排队任务。进度沟通聚焦训练状态、关键指标与异常。该偏好不取消必要的启动验收，也不放宽数据隔离或科学gate。

用户随后增加：每个实验完整实现后进行交叉审核与讨论。每个实验登记实现commit与对应配置，由实现者和独立审阅者核对执行正确性、设计语义及真实回执；具体疑点带file:line讨论并记录解决结果。审阅不能只看能运行或只复述设计。结论区分PASS、需修正、未完成；未实现模块不能登记PASS。实质修正后复核受影响处，再准入正式课程。已在本要求到达前启动的D pair先标为审阅候选，不能在问题未解决时用其结果解锁下游。无需为每个审阅项另建GPU任务，也不要求用户逐项批准。

目标为完整80轮训练、211视频/792窗口测试，以及经过独立验证的Graph/RISE创新结果。最终选择在独立证据约束下达到最佳accuracy–compute Pareto的最小模型；不预设所有模块必须进入。最终候选为 Grounded Value + 经证据筛选的Graph、RISE/FVD、DB、Raw。Cheap Evidence → Budget → T → D → S → Physical Recovery → TAD 的执行顺序保持。

Atlas由原owner继续执行，不重复采集Atlas，不将其publication记录用于Value训练。用户随后明确授权停止价值较低的4090旧训练：已保留V2-S完成的80轮/20、40、60 checkpoint，保留V2-B与Uniform S/B收尾；旧Full-V1、PBD、旧Graph组合及额外分散评测已退役，9个活动作业停止、384个旧阶段关闭自动启动。具体以receipts/FASTTRACK_RETIREMENT.json为准，其他成员的作业不在本次范围内。Raw已开始的独立工程原型继续推进，主训练就绪后资源优先级低于Core。

## 冻结与版本

已冻结当前固定容量科学模型版本40552945ad5d56b7404f833cd86f998e8558e8b1；旧578候选独立归档。每个新配置、checkpoint、预测、回执保留science SHA及实际source revision。11:10发现的首次评测日志问题已由6e2fc7f修复，仅记录层改变，EVALUATION_REVISION单列，保留405模型、优化器/EMA/RNG和已有10轮。修复确实改变科学执行时另登记版本，不把跨版本结果混合。旧Atlas/V2/Graph保留原source_revision。源码Git SHA用于可复现身份，不新增没有用途的逐文件校验文件。

主配置：THUMOS14、VideoMAE-S、seed42、K384；路由层0-based[4,6,8,10]；selected-Q/full-KV；global TIA；Cross multidepth；S⊆D；80 epoch；完整评测点10/20/40/60/80。基础课程关闭Graph/FVD/DB/Raw。10/20轮属于同一80轮配置的预注册开发gate，不另建短程课程或改resume配置。

## 主课程

|ID|T|D heavy attention|S heavy FFN|执行角色|
|---|---|---|---|---|
|C0|Uniform K384|100%|100%|严格parity后可只跑全量reference|
|D-U|Uniform K384|Uniform 75%|D内全部heavy|第一波控制|
|D-V|Uniform K384|Value 75%|D内全部heavy|第一波主训练|
|S-U|Uniform K384|100%|Uniform 75%|第二波控制|
|S-V|Uniform K384|100%|Value 75%|第二波主训练|
|DS-U|Uniform K384|Uniform 75%|nested Uniform 50%|第三波控制|
|DS-V|Uniform K384|Value 75%|nested Value 50%|第三波主训练|
|TDS-V|Value K384|Value 75%|nested Value 50%|第四波完整Core|

另保留T-U/T-H65/T-V；优先冻结模型或小头对照，需要时T-V与TDS-V进入正式课程。S配额必须明确分母：DS的50%指全部合法token容量，在D内按native-time分配守恒的整数quota，不能默认为D内再乘50%。D/S每个packed group记录实际计数。若开发侧真实CF没有headroom，降低对应轴优先级；不根据test早期成绩改方法或终止既有有效课程。

每条新正式课程依次通过CPU contracts → dry-run → GPU preflight → 两次optimizer更新 → fresh instance严格reload → 正式80轮。现有训练器只有原实例reload的标志，必须补独立新实例重载验收，不能沿用该标志冒充已满足新要求。复用paper_train及现有dispatch逻辑，不另建竞争调度器。

D-U/D-V是policy-level D：选中token执行A_H+F_H，未选中token执行A_L+F_L。它回答完整heavy block refinement分配，不能声称attention-only收益。另登记固定support的A75/F100（direct attention effect）与A100/F75（direct FFN effect）机制诊断，标签与主策略结果分开。

DS-V只有D-V>D-U或S-V>S-U至少一个开发gate通过后才占GPU；TDS-V至少需要T-V或DS-V中的一条learned allocation优于matched simple control。可提前注册WAITING，但依赖必须读取实际证据回执。Joint进入最终模型还需优于最佳单轴，而不是“曾启动过”即保留。

## Value bank版本合同

每条action record同时绑定checkpoint身份（课程、实际更新数/epoch及保存的权重文件）、真实support、D/S policy版本、recovery版本、light路径版本、Graph上下文版本、candidate set和完整continuation语义；raw frame IDs与episode/augmentation身份一起保存。D动作两分支分别重执行同一版本S与后续策略；S动作保持共同D及post-attention状态，再执行后续策略。

改变S策略、recovery、light或Graph语义后必须重新查询一批training/dev标签并校准，记录re-query范围和旧bank的适用边界。旧bank可以作为冻结策略离线对照，但不能改名为新策略的actual value。Online最新标签只用于其对应参数版本的更新；跨checkpoint forecasting以固定action bank另行重放。

## Graph、RISE、Raw、DB

- G0：固定真实support、D/S masks、backbone/head，比较Cross_existing、Cross_fresh、StaticGraph、DynamicGraph、DynamicGraph+Referral。所有新训练机制匹配teacher、training budget和optimizer steps，Cross_fresh是必要架构控制；历史Cross只作外部参照。报告全量AP/@0.7、短动作AP、边界误差、恢复计算量与延迟。不得拿随机未训练Graph的无增益作为方法证伪。
- G1：同一training/dev actual-CF action bank比较Plain、Graph、参数量匹配MLP；D优先，随后T/S；看Spearman/NDCG/topK/regret。G1通过才解锁TDS-V-GCTX；G0独立通过再解锁GREC。GraphKV保持关闭。
- RISE-A（独立文件/图）：严格固定(video,window,support,action,candidate set)，在detector θ20/θ40/θ60上完整重执行，测actual value drift。这一阶段不叫routing function forecast。
- RISE-B（独立文件/图）：固定相同detached原始decision state，分别由anchor/post完整routing函数（adapter/context/head/normalization均匹配快照）预测z_A、z_P，再构造z_F=z_P+(β−1)(z_P−z_A)。比较current、真正累计EMA、post β=1与β∈{1,1.05,1.1,1.2}，β只用开发侧选择。必须同时存在actual target non-stationarity与future ranking可预测性；forecast胜current/EMA且NDCG/topK不劣才解锁FVD。不得把两个checkpoint的平均值冒充真正EMA。
- Graph与FVD均通过才训练TDS-V-GCTX-FVD；保留Base/Graph/FVD/Graph+FVD四格及联合video-cluster交互统计。
- Raw：冻结detector，O-U/O-H65/R+-U/O-CF/R+-CF/O-V/R+-V；CF从同一个Official K384 support及相同合法扰动状态开始。先384-swap mini-bank，实测检查后才扩到最多6400；preview192×64只是pilot；同一无domain-one-hot Value head；记录pair/pack重组与完整decode成本。holdout优于proxy才能全量Value测试。Raw有限域和Value均有增益才解锁RAW-TDS-V匹配重训。
- DB：首个可用epoch40 Core出现后，先验证40/50/60/70/100%各tier的支持证据：mixed-plan训练曝光、每tier匹配适配，或至少证明它是经训练支持的合法执行路径。只有形状可运行不等于训练支持。未获支持的tier不得用于computation-demand解释。随后在相同平均实测成本比较fixed与GT-assisted dynamic的完整plan-response；有gap且learned planner能利用后才进入最终模型。

所有特征、β、Graph degree/referral、proposal、budget menu只在training/dev选择。正式test配置冻结后全量运行；统计按video cluster重采样并每次重算dataset AP。单seed视频CI不代表训练seed不确定性。

## 服务器安排与当前事实

1. AutoDL 44909：现有2×4080 SUPER由Atlas owner使用；保持既有采集。可使用CPU及独立目录部署，GPU短验收需由现有owner在合法阶段边界安排。
2. 4090 Slurm：已成功连接；旧owner已按用户授权清理，只保留有参考价值的收尾。Core在独立目录运行，由原owner管理D-V/D-U内联课程；Raw首个完整视频3窗口运行已通过，正式科学效果尚未验证。
3. A100 Slurm：已完成代码和资产部署；S pair长期Priority排队，已确认未启动后迁至4090，避免重复。后续A100资源可用时再分配互不重复的课程，不根据账户队列推断节点空闲。

第一个稳定S-Core开发candidate需同时满足：learned policy优于同预算simple control；actual routes确实使用声明的稀疏容量；完整实测compute下降；AP@0.7无异常定位崩溃；收益不是单个epoch峰值。应在启动前登记数值容差和相邻开发评测点，依据开发侧结果判断，不读取test调门槛。满足后立即并行B与ActivityNet的Strong Reference/Uniform/Final三个版本，不等待Graph/FVD/DB，也不复制全部消融。最终保留80轮endpoint、关键额外seed和独立系统延迟/decode测量。

## 最终模型登记与收敛

FINAL_MODEL_LEDGER.csv是组件证据登记表，字段固定为Mechanism、Gate、Evidence、Status、Final。未测为WAITING，不冒充FAIL；只有状态PASS且证据路径有效的机制才有资格标Final=yes。选择时仍比较成本收益，冗余PASS组件也可不进入最小最终模型。

先得到可证的Standard Core（可能是T、T+S、D+S或T+D+S）；再依据独立gate添加Graph、FVD及DB。Temporal Value统一为Q_T(s,C_T,m)→V_T，Standard提供C_T=O，Raw提供C_T=O∪Poff；Core不依赖前端域名称。Raw frozen domain/Value和matched retraining均成立才替换前端，失败不影响Standard Core。

解锁链：D/S单轴证据→DS；T或DS learned证据→TDS；G1→GCTX，另加G0→GREC；RISE-A与RISE-B→FVD；Graph与FVD独立通过→联合四格；trained-tier DB参考与planner→DB；Raw frozen与matched retrain→RawFrontend。既有Atlas与旧有效课程继续，新课程的等待状态不靠聊天推断。

## 当前实施状态

- 已完成：原文完整归档；Raw提交1259d25并部署A100/4090；4项CPU合同通过；4090首个完整训练侧视频3窗口的RGB/预测/官方AP缓存一致性通过。Raw科学domain/learnability gate尚未完成。Core已建独立codex/wtr-fasttrack工作树，精确D/S容量、真实交换监督、共享T Value API、内联课程和独立实例reload代码已实现，CPU合同通过。
- 最新11:33：Core D-V/D-U已训练10轮，首次内联评测日志故障修复后提交从断点补评/续训；S-V/S-U从长期未启动的A100迁到4090，当前四条均Priority排队。Raw六视频19窗口及346-swap mini-bank已完成，尚缺新增off-grid动作覆盖与Value学习证据，不直接扩6400。详见EXECUTION_STATE.zh.md和progress_20260915_1100/REPORT.zh.md。
- 待证据解锁：Graph/FVD/联合增强正式训练、Raw matched retrain、DB、B/ANet扩展。

每个正式评测点保存checkpoint、EMA预测、metrics、route/cost/value统计；Graph/FVD/Raw课程分别追加graph/snapshot/forecast/acquisition记录。报告始终区分“已实现、已通过、已提交、运行中、已完成”，不把队列登记当实验成绩。
