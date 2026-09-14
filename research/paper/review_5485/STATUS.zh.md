**Full-V1 / Full-V2 / Simple强控制实施状态**

14:11–15:03更新：epoch20六组全测齐。V1 S/B=63.4087/68.1542%，V2 S/B=64.4855/68.5014%，Uniform S/B=64.5400/68.0731%。V2-B平均3940.848G/窗，相对V1/Uniform同epoch更准更省；按10/20最佳，优势为0.3329/0.1692pp。S的Uniform略高于V2且略省，不能概括双骨干都获胜。V2-B已有519/792窗选择D75/S100，其余D100/S75；V2-S仍D100。实测S路由QK增加47.186G，超过重FFN节省45.298G，是净算量不足的明确代价。图表与原始profile见monitor_20260914_1403/UPDATE.zh.md。

为使P0强基线尽早启动，14:35对最早两条V1执行USR1计划断点轮转，80轮/seed42及optimizer/EMA/RNG/cursor全部保留。PBD-S1289576已正式运行；PBD-B1289577预检发生同支持统计OOM，已用等价分块统计/梯度重算修正并通过两项CPU前向/梯度/mask检查，重提1289883等待GPU复验（尚未宣称GPU通过）。运行修正版本NEW5b88b3c、OLD0e3c716，唯一控制器1867620；V1续跑1289884/85、当前Dense-S复测1289886、ANet-B技术预检1289888排队。当前Dense-B复测已完整完成71.1204080%/8082.154709G，累计84唯一全测=71历史+12当前epoch结果+1公开复测。ANet CPU step1289574.0持续运行，无新GPU分配；全量READY仍未产生。以下均为历史快照。

12:36–12:53更新：六条核心课程的epoch10 EMA全测已齐（211视频/792窗）。V1 S/B=63.2251/68.1685%，V2 S/B=64.2351/68.3708%，Uniform S/B=63.7713/68.3322%。V2相对V1平均窗成本增加5.11%/0.64%；Uniform-S支配V1-S，V2的所有测试窗D均为100%，暂不能宣称智能选帧或稀疏深度机制成立。六条课程仍继续80轮，当前6条新结果+71条历史；同轮次图及证据见monitor_20260914_1235/UPDATE.zh.md。

ANet已启动CPU step1289574.0，在本路线Uniform-S allocation内使用1 CPU/单worker/nice15/CUDA不可见，新增GPU0。取得准备锁后取消原PENDING1289684。12:53该step RUNNING、已有真实MKV转码成功，Uniform-S仍正常更新至1363。权威data_preparation_job.json的job_id现是step字符串，查询sacct不带-X；log_path为slurm/anet_cpu_1289574.log。宿主结束/切片可能终止step，后续按成功journal接续；仍未全量READY。数据代码旧d341db5/新19e02fa，CPU共享回执写入宿主cpu_colocation.json，耗时分析需标注。以下为历史快照。

11:54:50资源协调回执已核验：指定V3/DST协作任务占0 GPU，无可取消资源；账户16张中本路线6张，其他项目/归属未明分配10张。1289618申请4张但实际分配0张，且不是指定任务。PBD/Dense/ANet继续等配额。本次无取消或重提，详见coordination_20260914_1154/DECISION.zh.md。

2026-09-14 11:28–11:45更新：Full-V1、Full-V2、Uniform的S/B六条课程均已通过各自预检并正式运行。首个新完整测试为Full-V1-S epoch10 EMA：63.2251083% mAP，代表窗1170.152664G、全测试平均1169.595255G/窗；继续完成80轮，不以此做早停。BMCR80历史两条课程和24次完整测试、对应profile全部收尾，S最佳65轮63.8372101%、终点63.5533031%；B最佳/终点80轮68.3447613%。当前新结果1条，去除同目录metrics/completed重复后历史完整结果71条。详情monitor_20260914_1130/UPDATE.zh.md和bmcr80_final/。

ANet原修复1289554在13374/14752覆盖处以INCOMPLETE退出；发现准备工具忽略现有MKV/WebM格式，已修正（新4a89751/旧deee9f2）并提交1289684接续，当前等待GPU配额。未改数据协议、不重下载；仍须完整READY。以下10:14是保留的部署快照。

2026-09-14 10:14 SSH恢复后的真实部署快照：75个完整配置、364阶段已注册；唯一paper调度器PID4185066。首批8个长训练已取得Slurm回执，另提交两个官方Dense复测。Full-V1-S已通过本课程两步预检并进入正式80轮训练，快照时epoch1、9次真实更新。其余7个核心课程仍为Priority排队。回执和检查原始值见reconnect_20260914/。

| 课程 | Slurm作业 | 10:14状态 |
| --- | --- | --- |
| Full-V1 S / B | 1289570 / 1289571 | S正式训练；B排队 |
| Full-V2 S / B | 1289572 / 1289573 | 排队 |
| Uniform Full S / B | 1289574 / 1289575 | 排队 |
| PBD-style S / B | 1289576 / 1289577 | 排队 |
| 官方AdaTAD B / S复测 | 1289578 / 1289579 | 排队 |

Static S/B已进入持久队列，训练槽释放后自动补位。远端新模型版本db9c749c7bbbdcc1b5f1f14e10b1e1d0f087cacc；旧模型及调度器技术修正版本2176cdbdc1eed918f4f878f653df513713f4ab52。未改动其他任务的GPU作业。

当前独立工作树为support_review_20260914，分支codex/support-review-20260914。保留现有52个完整课程，复审18项加最新Full-V2 S/B、late/统一深度控制和完整shared-full移除，共23项新增、75个不同完整配置；全部seed42一次。重复的C2/C3、C6/C7用同一课程别名，不重复训练。

P0核心顺序：Full-V1 S/B、Full-V2 S/B、现有Uniform Full S/B、PBD-style S/B；Static S/B随后补位。其余按P1机制/decoder、可用泛化和P2补充依次进入空槽，不使用mAP启动门槛或早期成绩淘汰。现有Uniform Full仍是原来的动态D/S架构中的uniform frame控制，不能叫纯T-only；纯T控制另用独立axes课程。

Full-V2：保持原H65/BMCR时间策略；前4层dense，路由层原ID4/6/8/10（0 based），最后一层dense；selected-Q/full-KV，独立depth-attention/depth-FFN轻残差，空间heavy/light，冻结初始encoder的同选帧支持目标，原轴Cross、可训练head、原有shared-full与外部目标。

PBD/Static按最新Final-Simple要求采用空间容量1，关闭动态空间；终点保留9/12层。此次部署统一全部10个P0课程为80轮、seed42，保存10/20/40/60/80轮。PBD在同课程0/26/52 epoch处基于固定8个训练样本的候选loss逐次删层并恢复，重建optimizer但保持全局80轮LR日程。使用本框架Adapter/norm/head，不冒充官方LoRA配方faithful复现。Static和PBD终点同层数/全容量算子成本匹配，不能声称与动态N00有严格相同FLOPs；40与80轮分别进行配对分析。

实现已包含同支持TIA前后/attention状态、独立depth-bypass、实际曝光、支持参考成本、plan-conditioned单次frame refinement、fresh Cross、MAE输入统计适配、严格无外部查询、公共K400初始化稀疏/dense配对、完整训练和续跑、官方dense同口径复测、配对bootstrap及五张决定性图的生成入口。

CPU结果：16项旧/新合同测试全部通过（47.779秒）；另以checkpoint重算和直接反传比较中间状态loss梯度，1项通过（26.947秒）。真实权重下11种代表性模型构造通过，公共S识别预训练163 keys且无TAD Adapter/scout/head，官方MAE S/B资源可用。冻结参考采用从同一资产重新构造，避免框架包装器不可deepcopy的问题。

GPU技术阻塞已定位并修正：1288834三次输出均有限、选帧/预算/RNG不变，但相同输入重复差1.3351e-5，改GT差1.1444e-5。预检局部使用确定性cuDNN与math SDPA，保持atol=1e-6、rtol=1e-5，并新增重复输入也必须通过。1289556在真实4090上完成两次任务更新，重复输入/修改GT差异均为0，教师冻结、head更新及重载全部通过；正式训练和生产profile仍使用原算子路径。原9次技术失败保留attempts和诊断后重新入队。TadTR单进程distributed初始化修正已同步；各课程仍执行自己的真实预检，不将S通过冒充全部架构已通过。

SSH已恢复，可偶发关闭连接，重试后执行成功；未改全局网络设置。ANet准备作业1289554已运行：复用已传完的val.09，内容与发布者匹配，损坏原件保留，已替换并接续准备。不重复下载；全数据完成仍以10024/4728和anet_ready.json=READY为准，当前不能宣称READY。

绘图：已生成并视觉检查Full-V2网络结构的PNG/SVG/PDF；新实验性能图在无结果时明确列为等待，不填点。现有69条历史完整测试已读取归集。需在连接恢复后同步最新结果，不能将历史Cross或BMCR成绩作为Full-V2成果。

主要入口：tools/paper_review_plan.py、paper_support_check.py、paper_review_assets.py、paper_review_register.py、paper_review_diagnostics.py、paper_review_analyze.py、paper_review_plot.py、paper_public_dense.py。已实现与待部署状态分开，不以本文件替代实际Slurm回执。
