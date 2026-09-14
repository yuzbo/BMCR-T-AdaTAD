# 实现、部署与整体进度

核验时间：**2026-09-14 20:39:48 +0800**。依据是[publication_snapshot.json](publication_snapshot.json)中的实际调度阶段、作业队列、训练JSONL、评测回执；[20:15初始快照](live_snapshot.json)同时保留。结果图最近生成于20:00，20:39无新增完整测试。本文件不是对计划完成率的估计。

## 最新任务的落实

|最新要求|已落实内容|实际运行边界|
|---|---|---|
|Graph Machine启发的G-Repair、G-Context、G-Full S/B并行|7配置/36阶段，图KV、时间状态、可靠anchor、frame context、反馈、损失、计费及诊断全部接入|代码已推送并部署；G-Full S/B GPU预检已提交，其余课程持久入队；尚无GPU通过与mAP|
|固定局部图、无referral、full-KV机制对照|3个S控制，40轮，80轮LR前缀；每条独立执行，不设成绩门槛|等待资源；不是已经得到机制结论|
|用原版AdaTAD直接降采样|K384/K768 S/B共4配置；K384直接官方EMA评测及GT-only 80轮适配|K384 S/B零更新评测已提交，仍等待GPU；尚无native降采样mAP|
|Full-V1 / Full-V2 / 简单强基线并行|V1/V2、Uniform、PBD/Static S/B共10个80轮P0课程|V1及V2-S断点等待续跑；V2-B、Uniform、PBD-S运行；PBD-B技术重试等待；Static持久等待|
|所有完整配置仅seed42一次|当前84训练课程均seed42；保留各自40/80轮或泛化15/40轮日程|历史seed3407证据不重标、不重跑；续训不计新seed|
|性能与总计算量为主，报告全部里程碑|保存mAP/AP各阈值、峰值、终点、窗口平均/总GFLOPs、路由、时延与训练查询计数|时延不做科学门槛；完整训练反向FLOPs尚不能由前向代理替代|
|同步全部实现与进度到GitHub|重写根README；生成86配置、410阶段索引；整理代码、命令包、结果和论文待补项|运行代码与文档提交分开；不覆盖旧证据、不改正在训练的科学配方|

Graph技术验证完成10项远端Linux CPU测试、7种实际checkpoint/数据构建与优化器/状态重载。GPU预检要求真实训练更新、关系梯度、teacher冻结、student head更新、严格重载、无GT推理和完整计费；这些GPU回执尚未产生。详见[Graph实现](../paper/graph/IMPLEMENTATION.zh.md)、[CPU验证](../paper/graph/cpu_validation.json)、[资产验证](../paper/graph/assets_validation.json)。

Native AdaTAD在官方window/crop之后精确`[::2]`：768候选→384 RGB，masks同步下采样、GT坐标除2、物理stride4→8，仍为211视频/792窗口及相同物理覆盖。保留原12层encoder、TIA、head和官方tubelet线性上采样；不加scout、Cross、D/S router或KD。CPU协议检查及S/B各499键严格加载通过；这不是GPU性能验证。[协议](../paper/native_adatad/PROTOCOL.zh.md)

## 全部注册与当前资源

实际非空config_id去重为**86**，其中**84训练课程**，另两条K768仅评测配置；合计**410调度阶段**。

|口径|已完成|运行|已提交Slurm等待|持久等待|内联等待|
|---|---:|---:|---:|---:|---:|
|训练课程|0|4|1|79|—|
|全部阶段|11|4|6|357|32|

410阶段包含84 train、213 eval、39 inline_preflight、4 preflight、30 diagnostics、7 graph_diagnostics、23 analysis、10 calibrate。训练allocation内部的里程碑评测不一定另占一个stage，所以24次当前全测与11个COMPLETED stage不是同一个口径。

唯一控制器PID **3506502**，存活。max_live=10、max_train=8、账户配额16，保留原单owner，不启动第二个竞争控制器。当前没有FAILED状态；PBD-B历史OOM及重试仍保留。

|当前作业|课程/动作|状态|最近成功训练更新 / 8000|
|---|---|---|---:|
|旧1289572，等待续跑|Full-V2-S|WAITING，epoch79内；预定时间切片|7877|
|1289573|Full-V2-B|RUNNING，epoch75内|7497|
|1289574|Uniform Full-S|RUNNING，epoch69内|6812|
|1289575|Uniform Full-B|RUNNING，epoch69内|6877|
|1289576|PBD-style-S|RUNNING，epoch48内|4749|
|1289883|PBD-style-B，含自身预检|PENDING，AssocGrpGRES|尚未正式更新|
|1289969 / 1289968|Native AdaTAD K384 S/B直接评测|PENDING，AssocGrpGRES|零更新评测|
|1290143|Native AdaTAD K768 B参考评测|PENDING，Priority；20:36补交|零更新评测|
|1290083 / 1290082|G-Full S/B技术预检|PENDING，Priority|尚未运行|
|无当前Slurm请求|Full-V1 S/B断点续训|WAITING；旧checkpoint完整保留|3360 / 2916|

V1已完成20轮完整测试，训练实际分别到epoch34/30内。19:21延期的是尚未开跑的续训申请1289884/1289885；没有取消正在训练的V1进程，也没有删除80轮任务。进入资源队列不等于已经提交Slurm，报告分别标注。

20:39复查中，V2-S的continuations明确记录`planned_checkpoint_time_slice`，在7877更新保存完整状态等待后续资源。课程仍须完成剩余123更新及80轮完整评测；不能把此次切片算成训练结束或根据成绩淘汰。控制器随后自动提交K768-B参考1290143。

**PBD-B故障处理**：原1289577在技术预检中，同支持状态cosine临时张量导致OOM。已改为数学等价的分块FP32统计与梯度重算，CPU前向/梯度/掩码核对通过。原batch、损失和课程不变；1289883等待真实GPU复验。不能提前称“GPU问题已验证解决”。

**ActivityNet**：CPU step **1289574.0**位于Uniform-S的1289574 allocation内，是明确授权的1CPU共享，不是作业号冲突或第二个GPU训练。1worker、nice15、CUDA不可见；截至快照日志至少**14312/14752**准备成功，补齐阶段300/740、failed0。`anet_ready.json`仍不存在，旧preparation.json的13374摘要不是最新journal进度。宿主切片时若step结束，应按成功journal在其他自有allocation继续；仍需10024训练+4728验证完整READY后启动正式ANet。

## 所有实验家族

|家族|训练课程数|当前进度|
|---|---:|---|
|Full-V1 / Full-V2 / Uniform Full|2 / 2 / 2|V1及V2-S续跑等待；V2-B与Uniform S/B运行|
|PBD-style / Static|2 / 2|PBD-S运行、B重试排队；Static两条等待|
|Graph|7|G-Full两条技术预检排队，其余随课程预检；7训练均等待|
|Native AdaTAD K384|2|官方直接评测排队；适配训练等待；另有K768两条评测配置，其中B已补交|
|Dense adaptation|2|等待；不同于已完成的官方dense-B零更新复测|
|Support/depth状态机制|10|全部等待|
|Decoder|6|Interp、TCN、MAE-pretrained、MAE-random、fresh Cross、MAE输入适配，全部等待|
|独立axes控制|7|全部等待；完整TDS引用主模型，预算匹配仍需论文核对|
|监督消融|4|no-external-loss、no-self-feature、no-full-GT、去整个shared-full，全部等待|
|Teacher/初始化依赖|3|无外部查询、公共识别起点稀疏/dense，全部等待|
|其他路由/结构控制|18|全部等待，含固定plan、随机、全量微调等|
|ActivityNet|6|S/B × full/uniform/dense，15轮；等待数据及资源|
|InternVideo1-MQ|3|full/uniform/dense，40轮；等待技术预检及资源|
|TadTR|6|S/B × full/uniform/dense，40轮；等待技术预检及资源|

逐条配置、完整命令、依赖和历史attempts分别见[EXPERIMENTS](EXPERIMENTS.zh.md)、[STAGES](STAGES.zh.md)、[experiment_index.json](experiment_index.json)、[stage_index.json](stage_index.json)。静态/PBD的实际配置ID为P00/P01，它们已注册，不能因文件名不含“pbd”而判为缺失。

## 当前结果与边界

已归集**95次唯一完整测试=71历史+24当前**，去除了同次metrics/completed重复。当前测试覆盖211视频/792窗口，mAP为五阈值平均；当前路线均seed42，历史seed与配方分别保留。

|模型|骨干|最佳已测 mAP %|epoch|全测试平均 GFLOPs/窗|该检查点固定窗平均时延 ms|
|---|---|---:|---:|---:|---:|
|Full-V1|S|63.4087|20|1147.2530|133.26|
|Full-V2|S|64.9949|40|1225.9797|133.54|
|Uniform Full|S|65.3369|40|1226.4780|114.01|
|PBD-style|S|63.8488|20|1131.7412|110.84|
|Full-V1|B|68.1685|10|3988.3309|233.22|
|Full-V2|B|68.5014|20|3940.8480|228.42|
|Uniform Full|B|68.3322|10|4094.3159|147.65|
|官方dense AdaTAD复测|B|71.1204|官方EMA|8082.1547|131.23|

时延在RTX4090、resident RGB固定代表窗测量，排除视频解码与NMS，不是全测试平均或端到端时延，不作路线判决条件。FLOPs采用矩阵/卷积2×MAC口径，覆盖scout/router/encoder/TIA/light/decoder/head；非矩阵索引、排序、softmax等不等价于免费，不能称精确所有算子的总运算。保留逐窗平均和代表窗的不同范围。

同60轮结果：V2-S/B **64.8354/67.7427**，Uniform-S/B **65.1959/67.8910**；S成本基本相同，B的V2平均成本低4.49%。各自最佳比较中，V2-S低0.3420pp，V2-B高0.1692pp且成本低3.75%。V2-B相对当前dense-B约削减51.24%计算、低2.6190pp mAP；这是压缩折中，不能写成超过dense精度。

**Uniform AdaTAD问题的直接答案**：原版K384直接降采样目前没有结果；65.3369/68.3322是Uniform Full的S/B峰值，不能拿它冒充原版AdaTAD降采样的成绩。

**历史内部锚点**：修正BMCR80的36阶段及24次全测已完成。S峰值epoch65为63.8372%，终点80为63.5533%；B峰值/终点80为68.3448%。历史Cross S/B为64.5743/68.5792%，历史H65-S为65.3857%。它们与当前seed42/初始化/课程不完全相同，不能直接当当前训练增益或独立公开方法。详见[BMCR80完整收尾](../paper/review_5485/bmcr80_final/)。

官方S历史完整mAP为69.0126%，旧固定窗2347.894G；本轮同口径复测仍待运行。其存在也意味着不能只在B骨干子集画前沿后声称跨模型全局最优。

PBD-style-S40为62.6013%/1037.0044G，目前仅删至10层，80轮目标9层。Full-V2-S40对PBD40的配对bootstrap差值+2.3936pp，95%区间[1.2609,3.2037]pp；固定checkpoint、211视频重采样1000次，不是多训练seed置信区间，也不是同成本比较。[完整配对报告](../paper/graph/monitor_20260914_1959/UPDATE.zh.md)

## 代码与结果版本

|对象|科学来源或登记版本|
|---|---|
|Graph部署科学代码|`841307e3af9ad6547535a61c9c0b23b9f308f383`|
|Native AdaTAD部署代码|`0313f136d076552e84a459b91869ad59169f4c3b`|
|V2/PBD原运行记录|`db9c749c7bbbdcc1b5f1f14e10b1e1d0f087cacc`|
|PBD-B技术重试修复|`5b88b3cc4dfc216df29a659892caf7d6860036b1`|
|V1/Uniform原运行记录|`2176cdbdc1eed918f4f878f653df513713f4ab52`|
|本次归集所基于的已推送报告|`8e6f6c934eaa13c72a856802602baa920384fefd`|

所有原始路径、配置、任务来源都保留在快照/结果索引中。GitHub包含可核对的代码和证据；大型数据、权重与checkpoint在原服务器，未把“提交索引”称作“上传全部训练权重”。

**整体判断**：基础代码、并行课程、计费和绘图管线已经落地；当前是完整候选的训练与证据积累阶段。80轮终点、原版降采样、Graph、匹配消融和泛化结果尚缺，现有结果不足以支撑“完整三轴动态图方案已经优于简单基线”的最终论文结论。具体论据与待补图表见[PAPER.zh.md](PAPER.zh.md)。
