**最新部署：2026-09-13 22:33，单 seed 42**

按用户要求，每个完整配置仅运行 seed 42 一次。当前计划52个配置、217个阶段；旧论文控制器3972344已停止，新控制器456261已启动。9个旧seed作业经sacct确认全部CANCELLED：1288499、1288505、1288520–1288525、1288576。未完成的旧FPW训练已退役，不得自动重提；已有结果、权重与原seed标签保留。

新提交8个作业均PENDING：H65-S完整模型1288635，BMCR-B完整模型1288638，TadTR-B1288639，TadTR-S1288642，S全骨干微调1288640，S官方MAE decoder1288641，ANet-B技术检查1288636，ANet-S技术检查1288637。其他配置在持久计划内等待技术/资产依赖与资源，并未全部获得Slurm ID。80/40/15轮课程不变。分片接续仍属于同一次课程。

ANet准备1288466仍RUNNING；已完成BMCR80课程的S epoch75评测1288573保留。取消及新队列原始回执见seed42_transition/。ANet-B权重已完成并核验，InternVideo1-MQ仍由既有传输续传。自动跟进已同步单seed规则。新课程尚无完整测试结果，不能宣称跨随机种子稳定性。

以下为切换前历史部署记录，仅供追溯，不得据此恢复旧seed作业。

**部署回执：完整论文模型与实验**

2026-09-13 20:35:36 的权威快照：60 个训练配置、233 个执行阶段；6 条完整训练作业及 1 条 ANet-S 技术检查已经提交，均为 PENDING(Priority)，尚未产生新论文模型的训练更新或完整测试结果。控制器3972344健康接续，所有已提交job ID保持不变。其余配置位于持久调度队列，等待自己的技术检查、数据/权重或检查点；不是所有阶段都已拥有 Slurm job ID。

|正式提交内容|Slurm job|课程|
|---|---:|---|
|完整模型 VideoMAE-S / H65|1288520|技术检查后直接80 epoch|
|完整模型 VideoMAE-B / BMCR|1288521|技术检查后直接80 epoch|
|TadTR-B 完整模型|1288522|技术检查后直接40 epoch|
|VideoMAE-S 全骨干微调|1288523|技术检查后直接40 epoch，使用80轮调度前缀|
|VideoMAE-S 官方 MAE decoder 对照|1288524|技术检查后直接40 epoch，使用80轮调度前缀|
|TadTR-S 完整模型|1288525|技术检查后直接40 epoch|
|ANet-S 技术检查|1288499|在已准备真实视频上做技术检查，不能作为完整ANet实验|
|ANet完整数据准备|1288466|沿用既有journal与成品，保留43个源分片|

活动根为 `/data/run01/sczc063/yuzibo/h65_clean_adatad_20260910/paper_20260913`。权威回执为 `research/paper/deployment.json`，实际代码版本为 `source_revision.txt`，CPU验证及结果材料为 `research/paper/validation` 与 `research/paper/figures`。控制器PID以权威回执为准；不要使用旧PID重复启动。

新的控制器管理论文与继承的旧FPW队列，合计最多10个活动/排队作业、8条长训练，并遵守账户16作业上限。旧FPW控制器3944214已经停止；其8个原PENDING任务已退回后续队列，已有权重、日志、完整结果保留。当前旧TCN-B已在剩余位置重新排队为1288505。旧FPW只使用论文当前不需要的后续空槽，不另起独立控制器。BMCR80控制器保持独立继续评测；两条80轮训练早已完成，不重训。

初始6个THUMOS独立检查作业1288498、1288500–1288504已改为上表的集成课程，保留完整转换回执。两次技术更新不计入正式课程，也不作为正式初始化；验证通过后重新从固定初始化进入完整训练。长训练保存epoch、样本游标、optimizer、scheduler、EMA与RNG，以退出码75表示计划分片，由调度器继续；真实失败保留历史并等待诊断。

原始回执保存在 `raw_snapshots/20260913_2035/research/paper/`，较早的接管和集成课程回执保存在`raw_snapshots/20260913_1934/research/paper/`。包括deployment、控制器维护、队列、资源catalog及CPU结果。科学代码e55097d通过8项CPU测试（50.90秒）；7个已有资产组合已通过模型构建。新增验证覆盖了每个混合预算都有真实动作监督，完整参照相对自身的收益及不确定性为0。GPU真实技术检查尚在排队，不能用CPU成功代替GPU结果。

ANet共享状态：`/data/run01/sczc063/yuzibo/bcr_tad_v3_implementation/OpenTAD/reports/data/anet_preparation/preparation.json`；逐视频journal同目录`prepared_videos.jsonl`；成品位于`/data/run01/sczc063/yuzibo/bcr_tad_v3_implementation/activitynet/15fps_short256`。最终READY标记为本活动根的`research/paper/assets/anet_ready.json`。全10024训练/4728验证覆盖完成前，完整ANet训练和测试不会启动。

ANet-S官方TAD权重已实际读取核验。本地ANet-B与InternVideo1-MQ权重完整，远端还在传输；运行中的可恢复传输由FastCtx后台管理，当前B为`j-w0oc74`，MQ为`j-oz0xh1`。先查现有job，不重复并发同一文件；若已终止，使用`tools/paper_transfer_weights.sh`从远端已存在的部分续传。完成后由paper_assets.py实际读取tensor验证，资产错误只阻塞相关路线。

已有47条旧探索协议完整测试由paper_analyze.py重新汇总，结果与模型结构图均有PNG/PDF/SVG。`figures/REPORT.zh.md`、`full_test_records.json`和`cross_backbone_points.json`保留真实来源。论文协议本身的完整测试仍为0；真实case、OOF校准、matched-seed和paired-video图表将在各自所需模型结果产生后执行。

同一个heartbeat `bmcr-80` 已更新为跟进论文、继承FPW和BMCR80，保持30分钟检查和有意义变化通知。全部授权实验及报告完成前不删除；不恢复原DS3或原16候选clip选择路线。
