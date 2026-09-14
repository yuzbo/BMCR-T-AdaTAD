# 历轮建议、agents命令与当前有效要求

本文件是研究材料导航和会话决定摘要，不声称是会话逐字导出。附件中的执行建议保留为来源；实际运行以用户的最新授权和带时间戳回执为准。旧命令不能覆盖已经取消的路线或当前seed42约束。

|轮次|完整材料/任务入口|当前地位|
|---|---|---|
|最初两轮BMCR复核|[review一](../discussion_20260913/prior_rounds/20260911_external_bmcr_review/)、[review二](../discussion_20260913/prior_rounds/20260911_external_bmcrt_review_v2/)|保留原件、源码固定点和审查记录|
|DS3研究任务包|[总命令](../discussion_20260913/prior_rounds/20260912_ds3/bundle/H65_DS3_research_20260911/COMMANDS_zh.md)、[agents角色任务](../discussion_20260913/prior_rounds/20260912_ds3/bundle/H65_DS3_research_20260911/agents/)|历史，只读；原clip选择路线取消|
|239d098建议一|[原回复](../discussion_20260913/prior_rounds/20260912_external_239d098_dual_review/inputs/review_1/original_response.txt)、[内容包](../discussion_20260913/prior_rounds/20260912_external_239d098_dual_review/inputs/review_1/package/)|与建议二分开保存，不恢复已取消路线|
|239d098建议二|[原回复](../discussion_20260913/prior_rounds/20260912_external_239d098_dual_review/inputs/review_2/original_response.txt)、[内容包](../discussion_20260913/prior_rounds/20260912_external_239d098_dual_review/inputs/review_2/package/)|BMCR修正落实及后续研究来源|
|Pro三轴/H65_BMCR_3D包|[原回复](../discussion_20260913/prior_rounds/20260913_fpw_tad/inputs/original_response.txt)、[agents任务](../discussion_20260913/prior_rounds/20260913_fpw_tad/inputs/package/tad3d_agents/AGENTS_TASKS.zh.md)、[执行命令](../discussion_20260913/prior_rounds/20260913_fpw_tad/inputs/package/tad3d_agents/COMMANDS.zh.md)|frame时代资产延续到paper；不能将历史缺口照抄为当前缺口|
|首轮论文实现/外部讨论|[原统一交接](../discussion_20260913/README.zh.md)、[当时用户决定](../discussion_20260913/USER_DECISIONS_AND_COMMANDS.zh.md)、[外部深入讨论Prompt](../discussion_20260913/DEEP_DISCUSSION_PROMPT.zh.md)|固定5485c3c的研究上下文，保持原样|
|5485c3c独立复审|[用户原回复](../paper/review_5485/original_response.txt)、[原ZIP](../paper/review_5485/TAD_5485c3c_Independent_Review_Agents.zip)、[全部解包文件](../paper/review_5485/input_package/)、[agents总指令](../paper/review_5485/input_package/AGENTS_MASTER.zh.md)|对应两类失配、状态保护、PBD、decoder与依赖实验；原源码定位可核对|
|快速并行Full-V1/V2/Simple命令|[实施计划](../paper/review_5485/IMPLEMENTATION_PLAN.zh.md)、[生成器](../../tools/paper_review_plan.py)、[登记配置](../../configs/paper_review/)|在原52上增加23项，保留P0优先级；重复机制别名不重复训练|
|原版AdaTAD直接降采样|[协议](../paper/native_adatad/PROTOCOL.zh.md)、[登记](../paper/native_adatad/registration.json)|原版K384与Uniform Full严格区分；4配置，新增2训练课程|
|Graph Machine结合与部署授权|[研究稿](../paper/model_optimization_20260914/GRAPH_MACHINE_TAD.zh.md)、[计划](../paper/graph/PLAN.zh.md)、[实现](../paper/graph/IMPLEMENTATION.zh.md)、[登记](../paper/graph/registration.json)|从“研究候选未部署”更新为7配置/36阶段已部署，GPU尚待运行|
|本次最新全部实际命令|[stage_index.json](stage_index.json)、[20:39原始服务器快照](publication_snapshot.json)|410阶段完整args、依赖、路径、job_id、attempts；覆盖旧/Support/Graph三个runtime|

## 当前有效约束

- H65/BMCR优先作为内部起点；H65、BMCR、Cross、Carrier都属于作者方案，不能冒充公开论文竞争方法。
- 原DS3、T24/U24、原16候选clip选择保持取消；不重启旧controller，保留历史结果与断点。
- 每个当前完整配置仅seed42一次；主THUMOS候选80轮，机制通常40轮及80轮LR前缀，ANet15、MQ/TadTR40按登记执行。不能把旧seed3407报告改成seed42。
- 多阶段、多路线独立实现和并行排队；只保留技术/资产/数据/自身checkpoint依赖，不根据早期mAP设科学启动门槛或早停。
- 全模型实际计算与最佳完整测试性能为主；保留终点、全部里程碑和训练分布。时延、显存、额外训练计算必须报告，但不作路线淘汰条件。
- 当前已授权Graph实施和原版降采样，旧“只讨论、尚未部署”描述已成为历史。此次提交进度资料不改变正在运行的配方。
- 唯一owner/dispatcher调度全部当前任务；不任意取消其他项目。指定协作任务此前核验占0 GPU，不能把未知账户作业当成它取消。

## 保存范围

原始任务包、原回复、代码、配置、报告与结果索引都保留在本仓库。其完整性来源是已存档内容和当前Git树，不是本次重新下载或改写外部建议。大型训练数据、预训练权重和远端checkpoint继续位于服务器；对应路径和来源保留在回执中。

此前README已保存为[README_before_consolidation.md](README_before_consolidation.md)，其内部相对链接使用当时仓库根目录语义；可在提交`8e6f6c934eaa13c72a856802602baa920384fefd`的根README查看原链接。所有旧日期“当前进度”均受本次[STATUS](STATUS.zh.md)覆盖。
