# 当前约束与历轮agents命令

以下是本次交接对会话有效要求的整理，不声称是完整会话逐字导出。

|当前有效决定|适用范围|
|---|---|
|H65与BMCR均为作者自研方法|作为内部基线、前身和消融；公开论文基线独立列出|
|以H65/BMCR为第一基线|原DS3/T24/U24/原16候选clip路线取消；保存历史证据|
|完整课程并行推进|不以其他路线的mAP作为启动门槛；技术、数据、权重、同课程checkpoint仍有依赖|
|每个完整配置仅seed42一次|取消多种子重复课程；断点续训属于原课程；历史seed标签不改|
|主要关注完整模型计算量与最佳性能|报告终点与课程分布；延迟/显存/E2E报告，不作路线判决条件|
|深度参考A-MoD|交替层、首末dense、attention评分；是否需要改进由研究讨论与实验判断|
|核查官方VideoMAE decoder作为初始化|这是当前讨论问题，不预设优于自建Cross或随机同架构|
|重点讨论深度、空间稀疏的性能保持|包括Progressive Block Drop-TAD与跨领域机制；此轮只形成外部讨论问题|
|本次交付为GitHub材料及讨论prompt|整理资料和提问，不按文献建议直接改变正在部署的模型配方|

|轮次|原始内容入口|当前地位|
|---|---|---|
|早期固定源码复核一|[20260911_external_bmcr_review](prior_rounds/20260911_external_bmcr_review/)|历史原件和复核结果|
|早期固定源码复核二|[20260911_external_bmcrt_review_v2](prior_rounds/20260911_external_bmcrt_review_v2/)|历史原件和复核结果|
|DS3多agents研究包|[总命令](prior_rounds/20260912_ds3/bundle/H65_DS3_research_20260911/COMMANDS_zh.md)、[角色任务](prior_rounds/20260912_ds3/bundle/H65_DS3_research_20260911/agents/)、[实验注册](prior_rounds/20260912_ds3/bundle/H65_DS3_research_20260911/experiments.json)|路线已取消，只读历史|
|239d098第一条建议|[原始回复](prior_rounds/20260912_external_239d098_dual_review/inputs/review_1/original_response.txt)、[交接](prior_rounds/20260912_external_239d098_dual_review/inputs/review_1/package/bmcr_review/CODEX_HANDOFF.zh.md)|保留建议与逐项复核；不得重新启动已取消clip路线|
|239d098第二条建议|[原始回复](prior_rounds/20260912_external_239d098_dual_review/inputs/review_2/original_response.txt)、[交接](prior_rounds/20260912_external_239d098_dual_review/inputs/review_2/package/BMCR_239d098_review/CODEX_HANDOFF.zh.md)|同上，BMCR修正与研究分支执行状态另看回执|
|Pro三轴FPW包|[原始回复](prior_rounds/20260913_fpw_tad/inputs/original_response.txt)、[agents任务书](prior_rounds/20260913_fpw_tad/inputs/package/tad3d_agents/AGENTS_TASKS.zh.md)、[命令](prior_rounds/20260913_fpw_tad/inputs/package/tad3d_agents/COMMANDS.zh.md)、[27项计划](prior_rounds/20260913_fpw_tad/inputs/package/tad3d_agents/plans/experiments.json)|原始建议；实现已从h65/frame扩展到h65/paper，当前用户决定优先|
|第一代实际FPW完整性审计|[Pro完整性](legacy_runtime/fpw_3d_20260913/PRO_COMPLETENESS_20260913.zh.md)、[下一步指令](legacy_runtime/fpw_3d_20260913/NEXT_MODEL_INSTRUCTIONS_20260913.zh.md)|历史审计；其中部分缺口已在paper代码完成，不能照抄为当前缺口|
|最新论文完整课程|[计划生成器](../../tools/paper_plan.py)、[当前计划](../paper/plan.json)、[部署](../paper/DEPLOYMENT.zh.md)|52配置/217阶段，均单seed42；结果看本次快照|

原始任务包中的CPU toy结果、设计伪代码、计划CSV、曾经估算的成本和生成的历史图，均不自动成为当前模型的实测证据。引用时必须核对来源版本和配方。
