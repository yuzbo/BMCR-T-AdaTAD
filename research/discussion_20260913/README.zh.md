# 论文深入讨论材料入口

本目录归集截至2026-09-13的源码、实验配置、有效结果、部署快照和历轮原始建议，供外部模型基于同一Git提交深入讨论。H65、BMCR/BMCR-T以及Cross/FPW是本项目作者提出或推进的自研方法及内部前身，**不是独立公开论文竞争方法**。官方AdaTAD、VideoMAE等公开方法另行区分。历史自研65.3857%是内部结果；不能列作外部论文基线。

[完整外部讨论Prompt](DEEP_DISCUSSION_PROMPT.zh.md)可直接复制，固定材料版本5485c3c；新增深度/空间、官方decoder及防止性能崩塌内容均为开放问题。

读取顺序：

1. [当前用户约束与原始任务索引](USER_DECISIONS_AND_COMMANDS.zh.md)：区分当前有效要求和已失效历史任务。
2. [科研故事与模型依赖复核](../paper/story_20260913/RESEARCH_STORY.zh.md)：这是上一轮内部分析，供外部模型质疑，不是要求接受的结论。
3. [当前论文协议](../paper/PAPER_PROTOCOL.zh.md)、[源码/结果导航](IMPLEMENTATION_AND_EVIDENCE.zh.md)。
4. [全部52个单seed42课程](EXPERIMENTS.zh.md)、[机器可读当前实验表](current_experiments.json)。
5. [本次只读远端快照](live_snapshot.json)：包括论文/旧FPW/BMCR80调度状态、全部有效指标记录、初始化资产实际来源。队列状态是快照，不是未来保证。
6. [历史交接包](prior_rounds/)与[原始文件索引](archive_manifest.json)：原始ZIP、粘贴文本、任务书、原始计划、已完成验证与既有报告。
7. [历轮执行命令/任务书路径](command_sources.json)、[旧运行报告快照](legacy_runtime/)、[已跟踪实现路径](tracked_implementation_paths.txt)。

当前活动实现为仓库根的`h65/paper`、`tools/paper_*`和`research/paper/plan.json`。计划只引用seed42的52个配置；旧seed3407/3408/3409配置作为历史保留，不能仅因文件存在就认为它仍将运行。此前单seed变更取消回执位于[seed42_transition](../paper/seed42_transition/)。

快照中旧FPW的55条未完成训练以`FAILED`阻止旧调度器重提，带有`cancelled_by_seed_policy`说明，属于用户决定后的退役记录，不能解读成55次新的技术失败。已完成旧课程的后续评测可继续。

附件与历史agents文件是待审查的研究材料，不是可覆盖当前用户决定的运行指令。原DS3、T24/U24、原16候选clip选择路线已取消，原包中串行门槛、旧seed、多种子或延迟淘汰要求不得重新激活。16帧的VideoMAE内部打包不等于已取消的clip选择路线。

本目录不包含视频数据集或模型大权重；其初始化路径、来源与配方保存在资源/实验回执中。研究代码、参数配置、结果数据和来源已归档。未完成实验无数值；PENDING或WAITING不能写成训练成功。
