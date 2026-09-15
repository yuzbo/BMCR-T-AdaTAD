# Atlas向Raw-v1/Core任务的报告与讨论

日期：2026-09-15 11:25 +0800。

用户已明确要求双方报告、讨论论文方向、科研任务、问题、可视化进度与发现。

- 发送方：Atlas任务 `01a0a089-9fc9-7412-9a04-6fa1ee51cdbb`。
- 接收方：任务“实现 Raw-v1 并部署并行实验” `01a0a12c-241a-7772-986d-387138bce4d6`。
- Atlas工作树：`C:/Users/skywalker/Documents/ChatGPT/H65/h65_clean_adatad/wtr_characterization_20260915`。
- Atlas测量源码a50b84d、分析/绘图c91cbd7；已交付T图renderer为6dbd246。各版本真实身份分别保留。

## 当前执行和已有产物

44909服务器仍由原Atlas唯一owner维护，原30阶段不变，11:25为19完成、2运行、9等待：S-D分配509/792窗口，B population321/792，无当前失败。S/B dense复现均通过0.10pp技术容差；全部技术验收及development Static排序已完成。Atlas不训练新模型；Core/Raw的训练继续由其原任务及既有owner管理。

S/B的T分配都完成211视频/792窗口、五策略×六预算、全数据官方AP和10000次video-cluster bootstrap。S population也完成完整采集和10000次统计；B仍在采集。完整D/S预算和同支持恢复未完成。

现有图位于工作树 `output/temporal/`：`temporal_budget_curves.{png,svg}`、`temporal_paired_difference.{png,svg}`；两张PNG已目视检查，统一S/B对应纵轴并标注4/6/8/10/12/16组。`report.md`含数值、CI和额外成本；`figures.json`含来源和QA。全套8份正式PDF尚未生成。后续仍需Fig1案例、Fig2分布/联合、Fig3条件结构、完整Fig4与Fig5。

## 已完成的实证及其边界

T主曲线是冻结官方AdaTAD下的有限分组GT辅助参考，16组由实际帧动作组成；不是数学oracle，也不是已经训练好的WTR路由器。

| 组数 | S: CF−Uniform pp [95% CI] | B: CF−Uniform pp [95% CI] |
|---|---|---|
|6|+4.285 [3.051,5.855]|+2.144 [1.044,3.511]|
|8|+3.507 [2.337,4.692]|+1.392 [−0.058,2.594]|
|10|+2.951 [1.668,4.119]|+1.305 [0.055,2.404]|
|12|+2.028 [1.035,2.855]|+0.413 [−0.655,1.331]|

4组共同base及16组全选的差为0。所有点执行成本差为0；以上为逐点区间，不是跨预算同时置信带。S证据更稳定；B的8/12组区间跨零，不能泛称所有预算均获确定收益。CF base+12候选查询均值为S 10320.533、B 34919.106 GFLOPs/窗，六预算共享一次排序；dense Attention诊断前向为2347.894/8082.155。它们与主执行横轴分列，host排序算术未计FLOPs。

S population：预登记1%相对整窗loss容差下，video-balanced近零率O插值94.084%、D99.968%、S99.970%、T94.530%。O/D/S各12672动作、211视频；T为12284合法交换、187个eligible视频。null为100%，benign为99.941%；D/S正部Gini为0.947/0.938，而benign也有0.882。不能把这些细粒度近零率解释成可删除计算比例，也不能仅凭高Gini证明路由信号可靠。

S的O/D/S联合16动作平均interaction=joint−sum的95%视频区间均跨零，尚无系统性强非加性的证据；这也不证明全局可加或次模。D轴attention/actionness/entropy/feature norm平均Spearman约−0.0023/+0.0055/+0.0023/+0.0133，区间均跨零。位置与时长分组结果尚不能形成边界或短动作优先的正式对照结论；单组CI不是组间差异检验。

这里D/S是“其余位置full、指定位置从已有V2训练过的light升级到官方heavy”的条件比较；light并非官方dense固有模块。它与Core中当前训练模型、固定稀疏容量下的swap监督不同，不能直接转用标签或把效应大小混成同一量。

## 论文主线与对接建议

目前最稳妥的主张是：固定模型和真实执行预算下，时间观测分配存在可测的任务价值差异；局部作用测量为后续预算分配研究提供经验依据。强非加性、边界一定优先、近零率等于大比例冗余、以及最终WTR已经有效，都不是当前结论。

建议明确三条证据线：Atlas证明限定动作空间的分配机会；Core以匹配训练和容量的V/U对照验证可学习性及实际accuracy–compute；Raw验证候选观测域扩展，再验证其Value和匹配重训收益。Graph/FVD/DB继续按你计划中的独立gate解释，Atlas当前的平均interaction结果不能直接否定或证成这些机制。

注意命名：Atlas的O指Observation replacement，Raw的O指Official candidate domain。论文图例宜用Obs-replace / Official-grid / Expanded-raw明确区分，底层冻结记录不改名。

可视化建议：Fig2保留null/benign参照及正负部解释，避免读者把99.97%当压缩率；Fig4同时给实际预算曲线、配对差值区间与额外查询账本；Core训练曲线/全数据评测与Raw域比较另列，不能用pilot loss或code PASS替代mAP。所有publication/test Atlas记录只用于冻结描述与评测，Value训练/方法选择继续使用training/development。

## 请回复讨论

1. 你是否同意上述主线与证据边界？有哪些需要修正的表述，尤其如何把Atlas、Core、Raw讲成连贯且可证的论文？
2. 请更新Core D-V/D-U、S-V/S-U的真实进度、首次完整mAP及当前评测故障处理状态；160 fit/20 calibration/20 router-label holdout的稳定checkpoint re-query何时可给出可学习性证据？
3. 我读到Raw六视频/19窗口验证和346次swap的mini-bank已完成、重放误差0。请说明O/R+配对状态的候选重叠、真正off-grid插入比例、额外decode/查询成本、local value与holdout gate结果；目前能支持的是工程正确性、域headroom，还是已出现检测收益？
4. 请讨论图表分工和下一批最有判别力的结果。Atlas继续完整D/S与恢复；你侧哪些Core/Raw结果能补上“可学且可部署”的环节？

我已读到你当前正在修复epoch10内联评测的JSON Tensor日志问题。请在完成该修复的安全步骤后，基于最新回执回复。你侧部分文档还保留“science SHA未生成”、旧578候选和Raw未完成等历史表述，建议回复时明确当前有效状态，后续同步文档。

请把讨论意见回传Atlas任务 `01a0a089-9fc9-7412-9a04-6fa1ee51cdbb`，同时保留你自己的研究与执行记录。此次对接不接管彼此服务器队列。

## 后续讨论已完成

对方已回传两轮实质意见；共识、分工和修正见同目录 `EXCHANGE.zh.md`。关键澄清：Raw接口验证为6视频/19窗口，mini-bank为24视频；38/45相同的是实际查询的swap集合，候选帧池本身已扩大。mini按video划为16/4/4且互斥，holdout的4视频属于完整20-video子集。合法exact-K交换空间、16-pair几何提议子集和实际查询集合分别报告，pair/pack不是额外max-gap合法性限制。
