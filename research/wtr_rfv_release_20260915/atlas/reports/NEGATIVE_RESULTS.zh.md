# 当前负结果及其科学含义

核对时间：2026-09-15 13:23 +0800。正式证据来自完整S population与S/B T allocation；每模型211个测试视频、792窗口，10000次video-cluster bootstrap。T population只有187个视频存在合法单帧交换。Raw数字来自另一个任务此前完成的24视频mini-bank，不能混入正式Atlas样本。

## 1. 常用代理分数没有给出可靠的局部价值排序

S骨干D轴的attention、actionness、entropy、feature norm视频平均Spearman分别为−0.00233、+0.00550、+0.00227、+0.01334，四项95%区间均跨零。当前单点动作和冻结模型下，没有强排序相关证据。不能把这些简单代理直接当成真实计算价值，也不能反过来据此宣称新Value Head已经可学习。

## 2. 细粒度近零效应和高Gini不足以承担冗余主证据

以预登记的1%相对整窗loss容差衡量，D/S近零率99.968%/99.970%，轻扰动benign对照也达到99.941%。D/S正部Gini为0.947/0.938，benign同样高达0.882。因此，极小单点动作的尺度、整窗loss归一化和正部集中度限制了这些指标的区分能力。不能把99.97%写成可删计算比例，也不能把高Gini单独解释成可学习的强路由信号。

D/S原始负值比例约49.85%/49.15%，benign约49.90%；多数效应很小。保留有符号分布，但不能以接近一半的负号声称heavy计算有害。

## 3. 尚未测出系统性的强联合交互

S骨干O/D/S各轴16动作联合的平均I=joint−sum分别约−5.781e−5、−2.560e−6、−4.741e−8，95%视频区间均跨零。现有尺度上没有系统性强非加性的确证。区间跨零不等于等价性检验，不能据此证明全局可加、次模性或Graph无用。此处仅为同轴coalition，未测T×D/T×S/D×S。

## 4. 时间分配收益在B模型和部分预算上不稳定

有限GT辅助CF参考相对Uniform的B模型8组收益为+1.392pp，95%配对CI[−0.058,2.594]；12组为+0.413pp，CI[−0.655,1.331]。这两点未获得稳定正收益的统计支持，不能宣称所有模型、所有中间预算一致受益。S模型6/8/10/12组均为正且区间不跨零，收益约+2.028至+4.285pp；目前稳妥的结论限于所测模型、动作空间与预算。

CF选择还额外需要S/B约10320.5/34919.1 GFLOPs每窗的base及候选前向，六预算共享排序。这是测量成本；当前参考不能作为廉价可部署selector的结果。

## 5. 扩大Raw候选池尚未转化为验证侧域增益

此前Raw mini-bank每状态确实增加576个off-grid候选，但45个配对状态有38组实际查询swap集合仍相同。R+的180次查询仅24次off-grid；holdout的8个状态（4个独立视频）查询集合全部与Official-grid一致，没有看到该批验证侧的域增益。它暴露了提议与查询覆盖不足，尚不能据此判定扩大候选域本身无效。5个状态的局部收益改善来自fit，不能当成验证收益。

## 仍未得到证据支持的主张

- “边界/短动作应当优先”：S D轴位置和时长各组总效应区间跨零，但未完成组间配对差异检验，不能判定规律成立或被否定。
- “D/S完整预算分配有效、同支持恢复有检测收益”：本次核对D的30配置统计仍未全部齐备，S分配与恢复尚未完成，不能当作已有负结果。
- “Value可学、Graph/RISE带来收益”：需要独立training/development bank、匹配训练和完整AP。现有旧T calibration为group-upgrade记录，不是新RFV固定K局部swap训练bank。

这些负结果要求论文以实际同预算AP和独立可学习性验证为中心；近零率、Gini和直觉结构只提供带对照的辅助描述。

来源：receipts/population_s_analysis.json；receipts/population_s_progress_summary.json；output/temporal/report.md；discussion_raw_v1/EXCHANGE.zh.md；对方reports/wtr_fasttrack_20260915/progress_20260915_1100/RAW_SPLIT_CLARIFICATION.zh.md。正式测量a50b84d，当前分析c16bafc，已交付T图renderer6dbd246。

## 后续更新：14:11完整D预算统计已获得正证据

此前“D完整预算收益尚待验证”的状态已更新。S D的30配置全部完成211视频/792窗、官方AP与10000次bootstrap，回执receipts/allocation_s_D_summary.json。

|组数|CF−Uniform（pp）|95%配对CI|
|---|---:|---|
|4|+9.333|[7.574,10.956]|
|6|+8.255|[6.590,9.705]|
|8|+6.688|[5.196,7.741]|
|10|+5.714|[4.185,6.771]|
|12|+4.173|[3.070,4.816]|
|16|0|[0,0]|

这是GT辅助有限分组参考，执行成本差满足预登记容差，均值绝对差最大0.1651 GFLOPs；额外CF测量查询为28786.651 GFLOPs/窗。微动作近零与分组预算收益处于不同干预条件，两类结果可同时成立。现有结果支持D存在可测分配机会，Value可学习及可部署收益仍须独立验证。

## 论文图修订后的进一步限定

用户要求的Random参照与regret已补齐。attention/actionness/entropy的NDCG配对高于Random，feature norm无明确差异；所以“低Spearman”不能写成所有proxy全面失效。四项原始signed-value regret的proxy−Random配对95%CI都跨零，当前没有明确regret改善证据，也不证明等价。详见PAPER_NARRATIVE.zh.md与receipts/paper_controls_s.json。

新增benign95参照epsilon=0.00159304，超过它的D/S单点效应仅约0.206%/0.257%；这个后续描述阈值不用于过滤原数据或训练，也不是跨动作族统一的噪声下限。原1%容差与原版图均保留。
