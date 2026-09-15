# 论文版图修订后的科学表述

## 最新主文口径：具体的局部选择改变

用户随后明确要求主文优先使用“时序帧选择/帧交换、空间token选择/FFN分配、深度计算选择/层内refinement分配、一次局部选择改变、局部计算收益、边际任务价值”。方法定义仍可用delta，历史机器字段保留。actionness和人体动作边界/时长不改。

本次显示版将主p2(d)按既有benign参照带划分任务损失增加、带内、任务损失减少；原始负/恰零/正图完整保留附录。T/S/D三类选择与观测替换O分开，不翻转O的原始符号。D/S单点测量固定token和其他计算选择，但增加FLOPs；固定整体预算的比较来自Fig4。其余特征仍正常传播。该图不能单独否定“靠近人体动作区域更值得选”等规则，也不直接衡量每次选择的完整视频mAP。

建议最新英文段落：

> Changing one selected frame, one token's FFN allocation, or one token's refinement path at a particular layer can change the final TAD task loss in either direction. At the tested one-token granularity, most FFN and layer-local refinement effects lie within an empirical benign-effect reference band, so their raw signs alone do not determine where additional computation is useful. Positive measured effects are concentrated, and simple scores recover some top-k relevance, but no clear signed-value regret reduction over random ordering is established by the paired intervals. These observations motivate learning the marginal task value of local choice changes. They do not establish global additivity, safe deletion at large scale, or the effectiveness of Graph and RISE.

本图的直接结论是：“多算这里”不保证带来明确的任务收益，应估计当前状态下这次局部选择改变的边际任务价值。具体显示定义与符号方向见LOCAL_CHOICE_DISPLAY.zh.md。以下为保留的上一版数值与论证记录，数字不变。

2026-09-15。用户要求已落实到output/pdf_s/paper_revision：主图四页，条件分析单独附录一页；全部PDF已渲染检查。原版图与原统计保留，新控制阈值和regret属于明确标识的后续描述分析。

## 新增对照实际说明了什么

- epsilon_benign,95为0.00159304，即该窗口dense loss的约0.1593%；95%视频CI为[0.00113443,0.00281519]。这是相同publication cohort上的benign扰动经验参照，不是独立校准的纯噪声阈值。
- 超过该参照的比例：Observation interpolation 26.167%、D 0.206%、S 0.257%、T 30.528%、benign 4.885%、no-op 0%。不同局部改变的粒度和条件不同，不能用这个阈值统一判定统计显著性、可删除比例或值得增加计算的位置比例。
- Top10% D单点效应贡献91.4%的正部效应，95%CI约[83.1,95.6]；benign对照Top10%也贡献79.0%。集中度需要结合控制、效应尺度和后续实际预算AP解释。
- 原四proxy的Spearman仍接近0；新增Random后，attention、actionness、entropy的NDCG分别比Random高0.06975、0.15336、0.15427，配对95%CI均正。Feature norm差−0.00781，区间跨零。不能写成所有proxy在所有排序指标上都失败。
- 原始signed-value regret的proxy−Random差：attention −1.782e−5 [−5.739e−5,2.199e−5]，actionness −1.832e−5 [−5.884e−5,2.148e−5]，entropy −8.363e−6 [−4.771e−5,3.044e−5]，norm −7.387e−6 [−4.591e−5,3.081e−5]。四项均跨零，目前没有明确的regret改善证据；这不证明与随机等价。
- Regret诊断为D的16个已测单点效应中固定选top20%，即k=4；以有符号原始task loss定义差值。组合没有重新执行模型，也没有STOP。它与RFV同一state的单swap+STOP决策regret分开。

## 建议的英文段落

> Controlled comparisons reveal heterogeneous marginal task-loss effects, with positive measured effects concentrated in a small subset of sampled observation and computation changes. For one-token FFN and layer-local refinement choices, most effects fall below an empirical benign-effect reference. Common proxies show little global rank correlation with actual value; some recover above-random top-k relevance, yet none yields a clearly established reduction in signed-value regret relative to random ordering under paired video-cluster intervals. Within the recorded small combinations of local changes, average interaction residuals are small and uncertain. These observations motivate learning conditional marginal task value as a first-order target, without establishing global additivity, large-scale safe deletion, or the effectiveness of Graph and RISE.

用户提供的原段落与这一版方向一致；新增Random和regret后，应补上“部分top-k相关性存在，但regret收益未得到明确支持”这一层，以免把低Spearman扩大成全面proxy failure。

## Graph与RISE的独立科学判据

Grounded Value目前已有前置characterization动机：实测价值分布、带对照的集中度、有限预算分配机会及proxy的不足。Value能否在独立router-label holdout上学会并改善实际检测，仍由匹配训练实验验证。

Graph的下一项主证据应是同候选、checkpoint、数据分区和证据输入下，relation降低value-prediction/selection regret；不是用当前有限同轴I去声称强交互或Graph必需。RISE需要独立、对齐video/state/局部选择改变的跨训练checkpoint测量，证明value确实变化并检验相应学习收益；这四页没有该证据。本任务不因此扩Atlas测量矩阵或启动Graph/RISE训练。

叙事顺序：Measured value → concentration with controls → proxy limitations / regret gap → learned value → Graph? → future value / RISE?。

数据来源：receipts/paper_controls_s.json、原population_s_analysis.json及output/pdf_s/paper_revision/numeric_tables.md。统计与renderer为d44df777080332876c513688f2bbdabc866acf56，测量仍a50b84d；QA见receipts/paper_revision_qa.json。
