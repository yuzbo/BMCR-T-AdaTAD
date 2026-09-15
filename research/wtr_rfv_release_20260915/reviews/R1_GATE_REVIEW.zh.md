# Value-R1：本轮同硬件结果的独立判读

2026-09-15。来源：VALUE_R1_GATE.json，代码b644d870d1845abbc1e4fd5ab7780f29ff96a53a，分支codex/wtr-rfv-20260915。仅复核已有结果与预登记门槛；未改数据、阈值、权重或启动新拟合。旧CPU R0不参与本轮差值。

**结论：LEARNABILITY_FAIL判定正确。R1的跨视频Spearman有相对改善，但没有满足主指标regret和同视频未见帧交换的联合要求；正式T/G/F课程继续未解锁。** 这既不是“R1完全没有任何变化”的结论，也不是Graph/RISE框架普遍无效的结论。

## 1. 比较对象

本轮为full-mini/within两个视角，各含Plain-R0、Plain-R1、Plain-L-R1与3个seed，共18个head。同一视角共享硬件、数据、输入/候选、训练顺序、2000updates与归一化来源。

R1为masked state-wise JS＋0.1原component-normalized Huber；407/128/64/2 Plain-M结构不变，原始cls+loc总收益按共同fit尺度进入分布目标。tau=1、未中心化raw收益的STOP阈值0均冻结。full-mini仍是mini开发范围，不是完整160/20/20或官方test。

## 2. 主要结果

|视角|R0 regret|R1 regret|R1−R0，95%视频CI|R0→R1 Spearman|
|---|---:|---:|---|---|
|跨视频，mini inner10|0.003432762|0.003372831|−0.000059931 [−0.000785028,+0.000619910]|−0.10735→−0.03627|
|同视频，25 state的held8|0.001519199|0.001598842|+0.000079643 [−0.000063080,+0.000280013]|+0.01841→+0.005397|

regret单位为实际任务loss收益，不是mAP百分点。

- full的Spearman改善为+0.071078，CI [+0.008235,+0.139804]，应如实保留这一secondary信号。但R1平均相关仍为负；regret差值区间跨零，NDCG/TopK改善区间也跨零。
- within的regret点估计更差，且高于STOP的0.001413537和随机合法帧交换的0.001535185。同视频Spearman改善CI为[−0.083810,+0.062222]，未满足改善要求。不能称其regret显著恶化，也不能称已改善。
- Plain-L-R1的full/within regret为0.003756881/0.002121039，没有解决本次门槛；不据inner结果临时改选它，也不由此推断所有容量增加都无效。

R1−R0 regret的三个seed方向：

- full：−0.000974194、−0.000190073、+0.000984474。
- within：+0.000022993、−0.000691968、+0.000907905。

均不满足一致改善。

## 3. 独立复算

主代理从本次JSON逐一重算：每seed到每video的聚合、所有R1−R0指标差值和10000次paired video bootstrap、R1与三种简单控制的regret区间。结果全部与原文件一致，独立代入预登记门槛也得到false。

六项布尔条件中，只有within的R1平均Spearman>0成立；full regret CI、within regret CI、within rho改善CI、两个视角同时胜简单控制、所有seed方向一致均未满足。该FAIL不需要修改门槛才能成立。

这些是对固定3seed平均效果的视频区间，不是训练seed置信区间，也不是全部指标的同时置信带。

## 4. 后续检查的边界

同意实现任务只补**本轮已有权重的fit评价**及原8/8候选的orientation/normalization支持范围检查，不重训或扩大模型矩阵。

- 旧CPU实验的高fit相关性不能代替本次R1是否拟合的证据。
- 先确认本轮fit排序/收益校准，再判断是训练内也未学会，还是训练内可拟合但未见帧交换失败。
- 若发现fit/held方向或合法候选语义不同，应如实说明；仅有连续特征超出fit的min/max，不自动构成协议错误。不能根据已见held结果静默重划8/8或改尺度。
- 主指标仍用raw收益与STOP。中心化分布的相关性变化不能冒充实际决策收益提升。

fit及支持范围结果由实施任务负责，本任务没有重复评价或新拟合。未完成这些检查前，不把根因唯一归为没有信息、容量不足或优化失败。epoch80、outer20封存及正式课程准入规则保持。

## 证据

- [本轮原结果](C:/Users/skywalker/Documents/ChatGPT/H65/reports/rfv_sprint_20260915/VALUE_R1_GATE.json)
- [独立复算](C:/Users/skywalker/Documents/ChatGPT/H65/reports/wtr_rfv_review_20260915/R1_GATE_RECHECK.json)
- [生效v3协议](C:/Users/skywalker/Documents/ChatGPT/H65/reports/rfv_sprint_20260915/EXPERIMENT_PLAN.zh.md)

本文仅为结果判读复核，不新增代码或科学PASS。
