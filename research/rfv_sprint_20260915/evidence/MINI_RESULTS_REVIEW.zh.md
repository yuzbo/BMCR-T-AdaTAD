# RFV 首批结果：独立科学复核与最小可学习性诊断

日期：2026-09-15。捕获版本 `2ca4d4b5f2ad657410bfc5c2c4cebd3bc6a11a32`；学习/统计消费者 `b679ae0ec46510bd85dc303dfacd4b6a6fd5288d`。依据真实 bank、已保存 checkpoint、G0a/G1 回执与对应代码。没有修改源码、再训练、选择新阈值、操作GPU/队列或访问正式outer20标签。

**结论：当前暂停任务级 T/Graph/FVD 长训是正确的。G0a 已证明所测局部动作空间存在 loss 改善；Plain mini learnability 与 G1 尚未过门槛。新完成的只读诊断把主要症状定位为强烈的 fit→calibration 泛化落差，而非没有拟合能力、全零标签或单纯STOP阈值问题。** 这不是“Graph普遍无效”的结论。

## 1. G0a、G1 数值与定义核验

主代理重新从G0a的20个视频loss gain计算均值及10000次视频bootstrap，得到：

- mean gain = **0.0096271103**；95% CI = **[0.0052436054, 0.0149969226]**。
- 20视频/20代表窗、60个接受swap、1145次label forward；数值与原回执一致。
- G0a PASS 是有限greedy loss-space证据，不是穷举4步oracle或完整AP PASS。

后续已直接读回G0b的完整calibration20/46窗结果：Uniform **92.49616075%**、LocalCF **93.02457398%**，Δ **+0.52841322pp**，95% CI **[+0.08244043,+0.71468687]**，AP@0.7 Δ **+0.82774438pp**。G0b确认所测局部动作空间的开发侧AP headroom；Plain泛化未过，不能仅凭G0b解锁正式课程。

G1 的signed utility为实际`gain_cls + gain_loc`；预测最大值>0才执行，否则STOP。Regret为`max(0,max actual_gain) - chosen_actual_gain`；Uniform-swap取同一合法已查询集合上均匀抽样的期望，不是再次用标签选动作。Graph类型在calibration锁定后才评inner holdout。

独立重算每视频均值与paired video bootstrap，均与原JSON一致：

|inner holdout，3 seed均值|Regret（越低越好）|Spearman|
|---|---:|---:|
|Plain-M|0.00368166|−0.09108|
|Plain-L|0.00395215|−0.05304|
|Static Graph|0.00338687|+0.01814|
|Dynamic Graph|0.00346361|−0.07314|
|STOP|0.00360015|不定义|
|Uniform swap|0.00343533|不定义|
|Transition proxy|0.00429277|−0.00118|

Static−Plain-L regret = −0.0005652892，95% CI [−0.0012700129, +0.0000951362]；三个seed的差值为 −0.00066584、−0.00128749、+0.00025746。NDCG正部relevance的平均增量为+0.01990695，区间跨零。因此`plain_learnability=false`和`graph_value_gate=false`的现有判定成立。

另计算Static−Uniform swap：−0.0000484608，95% CI [−0.0005823555, +0.0004482352]。Static相对较差Plain-L的平均改善，不能替代它相对简单控制的稳定增益。

## 2. 独立样本量必须按实际动作计

|分区|注册视频/组|有合法动作的独立state/视频|有效actions|
|---|---:|---:|---:|
|mini fit|32|25|400|
|mini calibration|10|8|128|
|mini inner holdout|10|10|160|

一共52个组、43个有动作的state、688个动作。无动作state来自当前合法集合条件，不应人为补动作凑数；排序指标只对有动作state定义，因此不能把calibration结果写成10个有效视频。

inner10来自原fit160，且与mini fit32互斥；整个mini与outer20交集为0。合并JSON里的`state_count=30`指10个state×3个seed的评估行，不是30个独立state或视频。bootstrap重采样的是10个视频上的seed平均效果，正确地没有宣称训练seed置信区间。

inner10中两个视频贡献约63.8%的STOP/oracle headroom。它说明mini不确定性较大，不构成删视频、改权重或用holdout调参的理由。

STOP的NDCG约0.2508来自全相等分数下的stable index排序，没有独立的“STOP排序能力”含义；只使用它的regret/chosen gain作为决策控制。当前gate本来就未用STOP的NDCG，因此不改变FAIL判定。

## 3. 已直接完成的最小只读诊断

原训练history只是每200步保存的随机mini-batch loss，不能据末次loss判断完整fit是否学会。为解决这个缺口，主代理使用已保存的Plain-M/Plain-L三seed权重，在CPU重新前向全部有效fit/calibration states，未做任何优化或阈值选择。

|模型与分区，3 seed均值|Regret|Spearman|NDCG|state-centered NMSE|
|---|---:|---:|---:|---:|
|Plain-M fit|0.00002808|0.86757|0.94330|0.01572|
|Plain-M calibration|0.00447655|−0.03775|0.18851|1.56325|
|Plain-L fit|0.00003847|0.87745|0.95401|0.01609|
|Plain-L calibration|0.00414966|−0.05919|0.21137|1.29179|

state-centered NMSE先在每个state内对预测与实际gain分别去均值，再以实际gain的组内方差归一化；每state预测常数对应1。它是诊断尺度，不是可部署的额外oracle控制。

Plain-M在fit上只错失约1.08%的可用headroom，已能很好地区分这400个训练动作；Plain-L也能拟合。calibration上的组内误差反而高于常数预测，说明不能把问题归结为“没优化到位”后继续加步数或容量。

同时确认实际**组合收益**分布：

- fit：正200、负200、abs(gain)≤1e−8为0；25个有效state全部同时含正负动作。
- calibration：正65、负63、近零0；8个有效state全部同时含正负动作。
- fit/calibration组合gain RMS为0.00159570/0.00152812，平均组内range为0.00572166/0.00598574。这里统计的是cls+loc实际和，不是仅cls分量或单独的两维RMS。

因此没有支持“标签全零、严重正负失衡或完全无组内监督信号”的证据。fit→calibration落差已确认；具体是训练上下文记忆化、跨视频覆盖不足还是cheap descriptor的可迁移信息不足，尚未分开。

## 4. STOP校准不是充分解释

Plain-M三个seed在8个calibration state上全部选择了swap，其中17/24次选择实际负收益。确有决策校准问题，但它伴随更严重的排序问题。

为区分两者，固定模型的top-1排名，仅让诊断oracle知道该top-1真实收益的正负，在“执行该动作/STOP”间完美选择。这不改排名、不训练模型、不选可部署阈值。

Plain-M的该oracle-sign regret仍为 **0.00373649**，而STOP为0.00385645、Uniform swap为0.00375860。即使完美解决top-1符号，最多也只利用约 **3.11%** 的calibration headroom；Plain-L同样仅约3.42%。单独调STOP能避免部分负收益，但不能补回缺失的排序收益。

## 5. 唯一建议的下一步：现有fit内按action留出

**先不新增CF/GPU采集、不扩Graph/FVD矩阵。只做一个固定Plain-M的CPU诊断：**

1. 仅使用现有25个有效fit视频；每state按预固定的物理顺序交错拆分8个拟合动作、8个未见动作。划分不读取收益。
2. 保持现有架构、目标、2000 steps和3 seeds；descriptor归一化及target RMS只使用拟合动作，公共cheap节点仍属于可见输入。
3. 在每state留出的8动作集合上计算regret与centered error，其oracle/Uniform分母也只对应这8个动作。可同时查看原calibration，正式outer20仍封存。

这一次诊断回答“是否只能记住训练过的动作，还是能在已见视频中泛化到未见动作”：

- 若同视频未见动作可学、跨视频仍差，再预登记唯一的fit32→fit64跨视频覆盖增量，保持Plain-M、目标、proposal、2000steps和3seeds不变；不同时启动采集，不优先增加参数或训练步数。
- 若同视频未见动作也差，先检查descriptor对局部交换收益的可分辨性和记忆化；仍不直接加Graph/RISE救场。

该诊断不是新的detector短课程，不产生Value/task科学PASS，也不把inner开发结果升级成论文最终holdout。它是已确认fit/cal落差之后最小的进一步分岔检查。

实现任务已明确最终唯一顺序为“先8/8，符合上述条件才32→64”。8/8脚本`d62ea556`已经完成独立代码/协议点验，只有诊断范围PASS，尚无该诊断结果或科学PASS。capture两版等价与脚本复核详见[定点审阅](C:/Users/skywalker/Documents/ChatGPT/H65/reports/wtr_rfv_review_20260915/CAPTURE_ACTION_CODE_REVIEW.zh.md)。

## 6. 证据文件与交叉讨论

- [原始G1结果](C:/Users/skywalker/Documents/ChatGPT/H65/reports/rfv_sprint_20260915/GRAPH_G1_T_MINI.json)
- [独立重算的G0a回执](C:/Users/skywalker/Documents/ChatGPT/H65/reports/wtr_rfv_review_20260915/G0A_VERIFIED.json)
- [固定Plain权重fit/cal完整重放](C:/Users/skywalker/Documents/ChatGPT/H65/reports/wtr_rfv_review_20260915/PLAIN_MINI_FIT_CAL_REPLAY.json)
- [摘要与重新计算的paired CI](C:/Users/skywalker/Documents/ChatGPT/H65/reports/wtr_rfv_review_20260915/MINI_DIAGNOSTIC_SUMMARY.json)
- [只读CPU诊断脚本](C:/Users/skywalker/Documents/ChatGPT/H65/reports/wtr_rfv_review_20260915/replay_plain_fit_cal.py)

两代理分别核验了统计定义与既有训练/checkpoint信息；主代理复核源码、独立计算视频CI、亲自完成固定权重诊断并向实现任务反馈。确认当前stop rule处理正确，RFV各长训与Final状态不因本报告自动解锁。
