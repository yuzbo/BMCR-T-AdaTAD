# Atlas interaction v1：完成结果与 RFV 路线对齐

2026-09-15。主代理完整阅读 Atlas 的 INTERACTION_RESULTS.zh.md，并接收负责人交付。本文核对结论范围与路线关系；没有重新运行采集、完整AP/bootstrap或PDF视觉验收。

**接收 INTERACTION_GATE=INCONCLUSIVE。实验提供了局部价值依赖计算背景及条件选择代价的刻画；尚未提供稳定的AP联合增益、全局可加性或复杂联合规划必要性的证明。**

## 交付身份与范围

- 完整211视频/792窗口；T相关为187视频/768个有合法交换的窗口，D×S覆盖全部211视频。
- 3072个八格cube、96个D×S四格cube，负责人报告34416次模型执行。没有训练新模型。
- 测量source49f74bd5601bc298997921a93193109156adf546，主统计1b3850e6db36fbfaaf43fdc71ea1e39ea22af065；探索补充0a9380a，最终renderer5d2402a。
- 六页图集为5页主图及1页探索附录。负责人已报告逐页PDF QA通过；原S八页图及旧轴内两页图保留。

## 当前能支持的结论

同cube共同基准总loss归一化后，TS/TD/DS的平均绝对交互约为7.347e−5、7.743e−5、2.839e−6，总损失交互均值的95%视频CI都跨零。不同组合的样本总体和选择粒度并不相同，不能由绝对量级排序轴的重要性。

T改变后，沿用原背景的S或D两候选参考决策，平均regret分别为1.738e−5和1.429e−5。这是使用真实收益的条件选择诊断，不是learned router结果，也没有证明最优执行顺序或Graph必要性。

6336次no-op完全一致，使epsilon=0下的“超过阈值”成为严格非零比例。该比例不能叫显著比例；同状态重放一致也不能单独排除不同输入/计算选择下的有限精度影响。

四冻结背景条件的Avg-mAP交互为−0.023734个百分点，CI [−0.103427,+0.132906]；AP@0.7交互为+0.068083个百分点，CI [−0.080802,+0.294083]。均无稳定方向。S背景的计算量同时变化，不能将四条件当作匹配成本的最终learned系统对照。

## 与当前 RFV 的关系

Atlas加强了“条件化局部价值值得测量”的研究动机；廉价信息能否预测并利用这种变化，仍需独立训练/开发证据。

RFV的R1与反向一致性4aa mini仍未通过各自学习门。不能把Atlas交互当作这些失败的原因，也不能据此直接加入Graph、Joint Planner或FVD。该Atlas实验没有跨训练checkpoint证据，不能支持RISE的未来预测主张。

publication bank继续只作刻画，不回流训练/调参。Atlas GPU查询已由负责人完成并释放，B保持HELD；本任务没有新建Atlas查询、修改队列或操作该任务的自动跟进。

## 交付链接

- [Atlas完整中文报告](C:/Users/skywalker/Documents/ChatGPT/H65/h65_clean_adatad/wtr_characterization_20260915/research/atlas_20260915/INTERACTION_RESULTS.zh.md)
- [六页正式图集](C:/Users/skywalker/Documents/ChatGPT/H65/h65_clean_adatad/wtr_characterization_20260915/output/interaction_v1_s/interaction_atlas.pdf)
- [RFV反向mini完整审阅](C:/Users/skywalker/Documents/ChatGPT/H65/reports/wtr_rfv_review_20260915/REVERSE_4AA_EXPERIMENT_REVIEW.zh.md)
