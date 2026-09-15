# 用户要求的论文版图修订：新增描述统计合同

2026-09-15，用户在完整S population图交付后明确要求增加控制阈值、精确top-p数字、交互CI数值、Random/Actual参照及proxy regret，并将宽CI的位置/时长面板移至附录。以下定义先于新增统计的执行固定。

这是看过原结果后的用户授权补充分析，不冒充原预登记结果。原测量、1%相对loss容差、抽样、缓存和原版图保留；不增加模型查询，不按publication数据调整训练。

## Benign参照

对每窗口benign记录取abs(value)/该窗口dense总loss；每视频等权、视频内窗口等权。epsilon_benign,95采用这一加权经验分布的95%分位数，即累计权重第一次达到0.95的观测值，不做插值。

每类动作报告严格超过同一epsilon的比例，每视频内先求比例，再视频等权。10000次seed42的视频聚类bootstrap同时重估benign分位数和超阈值比例；T仅在该次重采样中的eligible视频上归一化。该阈值来自同一publication样本，是benign扰动的经验尺度参照，不是纯数值噪声、显著性检验或独立校准阈值。

原1%容差仍保留作原协议参照。新增阈值不用于过滤regret、proxy或其他后续结果。

## Top-p和交互

直接展示原缓存中的top-5/10/20正部贡献百分比，不重算或改变Lorenz/Gini定义。交互图明确标注最大已测coalition k=16的均值及95%视频CI，全部尺寸/排序的数值另附表。Caption使用“Failure to reject zero interaction is not proof of additivity.”，并明确同轴、有限动作尺度。

## Proxy参考与regret

保留D轴原有attention/actionness/entropy/feature-norm的Spearman和NDCG汇总。每窗口D的16个实测population单点效应为候选；k=ceil(0.2*N)，当前通常k=4，固定选k个。

- Spearman：随机排序的期望为0，实测value自身排序为1；常数truth的相关性无定义，沿用原coverage。
- NDCG：relevance=max(V,0)，原discount与k不变。Actual=1；Random的精确期望为mean(relevance)*sum(discount)/ideal_DCG。无正效应窗口为N/A并报告coverage。
- Regret_proxy=sum(top-k实测有符号V)−sum(proxy所选k个动作的有符号V)，单位为原始task loss。Random期望为sum(top-k V)−k*mean(V)；Actual=0。原始负效应保留，不先截成正部。也记录除以窗口dense loss后的相对数值，主图使用原始task-loss regret。

先视频内等权平均窗口指标，再视频等权；10000次seed42 video-cluster bootstrap。Regret是有限单点效应的first-order排序诊断，proxy组合没有重新执行模型。它不等价于检测AP regret，也不等价于RFV同一state中单swap+STOP的决策regret；这里不添加STOP。RFV已回传其独立定义：max(0,max actual swap value)−chosen actual value，预测全非正则STOP=0。

首次补充结果显示部分proxy的NDCG点估计高于Random，因此同时报告同窗口、同视频配对的proxy−Random差值及CI，不能用边际CI是否重叠代替差异检验。该配对分析仍属于本次明确标识的后续描述分析；保留初次输出，验证重构NDCG与原冻结汇总一致，不以结果调整阈值或训练。

## 展示与叙事

位置/时长与cls/reg条件效应已经来自完整211视频/792窗，当前不形成清晰主结论，移到单独appendix PDF。主图保留proxy相关性、NDCG及regret。所有参考限定为实测有限集合，不命名为全局oracle。

论文主线：Measured value → concentration with controls → proxy limitations → learned value；Graph须独立证明同证据条件下降低value-prediction/selection regret，RISE须独立证明跨训练checkpoint的value变化。这些characterization图不构成Graph/RISE已成立的证据。
