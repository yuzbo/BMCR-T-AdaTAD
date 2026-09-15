# WTR / RFV 统一术语：intervention／干预

生效日期：2026-09-15。依据用户最新术语指令，适用于后续论文、图表、汇报及研究讨论。研究中的受控计算或观测变化统一称为 **intervention／干预**；人体动作类别、动作实例、动作边界及actionness继续使用TAD本来的语义。

## 术语表

|English|统一中文|说明|
|---|---|---|
|intervention|干预|本研究所定义的受控变化|
|observation intervention|观测干预|改变可用输入证据|
|computation intervention|计算干预|改变一次合法计算路径或分配|
|depth intervention|深度计算干预|具体执行语义须随实验注明|
|FFN intervention|FFN计算干预|改变FFN计算路径或分配|
|frame exchange / frame-exchange intervention|帧交换干预|固定合法支持集上的移出/移入|
|joint intervention|联合干预|多个变化的合法组合|
|single / individual intervention|单项干预|一个已定义的干预单位，不强求只影响一个底层算子|
|candidate intervention|候选干预|当前允许评估的合法变化|
|intervention set|干预集合|须满足共同合法性与预算约束|
|intervention space|干预空间|合法变化的范围，不等同于实际查询子集|
|intervention identity|干预身份|绑定状态、支持、位置、配置与执行语义|
|intervention effect|干预效应|受控变化对指定测量量的影响|
|marginal intervention value|边际干预价值|本研究的主要V术语|
|marginal task value of an intervention|干预的边际任务价值|强调V用任务效用度量|
|task effect|任务效应|须说明指标、单位与方向|
|intervention interaction effect|干预交互效应|联合效应相对单项效应之和的偏离|
|future marginal intervention value|未来边际干预价值|涉及后续训练时点的价值|
|future computation value|未来计算价值|强调计算分配时的简写|

正文不再以action value或“动作价值”指上述研究量。`actionness`表示人体动作存在性线索，不改成干预相关概念。

## Atlas与RFV显示标签

|统一English label|统一中文|
|---|---|
|Fraction of interventions below tolerance|低于效应阈值的干预比例|
|Highest-effect interventions (%)|任务效应最大的干预比例|
|Number of jointly applied interventions|联合施加的干预数量|
|Sum of individual intervention effects|单项干预效应之和|
|Joint effect of individually low-effect interventions|单独效应较低干预的联合效应|
|Observation replacement intervention|观测替换干预|
|Depth-path switching intervention|深度路径切换干预|
|FFN-path switching intervention|FFN路径切换干预|
|Frame exchange intervention|帧交换干预|
|Benign control intervention|良性对照干预|
|No-op control|空操作对照|
|Intervention holdout|干预留出|
|Intervention regret|干预选择遗憾值|

路径类别默认使用switching／切换，避免upgrade／增强预先暗示heavy路径更优。当前D-U/D-V是heavy attention＋FFN对light路径的策略级切换，仍须注明，不能凭名称改称attention-only或整层early exit。

图注须保留原测量定义。例如按绝对效应排序时明确写effect magnitude／效应幅度；低于阈值的比例须说明是绝对变化、相对变化还是其他已登记尺度。术语替换不改变数值、排序、符号或阈值。

Atlas的O表示观测替换干预；Raw中的\(\mathcal O\)表示官方候选域。图例与正文明确区分二者。

## 符号与符号方向

新正文统一使用\(\delta\)表示一次候选干预，\(\Delta\)保留为差值符号。旧资料中的\(a\)若需要引用，应解释为旧版候选干预符号，不解释为人体动作实例。

在固定模型参数\(\theta\)、合法预算及后续执行策略\(\pi\)下，定义：

\[
V^\pi_\theta(\delta\mid s)
=\mathcal L^\pi_\theta(s)-\mathcal L^\pi_\theta(s\oplus\delta).
\]

\(s\oplus\delta\)表示合法施加干预后的状态；\(\mathcal L^\pi_\theta\)表示完成所声明后续计算后的任务损失。因此V是**损失减少量，允许负值**：V>0表示损失下降，V<0表示损失上升。泛称“干预效应”时，仍按相应图表注明测量方向，不能把所有变化量都默认当正收益。

RFV现有标签保留分类/定位两个分量；用于排序的价值为二者实际收益之和。归一化只按已登记规则服务回归，不借术语变更重定义目标。

固定预算的帧交换或计算slot交换属于重分配，不要求新增计算量。论文主句建议写为：

> 在给定已有证据、合法预算和后续执行策略的条件下，我们估计每个候选干预对最终任务带来的边际收益。

## 联合干预

当联合变化可合法、无歧义地作为集合施加时：

\[
I(\delta_1,\delta_2\mid s)
=V(\{\delta_1,\delta_2\}\mid s)
-V(\delta_1\mid s)-V(\delta_2\mid s).
\]

称为**干预交互效应**。三个效应都从同一基准状态、模型和策略定义。若顺序影响合法性或结果，则先定义有序组合\(\delta_{12}=\delta_2\circ\delta_1\)，再用\(V(\delta_{12}\mid s)\)；不能用无序集合隐藏执行顺序。若某单项在基准状态下不合法，不计算上述没有对应合法对照的交互量。

## Graph与RISE

当前Graph原型的节点是廉价计算状态，推荐准确写为：

> Graph models relational context among cheap computation states to predict the marginal value of candidate interventions.

> Graph建模廉价计算状态的关系上下文，并据此预测候选干预的边际价值。

可写\(\widehat V_\phi(\delta\mid s,G_s)\)，称为**关系条件下的边际干预价值**。这不声称当前实现已经把候选干预本身建成图节点。

RISE使用**未来边际干预价值**或**未来计算价值**。保持已冻结约定：

\[
\widehat V_F=\widehat V_P+(\beta-1)(\widehat V_P-\widehat V_A),\qquad \beta=1\Rightarrow\widehat V_F=\widehat V_P.
\]

各函数接收同一\((\delta,s,G_s)\)所对应的原始输入，并完成自身上下文和归一化。Current/Post、真实EMA及未来目标的时间边界继续按RFV协议记录。

## 历史记录与显示层映射

现有`action_pairs`、`action_ids`、`ACTION_HOLDOUT_DIAGNOSTIC.json`等属于可复现记录的历史机器标识。引用这些真实字段或路径时保留原字面量，同时在自然语言中称“干预对”“干预ID”“干预留出诊断”。新显示标题、图注与说明采用本规范；代码、数据与图表显示版本按各自实际修改登记。

明确保留TAD的人体动作术语：action class／动作类别、action instance／动作实例、action boundary／动作边界、short/long action／短/长动作、actionness／动作存在性线索。

用户原始建议见[完整记录](C:/Users/skywalker/Documents/ChatGPT/H65/reports/wtr_terminology_20260915/USER_PROPOSAL.zh.md)。
