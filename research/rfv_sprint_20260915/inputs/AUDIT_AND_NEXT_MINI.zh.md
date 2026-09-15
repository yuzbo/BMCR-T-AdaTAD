# WTR/RFV：6d1f924 源码—证据复核与下一轮最小实验规格

## 0. 身份与边界

- 审阅固定提交：`6d1f924093403362a9159072321fc4594014ac41`。
- 依据：该提交的RFV协议、执行状态、进度报告、指标JSON节选、核心源文件和交叉审阅记录；不是服务器实时查询。
- 本轮没有修改仓库、恢复或取消作业，也没有运行视频/权重级GPU复现。
- 独立执行的数值检查：D/S同epoch差值；Graph已公布10个视频差值的10000次bootstrap；RISE已公布逐视频regret差值的bootstrap；β公式边界检查。详见同目录Python和JSON。
- 未完成：全部视频/模型权重加载、原始NPZ bank重放、完整训练日志ZIP解包、全量预测重新匹配AP。不能称为整个仓库所有历史代码的动态认证。

源码链接的统一前缀：
`https://github.com/yuzbo/BMCR-T-AdaTAD/blob/6d1f924093403362a9159072321fc4594014ac41/`

## 1. 审阅结论

当前执行框架与RFV最小实验的主要监督、数据隔离、配对和统计合同基本吻合；未发现足以据此反转现有负结果的确定性核心错误。当前实现并未证明廉价Value能在保留干预上泛化，Graph独立收益未确认，历史RISE-B0没有降低选择regret。

应保留研究目标，但不能继续把“存在GT辅助机会”当作“当前Value模型可学习、可部署”的证据。下一轮只批准一个结构性Value修订及其匹配控制，不能直接解锁Graph/FVD/DS/TDS长训。

## 2. 源码覆盖与判断

|路径/入口|逐逻辑核对内容|判断及边界|
|---|---|---|
|h65/paper/operator_value.py|exact_mask、depth_mask、ff_natives_quota、spatial_mask、apply_exchange、features|D/S整数容量、S以全部合法token为分母、native quota守恒；D/S仍为unary差近似|
|h65/paper/engine.py|attention、value_score、capture_decision、重/轻A与F、TIA|D在A前、S在实际post-A后；selected-Q/full-KV；FFN路径不重复；D75/S100不是attention-only|
|h65/paper/operator_training.py|collect_operator_action|同预算交换，D后续策略重算，S保持共同D及native quota；loss为真实两分量差|
|h65/paper/readout.py|loss、components、commit_normalizer|cls/reg目标明确；配对loss不永久更新normalizer；不把repair当真实执行|
|h65/paper/model.py|构造、plan|新模块初始化使用fork_rng；当前固定容量与旧Graph recipe不能等同最终统一模型|
|tools/paper_train.py|split、初始化、恢复、CF频率、backward、EMA、fresh reload、inline eval|200视频detector训练；仅fit产生Value标签；保存真实更新数；注意全参数gradient clipping的间接耦合|
|h65/raw/contracts.py|公共episode、proposal、选择、swap、tubelet、geometric_swaps|有界合法逐帧交换；确定性打包；不是任意支持集优化|
|h65/raw/value.py|descriptors、TemporalValueHead|407维廉价公开输入、raw cls+loc排序、输出层zero-init；未显式传入受影响tubelet的完整配对上下文|
|h65/rfv/bank.py|FixedTModel、collect_state、assert_same_actions|真实固定策略重执行、同支持干预、元信息身份；标签是立即T交换后的收益，不含剩余T策略|
|h65/rfv/dataset.py|load_bank、normalization、collate|完成清单、版本绑定、JSON/NPZ标签对应、fit-only统计、sealed outer20；mini数据量按实际有效状态计算|
|h65/rfv/value.py|TemporalProbe、匹配Plain宽度、EMA/snapshot|Plain-L得到共同廉价节点；完整函数与normalization保存；Static也学习边权|
|h65/paper/edge_ops.py|initial_edges、EdgeRouter、GraphMessage|稀疏地址/连续边权是真实现；Static/Dynamic同时改变种子邻域和referral，不能单独归因于referral|
|tools/rfv_fit.py|train_one、controls、calibration锁定、holdout统计|同数据/步数/种子，正负收益及STOP明确；训练目标是两分量逐点Huber，不直接最小化决策regret|
|tools/rfv_action_holdout.py|8/8划分、fit统计、held评价|按物理顺序交错而非按收益分组；outer20未使用；是候选干预泛化诊断|
|tools/rfv_forecast.py|两阶段连续拟合、EMA、共同s40、β选择|正确β公式；future-fit标签未训练；β用了later-cal，属于回顾式跨视频诊断，不是严格时间预测|
|h65/rfv/metrics.py|regret、NDCG、Spearman、video aggregate、forecast|STOP=0；正部NDCG不替代regret；视频聚类、seed平均不是seed不确定性|
|tools/rfv_run.py|prepare、bank、local_cf、replay、smoke、finish_ap|Standard768预览；Graph192为派生节点；bank仅round0，local-CF最多4轮；完整视频合并/落盘路径|
|tools/rfv_analyze.py、h65/atlas/statistics.py|headroom、drift、AP cache、cluster bootstrap|配对视频重采样；缓存完整AP有官方对照检查；未在本机重新计算视频预测|
|evidence/descriptor_discriminability.py|固定距离、8/8身份、置换与bootstrap|只检验指定距离下局部一致性；没有精确碰撞不能证明信息充分|
|tests/test_rfv_contracts.py|padding、zero-init、β1、EMA、图参数、身份回放测试|已阅读测试定义；本轮未在完整项目环境运行这些测试|

## 3. 数值判断

### 3.1 D/S课程

|epoch20|mAP V−U / pp|AP@0.7 V−U / pp|
|---|---:|---:|
|D|−0.6444293581|−0.7169734552|
|S|−0.2609437024|−0.0952259194|

D训练已到40，不能把epoch20指标标为epoch40成绩。两个点方向不利于当前Value版本，但单seed中途差不能当epoch80或训练seed显著性结论。不同课程参数已共同适配，跨课程V−U不是“只切一个mask”的直接因果效应。

### 3.2 当前T局部动作空间

cal20全部46窗：Uniform 92.49616075%，Local-CF 93.02457398%，差+0.52841322pp，视频CI[+0.08244043,+0.71468687]。这是训练侧已见视频、有限标签辅助搜索；不是211-video官方test、不是部署Value策略、不是最优上界。

### 3.3 Value、Graph、RISE

- mini实际有效：fit25状态/400干预，cal8/128，inner10/160。登记32/10/10不等于所有视频都有合法干预。
- Plain-M 8/8：fit rho≈.9403，held≈.04444；held regret≈.00149207，高于STOP≈.00141354。
- Static−Plain-L regret：−.0005652892，CI[−.0012700129,+.0000951362]；种子方向不一致。
- Static−随机合法swap：−.0000484608，CI[−.0005823555,+.0004482352]。不能只与较弱的Plain-L比较。
- RISE Future−Current/Post regret=0，每视频差均为0；Future−EMA均值+1.4154613e−5，CI[0,+4.2463839e−5]。
- 当前Future选择β=1.1，不是β=1。正确公式为 `post + (beta - 1)*(post - anchor)`。

## 4. 必须在下一轮保留的审阅意见

### 4.1 跨课程负差不能唯一归因于打分

固定V/U checkpoint在开发视频上分别切换Uniform/learned规则，保持该checkpoint全部权重及后续策略版本；将“固定模型内策略切换”与“不同课程共同适配”分开。切换到未适配路由有distribution shift，因此仍是诊断，不是替代主表。

### 4.2 detach不等于优化完全独立

paper_train对全部requires_grad参数统一clip_grad_norm_(...,1)。即使Value输入detach，Value梯度仍可能改变全局clip系数，从而缩放detector的更新。必须先记录task/controller分组范数与clip系数；未测前不能宣称这就是掉点原因。不要在运行中的旧课程直接换成分组clipping。

### 4.3 旧配置的峰值字段仍需隔离

configs/wtr_fast/D-V.json保留best_full_test_mAP，RFV新协议规定epoch80 primary。当前报告按同epoch20比较是正确的。旧配置不原地篡改；新汇总器/最终选择清单须明确采用终点，并记录协议修订时间。

### 4.4 bank仅round0，不等于闭环状态覆盖

当前fixed-bank拟合与4轮部署的状态分布不同。现有8/8在round0已失败，不能把所有问题推给on-policy shift；后续若学会单步，再增加少量当前策略后续状态，而不是先扩海量bank。

### 4.5 输出目录复用需fail-closed

rfv_fit、rfv_forecast、rfv_action_holdout存在结果JSON已存在就返回的快捷分支。当前按版本独立目录未见误用证据；新实验应验证source/config/bank/seed/目标协议一致，不得复用旧结果文件名就显示新方案完成。

## 5. 下一轮唯一建议机制：可逆T交换的反向一致性

这是审阅后提出的候选，不是仓库已有功能，也不保证有效。

### 5.1 适用前提

固定同一detector/checkpoint、相同完整episode/RGB增强、相同D/S与恢复函数、相同loss normalizer。T干预是一对支持集S和S'=S−{i}+{j}之间的一次立即交换，随后执行固定检测函数，不包含额外剩余T策略或状态改变。

真实标签满足：

`g(S,i→j) = L(S)−L(S')`

`g(S',j→i) = −g(S,i→j)`

这里只对当前RFV的T交换使用，不自动推广到带执行历史、不可逆更新或变化剩余预算的D/S动态过程。

### 5.2 结构

设现有407维描述子为x+=phi(C,S,i→j)。从真实S'构造x−=phi(C,S',j→i)。同一个原Plain-M头f运行两次：

`g_hat = ( f(x+) − f(x−) ) / 2`

维持两个cls/loc分量，部署utility仍是两分量原始单位之和。参数共享，参数数量不增加；不强迫整个集合价值成为逐帧可加。该结构只保证合法互逆干预的反对称性与no-op=0，不保证循环一致性、全局最优或泛化。

### 5.3 为什么此修改有明确且有限的因果理由

当前自由pair MLP能记住已有候选，却不能泛化到同视频未见候选。真实标签还有一个当前自由MLP没有强制遵守的结构约束。新实验只检验利用这个约束是否改善候选泛化；不把近邻阴性结果当作信息不足证明，不新增Graph、感知输入、容量、超参数网格或teacher。

### 5.4 反向描述子可重用已有缓存

旧407块为[h_remove, h_insert, support_mean, global_mean, 23个标量]。同一cheap插值函数下：

`support_mean_after = support_mean_before + (h_insert − h_remove)/K_valid`

反向的remove/insert特征互换；global_mean不变；gap/membership必须用S'重算。时间、official/preview provenance、actionness/transition成对交换；合法membership仍是remove=1、insert=0。

不能只把旧S里的i/j交换而不改支持集合。FP32更新误差需与从原始cheap缓存直接重建的描述子比较。若扩到Graph，support_occupancy、support_distance和图状态也须基于S'重建。

### 5.5 最小匹配对照

- P0：现有Plain-M已冻结结果，仅作历史锚点；未改变条件可复用。
- P1：同Plain-M头，两个方向分别监督；loss=1/2[Huber(f(x+)/s,g/s)+Huber(f(x−)/s,−g/s)]。
- P2：同头的反对称输出；loss=Huber((f(x+)−f(x−))/(2s),g/s)。

P1/P2均两次头前向、2000更新、相同状态抽样次序、相同3个种子、相同原fit统计、相同CF标签。P1是计算与反向训练信息控制，P2是唯一结构性修订。P0/P1/P2不是三个新的detector课程。

P1推理只需正向、P2推理需双向，真实控制器成本不同，必须报告；不能以相同K声称全系统成本严格一致，也不能故意加无用计算填账。

### 5.6 验收与停止

1. 先检查少量合法T交换：正向+反向真实gain在重放噪声内为0；normalizer/权重不变。
2. CPU验证S→S'→S、有效索引、reverse descriptor、no-op及输出反对称。
3. 固定当前8/8及cal协议训练P1/P2；旧inner评估现在是开发证据，不能重新叫最终holdout。
4. 主指标regret/chosen gain/负收益执行率；rho/NDCG为辅。比较P2−P1及STOP/随机合法swap，而非只比明显较弱的旧Plain。
5. 不用调STOP掩盖排序失败。预先锁定阈值0；如要后续校准应独立登记。
6. 若仍无候选泛化信号，结束本修订，不持续扫loss、宽度、β或距离。下一阶段才讨论实际tubelet/pack变化表征或有成本的新cheap证据，重新定义独立实验。

## 6. Graph/RISE如何继续而不以故事代替证据

Graph：保留现G1负结果。若新Value规则有效，只在相同新学习规则、共同节点信息和预算上重测一个选定Graph与Plain-L。Static与Dynamic的referral因果问题须通过共同种子邻域的no-referral控制回答；当前不扩大矩阵。

RISE：先用已保存完整函数重新导出每个候选的anchor/post/EMA/future输出和选择ID，核对是否同选择、是否只是相同gain、STOP是否变化、预测变化是否跨越argmax间隔。此操作不重新训练、不生成新CF、不改β。未来teacher即便改变软分布，也不能据此宣称决策改善；当前FVD仍不解锁。

原RISE含grounded update、教师构造与distillation。当前B0是离线两阶段预测器轨迹的迁移前置测试，未运行递归FVD；应避免把B0未获益说成原RISE普遍无效，或把观测到drift说成FVD已经有效。

## 7. 并行推进指令（建议，尚未执行）

|责任|任务|验收|禁止|
|---|---|---|---|
|现有owner|保留D/S四课程，完成原定40/60/80评测|epoch与评测epoch分栏；终点及失败记录完整|从test早期负差改配置/取消/重启|
|Core诊断|固定开发checkpoint的policy crossover；记录分组梯度范数|分离mask选择、共同适配、全局clip耦合|直接将跨课程差解释成mask因果效应|
|输入/合同|实现合法反向描述子与身份检查|反向真实gain、S往返、no-op、padding|偷看heavy或GT；反向仍使用旧S|
|Value小头|只做P1/P2匹配mini|3seed、固定bank、regret与简单控制|新增长训或32→64采集|
|Graph|现有结果归档；新Value通过后一个匹配复测|共同信息、完整开销、独立inner/outer身份|扩大degree/referral扫描|
|RISE|已有snapshot的选择/间隔诊断|公式β1、选择ID、EMA与同state|扫更大β强迫改变选择|
|统计/部署|严格版本/输入/输出绑定；Atlas-S统计收尾|事实状态与证据清单|重启已完成Atlas或虚构当前空闲卡|

AutoDL空闲卡首先给需要真实执行的反向校验与Core开发re-query；小头拟合可CPU或一张卡。A100用户队列为空不等于物理卡空闲。调度以owner当时回执为准。本文件不承诺完成小时数。

## 8. 完整路线的最短剩余路径

现有四条D/S完成预登记终点；并行完成一个Value结构修订。若修订通过：在独立视频/新干预集确认 → 4轮真实闭环、完整开发视频AP → 同起点T-U/T-V适配 → 选定Graph或FVD独立收益 → 仅组合通过的机制 → 最终S/B/ANet与系统成本。

任何层级失败都只能否定所测方案与范围。若所有廉价Value方案持续不能超过简单控制，研究目标仍可调整，但不能宣称WTR learned allocation主方法成功。

## 9. 关键来源

- research/rfv_sprint_20260915/PROGRESS_20260915_1640.zh.md
- research/rfv_sprint_20260915/PROTOCOL.zh.md
- research/rfv_sprint_20260915/EXECUTION_STATE.zh.md
- research/rfv_sprint_20260915/evidence/DS_MILESTONES_20260915_1640.json
- research/rfv_sprint_20260915/evidence/GRAPH_G1_T_MINI.json
- research/rfv_sprint_20260915/evidence/RISE_B_T.json
- research/rfv_sprint_20260915/evidence/MINI_RESULTS_REVIEW.zh.md
- research/rfv_sprint_20260915/evidence/DESCRIPTOR_DISCRIMINABILITY.zh.md
- 代码审阅覆盖见第2节。
