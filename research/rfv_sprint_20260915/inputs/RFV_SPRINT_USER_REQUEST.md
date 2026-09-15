我建议现在明确提高 Graph / RISE 的优先级，但不是把它们未经验证直接塞进当前 D/S 长训，而是把接下来实验安排改成一次专门的 RFV Sprint（Relational Future Value Sprint）。

从当前公开代码和结果看，工程推进速度已经很快；真正落后的不是实现数量，而是决定论文核心方法形态的科学证据。目前最强证据仍然是 Temporal：S 上 6/8/10/12 groups 的 CF–Uniform 都有明显正 gap，8 groups 为 +3.507pp [2.337, 4.692]。相比之下，新 D/S learned Value 还没有完整 mAP，Graph G0/G1、RISE-A/B 在 ledger 中仍是 WAITING。

因此，如果你认为 Graph/RISE 对论文完整性更重要，我赞成把路线改成：

$$ \boxed{ \text{T Grounded Value} \parallel \text{Graph Relational Value} \parallel \text{RISE Future Value} } $$

尽快闭合：

$$ \boxed{ V\_{\text{future}}(a\mid s,G\_s) } $$

而 D/S 继续训练，但不再阻塞 Graph/RISE。

一、对当前进度的评价

1. 当前实验推进是合理的

而且过去这一阶段几个关键工程问题已经解决得比较扎实。

D/S 的真实交换并不是伪标签：代码会运行 baseline，强制一个合法 D/S exchange，再完整重新执行 downstream policy，用真实 base\_loss - changed\_loss 监督 Value；S 还显式要求 changed branch 保持共同 D support。

D/S 的精确容量也实现正确：D75 是实际整数 quota，DS 的 S50 是相对全部合法 token 的 50%，然后在 D 集合内按 native-time 守恒分配，不是 0.75×0.5。

160/20/20 的 router-label split 也已经真正进入训练入口，只有 fit 视频产生 Value supervision，而 detector 仍在全部200个训练视频上学习。

Raw 方向也已经从“概念”变成可运行接口：Episode、preview、bounded proposal、arbitrary frame read、tubelet metadata 和 384→192→384→768 recovery bridge 都已经实现。

所以：

$$ \boxed{\text{现在不是实现能力不足，而是科学结果还没跟上实现速度。}} $$
2\. 过去8小时工程推进足够快，但决定性科学推进还可以明显加速

我不会评价为“慢”。

你们已经完成了很多过去往往需要数天的事情：

新 Core science SHA 冻结；
D pair 10 epoch / 1000 updates；
evaluation serialization bug 定位、修复且证明只涉及 evaluator；
fresh-instance reload；
Raw GPU bridge；
CF mini-bank；
full-dataset temporal characterization；
代码交叉审阅；
160/20/20 leakage 修复。

而且我直接比较了 4055294→6e2fc7f，确实只有 evaluation serialization/metadata 变动，没有改变模型 forward 或训练科学语义。

但未来8小时如果仍继续主要产生：

更多接口、更多配置、更多 WAITING stages

价值就会下降。

接下来应该转成：

$$ \boxed{ \textbf{少写新模块，快速产出 Graph/RISE 的 PASS / FAIL 证据。} } $$
二、为什么现在应该把 Graph/RISE 提到主优先级

你这个判断我现在更认同了。

如果最后论文只是：

“T/D/S 上学一个真实 Value router”

这是一个不错的方法。

但如果实验支持：

$$ \boxed{ \text{Marginal Value is relational} } $$

以及：

$$ \boxed{ \text{Marginal Value is non-stationary during learning} } $$

那么论文会从一个 dynamic-computation framework，升级成一个更明确的方法论：

$$ \boxed{ \textbf{Grounded Relational Future Computation Value} } $$

这比“Graph + RISE 两个附加模块”强得多。

三、Graph 的优先实验对象现在应该从 D 改成 T

当前 Fast-Track 文档仍写着：

G1：D优先，随后T/S。

我建议现在正式改成：

$$ \boxed{ T\rightarrow D\rightarrow S } $$

原因不是理论偏好，而是新证据改变了优先级。

T 已经有全量：

$$ 211\text{ videos}/792\text{ windows} $$

的明确 allocation headroom。

而 D 当前仍处在：

Atlas population 未完整；
learned Value 无完整 mAP；
当前简单 proxy 相关弱；
policy-level D 还混合 A/F effect。

所以如果我们的目标是：

最快判断 Graph 能不能提升 Value，

应该把 Graph 放到已经知道“这里有东西值得学”的 T 轴。

四、未来 12 小时：只做四个 Graph/RISE 决定性实验

这是我建议实验 owner 直接执行的核心表。

实验	当前输入	新增代码量	回答的问题	通过标准
G1-T	T actual-CF bank	小	Graph context 是否改善 T Value	Graph regret < matched-MLP
R1-T	固定 T action manifest	小	actual T value 是否随 checkpoint 漂移	later ranking/sign 有实质变化
R2-T	相同 detached state	中	extrapolation 是否预测 later value	Future regret < current & true EMA
T-local-CF	当前 T-V 4-round action space	极小	当前 T-V action space 自己有没有 headroom	local-CF > Uniform

这里第四项非常关键。

因为 Atlas 的 +3.5pp 来源于较宽的 group action space，而当前 T-V 是：

$$ Uniform\ K384 + \le4\text{ swaps} $$

每轮只检查最多16个 bounded pairs。

所以不能把 Atlas +3.5pp 直接当作当前 T-V 的理论上限。

先测：

$$ M\_{\rm localCF}-M\_{\rm uniform}. $$

如果这个 gap 本身很小，就应该先扩 T proposal/action space，而不是拿 Graph 或 RISE 去修一个 action-space ceiling。

五、Graph G1-T 应该怎样实现

现有 Graph primitives 可以复用，但当前 Graph-conditioned Value 还不存在。

这是我代码审核得到的明确结论。

目前 Graph attention/recovery 的 sparse edges、referral、indexed gather 都是真实现，而且 GraphKVAttention 也诚实地保留全 K/V projection。

但当前 OperatorValueRouter 和 TemporalValueHead 都没有 Graph input；而 WTR recipe 与 graph\_tad\_v1 仍是两条执行路径。

所以 G1 第一版不要整合 GraphKV。

建议只实现：

$$ \boxed{ g\_i=\operatorname{GraphContext}(C^{cheap},t,S) } $$

然后：

$$ Q\_T(i\rightarrow j) = f( x\_{ij}, g\_i, g\_j, g\_S ). $$

比较三个模型：

$$ \text{Plain-M} $$

当前 Temporal MLP；

$$ \text{Plain-L} $$

参数量匹配的大 MLP；

$$ \text{Graph} $$

关系条件化 Value。

全部：

同一 fit bank；
同一 calibration；
同一 holdout；
同 optimizer steps；
同 action IDs。

第一关不要跑 detector 80轮。

先只看 holdout：

$$ Spearman,; NDCG,; TopK,; Regret. $$
六、Graph PASS 最好定义成两级
G1a：Value-level PASS

要求：

$$ Regret\_G < Regret\_{LargeMLP} $$

并且：

$$ NDCG\_G>NDCG\_{LargeMLP}. $$

至少在 video-level aggregation 上方向稳定。

这说明：

Graph 的收益不是“多了参数”。

G1b：Task-level PASS

只有 G1a 通过以后，才训练：

$$ T\text{-V-GCTX}. $$

然后 matched compare：

$$ T\text{-V} $$

vs

$$ T\text{-V-GCTX}. $$

要求：

full mAP 改善；
matched execution cost；
AP\@0.7 不出现异常损失；
Graph overhead 被完整计费。

只有两个 PASS 都有，Graph Context 才进入 FINAL\_MODEL\_LEDGER=yes。

七、GraphRecovery 暂时放第二优先级

不是不做。

而是：

$$ \boxed{ G\text{-Context} > G\text{-Recovery} > GraphKV. } $$

GraphRecovery 当前已有合理 primitive，包括 physical-time query state、contributor-aware deposit、reliability、residual Cross repair。

但它回答的是：

如何恢复 representation？

而 Graph Context 回答：

relation 是否决定 computation value？

后者更直接服务论文核心。

所以 G0 可以后台继续，但不要抢 G1 的主资源。

八、RISE 第一件事不是实现 FVD，而是证明 Value Drift 存在

RISE 必须先过：

$$ \boxed{\text{Existence Gate}} $$

否则没有理由做 Future Value。

固定一个 action bank：

$$ \mathcal A={a\_1,\ldots,a\_N}. $$

对完全相同：

video；
window；
support；
action；
candidate set；

分别在：

$$ \theta\_{10},\theta\_{20},\theta\_{40},\theta\_{60} $$

真实重执行：

$$ V^{actual}*{10}(a), V^{actual}*{20}(a), V^{actual}*{40}(a), V^{actual}*{60}(a). $$

这就是 RISE-A。

当前生效计划也已经正确把 RISE-A 与函数外推 RISE-B 分开了。

九、RISE-A 应该输出什么

不要只报相关系数。

至少输出：

$$ \rho(V\_{10},V\_{60}) $$ $$ \rho(V\_{20},V\_{60}) $$ $$ \rho(V\_{40},V\_{60}) $$

同时：

$$ TopKOverlap\_{10,60} $$ $$ SignFlipRate $$

和：

$$ Regret(V\_{10}\rightarrow V\_{60}). $$

最重要的是分层：

boundary；
interior；
background；
short action；
long action。

如果：

$$ V\_{10}\approx V\_{20}\approx V\_{60}, $$

RISE 就应该删。

这是一个完全可以接受的负结论。

十、RISE-B 必须严格做“同 state 函数外推”

不要拿不同 checkpoint 自己自然产生的 state，直接把两个 logits 相减。

必须 cache 同一个：

$$ s,\mathcal C. $$

然后分别运行：

$$ z\_A=Q\_{\Psi\_A}(s,\mathcal C) $$ $$ z\_P=Q\_{\Psi\_P}(s,\mathcal C) $$

再：

$$ z\_F = z\_P+\beta(z\_P-z\_A). $$

β：

$$ 1,;1.05,;1.1,;1.2. $$

只在 calibration 20 视频上选。

真正 holdout 20 视频只做一次最终判断。

十一、EMA control 必须是真 EMA

当前训练代码已经有真正的 trainable-state EMA：

$$ EMA\_t=\alpha EMA\_{t-1}+(1-\alpha)\theta\_t $$

，并且会追踪 requires-grad Value parameters 和 router mutable buffers。

所以 RISE 不能拿：

两个 checkpoint 简单平均

冒充 EMA。

必须比较：

$$ Current $$ $$ TrueEMA $$ $$ Post;(\beta=1) $$ $$ Future;(\beta>1). $$

只有：

$$ Regret\_{Future} < \min(Regret\_{Current},Regret\_{EMA}) $$

且 NDCG/TopK 不恶化，才解锁 FVD。

十二、接下来我建议马上的实际任务队列

为了符合你“更快确认 Graph/RISE”的要求，我会把执行顺序改成：

主 GPU 线

先完成：

$$ D\text{-V/U\@10 full eval} $$

因为 checkpoint 已经存在，不能浪费。

然后立即切：

$$ \boxed{T\text{-V technical gate}} $$

与此同时 S-U/S-V 继续排队/训练。

Graph 线

立刻实现：

$$ G1\text{-T offline}. $$

不用等待 T-V\@80。

可以先用现有训练侧 T actual-CF bank。

如果 bank 不够，则优先扩 T Standard-domain CF bank，而不是 Raw bank。

RISE 线

立刻实现：

$$ RISE\text{-A replay runner}. $$

它应该接收：

manifest
checkpoint\_A
checkpoint\_P
checkpoint\_F

确保 action identity 完全一致。

与此同时实现 RISE-B same-state runner，但不开始 FVD 训练。

Raw

降到低优先级：

修 query coverage，但不扩6400。

Atlas

完成已登记 D/S/recovery。

不要继续新增 characterization。

十三、如果资源允许，未来48小时应该出现四个模型

只要 preliminary gates 成立，就不要再等 TDS。

直接形成：

$$ M\_0=T\text{-V} $$ $$ M\_G=T\text{-V-GCTX} $$ $$ M\_F=T\text{-V-FVD} $$ $$ M\_{GF}=T\text{-V-GCTX-FVD}. $$

这是你最想要的论文完整性验证。

然后计算：

$$ I\_{GF} = M\_{GF}-M\_G-M\_F+M\_0. $$
十四、四种可能结果都能形成清楚故事
Graph+RISE 都成功

最终主创新：

$$ \boxed{ \textbf{Relational Future Marginal Computation Value} } $$

非常理想。

Graph 成功，RISE失败

论文主方法：

$$ \boxed{ \text{Grounded Relational Computation Value} } $$

RISE 作为负结果/analysis。

RISE成功，Graph失败

主方法：

$$ \boxed{ \text{Grounded Future Computation Value} } $$

关系结构不需要复杂 Graph。

两者都失败

仍然不推翻：

$$ \text{Grounded Value} $$

本身。

但论文方法会更聚焦。

十五、为了加速，我还建议马上修两个 T-V 实现问题

我刚刚逐行审计时发现，这两个问题会直接污染 Graph/RISE T 路线。

A. Temporal router random-init

当前 T Value head 默认 PyTorch init，第一次 forward 就可能产生随机正 gain，从 Uniform seed 随机 swap。

建议最后输出层：

$$ \boxed{\text{zero-init}} $$

这样：

$$ V=0 $$

初始严格退化为 Uniform/STOP。

或者先 prefit T Value bank 再打开 routing。

我更推荐：

$$ \boxed{ \text{offline prefit} \rightarrow \text{online refinement}. } $$

这样 Graph/RISE 的 trajectory 更干净。

B. short-window physical duplicate mapping

当前 to\_standard() 对重复 frame ID 使用 dict lookup，有可能把最后真实帧映射到 suffix padding occurrence。

正式 T-V 前应确保：

$$ candidate\_mask[ selection.indices ]=True $$

对所有：

$$ selection.valid=True $$

的 observation 成立。

这应该加成 hard assert/test。

十六、还有一个协议问题必须现在修，否则 Graph/RISE 主表会有偏差

当前配置 primary 仍有：

best\_full\_test\_mAP

而 evaluator 记录：

full-test milestone EMA peak。

这意味着有可能在官方 test：

$$ 10/20/40/60/80 $$

之间挑峰值。

我强烈建议现在冻结：

$$ \boxed{ epoch80 = primary endpoint. } $$

或者用 router calibration/development protocol 决定 checkpoint，然后 test 只用于最终报告。

不应该因为 Graph/FVD 的某个 test milestone 恰好最好就选它。

否则越多 optional components、越多 milestone：

越容易产生隐性 test selection advantage。

Graph/RISE 论文尤其需要避免这个问题。

十七、Graph+RISE 的最终 Value API 建议现在直接冻结

为了避免以后又重构，我建议设计成：

$$ Q\_a( x\_a, c\_{\rm cheap}, g\_a, h\_{\rm exec}, e\_l, e\_m ) \rightarrow [\mu\_{cls},\mu\_{loc}]. $$

其中：

(x\_a)：axis-specific local descriptor；
(c\_{\rm cheap})：Shared Scout context；
(g\_a)：Graph relational context；
(h\_{\rm exec})：执行历史；
(e\_l)：layer/operator embedding；
(e\_m)：budget/plan embedding。

然后 T/D/S 只使用不同 adapter：

$$ A\_T,\quad A\_D,\quad A\_S. $$

共享：

$$ \boxed{ \text{Value trunk} } $$

这样 RISE snapshot 的函数边界也很清楚：

$$ \Psi= { A\_a,; GraphConditioner,; ValueTrunk,; normalization }. $$

未来 FVD 就外推这个函数。

十八、但是第一版 G1 不要等这个最终 ValueNet 重构

为了速度：

现在 G1-T 可以只实现：

$$ 407D;TemporalDescriptor + GraphContext \rightarrow MLP. $$

先回答 Graph 有无独立价值。

如果 G1 FAIL：

就不用为了统一设计重构整个 ValueNet。

如果 G1 PASS：

再正式做 Shared ValueNet。

这叫：

$$ \boxed{\text{probe first, architecture second}.} $$
十九、我给接下来工作的优先级重新排序

如果目标是论文完整性优先，我现在会定成：

$$ \boxed{ 1.;T\text{-Value} } $$ $$ \boxed{ 2.;Graph\text{-conditioned T Value} } $$ $$ \boxed{ 3.;RISE\text{-A/B on T} } $$ $$ \boxed{ 4.;T\text{-Graph-FVD} } $$

然后才：

$$ 5.;S\text{-Value} $$ $$ 6.;D\text{-Value} $$ $$ 7.;TDS $$

注意不是放弃 D/S。

而是先闭合一条完整科学故事，再扩 action axes。

二十、我建议实验 owner 接下来直接执行的指令

可以直接把下面内容作为 Fast-Track v4 的执行原则：

A. 不再等待完整 D/S 才验证 Graph/RISE。Temporal allocation 已获得全数据 headroom，T 轴成为 Graph/RISE 的首要 scientific carrier。

B. 在 T-V 正式长训前，先完成当前 ≤4-swap action-space 的 GT-assisted headroom、short-window valid-index 回归、zero-init/prefit router control 和 matched T-U/C0 对照。

C. 立即实现 G1-T：Plain MLP、parameter-matched MLP、Static Graph、Dynamic Graph 在同一 160/20/20 actual-CF bank 上比较 holdout regret/NDCG。通过后直接解锁 T-V-GCTX，不等待 TDS。

D. 立即实现 RISE-A：固定 T action manifest，在对齐 checkpoint 上真实重执行，测 value drift。随后实现 RISE-B：同 detached state、同 candidates 下比较 current / true EMA / post / future。只有 actual drift 与 forecast 两关同时通过才训练 FVD。

E. Graph 和 FVD 各自通过后立即训练 T-V-GCTX-FVD，并保留 Base/Graph/FVD/Graph+FVD 四格。

F. D/S、Atlas 和 Raw 继续并行，但不得阻塞 RFV-T；Raw 暂不扩 full bank，先提高 off-grid diagnostic query coverage。

G. 所有 checkpoint selection 使用 development protocol；官方 full-test milestone 不作为主模型选择依据，epoch80 endpoint 永久保留。

最后的判断

你现在提高 Graph/RISE 优先级，我认为是合理而且及时的。

前8小时已经基本证明：

$$ \text{工程基础已经足够成熟。} $$

继续只堆 D/S/Raw 工程的边际收益正在下降。

接下来真正值得争取的是尽快回答两个论文级问题：

$$ \boxed{ \textbf{Is marginal computation value relational?} } $$

以及：

$$ \boxed{ \textbf{Can we predict its future evolution?} } $$

尽快能拿到：

$$ T\text{-Value}, \quad T\text{-Graph}, \quad T\text{-FVD}, \quad T\text{-Graph-FVD} $$

这四组的 holdout-value + full-TAD matched-control 结果，那么论文的最终方法形态基本就会第一次真正清晰下来。上面对于时间范围的表述不是固定时间范围，你应该在三个服务器上尽快推进我们的科研任务和路线可行性证明，并尽快实现和分析最终模型
