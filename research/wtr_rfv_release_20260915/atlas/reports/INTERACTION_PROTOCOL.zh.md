# Conditional Value / Interaction v1

2026-09-15，用户要求将跨轴交互加入核心 characterization，并允许只补缺失的关键组合。本协议在新跨轴 publication 数据采集前固定。它覆盖此前“Atlas 不再新增 GPU 查询”的限制，仅授权下述独立补测；旧 a50b84d 数据冻结，B 继续 HELD，不训练 WTR / joint router，不阻塞 RFV-T。

## 问题与证据边界

一个轴的局部选择价值是否依赖其他轴已经作出的选择？旧 Atlas 只有轴内 coalition，总 loss 联合值，不能拼成跨轴 factorial，也不能从单次 null/benign 得到 interaction null。旧轴内记录可以重新统计总效应分布，不能补造 cls/loc 联合值。

本轮固定官方 AdaTAD-S epoch59 EMA、已有 V2 epoch40 light、完整原时间轴恢复及同一检测 head，FP32 / TF32 off / math SDPA。正式测量全部211测试视频、792窗口；GT只计算任务损失、事后分层与评估，不参与位置选择。开发训练视频的技术验收不是论文证据。已查看过 test，不能称 untouched holdout，也不能用此 bank 训练/调参。

## 两种明确区分的尺度

### A. 固定预算的局部选择交换

每窗预定4个 triplet，覆盖0-based层4/6/8/10。T为Uniform K384中的一次合法帧交换；时间分层随视频/窗口轮换，避免层与时间固定绑定。若无未选有效帧，T不适用，仍测D×S四组合，不将重复T当成零价值样本。

D是同层、同native tubelet内两个空间位置交换：一个从完整attention+FFN改成训练好的depth-light，另一个相反。S是另外两个共同D-admitted位置的heavy/light FFN交换。D与S位置互不重叠。每个分支具有相同heavy query数量、heavy FFN数量及逐pack执行形状，TIA始终保留；记录真实GFLOPs核验固定预算。这样避免把D-light位置上的S再关闭误当作独立选择。结论仅适用于这一合法、近full的局部背景，不能外推全部稀疏预算。

T改变后，在新的选帧列表上重新计算native pair物理中心，并按同一版本、GT无关的最近物理中心规则重新选择D/S位置。记录每个分支的实际RGB pair、native index与空间坐标。因此T相关是固定规则的policy-level interaction；D×S是在同一T状态下的严格局部factorial。不是learned policy，也不假装原token ID语义不变。

每个合法triplet真实执行000、100、010、001、110、101、011、111。没有训练八个模型。保存每cell cls/loc/total loss、实际执行成本、执行时间、选择身份。统一G(A)=L000−LA；全部gain按同一个L000(total)归一化，同时保留raw。cls和loc不各自除以不同分母，以保留total=cls+loc。

### B. T换帧在两种S计算背景下的价值

每窗使用A中第一个预定T交换，D始终full。S-sparse为层4/6/8/10有效native位置上的固定空间棋盘格50%heavy FFN，其余为训练好的spatial-light；padding始终full。S-full全部heavy。测(T-uniform,S-sparse)、(T-swap,S-sparse)、(T-uniform,S-full)、(T-swap,S-full)。

这一对照改变S计算量，**不是匹配成本的S交换**。它直接比较V_T|S-sparse与V_T|S-full，附完整成本。没有合法T的窗口明确回退同一T输入，仅为全数据预测保留alias；局部T统计排除这些窗口。保存四条件全视频预测，可使用同一10k video bootstrap重算四组dataset AP的interaction；仅称冻结规则的背景对照，不充当learned joint model成绩。

## 真正的控制与统计

每窗另执行8次相同输入、相同D/S mask的no-op重放，构成独立记录的8-cell replay cube，得到各阶I_null。epsilon_null,95为同cohort、视频等权的|I_null|经验95分位，bootstrap中重估。它仅描述可测得的重放误差。若全部为0，明确报告0，不除以0、不把所有非零残差叫强交互或显著。

此前单次epsilon_benign不作为interaction阈值。新研究不凭空登记实际意义的等效界值；报告原始尺度、归一化尺度和超过replay参照的比例。没有实用等效界值时，不能给全局可加PASS。

以video为cluster、视频内窗口/triplet等权，seed42，10000次重采样。报告mean、median、E|I|及CI、P(I>epsilon)、P(I<−epsilon)、P(|I|<=epsilon)、分布与实际样本量，分别total/cls/loc。全部四/八cell共用基准和抽样。T不适用与缺失分开。位置/层/边界分层均标exploratory，不作未校正的多重显著性主张。

I_AB=G(AB)−G(A)−G(B)。纯三阶I_TDS=G(TDS)−G(TD)−G(TS)−G(DS)+G(T)+G(D)+G(S)。三阶与各二阶同时保存；不把总非加性误称纯三阶。负I只说明指定改变的次加性，不能自动断言机制上的冗余。

补充报告有限两选择的条件regret：基准背景按max(0,V_A)选择保持或交换，将该选择放入B背景，与B背景下真实两选择最优收益比较。它是真实候选集内的条件参考，不是训练router性能。很小的符号翻转不等于有实际影响。

## 图与Gate

主机制图优先TS、TD、DS：additive prediction vs actual joint gain及y=x；有符号分布/zero/no-op参照；mean/median/E|I|与CI数值；条件value和两选择regret。附录保留cls/loc、纯三阶、成本、分层。时间×层/空间图只标实测坐标，不插值伪造连续map。代表例按预定10/50/90分位|I|挑选，说明事后描述性选例；waterfall用同一真实cube的inclusion–exclusion。

INTERACTION_GATE不阻塞RFV-T。初始INCONCLUSIVE。可报告OBSERVED_CONDITIONAL_DEPENDENCE或WITHIN_MEASURED_REPLAY_SCALE作为范围受限的测量结果。JOINT_ALLOCATOR_REQUIRED还需要条件建模/预算分配在matched cost下稳定降低regret的独立实验；SEPARABLE_ALLOCATOR_SUFFICIENT还需要实用等效界值、预算范围和合法性约束下的验证。本轮不会用mean CI跨0宣称可加，也不凭大interaction直接指定复杂Joint Planner。

新源码版本写INTERACTION_REVISION，新原始目录results/interaction_v1_s，开发目录results/interaction_v1_development_s；分析analysis/interaction_v1_s、图output/interaction_v1_s。旧CODE_REVISION与所有旧记录不改。技术修复若改变科学定义，使用新版本目录，不覆盖已采publication数据。
