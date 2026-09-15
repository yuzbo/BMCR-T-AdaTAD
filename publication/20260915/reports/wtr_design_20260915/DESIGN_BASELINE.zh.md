# WTR 当前概念设计基线与接受意见

> 原始视频输入扩展：[采样时机与Raw输入评审](../wtr_raw_input_20260915/RAW_INPUT.zh.md)。Raw作为独立候选域/读取方式研究；逐帧Core、原轴恢复和Standard证据保留，micro-clip与整视频THUMOS不自动并入。

> 后续扩展记录：[动态预算设想与修订](../wtr_dynamic_budget_20260915/DYNAMIC_BUDGET.zh.md)。WTR-DB作为建议的可选扩展；固定预算Core及全数据实验优先级保持，顺序anytime不纳入默认执行合同。

日期：2026-09-15。依据：本轮新附件与用户消息正文，两份均完整保存。本文是科研设计记录，不是已实现规格、部署回执或效果证明。本轮没有修改模型代码、配置、训练任务或实验队列。

**结论：接受这一轮作为当前WTR概念设计基线。** 它比前几轮更合理，尤其补齐了后续策略条件、合法交换、真实执行域、成本上界和完整routing teacher。接受的是可实施、可证伪的研究候选；不将嵌套的必要性、具体超参数最优性或方法有效性写成已证结论。

## 1. 两份最新材料及解释优先级

- [附件全文](inputs/01_Attached_Full_Design.txt)：原始附件 `2a83db05-8810-4f06-952c-e6120f02ec38/pasted-text.txt`，逐字复制；明确完整前向、共享时间/空间特征、Vπ、分组配额、routing梯度及教师。
- [用户本轮评述与Core方案](inputs/02_User_Review_and_Core_Proposal.md)：保存本轮消息正文及末尾请求，保留原有公式、重复排版和措辞；进一步规定Core独立、交换监督优先、future先验证、共享空间证据必须便宜。

两份材料均是设计与评估对象。其中的实现建议不自动授权本轮执行。当前采用用户正文的收敛版本：**Core先独立成立，FVD和Graph各自证明增益，之后才考虑Full组合。** 附件中的Graph/RISE全开图保留为完整候选接口图，不作为Core的默认启用要求。

## 2. 本轮真正补齐了什么

|环节|本轮明确的内容|
|---|---|
|科学问题|计算收益条件化于证据、预算、执行历史和后续策略π|
|决策对象|计划选容量，T/D/S在合法域内分配身份；优先使用保持容量的交换|
|T|coverage-aware proposal＋显式set-conditioned swap；每轮按新集合重评|
|D|交换attention admission后，各分支按同一版本S策略重新选择合法FFN集合|
|S|在共同post-attention状态、共同D集合上交换heavy-FFN slots|
|预算|按层与packed组定义容量，native-time整数配额守恒；以形状相关成本上界约束|
|共享证据|复用同一次Scout stem的时间与小网格空间特征，后续价值头再读取当时已有重状态|
|教师|冻结完整routing函数，在同一detached raw decision state和共同候选域上比较|
|梯度|控制损失可训练routing graph与价值网络，不借hard index宣称精确反传|
|论文结构|Core、FVD、Graph分开主张；原轴恢复与全数据证据继续保留|

这些是设计完整性与可检验性的提升，尚未增加新的mAP、算量或泛化证据。

## 3. 当前核心问题与价值定义

给定已经获得的证据、合法预算和后续策略，选择哪一次真实计算动作最能改善最终TAD分类与定位。

`Vπθ(a|s) = E[ℓθ(s;π) − ℓθ(s⊕a;π) | s,a]`，其中 `ℓ=[Lcls,Lloc]`。

推理只预测该收益，不读取GT，不做真实反事实重执行。boundary/actionness/change/uncertainty/support gap等属于预测线索；它们本身不定义计算价值。保留原始有符号cls/loc分量，采用开发阶段固定的尺度与权重形成排序utility。风险项需要校准；未经校准时均值策略是默认清楚的控制。

π指固定版本的后续执行策略，不意味着两条分支的后续mask必须相同。它们看到不同已产生状态时，可以产生不同合法动作。策略条件总效应与固定后续mask的direct-effect诊断分开命名、存储和解释。

## 4. Core的前向过程

1. 对候选窗口运行一次Shared Coarse Scout，输出全轴时间上下文和小网格空间证据。在现有THUMOS接口中，768候选是窗口单位；整视频可能有多个窗口，累计成本需如实统计。
2. Joint Capacity Planner根据粗摘要、有效形状、预算和合法计划库选择容量；计划级目标来自完整方案在当前局部策略下的真实执行收益，不由局部预测简单求和。
3. T使用覆盖提议与集合条件交换，完成选帧后才进行一次RGB gather、patch embedding和dense prefix。
4. 每个可路由层先决定heavy attention身份，执行attention，再在共同D集合内决定heavy FFN身份；每个token只执行一种FFN，然后执行global TIA。
5. 保存真实contributor、时间跨度、计算历史和多层anchors，使用physical-time interpolation＋Cross multidepth恢复完整原轴表示，再接TAD head。

前缀和最后层保留dense；继承12层候选时暂用0-based `[4,6,8,10]`。这些是当前受控起点，不宣称最优。token的heavy更新路径与整层删除、early exit分开。中间dense层会刷新真实age；真实age、累计light更新和路由机会历史分别记录。

## 5. 共享粗空间证据

复用同一次Scout stem的池化前feature map，经小网格池化/投影获得空间cue，不新增第二套视觉backbone。`4×4`、96维可以作为首个候选尺寸，属于工程起点。

空间cue与heavy token按真实tubelet contributors及归一化空间坐标对齐。两帧不相邻的tubelet不能只按packed位置匹配原帧；低分辨率cue也不能被解释为已经观察到细节。

T只使用编码前可见证据；D使用当前层输入，S使用post-attention状态。共享证据不意味着所有动作在time0决定。池化、投影、缓存、对齐读取、候选搜索和Graph成本均不为零，进入统一账本。

## 6. 交换、嵌套和预算的当前接受范围

T保留积分式提议；正sampling floor不等于硬覆盖，覆盖修复和后续交换约束需明确。默认最多4轮交换是可调的工程参数，每轮更新membership与集合上下文，不累加旧集合下预测的收益后一次执行。

D标签采用保持容量的slot exchange，两个分支各自生成attention状态，并按固定版本πS在各自合法D集合内重新分配相同FFN总配额，之后继续执行。S标签固定D与共同post-attention状态，仅交换FFN身份后继续π。它们都是完整下游任务差，不互相冒充，也不直接相加。

**接受嵌套 `AS⊆AD` 作为Core默认候选。** 这不证明Light Attention＋Heavy FFN不可行，也不取消上一份算子方案的独立A/F关键对照。独立A/F一样可以只执行一次FFN。该对照用来检验嵌套是否排除了有用的算子组合，不要求在本轮重开架构讨论。

D/S unary排序是组合分配近似。用exchange标签训练/校准它时，不宣称任意交换的真实收益都能精确表示成两个unary分数之差。主要检验预算内chosen-action regret与完整检测收益。

容量计划为 `m=(KT,{KD(l,g)},{KS(l,g)})`。D在packed域分配；S的每组总数固定，再按admitted数量形成守恒的native-time整数配额。候选有效性、配额及计数域必须一致。

采用 `Cactual(x,A;m)≤Cbound(m,shape(x))≤B` 的声明口径。上界覆盖Scout、value/graph、选择、patch/prefix、heavy/light、TIA、恢复与head，并包含已约定的候选和交换次数。实测平均GFLOPs仅验证账本，不能单独充当硬上界。计算预算不保证延迟、显存或能耗。T的STOP表示不再交换；固定slot配额不能靠插入STOP后少选而继续称同容量。

## 7. 训练与教师

控制支路为 detached raw coarse/current state → 可训练axis adapter/routing graph/value heads。actual/future损失不经该输入更新上游Scout/heavy encoder；任务支路仍正常训练约定的adapters、head、decoder及可微粗证据/恢复模块。

Core默认不启用旧temporal transport的代理梯度桥接。若保留它，单列surrogate-gradient控制。hard index没有精确排序梯度，不代表所有未选择状态都没有梯度；任务支路的full-KV/TIA等作用另行区分。

Lactual需落实到计划级、T交换级、D/S交换级。初始proposal头的训练来源仍须在实现配方写清：可以先保留既有proposal、重点训练交换，或明确定义提议监督；不能声称交换loss会自动训练一个不在计算图内的独立unary头。

TAD、原轴recovery、same-support、value、Scout辅助监督各自保留明确职责。**上一份方案的同输入heavy/light算子替代实验继续保留为独立候选项Lop**；它与Lrec/Lsupport不同，不能因新损失式简写而被默认为同一项。是否进入Core正式训练配方，由任务收益与成本控制决定，不强行叠加。

FVD先完成离线可证伪检验，再决定训练投入。旧ZIP外推早/当前detector真实收益标签的实验，与这里外推完整routing函数预测输出是两个阶段；前者阳性不自动证明后者成立。相同物理支持/动作在later detector重放时hidden state自然变化，later真实收益只用于隔离诊断。

预测函数教师保存全部axis adapter、routing graph、trunk、unary/pair head及固定归一化状态。anchor/post各自在同一detached raw state上计算，不先通过当前在线Graph再送给旧head。共同域内中心化utility logits，限制外推增量；β=1回到post分布，不等于EMA。不外推方差、索引、mask、预算和running statistics。

保留no-FVD、post β=1、genuine EMA、future及成本对照。排序/预测regret改善是机制信号；进入最终方法还需全数据任务性能—成本改善。future失败只收缩FVD分支，不否定Core。

## 8. Graph与版本划分

Core使用简单粗上下文即可独立前向、训练和评测。Graph是可选增强，heavy relational operator默认仍为selected-Q/full-KV，GraphKV替换后置独立验证。

Graph可分别作用于T粗时间域、D/S已存在token的粗空间域、原轴query/anchor恢复域。第一版routing graph从当前raw state与几何重建，采用固定有界referral，不额外引入跨层持久隐图。这里的“progressive”首先指可用证据随真实计算增加；若将来持久化跨层图，需另定义状态重放和教师历史。

Graph relation、任务价值、anchor可靠性分别记录，不能互相替代。恢复保留原轴query、mask和来源的一致性，不承诺dense语义无损，也不要求边界处特征过度平滑。

|版本|概念范围|
|---|---|
|WTR-Core|共享粗证据＋联合容量计划＋集合条件T交换＋默认嵌套算子价值＋原轴Cross恢复|
|WTR-FVD|Core＋通过独立检验的future-value训练，允许只用于有效轴|
|WTR-Graph|Core＋独立验证的关系上下文或Graph恢复，允许只保留有效入口|
|WTR-Full|FVD与Graph分别有依据后再检验组合；不由两个独立阳性结果自动推出组合更好|

## 9. 对“更合理”和“完全接受”的回答

**更合理：是。** Vπ与D的后续S重决策解决了之前最重要的收益定义歧义；分组整数配额和形状相关上界使预算更完整；raw-state完整routing教师使状态、函数和梯度边界更清楚。用户提出的Core独立、交换监督优先、future先证伪、共享空间cue必须便宜，均接受。

**接受为当前概念基线：是。** 不再因Graph/RISE是否进入所有轴而反复改主线。后续实施应把剩余的proposal监督、Lop取舍、候选范围及成本合同落实成具体规格。

**无条件接受所有结构必要性和效果判断：否。** 嵌套、4×4、交换轮数、共享程度及各增强的价值仍须实测；帧级候选合理不等于粒度已被证明最优。标量排序本身不是问题，缺少条件定义与任务校准才是需要检验的部分。概念闭合不等于实现已完成或论文结论已成立。

## 10. 实验优先级与实施边界

此前[全数据实验规划](../wtr_intake_20260914/CURRENT_PLAN.zh.md)继续有效：32视频是工程验收；正式方向证据来自官方dense S/B全211视频、792窗口的必要性、TAD结构、coalition和CF–Uniform/Static改善空间。原轴恢复保留独立展示，Fig6留给训练后的WTR实际收益。

新概念应进入该证据框架，并保留Uniform＋算子适配强控制、独立A/F对照、固定容量与可学习容量的归因。新FVD/Graph研究不替代主问题论证，也不自动取消旧课程。代码参考仍是先前固定1955057及已知执行框架；本次没有生成新WTR提交。

本轮完成两份原文记录与概念吸收，没有应用ZIP、改模型/配置、运行测试、训练、部署或远端操作。
