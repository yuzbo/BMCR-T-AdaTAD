# 当前模型实现、实验证据与最终路线判断

依据：2026-09-15 11:50 +0800服务器快照、当前科学代码4055294、评测记录补丁6e2fc7f、Raw修正版27d557e，以及独立交叉审阅。完整SHA及回执见上级EXECUTION_STATE.zh.md和审阅目录。当前源码HEAD为6e2fc7f，模型/训练科学身份仍4055294，评测记录实现单独标记。

**判断：研究路线在概念上闭合，关键执行链已具备真实实现；目前仍不足以证明最终WTR完整模型有效。最关键的“真实价值可被廉价预测，并转化为匹配预算下的完整检测收益”尚未闭合。Graph、FVD、DB、Raw的最终必要性也未确定。**

## 1. 当前到底实现了什么

当前存在两条新实现线：固定容量的Standard Value refinement原型，以及冻结检测器的Raw acquisition原型。它们复用真实VideoMAE/TAD执行框架，不只是离线分数模拟。

|组成|实际实现|当前边界|
|---|---|---|
|VideoMAE执行|S骨干、16-observation pack、两帧tubelet、global TIA、真实重/轻算子路径|没有全量训练所有backbone参数；核心课程训练Adapter、使用到的light路径、Cross、TAD head和Value头|
|T Value|统一公开候选/物理几何描述子，集合条件remove→insert收益头，最多4轮、每轮最多16个提议，非正收益停止|接口与代码存在；T-V/TDS-V尚未独立完成GPU/科学准入；不是完整联合动作最优解|
|D Value|在当前层状态上预测两个任务收益分量，packed group内按精确整数容量选择heavy admission|固定容量原型；尚无显式共享Scout粗上下文、完整历史向量或跨预算泛化证据|
|S Value|使用实际post-attention状态，在D集合内部按native-time守恒quota分配heavy FFN|嵌套S⊆D是本批设计，不是数学必然最优|
|真实CF监督|改变一个合法slot身份后完整重执行，保持容量、输入、支持和当前策略版本口径|在线即时标签不是任意未来版本都能复用的持久bank|
|恢复/检测|多层anchors、物理时间插值、Cross multidepth、原detector grid与TAD head|维持轴与可运行已证明；新模型精度恢复程度仍待完整AP|
|Raw输入|EpisodePublic、前置192×64×64 preview、bounded proposals、任意合法帧ID读取、去重、显式padding与pair/pack元数据|64px preview是输出张量，不代表codec只解码64px；系统按需解码加速未证明|
|训练与运行|80轮固定配置，10/20/40/60/80内联评测，EMA/optimizer/scheduler/RNG恢复，fresh instance reload，逐配置review准入|首次完整评测暴露日志集成错误；已修复，实际全量补评仍待运行|
|Graph/RISE/DB|旧GraphRecovery/GraphFrameRouter/GraphKV代码可参考；新实验树和解锁规则已登记|符合新协议的G0/G1、RISE-A/B、DB runner尚未完成，旧Graph不能直接换名作新证据|

### Standard与Raw的实际读取区别

Standard仍由官方pipeline先构造768候选RGB，冻结Scout产生cheap特征，再选择K384进入编码器。它并没有因此变成真正按需获取RGB的系统。

Raw先确定公开episode，读取共同192-preview，构造Official-grid或Expanded-raw候选帧ID池，选择后读取heavy RGB。Raw中的O/R+共用cheap证据；Standard和Raw虽然共用T Value API，但当前preview密度/读取过程不相同，不能据接口相同宣称输入分布完全匹配。

两条路径保留：384 heavy frames → 192 native anchors → 384 Cross query positions → 768 detector positions。没有直接把Cross长度改成768来冒充原轴兼容。

### Value头及其监督

D/S头将384维当前token状态经LayerNorm和32维adapter处理，与pack均值和8维几何/合法性信息组合，形成72维输入，再经64维层输出cls/loc两个收益分量。每个启用轴约1.79万参数。D读该层执行前状态，S读实际attention之后的状态。

T头读取407维描述子：remove/insert的96维cheap上下文、选集均值、全局均值，以及23维时间、间隔、membership、provenance、覆盖及实际D/S容量等；MLP为407→128→64→2，约6.06万参数。描述子不读取候选heavy特征或GT，也不含official/raw域one-hot。

真实标签为

`G(i→j|s) = [Lcls(base)−Lcls(changed), Lloc(base)−Lloc(changed)]`。

排序和STOP使用实际cls+loc之和。Raw fit RMS仅用于回归的数值conditioning，不改变任务取舍。D/S当前是unary分数差拟合实际交换收益的组合近似；T为显式pair head。没有声称所有条件交换收益可精确分解为unary差，也没有全局最优分配保证。

D动作后，两支分别重算S与后续策略；S动作保持共同D，并在同一native-time quota内交换。TDS的D/S标签先执行当前T策略，changed支路复用其support。控制状态detach，Value通过明确的CF损失训练，不靠hard index声称精确梯度。

### 预算与D的科学身份

主配置为THUMOS14/S/seed42/K384，路由层0-based[4,6,8,10]，selected-Q/full-KV，global TIA，Cross multidepth。DS的S50以全部合法token为分母，再在D中分配守恒quota；不是先D75再乘50%。

D-U/D-V中的selected token执行heavy attention+heavy FFN，未选中执行light attention+light FFN。因此它是policy-level heavy block refinement，不是attention-only结论。A75/F100和A100/F75的独立机制诊断仍需实际证据。

### 初始化、训练和数据边界

新课程从相同Full-V2-S epoch40 EMA重构资产初始化，80轮指新增适配课程，不是从零训练80轮。当前主配置以TAD任务为主，辅以原轴self-feature及真实Value损失；外部teacher训练分支关闭，原始backbone和Scout冻结，Adapter、light路径、recovery/head及Value参数按配置更新。

detector训练覆盖200视频；Value监督仅160 fit。20 calibration/20 holdout按视频互斥，称router-label holdout，因为detector和旧V2可能已见过这些视频。Raw mini仅选16/4/4视频，是完整分区的子集；4个holdout视频产生8个配对状态，不是8个独立视频。

## 2. 当前真实进度与已确认发现

11:50快照：D-V/D-U均保留epoch10、1000更新；补评续训1290651/1290652为Priority排队。S-V/S-U迁至4090的1290654/1290655同样排队，尚无训练成绩。四个配置code/protocol审阅通过，不等于四个模型已经科学有效。

D-V累计103条在线CF标签，53正、50负、fit以外0。这说明真实监督被执行且没有该项标签越界，但不能代替稳定checkpoint上的holdout ranking/regret。新Core仍无完整mAP，因此“D-V胜过D-U”“S-V胜过S-U”“TDS有效”均未证。

评测故障发生在epoch10：执行plan中的wtr_geometry Tensor被写入JSON。修复只使用既有公开trace plan并另记evaluator版本，不改变模型、训练或标签，故可保留前10轮并从checkpoint补评。它也说明两更新预检不能代替完整课程集成验证；记录层修复的code PASS尚不能冒充792窗口已跑通。

### Atlas：T维的有限分配机会已有强证据

S/B各211视频/792窗口、五策略×六预算、官方AP和10000次video-cluster bootstrap已完成。以下是CF−Uniform的mAP百分点，最终执行成本相同：

|保留组数|S ΔmAP [95% CI]|B ΔmAP [95% CI]|
|---|---|---|
|6|+4.285 [3.051,5.855]|+2.144 [1.044,3.511]|
|8|+3.507 [2.337,4.692]|+1.392 [−0.058,2.594]|
|10|+2.951 [1.668,4.119]|+1.305 [0.055,2.404]|
|12|+2.028 [1.035,2.855]|+0.413 [−0.655,1.331]|

8组S：Uniform61.828、Attention62.381、CF65.335，约1225.31 GFLOPs/窗口。结论是S有稳定可利用的时间分配空间，B较弱且并非每个预算确定获益。

CF带GT及额外查询：S/B的base+12候选查询约10320.53/34919.11 GFLOPs/窗口，六预算共享排序。它是有限参考，不是廉价部署策略、全局oracle或mAP上界。逐预算视频CI也不是跨预算同时置信带或训练seed不确定性。

11:50 Atlas S-D643/792、B population341/792，无当前失败；完整D/S分配及恢复尚未完成。S population中，1%整窗相对loss容差下D/S近零率约99.97%，但benign也99.94%；不能解释成可删99.97%的计算。O/D/S平均interaction区间跨零，尚不支持强非加性，也不证明全局可加或次模。D简单proxy相关接近0不证明新Value必然可学。

### Raw：工程正确，当前域诊断仍不充分

6视频19窗口的RGB/恢复/预测与官方AP缓存一致。24视频mini-bank完成90个domain/state组、346次swap，replay error=0，正收益约48%。

候选池确实扩域：Official候选均值712.53，R+均值1288.53，每状态新增576个off-grid帧。然而，R+的180次实际动作仅24次off-grid（13.3%）；38/45配对状态查询动作完全相同，holdout的4视频8状态全相同。

这批数据不能检验充分的Raw增量，也不能证明Raw失败。它主要证明输入/标签流程正确。fit仅5/30状态的R+最佳单步loss收益更好，calibration/holdout没有最佳收益差；尚未拟合Value，也没有Raw全量检测或matched retrain结果。

所有实测swap均只改变1个pair/pack，属于当前局部提议规则。pair/pack是消费规则，没有max-gap额外限制；候选帧池、合法exact-K交换空间、16-pair几何提议子集、实际查询集合应分别展示。

bank实际含523次主体前向，约622061.56 GFLOPs；decode返回13820帧、decode/transform约135.79秒。GOP内部解码量未测，不能据此声称完整on-demand加速。

### 旧参考提供了完整终点，仍非新WTR结果

|80轮旧模型|Avg-mAP|mAP@0.7|平均GFLOPs/窗口|
|---|---:|---:|---:|
|Uniform S|65.154|43.642|1226.48|
|Uniform B|67.769|46.542|4094.32|
|V2-B|68.061|46.833|3909.50|

V2-B比Uniform-B高0.291pp、计算低约4.51%，但暂无多seed或相应配对CI支持强显著性结论。旧曲线不单调变好，必须保留80轮终点及各冻结里程碑，不按test峰值调整方法。

## 3. 目前能否说最终路线完整可行

应分三层回答：

1. **研究设计闭合：可以。** 统一任务价值、合法动作、执行/恢复和独立组件gate构成了可收敛的研究路径，局部失败允许收缩对应分支。
2. **完整工程实现：还不可以。** 关键执行链已运行，但共享粗上下文/联合预算、稳定checkpoint action bank、Graph/RISE/DB新实验链及全部配置验收尚未完成。
3. **最终方法科学有效：还不可以。** 核心缺口是cheap learned Value能否在保留视频上排序更好，并把机会转化为完整AP—计算/延迟收益。现有GT-assisted headroom不能替代这一步。

最终候选应是证据支持的最小Pareto模型，可只保留T、T+S、D+S或T+D+S中的有效部分。Raw/Graph/FVD/DB失败不必推翻已成立的Standard Core；但如果所有learned allocation都不能胜过匹配简单控制，则不能仅凭Atlas有headroom宣称WTR主方法成功，只能保留机制诊断结论或重做价值学习。

## 4. 尚待确定的实验和论点

|待确定问题|最有判别力的证据|当前缺口|
|---|---|---|
|Value是否可学且优于proxy|稳定同checkpoint、video-disjoint router-label holdout的ranking/regret，及完整V/U检测对照|Core re-query尚未完成，新AP未产出|
|D/S分别是否有价值|完整Atlas D/S预算曲线；D-V/U、S-V/U；A75/F100与A100/F75直接机制诊断|不能拿T headroom或单点近零率替代|
|light路径/恢复是否限制收益|相同输入的heavy/light误差与任务收益；same-support recovery；AP@.7和边界误差|排名改善不保证最终定位改善|
|Joint是否值得|DS优于最佳单轴；TDS比匹配控制有可归因增量|DS至少需一单轴Value信号；TDS至少需T或DS信号|
|Raw是否增加可用证据|提高开发侧off-grid实际查询覆盖的同支持/同预算小bank；共享Value holdout；完整四格AP；matched retrain|现mini的holdout动作没有域区分|
|Graph Recovery是否有效|冻结证据/backbone/head/D/S，Cross_existing、Cross_fresh、Static/Dynamic/Referral，匹配teacher和更新数|旧Graph训练不能充当新G0|
|Graph Context是否有额外价值|同一actual-CF bank的Plain、Graph、参数匹配MLP；regret及下游AP/成本|新G1未完成，不能把GraphKV当Value context|
|RISE/FVD是否成立|RISE-A同动作20/40/60真实drift；RISE-B完整函数快照在同raw state上forecast，胜current与真正EMA|两阶段不能混称；还需FVD训练后的下游增益|
|DB是否值得|trained/supported tiers的完整plan-response；同平均实测成本下oracle gap与learned planner利用率|单预算模型的未训练subnet响应不能叫计算需求|
|是否泛化、是否真实提速|完整80轮、关键额外seed、B/ANet、matched runtime latency/显存/端到端decode|GFLOPs下降不是延迟保证，视频CI不是seed CI|

## 5. 接下来的顺序

先完成D pair断点补评与S pair启动，并以已有epoch10冻结checkpoint补足calibration/holdout re-query。这比继续增加模块更直接决定主论点。Atlas继续D/S和恢复，不重复采集。

Raw先在training/development改善off-grid查询覆盖并保持共同support/预算，再复核小bank；不直接放大当前查询到6400。Graph G0/G1、RISE-A/B可并行实现，但正式增强课程只由各自证据解锁。

出现稳定单轴Value信号再解锁DS/TDS；出现真实稳定S-Core优势、实际稀疏计数/完整成本下降且@.7无异常退化后并行B/ANet。所有方法参数和早停判断使用开发侧，publication不反向调参。

论文现阶段最稳妥的论证是“存在机会→检验可学习性→验证可部署收益”，不宜提前写成“全部计算轴冗余、Graph/FVD必需、最终全模块WTR已获验证”。

来源入口：代码 h65/paper/operator_value.py、operator_training.py、temporal_value.py、h65/raw/；独立审阅 reports/wtr_code_review_20260915_578bab2/REVIEW.zh.md；原始实验回执见本目录快照和上级progress_20260915_1100/；Atlas讨论见wtr_characterization_20260915/research/atlas_20260915/discussion_raw_v1/EXCHANGE.zh.md。
