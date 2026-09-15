# 固定提交审查、文献迁移与方法方案

证据版本：1955057508af5a5dfd59a98bddf49302bee5972c。文献检索截至2026-09-14。本文区分“代码事实”“报告实测”“研究推断”“拟实现方法”。GitHub核心文件经过连接器读取，但未本地克隆全仓、未执行95次历史测试、未获得服务器实时作业状态。

## 1. 当前路线的事实基线

H65是对任务相关冗余的实验假设，不是已证明的前提。时间选择直接操作[B,T]上的候选frame index，随后gather真实RGB、按16个选中观测打包；VideoMAE tubelet含两个选中观测，原始时间映射通过Selection/contributor metadata保持。`//16`只是当前局部swap候选约束，不代表初始只能选择连续16帧。

V2在0-based4/6/8/10层做稀疏重计算，前四层和最后层保持dense；深度admitted由前层incoming attention排序给定，随后同时限制attention query与heavy FFN。full-KV和attention/FFN light bypass已存在。空间轴当前主要是token组heavy/light FFN，不是高分辨率ROI读取，也不是丢掉全部空间状态。global TIA在每层残差后完整执行。Cross恢复已经使用真实physical-time、scout context与L6/9/12多深度信息。

因此下一轮不要再把“加full KV”“保留TIA”“补cheap残差”“首次按原时间轴恢复”写成新增修复。

## 2. 比峰值更有解释力的结果

历史同epoch5相对Cross：A-MoD50 S降16.26%计算而mAP掉2.981pp；B降18.04%而掉4.856pp。joint S/B分别掉4.676/7.802pp。同checkpoint8因素诊断T×D交互S=-1.2252pp，B=-1.6719pp。出处：`research/frame/INTERPRETATION_20260913_1342.zh.md`。

当前同epoch60：V2S64.8354 vs Uniform65.1959，差-0.3605pp；V2B67.7427 vs Uniform67.8910，差-0.1483pp。它们不是最终80轮、也不是完全同配方因果对照，但足以否定“B峰值+0.1692已经证明稳定优越”。

**新发现：高分往往靠回避深度/空间压缩。** `current_results.json`记录：
- V2S epoch40峰值：783/792为K384_D100_S100，9/792为K384_D100_S48；完全没有D压缩。
- V2S epoch60：792/792为K384_D100_S100。
- V2B epoch20峰值：519/792为K384_D75_S100，273/792为K384_D100_S75；没有同时D/S压缩。

不能为了标题强迫router启用TDS。要通过可行、受限GT搜索区分“合理拒绝有害压缩”与“未学会利用有益压缩”。同时统计训练时与推理时plan占比，查看哪些子路径训练充分但未被部署、哪些部署路径几乎无对应干预监督。

## 3. 五个值得优先研究的瓶颈

### 3.1 全局计划价值不等于token级算子价值

BudgetRouter学习15个计划间的cls/reg收益，FrameRouter学习局部真实swap收益；而层内D/S身份仍来自attention或特征能量等评分。因此“已有VoC”主要存在于计划与局部换帧层，不等于所有三轴位置已经得到task-value监督。A-MoD这一路由定义不是实现bug，但它不能自动变成任务边际价值估计。

### 3.2 算子耦合与替代状态可恢复性

当前同一admitted同时限制attention与FFN，S又在admitted内压缩。即便保留global TIA，前面的状态缺失仍会传播，不能从TIA存在推断信息无损。先将attention access与FFN update拆开，并测输入相同、其他mask固定的真实task effect。

### 3.3 监督覆盖不足与状态错配

V2 action_interval=8，五类action轮换；8000 updates下名义约1000次probe、约200次/类，具体有效数需按日志核对。默认frame probe来自plan1 K384/D100/S100，但部署可能选择D75或S75；V2本身未启用plan_aware_frame。原本已有plan-aware实现是一个干净控制。新标签必须记录plan、state与operator，而不只记录video_name。

### 3.4 代理误差与真实效用不同

`actual_delta`是GT cls/reg差，`repair_delta`才是外部/共享full特征替换代理。旧诊断显示feature-repair代理与actual RGB-swap收益关系弱甚至负；因此不能把NMSE大、attention高、uncertainty高直接叫值得计算。

### 3.5 teacher与优化方向的可靠性

support_reference来自初始资产的冻结NativeEncoder，非当前EMA full分支。已有support loss是同选帧但不同计算轨迹的状态匹配；新增“同一个当前pre-op输入上heavy/light输出匹配”才是真正不同的局部替代训练。RISE外推必须同时面对移动scales、不同状态支持与当前/未来detector能力错配。GT训练改善平均loss，并不保证每个action value随时间单调改善。

## 4. 高价值论文及借鉴界限

一手标题、年份和标识见references.json。

| 论文组 | 借鉴原则 | H65落点 | 不能直接照搬 |
|---|---|---|---|
| A-MoD / 原MoD | 理解attention路由与可学习capacity路由的区别 | 审查当前D/S评分，并作uniform/task-value对照 | 分类或LLM结果不是TAD保证；不要再“新增MoD” |
| Router-Tuning / MindSkip | 不同子层具有不同跳过风险，单独研究attention | attention-only vs FFN-only vs coupled | 原模型是LLM层级策略，不是TAD逐token证据 |
| MoNE | 便宜但有效的计算替代，比完全跳过更可控 | heavy/light FFN共享/嵌套或局部函数适配 | V2已有light，不能当新增；需要额外同输入验证 |
| LayerSkip / OFA | 子网络可用性应在训练中建立，循序改变容量 | 避免突然T/D/S全压缩；低成本路径先可预测 | 已有mixed plans/full分支，不要泛泛重复sandwich/KD |
| SSA (2025) | 在同一层当前输入上比较full/sparse attention输出，训练时对齐两种计算模式 | 同输入局部counterpart比第二整条teacher轨迹更直接 | 默认冻结QKV/FFN；双向对齐不能声称更新了冻结权重；原任务是LM |
| EDDI | 价值以当前已获证据为条件，而非某帧固有属性 | 加plan/support/operator condition，学习额外task gain | expected information gain不是TAD loss差；不需要复制Partial VAE |
| SAGE / ROAR | 考虑coalition交互与移除后的再适应，分开敏感性和可恢复性 | 多帧/多算子联合干预，冻结与有限适配两组 | 操作效用不是严格互信息；不应把贪心搜索当最优上界 |
| Taylor pruning | 用任务梯度近似算子贡献 | 少量实际counterfactual校验后，扩展廉价候选排序 | 代理不能取代GT重执行；dense/cheap梯度基点不同 |
| mTAN / TadTR | 连续物理时间与稀疏reference-relative访问 | Cross/Graph恢复的朴素强对照 | sparse attention在TAD已有先例，不宣称首次 |
| RISE / Meta Pseudo Labels | teacher来源及teacher对student的实际帮助都需检验 | future-value teacher + GT grounding；独立校准与回退 | 不是每个外推checkpoint都更好；不偷用未来验证标签训练 |
| Graph Machine / NSA | 分离state/access/update，稀疏机制需真实计费 | Graph恢复可选；局部稀疏访问与硬件对齐 | fullKV投影/排序不免费；Graph from-scratch LM证据不等于TAD微调有效 |
| Uni-AdaFocus | cheap context引导昂贵局部计算、跨维度实验组织 | 复用已有Scout/逐帧路径，定义任务收益 | 它的sample-wise维不等于网络depth；不能只改任务名称 |

本次特别提高**SSA、MindSkip、MoNE/LayerSkip**优先级：它们直指A-MoD当前的算子损伤和替代状态问题。RISE优先用于已有真实价值目标的预测/学习；Graph是访问/恢复的独立候选，不作RISE前置条件。

## 5. 建议的最小方法

### 5.1 统一状态，但分开action语义

状态包含廉价全时间轴上下文、当前heavy证据、真实时间/空间支持、每个算子的age和已用预算。动作包括等成本frame swap、positive-cost新heavy证据、attention access替代、FFN heavy/light替代。每项价值是

`V_theta(a|s) = L_TAD(theta,s) - L_TAD(theta, execute(s,a))`。

这是当前模型、当前状态、当前动作下的有符号任务收益，不是输入固有信息量。一般预算目标使用整体执行成本C(x,A)，而非无条件sum(C_a)。等成本swap按收益排序；只有ΔC>0的追加计算可以看V/ΔC。

### 5.2 先让低成本路径可用，再学习分配

先冻结frame support和plan预算做attention/FFN解耦。记录哪种算子损害最大、位置是否稳定、贵算子能否被light条件模块逼近。用真实GT task优化保护任务，局部同输入counterpart loss只承担“可替代性”训练，不作为“哪里该算”的标签。

若FFN-light容易恢复而attention-hold伤害大，先保留attention不压，压FFN；反之亦然。必须允许实验给出非对称方案，而不是为了T/S/D整齐都压50%。所有阶段保持full原时间轴decoder与global TIA，除非有独立消融支持改变它们。

### 5.3 直接学习算子条件价值

轻量价值头输出attention与FFN各自cls/reg收益和置信度。特征只来自action前：hidden摘要、廉价任务状态、位置、operator ID、预算、上次更新距离。真实干预采样用于校准；Taylor residual proxy只有通过留出视频排序/反事实regret检验后才用于扩充。

真实grounding采样按状态/预算/轴分层，不只采最容易的K384 D100 S100。少量探索采样保留负收益与无效动作，防止只学习被现有router选中过的support。

### 5.4 future-value distillation独立成立

anchor与post模型在相同action集合得到原始单位value后外推，再在相同固定scale/temperature上归一化，并加入no-op。保留current GT value NLL + future JSD。在线版教师更新后必须reset/保存完整状态。

β=1是current，真正EMA需要EMA参数重新评估。训练轨迹低维只提供局部动机，不保证效用函数单调或外推更接近最优。需要同时证明：未来value预测更准；在同预算实际TAD中更好；收益不是更多probe/更新次数。

### 5.5 Graph为有条件的恢复机制

只在固定frame、fixedmask、同head下比较Cross / continuous-time / static sparse / referral sparse。仅当后者显示独立收益，再研究edge任务价值。不能同时替换backbone attention、recovery、framecontext和监督却把提升全归referral。

## 6. 实验排布

| 阶段 | 主要问题 | 执行范围 | 科学晋级要求 |
|---|---|---|---|
| E0 | 基线与代码合同可信？ | CPU + 两步GPU + 一窗口parity | 数据、状态、时间、代价均能复现 |
| E1 | 损伤来自哪里？ | S先行，固定bank，attention/FFN/joint | 能分辨直接损伤、交互、适配问题 |
| E2 | current/EMA/future能预测实际价值？ | 同3checkpoint、video-disjoint | 改善排序/反事实regret，非仅MSE或漂亮PCA |
| E3 | 更好的teacher改变实际routing吗？ | 冻结detector的FrameRouter pilot | 同预算完整任务收益；当前/后期detector分别测 |
| E4 | 算子light可替代与task-value路由 | 新recipe，S最小匹配控制 | 超过attention/uniform，且真实D/S启用并保住任务 |
| E5 | Graph恢复有独立价值？ | 同selection/masks decoder控制 | 超过静态/连续时间强对照且计费合理 |
| E6 | 完整论文证据 | B确认，THUMOS完整，ANet READY后，匹配detector/backbone | 冻结终点、对齐成本、关键seed复验、失败案例 |

不强迫原84课程按新性能门槛终止。新增实验也不自动铺开，先技术验收与研究者批准。所有成功/失败/待验状态分别记录。
