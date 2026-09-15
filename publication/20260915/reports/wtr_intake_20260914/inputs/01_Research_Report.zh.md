# Where to Refine：固定源码复审与跨领域研究报告

## 0. 结论与证据边界

本报告以 `yuzbo/BMCR-T-AdaTAD@1955057508af5a5dfd59a98bddf49302bee5972c` 为实现与实验依据，以用户上传的 Graph Machine、RISE 原文和本轮一手网络论文为文献依据。读取了 encoder/engine/gates、routing/interventions、model/decoder接口、objectives/support_targets/readout、training/profile、课程生成/训练/续训/登记/dispatcher，以及状态、论文与 MoD 诊断记录。不是对86个配置逐一完成动态验收，也不是服务器实时状态检查。没有真实 GPU 或视频/权重，因此没有产生新 mAP。

**核心判断：最应优先改变的是局部计算的训练目标与测量方式，而不是继续增加一种 MoD、Graph 或 teacher。** 当前系统已有真实稀疏执行、轻残差、原轴回写和全局预算监督，但局部 D/S 排序仍主要由 detached incoming attention 决定。全局“哪个 plan 较好”的监督，没有自动变成每层“哪个 token 值得更新”的监督。这是可被明确检验的缺口，不是性能下降的唯一已证原因。

“Where to Refine”应以开放问题开始：在明确数据、模型族和精度容忍范围内，TAD能否少算、少算在哪里、如何保持定位所需状态？观察冗余、访问冗余、状态更新冗余分别测量。H65 是实验平台；Graph 与 future-value 只作为有对照的候选。

## 1. 真实实现究竟做了什么

### 1.1 逐帧选择和三轴的真实语义

`h65/transport.py::sample_rates` 对 `[B,T]` 候选逐帧产生 exact-K indices，`gather_with_transport` 实际 gather RGB；有效前缀 mask 已有检查。`encoder.py::prepare` 在选完后每16个 observation 打包为 VideoMAE输入。两者不等价于从视频中选择连续16帧clip。

T 是 heavy RGB observation selection。D 是在特定层对 packed spatiotemporal token 选择 attention/FFN 的重更新。S 是在 admitted token 中按 native time 分配 heavy/light FFN；当前并不等于图像 ROI 高分辨率采集。三轴统一应是“计算动作的统一”，不能把三个轴都写成原始输入采样。

### 1.2 当前 A-MoD 是无参数 attention 排序，不是原版 learned router

`h65/frame/gates.py::incoming_attention` 复用上一层已投影的 detached Q/K，额外计算一次 QK，softmax后累加 query、平均 heads；分数表示某 token **被读取的程度**。`capacity_mask` 做精确容量 hard 排序。

`engine.py` 对 D 用该分数选择 Q/状态更新，对 S 也主要复用该分数。完整compact执行会真正 gather/scatter selected Q 和 heavy FFN；V2用 full KV，旧 compact 路径的 KV也随选中集合减少。被排除状态保留或获得轻残差，TIA仍全局执行。

必须区分：

`a token is often read` ≠ `recomputing this token will improve TAD`。

当前代码**没有**用 attention score 乘重更新残差；trace明写 `no output score multiplier`。不应凭原MoD论文的细节给当前实现编造“分数乘残差导致掉点”的bug。

### 1.3 全局价值监督和局部价值监督之间存在缺口

`interventions.py` 会真实执行frame/T/D/S/joint两条路径，记录 `actual_delta = [cls0-cls1, reg0-reg1]`。但训练对象是 FrameRouter 或有限菜单 BudgetRouter，并非每层D/S gate的任务收益。

当前 `repair_delta` 是把部分特征替换成外部/共享full特征的表示诊断，不是可部署compute action，不能当作actual_delta。同checkpoint的plan干预也不是独立训练的因子消融。

### 1.4 V2改进有效，但三轴选择尚未成立

固定记录显示：V2-S40 64.9949，Uniform-S40 65.3369，差−0.3420pp；计算量基本相同。V2-S40有783/792窗口使用K384_D100_S100，60轮全部D/S全量。这说明系统目前可通过避开D/S稀疏维持精度，而非已经成功联合压缩三轴。

同60轮V2-S/B为64.8354/67.7427，Uniform为65.1959/67.8910。B的各自峰值V2领先0.1692pp、计算少约3.75%，不能掩盖同轮排序。V2-B相对dense-B仍低2.6190pp，属于精度—计算折中。

诊断样本中，D预测收益为正、实际收益微负；这是特定记录的校准失配证据，不能推广为所有视频均错误。某S48代表窗中额外score-QK约47.19G、少算heavy-FFN约45.30G，说明评分开销必须计入；不能单凭两个分项代替完整系统成本。

V1→V2包含late routing、full-KV、depth-light、same-support等共同改变。组合提升不能逐一归因；已登记N00/N01/N02/N03/N04/N05/N06/N07应完成，而非再复制十套同义实验。

### 1.5 三个潜在机制需要分开验证

H1：incoming-attention centrality不是更新边际价值，导致错误选点。

H2：skip/light状态相对heavy状态产生task-relevant residual，经过TIA传播，单靠最终feature loss不足以修复。

H3：局部价值target稀疏、噪声且随detector变化，router学得滞后；future extrapolation是否有效未知。

三者都是假设。固定主干参数、初始化或训练长度也可能影响结果，但不能将全部掉点归结为“没训够”或预言蒸馏一定恢复。

## 2. 高价值跨领域文献：借思想，而非叠模块

完整一手入口见 BIBLIOGRAPHY.json。

### 2.1 Goal-oriented adaptive refinement / Dual-weighted residual

Becker & Rannacher 的后验误差估计以目标函数敏感度加权局部残差。对H65的启发是：heavy和light输出差异大，不一定值得修；差异小也可能影响边界。

 proposed proxy:

`r_i = heavy(h_i) - cheap(h_i)`；令 `u_i` 为该残差加入后的状态，取 `g_i = d L_TAD / d u_i`，得到 `V_Taylor(i) = - <g_i, r_i>`。梯度必须在修正注入点定义，不能拿normalized-input梯度直接替代。

这是对TAD loss的局部一阶近似，不是信息量、不是真oracle，也没有继承PDE误差界。优先做 FFN-only，同一当前state上比较heavy/light，用实际重执行测rank、符号准确率、选择regret。现有operator_diagnostics已能在当前normalized state计算heavy FFN差异；需补目标敏感度和实际干预校准。

### 2.2 Sparse DETR：下游需求监督哪些状态值得更新

Sparse DETR按decoder相关性稀疏更新encoder token，并使用检测训练信号。这比无条件attention centrality更贴近本任务。迁移不应把point head假装为DETR，而是把最终TAD cls/reg对状态的需求作为局部价值标签/辅助量。

关键控制：incoming attention、norm/entropy、task-gradient proxy、真实paired value，严格相同容量。引用它也意味着不能将“检测驱动token routing”本身声称首次提出。

### 2.3 MoD / Expert Choice：保留精确容量，替换局部排序依据

原MoD学习token router；Expert Choice以固定容量资源选择token。H65已经有exact-capacity，欠缺的不是另一个top-k，而是监督局部slot选择。

首版沿用当前计算内核，使用32维轻头预测D/S两类cls/reg收益。输入为当前层state、age、quality、层号、预算等推理可见变量。预测在重算前发生，不能读取尚未获得的heavy输出。训练用真实pair差，不用硬top-k伪装可微。

### 2.4 Learning-to-Cache / DeltaCNN：重算、复用和纠错是不同动作

扩散缓存学习何时复用层状态；DeltaCNN追踪视频变化并以稀疏更新控制误差。可借鉴的是“是否需要刷新”和累计失配，而不是直接缓存不同物理时间的视频token。当前H65已经保留状态/age和light分支，适合研究current-state residual refresh。

需测：误差在attention、pre-TIA、post-TIA何处放大；scheduled periodic refresh是否已足够；learned refresh是否优于相同数量的static refresh。空间/时间特征变化不是直接任务收益。

### 2.5 DAgger / contextual bandits：训练应覆盖策略实际遇到的状态

计算策略改变自己的输入状态分布。只在初始full模型或固定早期selection上收集标签，会偏离后期policy。借鉴DAgger的on-policy采集原则：从当前策略、uniform和少量随机合法mask混合采样状态，在这些状态上查询动作收益。

代价昂贵时可抽样，不必完整RL。新记录保存候选集、抽样概率、checkpoint和state/support hash。旧记录没完整propensity，不能事后宣称doubly-robust无偏评估。本包只记录给定层的pair概率，不代表完整动作过程propensity。

### 2.6 EDDI / costly feature acquisition：价值取决于已有证据

EDDI以已观察feature集合条件化下一次信息获取。H65可以把逐帧heavy evidence视为有成本的采集动作。借鉴set-conditioned utility，不必复制Partial VAE；信息增益也不等于TAD任务改善。

fixed-K frame exchange应学习替换价值，不应除以零成本。只有正增量成本的“增加计算”才适合收益率；全局预算可用 `V-lambda*DeltaC` 或合法组合优化。

### 2.7 RISE：以已验证训练变化构造教师，但不是收益保证

原文是RLVR-grounded更新→外推teacher→on-policy distillation。迁移到TAD时，GT监督是grounding，稀疏真实compute-intervention标签是局部价值grounding。首版应在同一当前state和候选域上对两个冻结value-head snapshot的函数输出外推：

`u_future = u_post + clip((beta-1)*(u_post-u_anchor))`。

然后蒸馏排序分布，而不是直接使用外推模型部署。β=1必须等于post teacher；只比较EMA与future不足以识别外推效应，所以本包有post控制。scale固定、variance不外推、teacher停止梯度，anchor/post/EMA全部持久化。

但value-head轨迹外推不自动等于未来detector真实价值。先在固定输入/action集合上用三时间点检测 `V_anchor,V_post,V_future_actual`，跨视频交叉拟合β，比较current、EMA、post和extrapolated预测。只在预测和下游预算收益都有证据时保留本模块。原RISE的RL性能与这里没有直接可转移的保证。

### 2.8 AERA：区分当前置信度与继续计算的收益

AERA研究后续推理块能否改善答案；其“未来”是同一次推理的额外计算，不是未来训练epoch。借的是expected residual opportunity，而不是对H65使用它的LLM结果作为证据。这是较新的预印本，只作为候选问题框架，不应凌驾于已核对的H65干预数据。

### 2.9 Graph Machine：稀疏访问不等于删除状态

GM保存node feature和可微权重/离散地址，SER通过稀疏多跳候选更新地址，SEA结合content与edge prior。H65当前简化degree16实现是真sparse gather，但不是完整multi-edge GM，KV仍全量投影，transition会并行dense与graph。

首选用途是原始时间轴上的证据读取，而非立刻换掉全部ViT层。固定同一selected frame/heavy feature，比较Cross、静态物理时间稀疏图、动态无referral、动态referral。保留查询栅格不意味着保留所有原始信息；边权也不是任务价值。

原论文预训练loss结果不能证明TAD有效；其理论算量降低与实际H100运行速度也不等价。本包默认拒绝Graph+WTR同时启用，避免因果混杂。完成对照后再讨论value-guided edges。

### 2.10 mTAN / Deformable DETR：Graph必须面对的原轴强控制

mTAN将不规则观测映射到reference time；Deformable DETR在reference附近读取少量点。可构造continuous-time attention和1D deformable recovery，对照Graph的地址传播。物理时间差、有效性、contributor跨度必须显式输入。

本包提供残差式continuous-time模块原语，不是完整mTAN复现、也未接入当前decoder。Cross很便宜，替代decoder应主要验证定位收益，而非预设削减全模型大量算力。

### 2.11 SAGE / coalition sampling：不要把LOO当集体删除证明

SAGE针对预测性feature importance并考虑interaction，比把training-data Data Shapley直接照搬更接近输入/计算分析。H65须在合法预算和支持集合中测条件边际值、替代/互补作用。实际网络utility未证monotone/submodular，greedy无自动近似保证。

具体可测joint removal interaction：`I(A,B)=DeltaL(A union B)-DeltaL(A)-DeltaL(B)`。与单点LOO同时给出；只报告正部Gini时明确负效应被分开统计。

### 2.12 US-Nets/OFA/data2vec2/NSA：训练与系统强控制

US-Nets/OFA提供sandwich/in-place distillation、progressive shrinking思想；当前已有mixed plans、full GT和self feature，新的收益必须超越这些既有对照。data2vec2提供full-context teacher向多个mask amortize target的思路，但不应当成不增加训练成本的免费信息。

NSA强调hardware-aligned sparse patterns。H65先证明局部价值和状态恢复，再考虑group/bucket/fused kernels。真实矩阵/卷积FLOPs、memory、resident latency、decode/NMS成本应分栏；不得用访问比例代替全系统加速。

## 3. 建议的最小方法：不是论文模块大拼盘

### 3.1 状态、动作与目标

状态 `s` 包括 cheap full-axis evidence、已选逐帧支持、当前局部hidden、执行历史与预算。动作是 frame exchange / D-update slot exchange / S-heavy-vs-light exchange /正成本refinement。原始物理时间不由packed rank替代。

`V_theta(a|s) = E[L_TAD(F_theta(s),y)-L_TAD(F_theta(s⊕a),y) | s,a]`。

保存cls/reg向量后以预登记固定尺度加权。添加动作正成本可用 V/DeltaC；交换动作用V或V−lambda DeltaC。一般组合不是sum独立V，先顺序重估或合法小组干预，报告搜索误差。

### 3.2 三个必须回答的机制

**Where：** cheap局部head预测task-conditioned update value，而非只看being-attended。

**How：** full-KV/轻残差/可靠原轴恢复维护定位状态；同支持目标与当前状态residual目标分开测试。

**How to learn：** actual paired interventions校准价值；on-policy采样减少分布漂移；future teacher仅作可删除的训练加速/泛化候选。

Graph提供可选access机制，RISE提供可选target机制，不是整个研究的存在前提。论文贡献应落在这三项的可验证问题与答案，不声称首次VoC或首次检测token routing。

### 3.3 最小代码路径

新增 `wtr_core.py/ValueHead` 和 `wtr.py`；engine在重算前调用小头，使用既有capacity/gather/scatter，不改变residual幅度；新路径关闭旧incoming-attention额外QK。原配置完全不启用WTR时保持原行为。

`paired_probe`固定selection、其他层路由和随机态，只交换同等预算slot；传递removed slot的FFN角色保持计数。其标签是真重执行，不是teacher repair。D的竞争域是packed clip，S是native time。

第一次D实验固定H65逐帧selection/scout、Cross、late layers、fullKV、light、support loss、teacher与训练预算；只改局部排序和teacher策略。局部head使用当前状态detached输入，明确不是完整end-to-end differentiable hard selection；可训练的是收益预测器及正常TAD分支。

## 4. 冗余证据与统计修正

“未显著掉点”不等于等价。先定义允许精度下降margin、最小有意义compute节省和bootstrap方法。对global mAP差做video-cluster重采样并重算全部AP，观察CI是否落在非劣/等价范围内。不要平均per-video AP来替代mAP。

首先区分：固定模型干预的敏感度、适配后的可压缩性、可部署selector的收益。固定dense模型一删就掉点并不证明不存在可训练稀疏模型；适配恢复也不证明被删信息完全没用。

全量强dense特征作为diagnostic特权访问必须标记训练/分析用途，不能声称省掉其生成成本。用GT搜索出来的reference只能叫label-assisted search；如果使用loss近似+greedy，它不是数学上界。正结果说明找到一个更好配置；搜索未提升不能直接证明无headroom。

已使用的THUMOS official test不得重新称为新holdout。train-video诊断适合机制，不等于未见detector泛化。严格model selection可在训练视频做OOF重训或预留真正未用于训练的新dev；所有β、预算、阈值、case选择规则在最终评测前冻结。

## 5. 达标条件与可能的否定结论

- Local-value ranking优于attention但mAP不改善：检查loss/AP错配、动作相互作用或未被控制的支持变化。
- 同时减少QK评分但uniform也变强：首先承认cheap routing的系统贡献，不能把全部收益归给value预测。
- 真实same-support计算oracle也不优于static：收缩该轴/预算的dynamic主张，保留简单方案。
- 当前状态residual监督改善teacher NMSE却损害TAD：特征逼近不是任务目标，降低/删除该loss，不强行保留。
- Future teacher只提高置信度不提高未来actual value预测：不保留RISE-inspired主张。
- Graph不优于静态物理时间边或Cross：不把Graph放进最终模型。
- 只有T有效：写temporal-focused论文并报告D/S负结果；不能用标题倒逼三轴成功。

这套流程不是为了证明预设方法一定正确，而是让最终方法由证据筛选。

## 6. 固定证据导航

源码：`h65/paper/{model,encoder,engine,routing,interventions,objectives,support_targets,profile,training,readout,graph_recovery,edge_ops,graph_frames}.py`；`h65/{transport.py,frame/gates.py}`。

实验：`research/project_status_20260914/{STATUS.zh.md,PAPER.zh.md,IMPLEMENTATION.zh.md}`；`research/paper/review_5485/monitor_20260914_1403/UPDATE.zh.md`；`research/paper/review_5485/EXPERIMENTS.json`；`research/paper/graph/monitor_20260914_1959/analysis/{full_results,best_results}.json`。

部署：`tools/{paper_train,paper_course,paper_dispatch,paper_review_plan,graph_deploy}.py`。后者仅审阅，不用于新方案登记。

读取固定文件的标准地址为 `https://github.com/yuzbo/BMCR-T-AdaTAD/blob/1955057508af5a5dfd59a98bddf49302bee5972c/<path>`；运行记录自身的source_revision优先于文档提交，不能给旧checkpoint回填新SHA。
