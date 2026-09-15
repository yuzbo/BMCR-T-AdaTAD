# Fast-Track v4：RFV Sprint（当前生效）

2026-09-15，按用户完整原文 inputs/RFV_SPRINT_USER_REQUEST.md、inputs/ATLAS_FINITE_ENDPOINT_USER_REQUEST.md 和最新附件 inputs/RFV_FAST_SPRINT_REFINEMENT.md 执行。最新附件覆盖原计划的 mini-bank、192-node/64-width Graph、完整视频 smoke 与直接80轮安排；β=1/Post 的一致约定及独立审阅提出的强 Plain-L 对照同时保留。目标是尽快取得可判别的科学证据，时间表不作为固定截止或停止条件。RFV 源码从 Core 6e2fc7f 分出，独立 codex/rfv-sprint-20260915 分支，以 WTR_RFV_SCIENCE_SHA 记录实际不可变提交；现有 D/S 4055294 科学版本继续，新的实验和真实回执保存在独立目录，不覆盖旧课程。

## 优先级与有限终点

主线为 T Grounded Value ∥ Graph Relational Value ∥ RISE Future Value。G1 的 axis 顺序正式改为 T→D→S；G-Context > G-Recovery > GraphKV。T-V、G1-T、RISE-A/B 不依赖 D/S 长训完成。D/S 保留当前课程和必要 S 对照，先回收已存在 D@10 的完整内联评测。Raw 只修 off-grid query coverage，不扩 6400 bank。

Atlas-S 已完成登记 GPU 测量并由 owner 在15:40释放最后一张卡，进入 GPU FREEZE；只继续统计重算、bootstrap、绘图、分析修复和 action manifest 导出。禁止新增 heuristic/proxy/budget 网格、Graph 变体、蒸馏、selector 或 recovery family 矩阵。B population 已保留459/792窗，B各阶段 HELD，不自动恢复、不新排长队。AutoDL 可接收经过准入的 RFV 工作；4090/A100 保留现有 D/S 课程并优先后续 RFV-T。跨轴 T×D/T×S/D×S 仍是明确证据缺口，不能拿同轴 interaction 充数，也不因此重新扩张 Atlas。由原 Atlas owner 管理自己的队列，禁止重复 owner 或影响其他成员作业。

## 第一轮四项实验

| 实验 | 输入与执行 | 核心判据 |
|---|---|---|
| T-local-CF | 当前 Standard 768-preview、Uniform K384、≤4 轮、每轮≤16 bounded pairs，GT-assisted greedy 每轮完整重执行 | calibration 全视频 AP 的 local-CF−Uniform gap，同时报告 STOP、查询成本和真实动作轨迹 |
| G1-T | 同一训练侧 actual-CF bank，Plain-M、参数匹配 Plain-L、StaticGraph、DynamicGraph | video aggregated regret/NDCG/Spearman/topK；Graph 先在 calibration 选择，holdout 最终一次比较 |
| RISE-A | 固定视频/窗口/支持集/动作/候选 manifest，在同一模型家族 checkpoint 上真实重执行 | actual value 排序、符号和选择 regret 是否发生超出 replay/no-op noise 的漂移 |
| RISE-B | 同一 detached decision state 与候选，完整同家族 Value 函数 A/P/EMA 快照 | Future 对真实 later value 的 regret 优于 Current/TrueEMA，另报告并要求外推相对 Post 的独立收益 |

T-local-CF 分 G0a/G0b：先用 calibration20 每视频1–2个固定代表窗完整执行4轮×16候选，报告 loss headroom/regret；有真实信号后，在同20视频的全部官方滑窗做 Uniform/LocalCF 完整 detection AP。不能把代表窗 loss gain 当作 full mAP，也不称为穷举4步oracle。每轮保存决策前 state 和重新生成的完整候选，非正收益 STOP。

先建立 T_VALUE_BANK_V1 mini：32 fit、10 calibration、10 inner-probe-holdout，约1–2 states/video、8–16 actions/state，总量约500–1500，完整 bank 再扩到160/20/20。为同时满足“mini结果用于决定推进”和“正式holdout只判断一次”，inner-probe-holdout 的10视频从160 fit pool预留、与mini fit32互斥；mini calibration取既有calibration20的固定10个；正式outer holdout20保持封存。mini不是论文final bank，不能把它的10视频重称正式outer holdout。扩展后的160 fit可使用这10个内层开发视频，所有角色在protocol.json预先列明。

bank 使用预登记中间官方窗口；每个 state 记录所有被查询 action IDs。local-CF 与 bank 可复用相同已查询状态的结果，不变更其实际参数/前端/continuation身份。四个头共享同一bank，至少3个head-training seeds；不为Plain/Graph/RISE另造不同数据。

## 数据与实际 Value 合同

沿用 seed42 的 160 fit / 20 calibration / 20 holdout router-label split。detector 原训练可能见过全部 200 视频，因此称 router-label holdout。所有拟合、输入归一化、target conditioning 只使用 fit；架构/β/checkpoint 选择只使用 calibration；最终 holdout 评估保存单次冻结选择回执。官方 211 test 视频及 Atlas publication labels 不进入拟合或调参。

Standard 前端使用 768 个 official preview observation、当前完整 Cross/decoder/readout 和固定 T 容量。已有 Raw mini-bank 使用 192 preview，不能只重新计算 descriptor 就复用其 gain。必须重新执行 baseline 与每个合法交换的完整 downstream graph，回归 `[gain_cls, gain_loc]`，排序统一为两者实际和；RMS 仅作回归 conditioning。

每条 bank 记录保存：真实参数来源/epoch/update/state（learned 或 EMA）、源码版本、policy/recovery/light/preview 配置、video/window/frame IDs/valid、当前 support、候选集、round、固定 action IDs、pre-decision cheap hidden/actionness/transition、407D local descriptor、base/changed losses、查询成本和 replay/no-op 结果。GT strata 与模型输入分开；Graph 不得接收 annotations、future targets 或 GT-derived actionness。跨 checkpoint 匹配直接比较物理支持、候选和 action 内容，身份不一致则不计算 drift/forecast。

正式代码 `FEATURE_DIM=96×4+23=407`。当前 `to_standard` 逆序字典已保留最早重复 occurrence；新版本增加仅 valid candidate 的映射与 `candidate_mask[selection.indices][selection.valid]` 硬检查。Value 输出层零初始化，初始严格 Uniform/STOP。T-V 采用 offline prefit→online refinement；没有有效 prefit/当前 action-space headroom 证据时不把随机路由启动作为正式课程。

## Graph 的边界与准入

第一版使用192个cheap temporal nodes、64维Graph context、degree16；节点为96D cheap hidden加physical time/actionness/transition/support-distance/occupancy。节点从同一Standard768 preview按物理时间取得，不改变decoder使用的Standard768 cheap前端。remove/insert/support context均按物理时间插值。只拼这些上下文与407D descriptor，不连接GraphKV、不改变合法actions、D/S masks或recovery。

Static固定几何拓扑，Dynamic使用有界sparse referral/learned edges；节点、最大degree、层数、width和训练预算相同。Plain-L使用相同节点adapter、无边node MLP和相同physical interpolation/support pooling，提供与Graph相同的信息和聚合机会，再将实际参数量匹配至Graph的1%内。Plain-M仍保留原407D紧凑头。预登记head seeds为42/43/44，所有模型共享bank、optimizer steps、数据顺序和fit-only归一化来源。

G1a以regret为主，Graph须优于同信息/参数匹配Plain-L，paired video bootstrap方向稳定，且NDCG不低于控制。Graph type只在calibration选择；Dynamic没有相对Static的清楚增量则保留Static。固定bank排名与真正4轮闭环rollout分开报告。G1a与Plain learnability通过即启动T-V-GCTX，不等D/S/TDS。G1b要求epoch80完整mAP改善、匹配body execution budget、AP@0.7无异常损失，Graph完整开销另加到total FLOPs。只有G1a/G1b都通过，GCTX才可入Final=yes。

## RISE 两阶段

RISE-A 优先复用真正 V2-S20/40/60（相同 source db9c749）作历史存在性诊断，不用 Uniform 路径冒充 V2。V2-S80 source 不同，需确认差异后才并入。其 drift 包含 adapter/scout/recovery/head 等训练变化，不归因于 Value head 本身，也不冒充新 T-V trajectory。

报告各 checkpoint pair 的 Spearman、topK overlap、sign flips 和 earlier-choice→later-target regret。GT 只用于统计分层：boundary、interior、background、short/long action；同时报告噪声阈值以下的模糊符号，不把浮点噪声当 drift。当前 RFV T 轨迹产生后复用同一 runner；若 actual values 无实质漂移，RISE 不进入最终模型。

RISE-B 必须创建新的同家族 RFV Value 训练轨迹和每次真实 optimizer update 累计的 EMA；旧 294D router EMA 不能冒充 407D/Graph EMA。所有 routing snapshots 包括 adapter、Graph conditioner、head/trunk 和 normalization。冻结相同 detached state/candidates 后分别执行全部函数，禁止相减各 checkpoint 自然产生的不同 state logits。

β 的唯一约定：β=1 为 Post，`z_F=z_P+(β−1)(z_P−z_A)`，候选 `{1,1.05,1.1,1.2}`。原文中直接乘 β 的一处公式与 Post=1 不一致，以此约定消除歧义。β 只在 calibration 的 later targets 上选择；holdout 只用于冻结选择后的最终一次判定。报告 Current、真实 EMA、Post、Future 四项，清楚标明 Current 的预测时点。任何未来 fit labels 不得用于生成用于预测其自身的历史函数。

第一轮 θ20→40→60 的明确对照为 Anchor(Q20)、Current=Post(Q40,β=1)、真实 EMA40、Future(β>1)。Q40 是当前可用函数；Current/Post 是同一预测，不伪造为两个独立控制。Future 必须同时优于 Q40 和 EMA40，不能只胜陈旧 Q20。所有函数从同一 s40 原始 cheap nodes 重算自身 adapter/Graph/normalization，不能把 Q40 已加工 Graph context 交给其他快照。使用 θ60 calibration labels 选 β 的结果称离线跨视频 forecast 诊断；严格时间前瞻还需提前固定 β，不能混称。

actual drift 与 forecast gate 同时通过才解锁 FVD；不提前蒸馏一个尚未证明有用的 forecast。Graph/FVD 独立通过后，直接比较 T-V、T-V-GCTX、T-V-FVD、T-V-GCTX-FVD 四格，不等待 TDS，并报告 `I_GF=M_GF−M_G−M_F+M_0` 的配对视频统计。

正式RISE-B随新T-V的10/20/40/60完整Value快照出现即执行，不等80轮。历史V2＋新离线head的RISE-A0/B0仅为preliminary。FVD通过后仅外推Value function，不外推detector；默认epoch20开始、每10epoch更新anchor/post、Huber(Q,stopgrad(zF))，记录实际snapshot身份、β、teacher查询和额外MACs。必须有β1蒸馏控制，以区分self-distillation与外推收益；若FVD成为论文核心，最终该控制需完整匹配。Graph conditioner和所有normalization包含在同一快照函数中。

## 正式课程与资源安排

T short-window correctness、local-CF headroom和Plain mini learnability通过后，T-U/T-V直接启动匹配的80轮课程，不反复重启短pilot。两者共享V2-S40 EMA、Adapter/Cross/head、optimizer/schedule、augmentation/self-feature、200训练视频，主区别仅Uniform与learned T policy。冻结C0只作parity，不替代matched adaptation T-U。

每个新recipe在同一allocation内完成：CPU contracts/dry-run→1个完整训练侧视频的全部windows→postprocess/JSON/metrics→2次optimizer更新→fresh-instance strict reload→no-GT inference→正式80轮。此smoke须走正式序列化/metrics路径；不能用两步训练检查替代。10/20/40/60/80评测内联，不另起每checkpoint作业。

若两张RFV主卡可用，优先T-U/T-V；Graph/RISE离线用CPU或额外A100/4090。既有集群配额允许时申请额外两张同级GPU，分别在gate通过后运行T-G/T-F，再以首个释放槽位运行T-G-F。不能取消D/S或其他成员作业获取配额，也不让B-Atlas扩展占用RFV资源。调度器必须消费真实gate回执，不能仅以WAITING/配置存在作为启动依据。

## 模型选择与最终收敛

所有新课程永久保留 epoch80 EMA 作为 primary endpoint。10/20/40/60 的 full-test milestone 只作描述，不选峰值、不调组件、阈值或 β；新报告的 primary 字段不使用 best_full_test_mAP。既有 D/S checkpoint/config 不重写，终点报告统一按实际 epoch80 提取，历史 peak 仅保留为历史描述。

Atlas 若显示 D/S 无分配 headroom，则降低对应轴，不能把空间天花板误判为 predictor 失败；S 有 headroom 则保留 S-V；只有多轴有实质 headroom 才提升 TDS 联合优先级。最终模型按真实 gates 取最小有效组合，Graph/RISE 的负结果均接受。代码 PASS、技术启动 PASS、Value-level PASS、task-level PASS 分开记录；每个实验完整实现后独立交叉审核再部署正式科学课程。

## 2026-09-15 实测结果后的当前队列

| 已完成实验 | 结果及边界 | 当前决定 |
|---|---|---|
| T-local G0a/G0b | G0b在训练侧cal20全部46窗：LocalCF−Uniform +0.5284pp，视频配对95% CI [+0.0824,+0.7147]pp，AP@0.7 +0.8277pp | 当前4×16动作空间有headroom；这不是learned router或官方test成绩 |
| Plain / G1 mini | 32/10/10登记视频中43个状态可执行；Plain未稳定胜STOP/随机合法交换，Static−Plain-L regret CI跨0 | mini learnability及G1均未过；不启动T/Graph正式80轮 |
| 同视频8/8动作留出 | 现有25个有效fit视频；fit Spearman .9403，held .0444；held regret .0014921，STOP .0014135 | 尚不能归因于只有跨视频覆盖不足，32→64扩展条件不满足 |
| RISE-A0 | 固定同一批动作20→60：Spearman .7764、TopK overlap .7267、sign flip .1439 | 历史actual drift已测到；不等于新T轨迹或forecast收益 |
| RISE-B0 | cal选β=1.1；Future与Current/Post regret相同 .0032777，EMA .0032635 | 本历史forecast gate FAIL，FVD不解锁 |

已完成一次使用现有 fit bank 的 CPU descriptor 可分辨性诊断，由讨论任务独占执行。固定原8/8划分，使用已保存seed42头的fit8 input_mean/input_scale；核对动作行与407D输入对应、state内full407精确碰撞，以及held8到fit8的标准化欧氏近邻。主统计是近邻gain差（按该state gain range归一化）是否小于同state随机配对，辅以符号不一致率，用state内标签置换及video bootstrap作参照。不生成NN router/regret，不调距离或阈值。此分析不新增GPU标签、不拟合新的router，不重复8/8训练，不扫描容量或步数，不触碰正式outer20。相同描述对应超噪声不同标签才是确定的观测缺失证据；近邻关系弱本身不能证明Value不可学。

诊断实测25个state、200个held→fit近邻：无full407精确碰撞；近邻/随机的归一化gain差分别.203867/.210178，差−.006311，video95% CI[−.029751,+.020092]、置换p=.334；异号率51.0%/50.875%。未检出该固定度量下稳定的局部gain一致性，仍未证明信息论上的表示不足，也没有定位到会反转已有结论的代码错误。此诊断结束，不重复运行。

下一安排限于定稿一个有具体因果理由、可证伪的最小Value学习或状态表示修订，并明确保持同数据、同计算预算的Plain对照；独立讨论通过后才登记新的mini开发实验。不能把近邻诊断本身当作新架构有效的证据，不扩大Graph/RISE矩阵。只有同视频未见动作能稳定泛化、而跨视频仍失败，才解锁一次预登记32→64覆盖增量。此前未通过的mini结果继续保留，不能改阈值后重称PASS。

主线仍是T→Graph/RISE，但优先解决已显现的Value泛化瓶颈。通过后按照原协议直接进入匹配80轮、边训边测；当前没有获准启动的RFV detector长训。历史G0b报告的旧字段`task_course_eligible`仅表示headroom，禁止单独用于课程准入；新分析程序已改名`headroom_gate_passed`。最终路线目前只有局部余量和实现可运行的证据，完整最终模型尚未证明。
