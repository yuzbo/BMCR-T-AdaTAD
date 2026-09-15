# RFV Fast Sprint v3：当前生效实施协议

用户v3推进命令优先于此前v2及Fast-Track v4。两份完整原文分别保存在inputs/RFV_FAST_SPRINT_V2_USER.md与inputs/RFV_FAST_SPRINT_V3_USER.md。工作分支固定为`codex/wtr-rfv-20260915`；原D/S、Raw、Atlas与第一轮RFV结果均保留原科学身份。新代码只有提交冻结后才部署，部署包中的WTR_RFV_SCIENCE_SHA/source_revision.txt对应实际不可变提交。

## 顺序与资源

T-local-CF → T-Value-R1 → Graph-T与RISE-T → 单独G/F课程 → 有必要时GF。D/S四个现有405课程继续80轮，但不新增变体；Atlas只完成S统计、recovery、既有noise/interaction和图表，B暂停。Raw只做official/off-grid coverage development诊断，不扩6400、不新长训、不做Graph+Raw或preview sweep；DB、B/ANet扩展等候第一个成立的RFV candidate。

- 4090：保留D-V/D-U/S-V/S-U，收集40与80完整评测、已登记calibration/holdout重新查询和真实route/cost。
- AutoDL GPU0/GPU1：当前空闲期可执行短小离线head工作；正式准入后分别T-U80/T-V-R1 80。先查真实占用与owner交接，不启动重复任务。
- A100：使用既有/HOME/pxyai/...RFV部署与opentad环境，不使用/usr/bin/python冒充项目环境。优先独立CF bank、Graph离线、RISE重执行及后续G/F课程，不等待D/S。

正式训练复用`tools/paper_train.py`及现有唯一owner，不另造trainer。所有课程同allocation边训边测；epoch80为primary，10/20/40/60仅轨迹记录。官方test不用于选择超参数、阈值、Graph类型、beta或checkpoint。

## P0与既有headroom

已有代码完成valid candidate映射硬检查、Value末层zero-init和epoch80协议；原反序lookup已保留最早occurrence，不能再宣称复现了最后padding覆盖错误。第一轮完整fit视频全部3窗已跑过正式postprocess/JSON/metrics/noGT。新正式recipe仍须自己的完整视频集成、2次更新、fresh strict reload/noGT及prefit接入，不能把旧smoke或2-update单独当新recipe通过。

当前T合法选帧改变保持Uniform K384、≤4轮、每轮≤16个同源geometric proposals、非正收益STOP。G0b已有cal20/all46窗LocalCF−Uniform +0.528413pp，配对视频CI[+0.082440,+0.714687]pp，T_ACTION_SPACE=HEADROOM_PASS。来源为capture2ca4d4b，AP汇总修复dca6564。此次R1不改descriptor/proposal/decoder，因此不重复查询已完成的headroom。它是有限标签辅助参考，不是穷举oracle、learned模型或官方test增益，也不等同Atlas宽分组空间的上限。

## 同一bank和候选身份

复用theta40共享mini，登记32fit/10cal/10inner，共52视频、43个可执行state、688条真实帧交换收益。inner10来自原fit160，与mini fit32互斥；正式outer20始终封存。mini结果用于开发准入，不能称最终holdout。初步信号成立后再完成160/20/20。

每条数据保留物理视频/窗口/支持/候选/交换位置、原checkpoint/source、407D描述、Standard768 cheap证据（Graph192只是其派生）、分类/定位实际损失减少量、后续D/S计划、query与noise记录。GT不进入预测输入。

v3按用户要求导出state_hash与candidate_hash：前者绑定物理/支持元数据和全部声明的预测输入，后者绑定有序交换对和固定ID，均不含target。它们写在新的输入视图manifest，不覆盖旧bank。跨checkpoint cheap表示可能变化，所以RISE-A只要求物理选择身份一致；RISE-B为所有函数绑定同一个current-state hash，不能错误要求不同自然checkpoint的cheap-state hash相等。

## Value-R1：只改变监督目标

Plain-M架构仍为407→128→64→2，输入、proposal、R0的AdamW(lr1e-3/weight_decay1e-3)、batch8、2000updates及seeds42/43/44保持相同。

排序依据保持真实`gain_cls + gain_loc`，避免无意改变TAD任务目标。冻结正尺度r=fit中raw总收益的RMS；g=(g_cls+g_loc)/r，q=(q_cls+q_loc)/r。在用户component尺度表示中，这等价于w_c=s_c/r、w_l=s_l/r。component RMS继续服务原Huber回归。

每个state内只对合法候选中心化并softmax。第一轮tau_g=tau_q=1固定，不扫描温度/阈值；R1损失=JS(teacher,student)+0.1×原component-normalized Huber。teacher detach，padding不参与分布、中心或损失；JS按state等权。R0保留完全相同的原Huber。STOP执行仍比较未中心化raw预测收益与0，不能用中心化后“必有正值”绕过STOP。

首轮固定三个arm：Plain-R0、Plain-R1、同cheap证据/参数匹配的Plain-L-R1，各3seeds。需要两个预登记视角，共18个head拟合：

1. full-mini：同fit32全部可查询候选，cal10，锁定后inner10。
2. 同视频未见候选：原fit25每state按物理(insert,remove)顺序交替8fit/8held，cal保持同一组。所有normalization只由fit8产生；不从held8或cal拟合尺度。

每个视角的各arm共享同一训练数据、归一化、数据顺序和updates；两个视角的fit统计各自冻结，避免held8标签进入normalization。绝不重用已看过held8标签的full模型去冒充8/8泛化。

## R1判定与指标

Primary为有限候选集、允许STOP的真实收益regret：max(0,max actual gain)−selected actual gain。Secondary为NDCG/TopK/Spearman，K沿用ceil(0.2×合法候选数)。统计单位为视频，不把3个seed当成额外视频。

预登记mini LEARNABILITY_PASS要求：

- full-mini的R1−R0 regret视频配对95% CI上界<0；
- 同视频held8的R1−R0 regret CI上界<0，Spearman改善CI下界>0且R1平均Spearman>0；
- 两个视角R1平均regret均优于STOP/随机合法交换/既有transition proxy，三个seed相对R0的regret方向一致。

否则写LEARNABILITY_FAIL，保持task-level Graph/FVD关闭，不通过重新解释阈值改名PASS。Plain-L是容量/信息对照，不从inner/outer结果中临时改选它为baseline。mini PASS之后完成正式fit160预拟合与cal20/outer20冻结判断，再与完整recipe技术准入共同解锁匹配80轮；不会把mini当成最终泛化证据。

## Graph-R1

节点固定为192个cheap时间状态、宽64。复用paper的indexed message kernel，不更换backbone、不启GraphKV/GraphRecovery、不做referral扩张。

共同候选地址为self与物理时间offset ±1/±2/±4/±8，共9，位于原degree16上限内。Static保留全部合法地址；Dynamic在完全相同的9候选中保留self及4个非self最高分地址。连续权重由content compatibility和时间/coverage几何学习，硬选择只把梯度传给保留的连续边权。两者trainable参数形状相同；实际保留度9/5不同，完整开销分别报告，不能声称开销相同。

Plain-M-R1 / Plain-L-R1 / Static-R1 / Dynamic-R1用同bank、同normalization、同updates及3seeds。Plain-L保留相同cheap节点信息、无边node MLP和相同插值/支持集pooling，并与Graph参数量匹配1%内。可复用Value-R1已完成的对应full-mini heads，避免重复优化。

Graph类型只在calibration选择；Dynamic对Static无稳定增量则选Static。holdout比较选中Graph与Plain-L的regret配对CI，须稳定负且NDCG/TopK不恶化。只有Plain可学与G1通过才解锁T-V-GCTX80；GraphRecovery不同时加入。

## RISE-A与同state预测

历史20/40/60固定选择重执行已测到drift，但只是A0，不能替代新T轨迹10/20/40/60。旧B0未胜Post/EMA的FAIL保持原记录。

新B用R1的连续训练轨迹与每次真实optimizer更新累计的EMA；完整函数包含head、Graph（若有）、所有normalization和代码/候选版本。Anchor/Post/EMA都在同一current raw state、同一候选集合上完整重算。beta只取1/1.05/1.1/1.2，在calibration选择后锁定；未来fit标签不训练用于预测自己的历史函数。

中心化与线性外推可交换：center(P)+(beta−1)[center(P)−center(A)]=center(P+(beta−1)(P−A))。因此相同旧函数仅中心化不会改善候选排序或softmax，不能换名重跑后归因于中心化。新证据必须来自R1训练轨迹。STOP需一起变换或恢复完整外推均值；beta1的实际决策严格等于Post。候选分布JS蒸馏和raw零收益STOP分别保留其语义。

Future必须在真实later targets上优于Current/Post和真EMA，并保持NDCG/TopK；否则FVD=NOT_UNLOCKED。使用later calibration选beta的历史离线B仍不声称严格时间前瞻。只有新T对应证据通过才启动FVD80：固定beta、每10epoch更新快照、JS分布蒸馏，同时保留beta1/Post控制；不加入adaptive-beta或其他teacher矩阵。

## 正式T课程与后台

T-U/T-V-R1共享V2-S40 EMA、K384、D100/S100、Cross、optimizer/schedule、augmentation和200训练视频、80新适配epoch。T-V先加载fit160的R1 prefit，再用低频current-policy真实CF在线更新；Value loss仅来自fit160。每个milestone保存raw/EMA Value、normalization、descriptor/candidate revision、science SHA和固定选择manifest。任务身份、真实device、receipt、route/cost与完整JSON/metrics必须落盘。

Graph/FVD各自通过后才产生GF四格；GF不优于最佳单项则删冗余模块。D/S只保留现有课程、40/80及既定标签holdout复核；epoch20负差只是TASK_FAIL_so_far，不当最终判定，不增加DS/TDS。

## 交付与记录

必须产生T_LOCAL_CF.json、VALUE_R1_GATE.json、GRAPH_G1_T.json、RISE_A_T.json、RISE_B_T.json以及T_U_train_receipt.json/T_V_train_receipt.json。条件不满足的课程回执明确NOT_UNLOCKED且无PID，不能伪装已提交。阶段分别记录IMPLEMENTED、TECH_PASS、HEADROOM_PASS、LEARNABILITY_PASS、TASK_PASS、FINAL，并保留FAIL及其具体范围。既有负结果不清零成WAITING。

24小时清单是尽快交付目标，不以钟点替代科学条件；每个完整实现交叉审核后部署，不重复已完成的无变化实验。正文优先具体的帧交换、FFN名额或层内计算选择、局部计算收益/边际任务价值。δ与历史机器字段保留，Atlas的增算单点切换、观测替换符号及benign参照带不冒充RFV等预算交换或数值噪声阈值。

方法参考仅作为适配来源，不移植论文的实验结论：[RISE](https://arxiv.org/abs/2609.05295)、[Graph Machine](https://arxiv.org/abs/2609.02881)。
