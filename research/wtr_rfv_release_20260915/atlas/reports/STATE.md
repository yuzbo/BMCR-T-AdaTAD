# WTR Publication Atlas - 执行交接与当前入口

## 最新有效状态：2026-09-15 20:24 +0800 — 本轮完成

**本轮已授权的原Atlas-S收尾与interaction v1测量/统计/绘图均完成。** 最终6页interaction图（5主图+1探索附录）已逐页渲染/目视检查；第5页数值拥挤已改为明确G/I标记与2×4小表格，修订页再次QA通过，其余5页通过后未改内容。当前renderer为5d2402a11873f81ddccb29c861c4be80c9289378，supplement为0a9380a，primary analysis仍1b3850e，measure仍49f74bd。QA回执receipts/interaction_final_qa.json，两端figures.json=passed，图集和完整结论已向两个对接任务回传。

最终文件：output/interaction_v1_s/interaction_atlas.pdf及6份单图PDF/PNG/SVG；同目录captions.md、numeric_tables.md、case_values.json。完整结论INTERACTION_RESULTS.zh.md已标统计/图QA完成。analysis/interaction_v1_s/INTERACTION_GATE.json明确INCONCLUSIVE：测得条件两选择代价，但无稳定AP交互或全局可加/复杂Joint Planner必要性证据。原始loss尺度、探索分层及人数均补齐，不能将未校正分层区间作组间检验。

所有本任务新GPU查询已于19:10:33结束并冻结；primary CPU和最终展示CPU均结束，不再运行或重复分析。已通知RFV GPU1释放；旧B维持HELD，原a50b84d数据、S八页图集与旧轴内两页图均保留。两端QA标记/报告已同步，20:24已通过automation_update暂停既有wtr heartbeat，未归档。当前授权范围没有剩余测量、统计或绘图任务；不称原全S/B矩阵都完成，也不自动恢复B或新增训练。

最终对接任务已接收INTERACTION_RESULTS.zh.md并保存其路线对齐记录。对方另外报告reverse-consistency 4aa六head为STRUCTURE_SIGNAL_FAIL：P2−P1-bi regret CI跨0、seed方向−/+/+、P2仍差STOP、39/75实际选择为负；出处为对方reports/wtr_rfv_review_20260915/REVERSE_4AA_EXPERIMENT_REVIEW.zh.md。此处仅记对方复审报告，不声称本任务独立验收。Atlas条件价值刻画与RFV学习能力结论继续分开。

## 19:56收尾过程（由上节覆盖）

**跨轴测量与全部主统计已完成，禁止再启动模型查询。** 19:10:33完成211视频/792窗口；3072个真实8cell cube、96个D×S四格、6336次null重放、3120次context执行与48个明确alias，共34416次真实前向，local各格成本差为0。采集PID284842已退出；19:20:23实测GPU均无compute-app，已告知RFV GPU1释放。旧B仍HELD。完整执行核对analysis/interaction_v1_s/execution_audit.json；source49f74bd，原CODE_REVISION a50不动。

**主统计/四条件AP已经全部完成，不重跑。** finish PID287713于19:25:10进入FIGURES_READY_FOR_VISUAL_QA并结束；analysis/interaction_v1_s/{interaction,context_ap}.json已下载。四组均211/792、official_AP_reproduced、两指标10000 draws，paired AP交互Avg−0.023734pp [−0.103427,0.132906]，AP07 +0.068083pp [−0.080802,0.294083]。INTERACTION_GATE仍INCONCLUSIVE；均值跨0≠可加，非零比例≠显著性。完整解释见INTERACTION_RESULTS.zh.md，报告目前仍标QA待完成。

**最终展示修订已生成，正在最后一次下载与PDF QA。** 为明确epsilon=0、两个方向条件regret、近零实测点、waterfall小项和既定探索分层，新增INTERACTION_DISPLAY_SUPPLEMENT.zh.md。supplement及renderer source0a9380a6560a789bd9b58a534a8c8329f7fb55f6；primary analysis仍1b3850e、measure仍49f74bd。CPU display PID315973已退出，日志无失败，19:56确认6个单图完整：conditional_value_interaction、interaction_matrix、conditional_choice_regret、temporal_spatial_context、interaction_cases、appendix_interaction_strata；合并interaction_atlas.pdf共5主图+1探索附录。最后下载/渲染FastCtx job j-9j73q7，先查结果，不重复生成；目录output/interaction_v1_s，qa_final-1..6.png。原S八页图和旧轴内两页均早已QA/交付，不重画。

剩余：逐页目视六页最终PDF；必要时仅修显示，不能改测量/阈值/主统计。通过后写QA回执并同步两端figures.json，补齐INTERACTION_RESULTS.zh.md完成状态与原始loss/分层引用，向用户和两个对接任务交付；然后暂停既有wtr heartbeat，不归档，不恢复B。当前工作树有本轮STATE/RFV记录的未提交文档变更。

## 18:54采集阶段快照（由上节覆盖）

**原S完整图集与轴内重分析两页图均已完成并通过PDF视觉QA，本轮交付。** 原S：output/pdf_s/local_choice_revision/publication_atlas_with_appendices.pdf，6主图+2附录共8页；原单图/PDF/PNG/SVG与精确表保留，source49f74bd，QA receipts/local_choice_final_qa.json。轴内：output/interaction_within_s/interaction_atlas.pdf共2页，单图within_axis_additivity与within_axis_distribution；renderer a66c70a2e7a1df12c215c115af34ffc1463ba272，QA receipts/interaction_within_qa.json；两端figures.json已同步passed。不要再次交付旧拥挤初稿或重复绘制已通过页面。

新跨轴采集仍在继续，18:54:46为620/792，窗口文件数与进度一致；GPU1 PID284842、finish观察器287713均正常，后处理状态WAITING_FOR_MEASUREMENTS。无暂停请求或失败，保留现有作业，不重复启动。当前没有跨轴完成结果，INTERACTION_GATE=INCONCLUSIVE。后续仅继续新interaction完整采集/CPU分析/四页图QA，原S收尾与action导出均已完成。新的INTERACTION_PLOT_REVISION=a66c70a，INTERACTION_ANALYSIS_REVISION=1b3850e，测量INTERACTION_REVISION=49f74bd；原CODE_REVISION=a50b84d及原数据不动。

## 18:10执行入口与验证依据

后续RFV owner来信：R1 18-head已完成并为LEARNABILITY_FAIL，下一项为reverse-consistency mini及D/S固定模型诊断；GPU0保留给其短合同验证/读回任务。我方GPU1继续既定interaction，若需要让出，对方写pause并通知。此R1结果仅记录为对接任务报告，未声称本任务独立重审；不与Atlas CF headroom混用，不因此扩大当前矩阵或恢复B。

**新interaction已启动，不要重复运行。** 17:59:27 GPU1 PID284842，results/interaction_v1_s，source49f74bd5601bc298997921a93193109156adf546；18:06:12为74/792，正常保存。开发4个完整/尾窗及1个真实67帧短视频通过；analysis/interaction_{tail,short}_validation.json，原T all-heavy路径特征/损失/成本差0。原CODE_REVISION仍a50b84d，新的INTERACTION_REVISION单列。GPU0专属RFV，GPU1可在完整窗口处让出：若queue/interaction_pause.json存在，runner保存后退出且不自动恢复；不要删除该文件或越过RFV请求。

**自动CPU后处理已运行等待，勿启动第二个写者。** PID287713，analysis/interaction_finish_process.json与interaction_finish_status.json，日志logs/interaction_finish.log；采集完成后自动运行atlas_interaction_analyze.py --mode cross --input results/interaction_v1_s --output analysis/interaction_v1_s，再运行atlas_interaction_plot.py --mode cross --analysis analysis/interaction_v1_s --output output/interaction_v1_s。最后状态FIGURES_READY_FOR_VISUAL_QA仍需下载/渲染/目視QA；PAUSED_FOR_RFV时需协调后才可续跑，FAILED时读具体错误。分析源1b3850e，原10k dataset AP数学复用不变。

旧轴内重分析已完成：analysis/interaction_within_s/within_axis.json，仅总loss，不能重建cls/loc或I_null。新派生I统一为G方向，原O/D/S的R号翻转、raw不改；median/Eabs/CI齐备。详情INTERACTION_REANALYSIS.zh.md。初版8窄面板刻度拥挤，正在改成两张2×2 PDF并重查；不交付初版。此分析不运行模型，旧统计不重复。

原local-choice完整S图的六主图与两附录均已逐页目视检查；本轮49f74bd只修复D-proxy标题将模型S误读为空间轴S的歧义，更新第4页已再次渲染核对。其余页使用完整固定S统计；待写QA回执及正式交付。目录output/pdf_s/local_choice_revision。INTERACTION_GATE仍INCONCLUSIVE，不能以正在采集的部分数据提前写方向结论。

## 17:45协议登记状态（后续执行以上节为准）

用户最新明确授权加入INTERACTION_GATE与缺失关键组合补测。本节和INTERACTION_PROTOCOL.zh.md覆盖下文“禁止所有Atlas模型查询/不扩跨轴”的旧范围；仅在独立协议/源码/结果目录开展新interaction v1，原a50b84d记录冻结，B四项HELD不变，不训练模型，不阻塞RFV。已核对旧数据没有跨轴factorial；旧coalition仅总loss，不能事后补cls/loc；单次null/benign不能构造I_null。新协议包含固定成本局部D/S交换的8cell与T在S-full/sparse背景下4cell、实际8次no-op cube。尚未启动新GPU测量，等待技术验证；不能将新协议当成已有结果。

原S全部统计已完成：T/D/S allocation共90配置、recovery四项均211视频/792窗及10000次完整dataset AP bootstrap，analysis/models/s/{allocation,recovery}.json已下载。原S-S CF−Uniform在4/6/8/10/12组为+5.004/+4.438/+4.451/+3.649/+3.304pp，配对CI均正，16组0，S成本精确匹配。Recovery physical 64.2986%、packed61.1919%、cross64.5531%、cross-MDoff64.5526%；不得用各自CI重叠判断配对差值。

新619a882 renderer已生成完整local-choice图集并下载：output/pdf_s/local_choice_revision，共6主图+2附录+主图合并PDF。main p2(d)以已有benign经验带区分loss增/带内/降，旧raw正负零图独立appendix_raw_signs，O另列；human actionness/动作边界术语保留。此版正在PDF目视QA，尚未交付。Action manifest此前已完成下载，不重复导出。原S closeout完成后仍保留wtr自动跟进新interaction任务，不据旧段落提前暂停。

## 16:25及以前历史快照（以下旧待办由上节覆盖）

16:25:08检查：唯一owner232154，22 COMPLETED、4 HELD、4 WAITING，全部S GPU测量维持FREEZE，B四阶段仍HELD。S allocation CPU PID248899正常运行（约79分钟、RSS约24GiB），S轴22/30 AP配置完成，最近static:4；analysis/models/s/allocation.json与recovery.json均尚未生成，无新故障。保留当前进程，完成后再接recovery CPU统计；不要重复启动或把部分配置当成完整Fig4数据。Action manifest已经完成并下载，已交付论文修订图无须重绘。

**S的五项GPU测量已全部完成，现正式FREEZE。** 15:40:19确认GPU0无compute-app、1MiB/32760MiB、0%利用率；queue/atlas_s_gpu_release.json已有实际回执，已通知RFV可供后续安排。GPU1仍由RFV使用，B四阶段继续HELD。之后只做CPU统计、图表/分析修订和manifest，不再启动Atlas模型查询。

Action exporter已于15:40:19启动PID278692，analysis/action_export_process.json，日志logs/action_export_s.log；写output/pdf_s/action_manifest。15:44仍在运行、最终manifest.json尚未出现。先检查此receipt/PID及最终文件，不能重复启动两个导出写者。当前S allocation CPU PID248899也仍运行；同机内存15:44约88GB/133GB，保持恢复统计在allocation之后接续的安排，优先留资源给RFV。

**Action export已于15:44:51成功完成，已下载本机，不再启动。** output/pdf_s/action_manifest含五个gzip JSONL及manifest/protocol，每阶段792窗口/211视频；本机实际解析了每阶段首个窗口，核对action/value/support字段与来源，receipt为action_export_local_check.json。生产完整校验及manifest副本在receipts/action_export_manifest.json；GPU释放和恢复采集回执也已下载。后续最终交付需附此本机manifest目录，当前只剩S allocation/recovery CPU统计与完整Fig4/5及图集交付。

论文修订QA标记已同步远端，当前output/pdf_s/paper_revision主四页和条件附录一页都已通过视觉检查。最终完整S图集现在应包含6张主图、1张条件附录及1个主图合并PDF；保留本次post-hoc控制、Random/Actual、regret及精确数值表。新完整Fig4/5仍待相应CPU统计完成后运行与QA。

**15:34论文修订与统计详情（采集状态已由15:40更新）：** 用户本轮论文图建议已落实。新增后续描述分析合同PAPER_FIGURE_AMENDMENT.zh.md、科学表述PAPER_NARRATIVE.zh.md；统计/renderer现为d44df777080332876c513688f2bbdabc866acf56，测量a50b84d与队列56734b5不变。原T与原population图保留。

- 论文版在output/pdf_s/paper_revision：主图四页population_atlas.pdf、五张单图PDF/PNG/SVG（含单独appendix_tad_conditions），完整numeric_tables.md及captions.md。主文去掉原p4(a-d)条件面板，新增Random、Actual参考与signed-value regret；p2含benign95阈值/超阈比例/精确top5/10/20数字，p3含k16的I均值/CI。全部五页已Poppler180dpi目视检查，QA为receipts/paper_revision_qa.json；本机figures.json已passed，需同步远端同文件。
- 新统计analysis/models/s/paper_controls.json，副本receipts/paper_controls_s.json；初版仅参考统计保留paper_controls_initial.json及本机副本。新增配对差值后，初版benign和ranking原指标逐项完全一致。新脚本tools/atlas_paper_controls.py只读取既有population，不运行模型；默认图入口含population时现在需要paper_controls.json。
- epsilon_benign,95=0.00159304；超阈O/D/S/T约26.17%/0.206%/0.257%/30.53%。阈值在同一publication cohort估计，bootstrap每次重估，是经验扰动参照，不是纯噪声或显著性阈值。NDCG方面attention/actionness/entropy配对优于Random；四proxy的signed-value regret差值CI全部跨零。因此写proxy limitations，不写所有指标全面失败。Graph与RISE各自独立验证要求已记录，不能用Atlas数据训练/调参。
- S_S采集已完整792/211，回执receipts/allocation_s_S_collected.json、allocation_s_S_manifest.json。15:05:50启动单一S allocation CPU分析PID248899，analysis/s_model_analysis_process.json，本机已有副本；只复用T/D缓存并补S，日志logs/analysis_s_allocation.log。15:34为S轴6/30 AP配置，完整analysis/models/s/allocation.json尚未生成。此进程活着时不并行起第二个相同AP任务。
- GPU0已进入recovery_s，PID244836；15:34:05为743/792，尚未完成。owner232154，21完成/1运行/4 HELD/4等待。GPU1由RFV占用，B459窗及其后续保持HELD。恢复齐备后全部S GPU测量FREEZE，可按原计划导出action manifest并报告GPU0释放；恢复CPU统计等当前allocation CPU完成后接续，或明确采用不重复缓存写者的独立安排。

原来的“等待S_S采集/启动S allocation分析”已完成，不重复启动。当前剩余为恢复采集、S allocation与recovery统计、最终完整S Fig4/5运行/QA及action manifest。最新论文版主图需保留本次控制、regret和附录安排。

本节覆盖下文历史进度与旧待办。最新资源安排、完整S统计和绘图入口如下，原科学协议不变。

- RFV已准备好具体作业并请求GPU1。Atlas在B下一窗口保存后暂停其独立进程组，保留window0–458共459/792窗；未写B完成标记。GPU1实测1MiB/32760MiB、0%利用率、无compute-app，已向对方确认交接。population_b及其B-D/B-S/B-recovery四项HELD，不能自动恢复B或占用RFV显卡。
- 新唯一owner PID232154；14:45:16仍为20完成、1运行、4 HELD、5等待。GPU0 S_S原PID205345继续，707/792；后继recovery_s尚未开始。B仍459窗，四个B阶段与resource_policy均HELD。交接回执receipts/rfv_gpu1_handoff.json，当前配置receipts/rfv_resource_policy.json。
- RFV已回传实际接管信息：14:41:29启动AutoDL direct-process shell PID238069、CUDA_VISIBLE_DEVICES=1、源2ca4d4b，依次执行完整fit-video evaluator smoke、T-local-CF cal20代表窗、mini32/10/10。详情receipts/rfv_gpu1_acceptance.json。RFV运行细节来自其owner回执，本任务独立核对的是Atlas B仍暂停、S继续；等待RFV归还回执，不自行使用GPU1。
- 暂停是在观察到下一次原子progress提交并读取对应窗口JSON后触发；可能刚开始的未提交下一窗口留待续跑，不声称精确停在Python循环指令边界。未中断S测量，未删结果。
- 当前分析/renderer为c65e6d36603c3d570a534f966a79ba2e9d2bc811，含79f8632单模型/section入口和d4b869f、c65e6d3视觉修订；继承c16bafc AP写锁/缓存，统计数学不变。测量仍a50b84d、队列控制56734b5、已交付T图renderer6dbd246分别保留。Action exporter为251a00b。
- **S D完整预算统计14:11:13已完成**：原辅助PID207315完成30配置；完整211/792、两指标各10000 draws、official_AP_reproduced逐项核对通过。4/6/8/10/12组CF−Uniform为+9.333/+8.255/+6.688/+5.714/+4.173pp，95%配对CI均正；16组差0。执行成本均值绝对差最大0.1651 GFLOPs，满足原容差而非精确相等。额外CF查询28786.651 GFLOPs/窗。回执receipts/allocation_s_D_analysis.json及allocation_s_D_summary.json；不要重复D统计。
- 以上D结果仍为冻结模型、GT辅助、有限组参考，不是新WTR/Value训练性能。不能将单点近零率当成删减比例；微动作测量和cheap-base分组分配条件不同。
- **完整S population四图已通过PDF QA**：output/pdf_s/population下fig1_controlled_interventions、fig2_task_necessity、fig2_joint_interventions、fig3_localization_structure，均PDF/PNG/SVG，合并population_atlas.pdf四页。Poppler180dpi逐页目视，修复标题重叠和案例量级压平；两端figures.json为passed，回执receipts/population_s_figures_qa.json。
- Fig2已包含null/benign、正部Lorenz/top-p及显式有符号比例；coalition新增I=joint−sum及95%CI；案例观察/深度分开纵轴并标明真实尺度。Fig3明确D轴，O/S条件统计仍在数据中。原T两图未重绘。
- S population的新analysis/models/s/population.json与旧cache.summary完全一致，未重新bootstrap，也未写原全S/B population.json。

接下来必须完成：

1. S_S采集齐备后，用单一CPU运行python tools/atlas_analyze.py --mode allocation --backbone s --input results --output analysis --resources resources.json --bootstrap 10000。环境CUDA_VISIBLE_DEVICES=''，OMP/MKL/OPENBLAS线程数4；复用原analysis/ap的完整T/D缓存，仅补S。原D辅助已结束，不重复或并发起相同切片。
2. recovery_s齐备后同入口--mode recovery。单模型汇总写analysis/models/s/{allocation,recovery}.json，不能伪造原全S/B完成标记。
3. 完整S图：python tools/atlas_plot.py --analysis analysis/models/s --raw results --output output/pdf_s --backbone s --section all。新增Fig4配对差值行、Fig5各项CI，代码仅语法验证，必须用齐备实际数据完成运行和PDF QA。S完整图集为6单图加1合并PDF；不等待暂停B项。
4. tools/atlas_export_actions.py已实现/部署（251a00b，语法通过，完整运行待S恢复齐备）。用--backbone s --input results --resources resources.json --output output/pdf_s/action_manifest，要求五个S阶段都792/211后导出gzip JSONL和manifest，保留原动作/组选择/支持/损失/成本/来源，省略预测、图片、逐query保真向量，附冻结协议。它不是RFV训练bank。
5. 只完成上述已登记S范围；T×D/T×S/D×S仍为RFV/TDS缺口，不扩Atlas。完成S统计、图QA、manifest与交付后暂停既有wtr自动跟进，不归档；B明确仍暂停。

以下为保留的原协议、历史执行依据与回执索引；旧快照中的PID、B继续运行、D统计待完成、S-only待适配及Fig2未加控制等描述已由本节更新。

## 用户授权与目标

用户2026-09-15明确授权在 `ssh -p 44909 root@connect.nmb1.seetacloud.com` 实施第一阶段正式实验与CVPR标准科研绘图。后续两次异步回复又明确确认：

1. Fig.4采用16个真实primitive并集的有限分组参考，形成完整预算曲线；逐点统计每轴16个不变。
2. 自动跟进当前任务，实验完成后继续进行PDF版面检查并回传正式图。

本轮不训练新WTR，不接Graph/RISE，不修改原Slurm队列。全数据测量和所有统计允许继续；技术故障可修复并续跑，不能按publication结果改科学定义或阈值。

## 本机和服务器

- 本机独立工作树：`C:/Users/skywalker/Documents/ChatGPT/H65/h65_clean_adatad/wtr_characterization_20260915`
- 分支：`codex/wtr-characterization-20260915`，从1955057508af5a5dfd59a98bddf49302bee5972c创建。
- 当前已部署代码：a50b84d（完整SHA见工作树Git及远端CODE_REVISION）。后续仅文档提交不要回填为旧测量源码。
- 当前分析/绘图代码见首节；原c16bafc新增完整D/S/恢复切片CPU预计算和AP写锁，继承c91cbd7成本账本、d23aff6 population缓存修复、658f9e3等价bootstrap。已交付T图renderer仍6dbd246，测量保持a50b84d。
- 旧 `graph_tad_20260914` 及其未提交characterization原型未改；只在新工作树复用、修订相关代码。
- 远端根：`/root/autodl-tmp/wtr_characterization_20260915`
- Python：`/root/autodl-tmp/envs/opentad/bin/python`；数据 `/root/autodl-tmp/thumos14`，200训练/211测试，正式792窗口。
- 两张RTX4080 SUPER，实际各32760MiB；容器memory.max约124GiB，CPU配额32核。每卡最多一个本队列GPU任务。
- Matplotlib3.10.9、SciPy1.15.3、Pillow12.3.0已存在；远端pdftoppm尚未安装。最终可下载PDF到本机用FastCtx PDF image模式或本机PDF运行时渲染。
- SSH偶发connection closed，重试相同读/复制命令一般成功。scp使用 `-O`。长传输用FastCtx后台任务；切勿在30秒超时前台命令里传100MB文件。

## 已完成的真实验证

- Dense-S官方EMA严格加载499键，完整211视频/792窗口已完成。
- Avg-mAP=68.9767271877%，mAP@0.7=48.2622113548%，mean GFLOPs/window=2347.894038528。
- 历史S69.0126%，差-0.0358728123个百分点，位于事先固定0.10pp技术容差内。
- 回执：远端 `results/baseline_s/completed.json`，测量科学代码75ba823；完成时间2026-09-15 00:52:41 +0800。
- Dense-B也已完成211视频/792窗口：Avg-mAP=71.1416637624%，mAP@0.7=49.5514356021%，mean GFLOPs/window=8082.154708992。
- B相对历史71.1204%为+0.0212637624pp，位于0.10pp容差内；回执 `receipts/baseline_b.json`，实际测量源码a50b84db1bcdafd86ceec4d9737156807f9daa9a，完成时间2026-09-15 01:50:20 +0800。两套官方dense参考均已通过。
- S真实完整开发视频5窗口，null、D/S light升级、T单帧交换、三轴有限分配及成本ledger均通过；15个policy/axis组合的完整视频AP与官方AP缓存复算一致。
- 回执：`results/validation_s/passed.json`，仅技术验证，不能作为正文方向证据。
- 原轴恢复S预检已通过，完整V2 checkpoint重构和四恢复器同支持执行成功：`results/recovery_preflight_s/shard_0_done.json`。
- B的完整开发视频5窗口干预与15组官方AP复算也已通过，回执已保存为 `receipts/validation_b.json`；B恢复预检及T/D/S三个32视频开发测量已完成。两模型的Static排序均已冻结，副本 `receipts/static_orders.json`。
- S的正式T分配采集已覆盖全部211视频/792窗口，回执 `receipts/allocation_s_T_collected.json`、`allocation_s_T_manifest.json`。07:39已确认五策略×六预算共30组完整官方AP和10000次video bootstrap全部完成，摘要 `receipts/temporal_s_summary.json`，原完整统计远端 `analysis/temporal_s_ready.json`。
- B正式T分配采集也已完成211视频/792窗口，回执 `receipts/allocation_b_T_collected.json`、`allocation_b_T_manifest.json`。09:20已确认30组完整AP/10000次bootstrap也全部完成，摘要 `receipts/temporal_b_summary.json`；S/B合并统计远端 `analysis/temporal_only.json` 已存在。
- S正式population采集及10000次bootstrap统计已完成211视频/792窗口。采集回执 `receipts/population_s_collected.json` 与 `population_s_manifest.json`；官方EMA严格499键、实际源a50b84d。完整统计副本 `receipts/population_s_analysis.json`，简明数值 `population_s_progress_summary.json`；远端缓存 `analysis/population_models/s.json`。
- S正式D分配已完成211视频/792窗口采集，回执 `receipts/allocation_s_D_collected.json`、`allocation_s_D_manifest.json`。CPU已开始完整AP/10000次bootstrap，当前不能用采集完成代替D分配收益结论。
- 曾因遗漏 `references/ASFormer/model.py` 导致恢复预检失败，已经把原Git跟踪的ASFormer源码依赖加入部署包后通过。历史失败保留在queue/failures及日志。
- 两个独立只读代码核验确认：D/S masks与真实成本、T交换/组预算、population/conditional分离、video bootstrap和官方AP缓存逻辑正确。AP缓存增加backbone/source/replicates合同校验。归一化boundary横轴标注本来正确，未按错误建议改成秒。

## 队列

唯一owner：远端 `tools/atlas_queue.py`；当前PID 32242（只是01:17快照，后续必须读launch/status，不复用陈旧PID）。

- `queue/status.json`：当前状态。
- `queue/plan.json`：30个依赖阶段及命令。
- `queue/launch.json`：owner回执；`queue/owner.log`、`logs/<stage>.log`：日志。
- `queue/measurements_and_figures_ready.json`：数据与图已生成，但视觉检查仍需完成。
- `queue/failed.json`：当前停在技术/执行失败；重启时归档到queue/failures，旧失败不能覆盖新的RUNNING状态。

12:56:28 +0800队列快照：20 COMPLETED、2 RUNNING、8 WAITING。唯一owner及活动进程未变，S轴分配、B population和D轴CPU统计持续推进；未发现新的执行故障，完整图集尚未就绪。

- GPU0：allocation_s_S，PID205345；12:56读到166/792。此前population_s和allocation_s_D均已完成792/792。
- GPU1：population_b，04:51:25启动（PID133234），本轮最近读到395/792。
- 两套模型的技术验收与开发校准均已完成；S/B T均已有完整数据统计和已通过视觉检查的首批图。其他轴/population尚未齐备；全套8份PDF尚待测量完成。
- 本轮未发现新的失败；未改变科学协议、源测量或队列。普通进度不作方向结论。

流水线：S完整baseline/技术/AP/恢复预检 → S三个32视频开发校准 → Static冻结 → S全量T分配、population、D/S分配、恢复；B在S技术通过后沿同流程独立推进。最后10000次video bootstrap、官方AP重算与图表生成。没有科学早停。

队列重启示例（先确认owner已退出）：

`cd /root/autodl-tmp/wtr_characterization_20260915 && /root/autodl-tmp/envs/opentad/bin/python tools/atlas_queue.py --launch --revision 5236ae4`

`--revision`标记原科学实现；每个新测量进程从远端CODE_REVISION读取实际部署源码并写入manifest/record。队列可接管仍活着的旧子进程；不能另起第二个owner。已完成窗口JSON不覆盖，续跑跳过已写文件。

## 代码与协议

- `research/atlas_20260915/PROTOCOL.zh.md` 与 `protocol.json` 是本轮冻结协议，原用户文本在inputs。
- `h65/atlas/`：official heavy reference、trained light substitutions、T真实RGB选择、D/S primitive与group、coalition、同支持恢复、数据/统计。
- `tools/atlas_run.py`：baseline/preflight/calibration/population/allocation/recovery，可续跑；population会保存实际输入缩略图。
- `tools/atlas_validate.py`：开发完整视频AP重算验收。
- `tools/atlas_analyze.py`：Static冻结、10000次视频聚类统计、全数据AP、配对CF-Uniform区间。
- `tools/atlas_plot.py`：只读测量JSON，不执行模型。7张单独科研图PDF/SVG/PNG＋1个合并PDF，共8份PDF；Fig6没有新WTR数据，不生成。
- `tools/atlas_temporal_analysis.py`（0a2b105）：已齐备T切片的CPU预计算，调用原 `evaluate_variants` 和同样10000次bootstrap，写入原 `analysis/ap/allocation_{s,b}_T/` 缓存，未来原analyze_allocation阶段直接复用。不是第二个队列owner，不改变原30阶段或GPU任务，不改原始测量。
- 正式输出远端 `output/pdf/`；最终复制回本机工作树同目录，渲染检查后再发给用户。

官方dense本身无light：D/S比较用V2 epoch40训练过的light算子，明确标注来源。full-heavy仍是官方模型。恢复用完整V2 EMA与精确初始backbone差异重构；不需要随机初始化light或重新训练。

Fig4只说明有限分组空间的冻结模型headroom，不是oracle。Attention使用dense诊断分数，特权评分/CF查询成本单列，不能称廉价可部署router。D/S允许全局成本下分层配额变化；每最终方案实测成本并匹配dense0.5%容差。T的padding在所有策略固定保留，真实执行成本照计。

所有后续Value Head训练只能读取training/development标签，publication/test atlas不得用于训练或反馈式调整方法。当前代码没有任何优化更新。

## 自动跟进与剩余工作

用户已同意当前任务自动跟进。已创建ACTIVE heartbeat：automation id `wtr`，名称“WTR全数据实验与科研绘图跟进”，每15分钟回到当前任务。不要建立重复自动化、新任务或cron替代heartbeat。

**13:29后的有效范围与调度以 `RFV_CLOSEOUT.zh.md` 为准。** 用户已授权RFV优先、Atlas-S有限收尾，对接任务本轮已再次确认。旧30阶段仍保存，但未启动的B-D/B-S/B-recovery已暂停，不能依据下文历史“全S/B”计划自动恢复。S完整测量后只做统计/图表/分析修复/manifest。T×D/T×S/D×S明确未测，对方同意列RFV/TDS缺口，本轮不扩Atlas。

13:27:42控制代码56734b5仅重启owner：PID32242→221333，原S_S PID205345、B population PID133234保持运行；`receipts/rfv_closeout_applied.json`有前后验证。13:29状态20完成、2运行、3 HELD、5等待；S_S330/792，B population422/792；D CPU17/30。GPU0还需S_S→recovery_s后才可交接；RFV目前不要求抢占GPU1。远端queue/resource_policy.json在owner启动时读取。测量a50b84d、分析c16bafc保持原身份，控制版本另记。

S齐备即须交付S-only正式图，原analysis/renderer硬编码S/B部分尚需只读修改并复用AP缓存，不能等待被暂停的B补充或伪造B完成。queue/atlas_s_measurements_ready.json只标记已登记S采集完成，不代表统计、QA或跨轴证据完成。既有wtr自动跟进已更新此范围；完成S统计/图QA/action manifest与交付后暂停，B未完成部分明确保留为暂停。当前负结果详见 `NEGATIVE_RESULTS.zh.md`。

每次跟进先读本文件和协议，再SSH读当前queue状态及活动stage进度。需要时修复技术问题，保持原始测量/科学协议；不取消别的任务或新建训练。只在实质里程碑、错误/需要操作或最终交付时报告，普通进度变化无需逐轮通知。

用户另行明确授权与任务“实现 Raw-v1 并部署并行实验”（01a0a12c-241a-7772-986d-387138bce4d6）报告讨论；已完成双向报告及两轮实质反馈，记录 `discussion_raw_v1/BRIEF_1125.zh.md`、`EXCHANGE.zh.md`。不存在尚待回答的关键口径问题，不要重复轮询/催答；后续真实里程碑可再同步。论文证据分工为Atlas有限分配机会→Core匹配V/U与router-label验证→Raw独立域增益及匹配重训。对方按其另行授权的Fast-Track课程运行，Atlas范围不接管或取消对方训练。

对接澄清：Atlas O=Obs-replace，Raw O=Official-grid；单点升级和当前稀疏状态swap条件不同，不混标签。Raw验证6视频/19窗口，mini另为24视频、346动作；候选帧池每state确实新增576个off-grid候选，但38/45相同的是实际查询swap集合，180个R+查询仅24 off-grid。mini视频划分16/4/4互斥，4个holdout视频属与Core相同的完整20-video holdout子集，不能称完成Core holdout验证。合法exact-K交换、16-pair提议子集、实际查询集合分层；不引入不存在的max-gap限制。对方新Core mAP、稳定checkpoint re-query与Raw Value均未获科学PASS，详见其11:10–11:33 REPORT和RAW_SPLIT_CLARIFICATION。

展示层后续动作：双方同意Fig2完整图保留null/benign参照及正负部，Fig4分开执行与特权查询成本；Core Fig6和Raw主性能图等待相应完整证据。当前atlas_plot.figure2尚未把control曲线/显式负部面板加进图内（数值已保留）；完整绘图时需落实并QA，不把本次口径共识写成这项绘图改动已经完成。测量/统计定义与冻结阈值不变。

完整切片CPU入口已实现并部署：`python tools/atlas_slice_precompute.py --stage allocation_s_D --launch`；可选S/B的D、S以及recovery。只处理有完整采集回执且精确匹配211视频/792窗口的数据，allocation先检查全部matched；调用原evaluate_variants和10000次bootstrap，不执行模型。AP写入原 `analysis/ap/<stage>/`，单切片汇总为 `analysis/precomputed/<stage>.json`，不会写全模型完成标记。原对应analyze已启动或该类全部采集齐备时，让原owner接手，辅助不再新起。

当前唯一辅助CPU进程：12:29:46启动PID207315、stage=allocation_s_D，回执 `analysis/slice_precompute_process.json`（本机receipts同名），日志 `logs/analysis_precompute_allocation_s_D.log`。12:56为7/30配置完成，最近marginal_cf:10，进程运行26:48、RSS约24GiB；整切片汇总尚未生成。首个真实配置已核对完整211/792、10000次、official_AP_reproduced=true。此进程活着时不要另起切片辅助；S轴/恢复齐备后等它结束再接续。切片入口只需启动一次，后续检查该receipt、日志和precomputed输出。

c16bafc用 `analysis/ap/<stage>/writer.lock` 串行化辅助与最终分析对同一缓存的访问，原数学函数体未改，finally显式unlock以覆盖官方AP fork期间的FD继承。12:35实际非阻塞第二写者探针被阻塞，且首个真实AP缓存成功写出；回执 `analysis/slice_lock_probe.json`。原30阶段和GPU任务未变，最终分析直接复用这些缓存。服务器Linux是统计执行环境。

S population CPU预计算已成功结束，不要重复启动。首次PID172075在保存时因cache路径变量target被病例分位值覆盖而失败，未写出任何缓存或全模型标记；旧日志/回执留在远端 `logs/population_s_precompute_attempt1.log`、`analysis/population_s_precompute_attempt1.json`。d23aff6把缓存路径独立命名为population_cache；10:03:57启动PID173893，以相同原始数据和10000次bootstrap重算，已成功写出约3.2MB的完整 `analysis/population_models/s.json`。当前process回执已同步本机，日志明确 `Full population statistics cached: s`；`analysis/population.json`仍不存在，原队列不会提前判定全模型完成。

每模型缓存只改变分析调度，原归一化、抽样、CDF/Lorenz/Gini、coalition/proxy、训练集时长阈值与10000次bootstrap公式保留。S缓存含analysis_contract和summary，已检查211/792、10000次、source a50b84d、训练时长阈值[1.9,4.5]秒。原30阶段的全模型analyze_population不指定backbone，将验证并复用S缓存、计算B后写原合并完成标记。未增GPU任务或第二个owner。B不完整时不生成完整Fig2/3。

S总体统计的当前边界：以预登记的1%相对窗口loss容差，video-balanced近零率为O插值94.084%、D99.968%、S99.970%、T94.530%。O/D/S各12672动作、211视频；T为12284合法交换、187个eligible视频，不能伪称T覆盖211个有合法交换的视频。null=100%，benign=99.941%；D/S正部Gini为0.947/0.938，benign也达0.882。这里只说明所采primitive尺度上的边际效应，不能把近零率直接当成可删除计算比例，亦不能仅凭Gini高宣称路由信号可靠。

S的16动作联合验证：O/D/S平均interaction=joint−sum分别−5.781e−5/−2.560e−6/−4.741e−8，95%视频CI均跨零。当前不能宣称已证明系统性强非加性，也不能据此证明全局可加或次模。16动作时两种排序选中同一完整16动作集合，结果相同是预期。B和D/S完整预算AP仍须继续。

用户报告复核的S条件数据：D轴的attention_score/actionness/entropy/feature_norm视频平均Spearman依次为−0.00233/+0.00550/+0.00227/+0.01334，四项95%区间均跨零。这些代理在当前primitive尺度上未给出强排序相关证据，但不能据此证明新Value Head可学习。D轴四个位置组及三个时长组的总效应区间也均跨零；这些单组区间不是组间差异的配对检验，不能替代边界/短动作优先性的正式对照结论。Fig3的跨模型论证仍待完成。

c91cbd7使最终allocation/recovery汇总包含cost_ledger，并让最终图集附 `compute_cost_ledger.json` 与 `compute_cost_ledger.md`。CF主曲线账本由base+每个候选的(base+delta)直接恢复，Attention用dense_gflops；恢复分开shared_gflops、variant.gflops−shared_gflops、dense_target_gflops。全观察target额外编码/插值复用共享preview，不是单独完整detector前向。已用一个真实T窗口和一个开发恢复窗口做字段/执行烟测（仅技术验收，回执 `receipts/cost_ledger_schema_check.json`），未新增模型查询；全数据账本随原分析阶段生成。

CPU时间轴预计算已完成：`analysis/temporal_only.json` 记录完成时间09:19:14，60组正式统计齐备。保留 `analysis/temporal_precompute_process.json` 与 `logs/analysis_temporal_precompute.log` 为来源记录，不要重新计算。旧PID134772切换等价内核后，由05:52:43启动的PID141411完成全部S/B。过程回执本机副本为 `receipts/temporal_precompute_process.json`。

T-only首批图已完成并在本轮交付。09:20:04启动的只读CPU绘图PID167021已退出；之后以6dbd246只重绘版面，未重跑模型或统计。远端和本机均有 `output/temporal/temporal_budget_curves.{png,svg}`、`temporal_paired_difference.{png,svg}`、`report.md`、`cost_ledger.json`、`figures.json`。两张PNG已亲自目视检查，S/B对应面板采用共同纵轴、预算组数已标明，图例/标签无截断重叠；visual_review通过记录已同步两端。SVG由同一Matplotlib figure导出。当前明确只覆盖完整T切片，原8份最终PDF仍须继续完成。不要无故重新生成或重复交付这两张图。绘图进程回执 `receipts/temporal_plot_process.json`，原日志 `logs/temporal_plot.log`。

T额外评分成本已逐窗完整汇总：CF共同base＋12候选前向的平均成本为S 10320.533、B 34919.106 GFLOPs/窗；这套参考排序由六预算共享，位于图中的所选执行成本之外。Attention的dense诊断前向为S 2347.894、B 8082.155 GFLOPs/窗；host侧排序/评分算术未计入FLOP模型。JSON/英文report中已明确该口径。

05:26检查发现CPU统计24分钟只完成2个T配置（S attention:10/12，均通过全数据官方AP核对）。现已验证并部署数学等价的FP64 Numba bootstrap加速；10000次、原NumPy Generator(42)权重、固定排序/匹配和指标不变。Numba0.57.1/llvmlite0.40.1仅安装在项目analysis_runtime，不改OpenTAD环境/NumPy；依赖见 `analysis_runtime_requirements.txt`。内核使用workqueue，因为官方AP每次会fork，Linux GNU OpenMP不能安全支持这一连续调用。

加速验证和集成源码658f9e3：`h65/atlas/fast_bootstrap.py`、`tools/atlas_bootstrap_validate.py`、`h65/atlas/statistics.py`。验证PID140897已成功退出，回执 `receipts/bootstrap_workqueue_validation.json`（远端 `analysis/bootstrap_validation/passed.json`）。以原attention:10完整211/792的全部10000次已完成统计逐项对照，并核对65组权重的全部五阈值、全部per-video诊断及零GT/重复权重，最大指标误差7.11e-14个百分点；workqueue共10212组权重含首次编译35.43秒。还实际执行了“并行bootstrap → 官方多进程AP → 再次并行bootstrap”，全部通过。不能把内核计时说成整个分析流水线耗时。

05:59首次生产结果 `analysis/ap/allocation_s_T/attention_6.json` 已写出，完整211/792、10000次bootstrap、official_AP_reproduced=true、kernel=fp64_numba_fixed_order_equivalent；本机回执 `receipts/allocation_s_T_attention6_numba.json`。09:20确认S/B各30/30已完成，均已逐项核对cohort、backbone、两指标10000次样本及官方AP验证标志，摘要已复制本机。原始GPU任务和唯一owner始终未中断。

首个完整S T证据：CF−Uniform在6/8/10/12组同执行成本预算分别为+4.285/+3.507/+2.951/+2.028pp，逐点95%配对CI分别[3.051,5.855]/[2.337,4.692]/[1.668,4.119]/[1.035,2.855]；4组共同base和16组全选的差为0。所有预算cost_difference=0。这只支持冻结S模型、预登记有限帧组中的GT辅助分配参考空间，不是新WTR/router性能，也不能直接推广到D/S。该里程碑已向用户报告，并已结合独立B统计完成首批图与额外成本账本。

完整B T证据：相同6/8/10/12组的CF−Uniform分别+2.144/+1.392/+1.305/+0.413pp；逐点95%配对CI为[1.044,3.511]/[-0.058,2.594]/[0.055,2.404]/[-0.655,1.331]。8和12组区间跨零，不能写成B所有预算均有确定提升。4/16组差为0，各点cost_difference=0。后续汇总必须保留S/B这种不同的证据强度。

数据齐后，检查analysis与8份PDF/7张PNG/SVG确实完整；下载到本机，逐页用PDF渲染PNG进行视觉检查，修正重叠、字号、图例等展示问题而不改数据定义。最终报告Fig2和Fig4的实测结论、失败/负结果、完整成本、CI与来源。

PDF技能已读取。`mark_artifact_operation_started.mjs --operation-kind create --expected-output-count 8 --output-format pdf`已在本机成功执行一次（2026-09-15本轮），不要无故重复；只有最终图产生后才可宣称已完成视觉QA。

全部交付并说明实测边界后暂停此heartbeat，不自动归档当前任务。
