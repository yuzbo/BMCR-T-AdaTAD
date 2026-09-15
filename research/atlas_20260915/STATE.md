# WTR Publication Atlas - 执行交接与当前入口

## 用户授权与目标

用户2026-09-15明确授权在 `ssh -p 44909 root@connect.nmb1.seetacloud.com` 实施第一阶段正式实验与CVPR标准科研绘图。后续两次异步回复又明确确认：

1. Fig.4采用16个真实primitive并集的有限分组参考，形成完整预算曲线；逐点统计每轴16个不变。
2. 自动跟进当前任务，实验完成后继续进行PDF版面检查并回传正式图。

本轮不训练新WTR，不接Graph/RISE，不修改原Slurm队列。全数据测量和所有统计允许继续；技术故障可修复并续跑，不能按publication结果改科学定义或阈值。

## 本机和服务器

- 本机独立工作树：`C:/Users/skywalker/Documents/ChatGPT/H65/h65_clean_adatad/wtr_characterization_20260915`
- 分支：`codex/wtr-characterization-20260915`，从1955057508af5a5dfd59a98bddf49302bee5972c创建。
- 当前已部署代码：a50b84d（完整SHA见工作树Git及远端CODE_REVISION）。后续仅文档提交不要回填为旧测量源码。
- 当前分析/绘图代码：7c9c2b3（远端PLOT_REVISION），增加已完整模型的population统计缓存；统计公式未改。包含658f9e3等价CPU bootstrap内核与6dbd246的T-only绘图。已交付T图的renderer_revision仍是6dbd246；测量执行代码保持a50b84d。
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
- S正式population采集已全部完成211视频/792窗口，回执 `receipts/population_s_collected.json` 与 `population_s_manifest.json`；官方EMA严格499键、实际源a50b84d。只代表完整干预记录齐备，分布/coalition统计正在CPU预计算。
- 曾因遗漏 `references/ASFormer/model.py` 导致恢复预检失败，已经把原Git跟踪的ASFormer源码依赖加入部署包后通过。历史失败保留在queue/failures及日志。
- 两个独立只读代码核验确认：D/S masks与真实成本、T交换/组预算、population/conditional分离、video bootstrap和官方AP缓存逻辑正确。AP缓存增加backbone/source/replicates合同校验。归一化boundary横轴标注本来正确，未按错误建议改成秒。

## 队列

唯一owner：远端 `tools/atlas_queue.py`；当前PID 32242（只是01:17快照，后续必须读launch/status，不复用陈旧PID）。

- `queue/status.json`：当前状态。
- `queue/plan.json`：30个依赖阶段及命令。
- `queue/launch.json`：owner回执；`queue/owner.log`、`logs/<stage>.log`：日志。
- `queue/measurements_and_figures_ready.json`：数据与图已生成，但视觉检查仍需完成。
- `queue/failed.json`：当前停在技术/执行失败；重启时归档到queue/failures，旧失败不能覆盖新的RUNNING状态。

09:51:15 +0800队列检查：19 COMPLETED、2 RUNNING、9 WAITING。唯一owner未变，S population完成后自动衔接S D分配；当前failed/figures_ready标记均不存在。SSH前3次握手关闭，等待15秒后恢复；服务器任务未中断。

- GPU0：allocation_s_D，09:45:14启动，PID170038；09:51读到22/792。此前population_s已完成792/792。
- GPU1：population_b，04:51:25启动（PID133234），本轮最近读到244/792。
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

每次跟进先读本文件和协议，再SSH读当前queue状态及活动stage进度。需要时修复技术问题，保持原始测量/科学协议；不取消别的任务或新建训练。只在实质里程碑、错误/需要操作或最终交付时报告，普通进度变化无需逐轮通知。

新增S population CPU预计算：09:56:28启动PID172075，回执 `analysis/population_s_precompute_process.json`（本机副本receipts同名），日志 `logs/population_s_precompute.log`。使用 `tools/atlas_analyze.py --mode population --backbone s --input results --resources resources.json --output analysis --bootstrap 10000`。09:59检查运行2:36、约1.3GiB RSS、持续CPU计算，cache及全模型完成标记均尚未生成；进程活着时不要重复启动。

7c9c2b3只增加每模型缓存与原有--backbone参数的population作用域，原归一化、抽样、CDF/Lorenz/Gini、coalition/proxy、训练集时长阈值与10000次bootstrap公式逐行保留。源码经过独立只读核验及服务器Python语法检查。成功写 `analysis/population_models/s.json`，包含analysis_contract和summary；S-only调用不会写 `analysis/population.json`。原30阶段队列的全模型analyze_population不指定backbone，将验证并复用S缓存、计算B后写原合并完成标记。未增GPU任务或第二个owner。后续检查缓存完成再报告数值，B不完整时不生成完整Fig2/3。

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
