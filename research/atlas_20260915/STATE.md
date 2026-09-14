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
- 当前分析/绘图代码：571ca06（远端PLOT_REVISION），包含658f9e3的等价CPU bootstrap内核和T-only绘图入口；测量执行代码仍保持a50b84d。
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
- S的正式T分配采集已覆盖全部211视频/792窗口，回执 `receipts/allocation_s_T_collected.json`、`allocation_s_T_manifest.json`。这是全窗口预测与成本数据采集完成；完整数据集AP/10000次CI尚待统一分析，不能据局部loss宣布headroom结论。
- B正式T分配采集也已完成211视频/792窗口，回执 `receipts/allocation_b_T_collected.json`、`allocation_b_T_manifest.json`。S/B时间轴全量输入均已齐备。
- 曾因遗漏 `references/ASFormer/model.py` 导致恢复预检失败，已经把原Git跟踪的ASFormer源码依赖加入部署包后通过。历史失败保留在queue/failures及日志。
- 两个独立只读代码核验确认：D/S masks与真实成本、T交换/组预算、population/conditional分离、video bootstrap和官方AP缓存逻辑正确。AP缓存增加backbone/source/replicates合同校验。归一化boundary横轴标注本来正确，未按错误建议改成秒。

## 队列

唯一owner：远端 `tools/atlas_queue.py`；当前PID 32242（只是01:17快照，后续必须读launch/status，不复用陈旧PID）。

- `queue/status.json`：当前状态。
- `queue/plan.json`：30个依赖阶段及命令。
- `queue/launch.json`：owner回执；`queue/owner.log`、`logs/<stage>.log`：日志。
- `queue/measurements_and_figures_ready.json`：数据与图已生成，但视觉检查仍需完成。
- `queue/failed.json`：当前停在技术/执行失败；重启时归档到queue/failures，旧失败不能覆盖新的RUNNING状态。

06:17:30 +0800队列检查：18 COMPLETED、2 RUNNING、10 WAITING。唯一owner及两个GPU任务PID与前次相同，日志持续推进，当前failed/figures_ready标记均不存在，无GPU故障。

- GPU0：population_s，02:55:16启动，本轮最近读到392/792。
- GPU1：population_b，04:51:25启动（PID133234），本轮最近读到70/792。
- 两套模型的技术验收与开发校准均已完成，尚无完整allocation/population科学结论或正式图。
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

还需检查CPU时间轴预计算：`analysis/temporal_precompute_process.json`、`logs/analysis_temporal_precompute.log`、`analysis/temporal_only.json`。旧PID134772在确认命令及独立进程组后，仅终止其CPU组并切换等价内核；05:52:43启动新PID141411（只是快照；读当前receipt确认）。CPU/GPU隔离，4 CPU线程，约24GiB RSS。若已运行不要重复启动。入口 `python tools/atlas_temporal_analysis.py --launch`。过程回执本机副本为 `receipts/temporal_precompute_process.json`。

`temporal_only.json`生成后，运行已部署的 `python tools/atlas_temporal_plot.py`，先交付完整S/B时间轴部分的PNG/SVG曲线和配对CI，明确只覆盖T，不能假装D/S/population已完成；原8份最终PDF安排不变。入口571ca06已通过服务器Python3.10语法检查，尚未生成或视觉检查实际图。输出 `output/temporal/temporal_budget_curves.{png,svg}`、`temporal_paired_difference.{png,svg}`、`report.md`、`cost_ledger.json`、`figures.json`。它检查全部60组AP/10000次CI，并逐窗只读汇总原始额外评分成本：CF共同base＋12次candidate前向、Attention的dense诊断前向，与所选执行成本分列。解析全量原始JSON可能数分钟，用后台运行；不执行模型。复制本机后目视检查两张PNG，再交付。这一预计算与原统一分析的数学定义一致，无新增模型查询。

05:26检查发现CPU统计24分钟只完成2个T配置（S attention:10/12，均通过全数据官方AP核对）。现已验证并部署数学等价的FP64 Numba bootstrap加速；10000次、原NumPy Generator(42)权重、固定排序/匹配和指标不变。Numba0.57.1/llvmlite0.40.1仅安装在项目analysis_runtime，不改OpenTAD环境/NumPy；依赖见 `analysis_runtime_requirements.txt`。内核使用workqueue，因为官方AP每次会fork，Linux GNU OpenMP不能安全支持这一连续调用。

加速验证和集成源码658f9e3：`h65/atlas/fast_bootstrap.py`、`tools/atlas_bootstrap_validate.py`、`h65/atlas/statistics.py`。验证PID140897已成功退出，回执 `receipts/bootstrap_workqueue_validation.json`（远端 `analysis/bootstrap_validation/passed.json`）。以原attention:10完整211/792的全部10000次已完成统计逐项对照，并核对65组权重的全部五阈值、全部per-video诊断及零GT/重复权重，最大指标误差7.11e-14个百分点；workqueue共10212组权重含首次编译35.43秒。还实际执行了“并行bootstrap → 官方多进程AP → 再次并行bootstrap”，全部通过。不能把内核计时说成整个分析流水线耗时。

05:59首次生产结果 `analysis/ap/allocation_s_T/attention_6.json` 已写出，完整211/792、10000次bootstrap、official_AP_reproduced=true、kernel=fp64_numba_fixed_order_equivalent；本机回执 `receipts/allocation_s_T_attention6_numba.json`。06:17更新：S当前10/30个配置完成，最近为marginal_cf:4；前4个正确的NumPy结果继续复用，B尚待处理。PID141411运行24:53、RSS约24GiB，连续官方AP/加速bootstrap顺序已在实际配置上正常执行。没有完整T曲线结论。日志中早期TBB警告属于已完成的首次验证，不是当前workqueue故障；原始GPU测量和唯一owner未中断。

数据齐后，检查analysis与8份PDF/7张PNG/SVG确实完整；下载到本机，逐页用PDF渲染PNG进行视觉检查，修正重叠、字号、图例等展示问题而不改数据定义。最终报告Fig2和Fig4的实测结论、失败/负结果、完整成本、CI与来源。

PDF技能已读取。`mark_artifact_operation_started.mjs --operation-kind create --expected-output-count 8 --output-format pdf`已在本机成功执行一次（2026-09-15本轮），不要无故重复；只有最终图产生后才可宣称已完成视觉QA。

全部交付并说明实测边界后暂停此heartbeat，不自动归档当前任务。
