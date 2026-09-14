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
- S真实完整开发视频5窗口，null、D/S light升级、T单帧交换、三轴有限分配及成本ledger均通过；15个policy/axis组合的完整视频AP与官方AP缓存复算一致。
- 回执：`results/validation_s/passed.json`，仅技术验证，不能作为正文方向证据。
- 原轴恢复S预检已通过，完整V2 checkpoint重构和四恢复器同支持执行成功：`results/recovery_preflight_s/shard_0_done.json`。
- 曾因遗漏 `references/ASFormer/model.py` 导致恢复预检失败，已经把原Git跟踪的ASFormer源码依赖加入部署包后通过。历史失败保留在queue/failures及日志。
- 两个独立只读代码核验确认：D/S masks与真实成本、T交换/组预算、population/conditional分离、video bootstrap和官方AP缓存逻辑正确。AP缓存增加backbone/source/replicates合同校验。归一化boundary横轴标注本来正确，未按错误建议改成秒。

## 队列

唯一owner：远端 `tools/atlas_queue.py`；当前PID 32242（只是01:17快照，后续必须读launch/status，不复用陈旧PID）。

- `queue/status.json`：当前状态。
- `queue/plan.json`：30个依赖阶段及命令。
- `queue/launch.json`：owner回执；`queue/owner.log`、`logs/<stage>.log`：日志。
- `queue/measurements_and_figures_ready.json`：数据与图已生成，但视觉检查仍需完成。
- `queue/failed.json`：当前停在技术/执行失败；重启时归档到queue/failures，旧失败不能覆盖新的RUNNING状态。

01:17:24 +0800快照：6 COMPLETED、2 RUNNING、22 WAITING。

- GPU0：calibration_s_S，14/32；S的T/D开发组测量各32已完成。
- GPU1：baseline_b，125/792。

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
- 正式输出远端 `output/pdf/`；最终复制回本机工作树同目录，渲染检查后再发给用户。

官方dense本身无light：D/S比较用V2 epoch40训练过的light算子，明确标注来源。full-heavy仍是官方模型。恢复用完整V2 EMA与精确初始backbone差异重构；不需要随机初始化light或重新训练。

Fig4只说明有限分组空间的冻结模型headroom，不是oracle。Attention使用dense诊断分数，特权评分/CF查询成本单列，不能称廉价可部署router。D/S允许全局成本下分层配额变化；每最终方案实测成本并匹配dense0.5%容差。T的padding在所有策略固定保留，真实执行成本照计。

所有后续Value Head训练只能读取training/development标签，publication/test atlas不得用于训练或反馈式调整方法。当前代码没有任何优化更新。

## 自动跟进与剩余工作

用户已同意当前任务自动跟进。已创建ACTIVE heartbeat：automation id `wtr`，名称“WTR全数据实验与科研绘图跟进”，每15分钟回到当前任务。不要建立重复自动化、新任务或cron替代heartbeat。

每次跟进先读本文件和协议，再SSH读当前queue状态及活动stage进度。需要时修复技术问题，保持原始测量/科学协议；不取消别的任务或新建训练。只在实质里程碑、错误/需要操作或最终交付时报告，普通进度变化无需逐轮通知。

数据齐后，检查analysis与8份PDF/7张PNG/SVG确实完整；下载到本机，逐页用PDF渲染PNG进行视觉检查，修正重叠、字号、图例等展示问题而不改数据定义。最终报告Fig2和Fig4的实测结论、失败/负结果、完整成本、CI与来源。

PDF技能已读取。`mark_artifact_operation_started.mjs --operation-kind create --expected-output-count 8 --output-format pdf`已在本机成功执行一次（2026-09-15本轮），不要无故重复；只有最终图产生后才可宣称已完成视觉QA。

全部交付并说明实测边界后暂停此heartbeat，不自动归档当前任务。
