# RFV Fast Sprint 当前执行状态

2026-09-15。权威计划为 EXPERIMENT_PLAN.zh.md，执行队列为 EXPERIMENT_QUEUE.json，完整用户原文在 inputs/。当前任务仍是实现并验证 RFV 最终路线；更新计划并不意味着最终模型已经证明。旧状态覆盖记录已归档至 EXECUTION_STATE.history_before_completed_B0.zh.md，不再从旧PID启动重复任务。

## 当前结论与唯一下一步

T-local headroom 已过，但 Plain mini、同视频未见动作泛化、G1 mini、历史 RISE-B0 尚未过相应科学条件。不得启动T/Graph/FVD detector长训，不扩32→64，不重复8/8，不加容量/步数。下一步只使用已有mini fit bank做一次CPU descriptor可分辨性诊断：检查动作行对应、相同/相近407D输入与真实gain关系；不新增GPU标签，不训练新router，不读outer20。精确重复输入出现超噪声不同标签才证明观测缺失，近邻关系弱不能单独证明不可学。讨论任务已收到全部新结果并被请求核验这一步是否必要。

## 已完成科学结果

- T_LOCAL_CF.json：G0b cal20全部46窗，Uniform92.49616075%、LocalCF93.02457398%，Δ+0.52841322pp，10000次视频配对bootstrap CI[+0.08244043,+0.71468687]pp，AP@.7 Δ+0.82774438pp。训练侧GT辅助bounded greedy参考，不是官方test或learned router。G0a平均loss gain .00962711、CI[.00524361,.01499692]。旧报告task_course_eligible仅指headroom，不能作为长训准入；新分析已更名headroom_gate_passed，不改原GPU证据。
- GRAPH_G1_T_MINI.json：32/10/10登记视频，43个可执行state、688actions；fit25/400、cal8/128、inner10/160。3seed内层regret：PlainM .003681657、PlainL .003952155、Static .003386865、Dynamic .003463607；STOP .003600146、随机合法swap .003435326。Static−PlainL CI[-.001270013,+.000095136]跨0，seed方向不稳。Plain和Graph mini均未过，不能推断所有Graph方法无效。
- ACTION_HOLDOUT_DIAGNOSTIC.json：同fit25视频按物理(insert,remove)排序，交替8fit/8held；PlainM 2000steps×3seeds，零新CF。fit rho .9403；held rho .04444、regret .001492071、STOP .001413537；cal rho−.17672。此前“仅跨视频失败才扩32→64”的条件不满足。现有报告不含held随机控制CI，不能伪称已经比较。
- RISE_A0_ALL.json：同一固定manifest在真正V2-S20/40/60重执行，20→60 rho .776412、TopK .726744、signflip .143895。actual drift已观测；早期实际排序仍保留多数收益。它不是新T轨迹，也不证明可学习预测。
- RISE_B_T.json：θ20拟合→θ40连续同Adam+真EMA4000updates；同s40运行全部函数，θ60fit labels不优化。cal选β1.1；Current=Post与Future regret均 .003277700146，EMA .003263545533。Future−Post regret CI[0,0]，Future−EMA mean+1.4154613e−5、CI[0,4.2463839e−5]；forecast_value_gate=false，fvd_unlocked=false。β见到later cal labels，只称历史离线诊断。

## 资源与已退出进程

2026-09-15约16:00 AutoDL nvidia-smi compute-apps为空，当前没有RFV GPU计算。CPU action-holdout PID280190和GPU0 B0 PID280193都已完成，回执value_diagnostics_launch.json，状态/日志value_diagnostics_status.json。RISE-A0 GPU1 PID272682此前已完成；不要重新启动任何一个。

AutoDL root /root/autodl-tmp/rfv_20260915，Python /root/autodl-tmp/envs/opentad/bin/python，SSH -p44909 root@connect.nmb1.seetacloud.com。GPU0 UUID GPU-da4cac68-a006-0219-7e01-306b04c2450a；GPU1 UUID GPU-140dfb95-91d3-eb14-bcd7-68de19f0ba9d，实有RTX4080SUPER/32760MiB。Atlas已释放GPU0，GPU1已明确交接RFV；均以真实physical CUDA_VISIBLE_DEVICES和direct allocation_id计账，不冒充Slurm。

4090/A100均已重新SSH核验：16:04时4090 D-V1290651/D-U1290652/S-V1290654/S-U1290655继续RUNNING；A100当前用户Slurm队列为空，不能由此断言集群卡一定空闲。结果cluster_queue_latest.json。既有D/S仍使用science4055294/evaluator6e2fc7f，未改配置或停止。D@10诊断全211/792：D-V64.2512、D-U64.7122，不调参、不替代epoch80。

RFV最初4090作业1290788/A100248292均在确认PENDING后取消，已转AutoDL完成同一工作，不得重复恢复。其他成员作业不归RFV调度器取消。

Atlas-S登记GPU测量已全部完成、GPU FREEZE。B保留459/792窗、各阶段HELD，不自动恢复。Atlas owner仍在CPU完成S/recovery AP与最终绘图，不能把GPU完成冒充所有论文图表完稿。五阶段action manifests已导出，仅为Atlas publication审阅，不能直接作为当前RFV监督。跨轴T×D/T×S/D×S仍是RFV/TDS后续缺口。

## 不可变代码与输出来源

RFV工作树 h65_clean_adatad/rfv_sprint_20260915，分支codex/rfv-sprint-20260915；从Core6e2fc7f分出。现有Raw27d557e、Core405科学版本均保留。

- b611a41：Standard T bank/local-CF/replay/smoke与Value/Graph primitives。
- 8341588：smoke连metas也去GT；4090/A100各13合同测试通过。
- 2ca4d4b5f2ad657410bfc5c2c4cebd3bc6a11a32：明确AutoDL硬件；实际G0a/G0b/theta40捕获。结果results/2ca4d4b。
- b679ae0ec46510bd85dc303dfacd4b6a6fd5288d：实际G1 fit/初始analysis，6项optimizer/normalization集成测试。结果results/b679ae0/G1_MINI。
- dca6564e77e13717adf461dcf1feabf8bc18bbda：仅finish_ap补sliding_window；CPU复用全部原预测补G0b，零GPU重算。
- 3a7e93f3f77350e9598011db1dd821cedaee8f0b：list/tuple物理身份按值比较；真实JSON roundtrip通过、实际frame变更仍拒绝。实际theta20/60捕获结果results/3a7e93f。
- 24fd722：同state完整函数forecast、累计真EMA；8项CPU集成测试通过。
- d62ea556c56ca6157289a866aaafc79c4fa427c6：混capture必须有精确等价审阅回执，新增8/8诊断；等价guard测试通过。实际ACTION_HOLDOUT与RISE_B0结果results/d62ea55。

2ca/3a在label forward上的等价回执为源码research/rfv_sprint_20260915/CAPTURE_EQUIVALENCE.json，独立讨论已复核。只授权这两个版本的CF标签比较，不包含AP/FVD/任意新版本。

部署脚本deploy_probe.py的package(ref)用git archive精确ref，并合成同ref revision文件；不混未提交源码、不漏references、不覆盖活跃版本目录。本地未跟踪revision指针仍可能为旧capture；以实际不可变部署和报告字段为准。

## 实现与交叉审核边界

已实现407D zero-init Temporal Value，valid-only映射及硬索引检查；原逆序lookup本来保留最早occurrence，未复现所谓最后padding覆盖错误。192-node/64-width/degree16 Static/Dynamic Graph和同信息Plain-L是实际实现。已有完整fit视频全部3窗正式JSON/metrics/noGT smoke通过；正式课程还需prefit接入、2更新/fresh strict reload和科学准入控制，不能谎称已完成。

独立审阅材料在 reports/wtr_rfv_review_20260915。完成B0及8/8后再次由两个只读子代理交叉核验，主代理点验实际报告/代码；未发现会反转FAIL结论的实现/统计错误。拒绝重复已完成8/8或把seed描述性区间冒充video CI。每个后续完整实验仍须独立讨论与审核。

## 后续准入与持续推进

主线不变：T Grounded Value ∥ Graph Relational Value ∥ RISE Future Value；不等D/S/TDS，但当前先解决T Value泛化。headroom、Plain可学性、完整recipe技术准入共同通过后直接T-U/T-V匹配80轮，10/20/40/60/80内联评测。Graph及新T RISE各自通过再解锁四格，不提前FVD。epoch80永久primary、outer20封存、test不用于选择。Raw仅query coverage，不扩6400。

当前任务的15分钟RFV heartbeat id=rfv继续按本文件处理有意义变化，静默处理无变化；Atlas独立automation=wtr由其owner管理。代码、图表和日志的公开旧快照保持wtr-snapshot-20260915/2d1072b，不能覆盖历史tag。新RFV发布状态见PUBLICATION.json（存在时以它为准）。
