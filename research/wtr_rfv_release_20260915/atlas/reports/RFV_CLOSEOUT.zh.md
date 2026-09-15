# Atlas-S有限收尾与RFV资源协作

## 15:40有效更新

S五项已登记GPU测量全部完成，正式FREEZE。15:40:19 GPU0实测无compute-app、1MiB/32760MiB、0%利用率，queue/atlas_s_gpu_release.json为回执，已通知RFV可供后续安排；GPU1仍由RFV使用。Atlas不再启动模型查询，B四阶段保持HELD。剩余仅CPU统计、最终图表/QA和action manifest。当前allocation CPU PID248899；action exporter PID278692，15:44还在运行，检查receipt与最终manifest后再采取后续动作，不能重复写者。

用户的新论文图建议已落实并QA：主图四页、条件附录一页、精确数值表，见output/pdf_s/paper_revision。新增benign95后续参照、Random/Actual、signed-value regret及配对差值；部分proxy NDCG优于随机，而regret差值CI均跨零。Graph/RISE独立判据及修订段落见PAPER_NARRATIVE.zh.md，不扩Atlas矩阵。

## 14:18有效更新

RFV已完成其核心probe准备并明确请求GPU1。本任务按唯一owner流程等待B下一已保存窗口后暂停其独立进程组，保留459/792窗（最后window458）。新owner232154接管原S_S PID205345；population_b及其后续三个B阶段均HELD。GPU1实测1MiB/32760MiB、0%利用率且无compute-app，已通知RFV可启动direct-process任务，不能自行恢复B占用。回执receipts/rfv_gpu1_handoff.json；当前policy副本receipts/rfv_resource_policy.json。下面13:29时的“不抢占B”仅为历史安排，已由这次明确请求更新。

S-only分析/renderer已在79f8632实现，最终视觉修订c65e6d3；统计数学不变，单模型输出analysis/models/s，仍复用原analysis/ap缓存。S population四张正式图已在output/pdf_s/population通过PDF QA。D的完整30配置AP/10k统计于14:11:13完成；S_S14:18为574/792，恢复尚待开始。Action exporter 251a00b已部署，完整导出等待五项S采集全部完成。后续入口和当前版本以STATE.md首节为准。

2026-09-15 13:29 +0800。用户在对接任务明确提出RFV优先；主任务已读取该授权，对方又在本轮明确确认以下范围。

## 已生效的执行安排

- Atlas保留已登记的S完整population、T/D/S有限分配及同支持恢复。S采集完成后FREEZE，只允许统计、bootstrap、绘图、分析修复及action manifest导出，不再扩展新的GPU查询矩阵。
- 当前allocation_s_S继续，随后recovery_s；有效S测量不中断。GPU0自然交接节点是recovery_s完成，不能把S_S完成说成整卡空闲。
- 尚未启动的allocation_b_D、allocation_b_S、recovery_b已设为HELD；当前population_b继续。对方目前仍在实现/审阅RFV，明确不要求抢占GPU1；实际就绪且确需资源时再请求本唯一owner协调。
- T×D/T×S/D×S没有测量。对方同意移入RFV/TDS证据缺口，不以同轴coalition冒充跨轴结果，不阻塞本轮RFV-T，不扩本轮Atlas。

队列控制代码56734b58f9a814f3b3a433c53bc063d20f16487d，配置为远端queue/resource_policy.json。配置在owner启动时读取；需要变更时控制重启owner并接管现存worker。测量CODE_REVISION仍a50b84db1bcdafd86ceec4d9737156807f9daa9a，分析PLOT_REVISION仍c16bafc7d2066aae167d1f58a66d8a054ac6a9a5。

13:27:42仅向旧owner PID32242发送SIGTERM，随后新owner PID221333接管原allocation_s_S PID205345及population_b PID133234。未向worker发送信号，未删数据。原30个阶段定义和依赖保留，三个B阶段被显式暂停；S恢复仍WAITING。实际回执为receipts/rfv_closeout_applied.json，含前后状态与验证。

13:29快照：20 COMPLETED、2 RUNNING、3 HELD、5 WAITING；S_S330/792、B population422/792；D完整采集后CPU统计17/30配置，单独汇总未完成。原CPU辅助PID207315继续，不重复启动。

近期S_S约12秒/窗，阶段完成估计约15:05；这是按当前吞吐外推，随后恢复阶段尚无可靠正式耗时，整卡释放时间待恢复运行后更新。B population近期约73秒/窗，剩余约7.5小时；负载与视频长度会改变估计。

## 统计及图表收尾

S完整切片应先交付，不等待B暂停的补充。原analyze/renderer存在硬编码S/B的部分，尚需修改只读分析和绘图以支持完整S选择及复用原AP缓存，不执行新的模型查询，不写伪造的全S/B完成标记。新queue/atlas_s_measurements_ready.json只表示五项已登记S采集齐备，不能替代统计或视觉QA。

完整S图继续满足211视频/792窗口、官方AP、10000次视频聚类bootstrap、真实执行与查询成本；population T另报告187个eligible视频。Fig2补null/benign与显式正负部，Fig3不预设边界/短动作结论，Fig4显示配对区间，Fig5保持同支持恢复。T两张已交付图无故不重绘。最终单独说明暂停B项及跨轴缺口，完成S图QA、manifest和交付后暂停既有wtr自动跟进。

## 可复用资产的真实身份

远端根为/root/autodl-tmp/wtr_characterization_20260915。

- assets/adatad_s_ema.pth：官方EMA，epoch59，200938640字节。
- assets/adatad_b_ema.pth：官方EMA，epoch51，475085520字节。
- assets/v2_s_epoch040_light.pth：126878675字节；assets/v2_b_epoch040_light.pth：158300627字节。两者为epoch_index40、4000次成功更新的完整V2重构bundle，source_revision=db9c749c7bbbdcc1b5f1f14e10b1e1d0f087cacc。
- bundle含完整EMA、初始backbone相对官方差异、scout_initial和元数据；按h65/atlas/recovery.py的build_probe重构，不当作普通官方state_dict加载。
- assets/v2_s_initial_source.pth和v2_b_initial_source.pth仅初始化来源。AutoDL本次盘点没有找到实际theta20/theta60资产，RISE相邻阶段checkpoint需原训练侧取得。

S原V2来自/data/run01/sczc063/yuzibo/h65_clean_adatad_20260910/support_review_20260914/research/paper/runs/review5485_full_v2_s_seed42/epoch_040.pth；B为同runs目录review5485_full_v2_b_seed42/epoch_040.pth。

## 旧训练侧T记录的使用边界

results/calibration_s_T及calibration_b_T均完成32个development视频、每视频一个中间窗口，使用上述官方S epoch59/B epoch51。动作是mandatory groups[0,4,8,12]之上的单组cheap-base升级。字段为meta、axis、marginal_actions、mandatory_groups、base_gflops、dense_gflops。

这些实际CF记录缺少RFV当前support/action/candidate/continuation/state descriptor合同，不能直接认作T-local-CF/G1-T/RISE训练bank。results/preflight_s_full_video与preflight_b_full_video各1视频5窗，population部分有固定K actual RGB单帧交换，只作技术参考。尚未拿到当前Core resources.local.json核对32个视频的160/20/20分区；publication/test Atlas数据不得补入训练bank。

资产和bank盘点由两个只读探子独立完成，未执行GPU查询、训练或修改。完整结果和科学负结果已发送到任务“实现 Raw-v1 并部署并行实验”（01a0a12c-241a-7772-986d-387138bce4d6）。
