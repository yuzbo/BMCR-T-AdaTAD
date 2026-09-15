# RFV 实施任务停止交接

用户指令：先整理当前实现与工作进度，报告并停止任务。

执行状态：STOPPED_BY_USER。RFV heartbeat（id=rfv）已设为PAUSED，不再自动跟进或启动实验。既有4090四条D/S课程保留运行，未把停止本任务解释为取消已有训练。

## 当前实现与结果

- Value-R1：b644d87，18个head已完成并复核；LEARNABILITY_FAIL。原407、统一bank/归一化/预算，state-wise JS+0.1 Huber未恢复未见候选泛化。
- Reverse：800bcd1完成6pair/24次真实重放，replay/缓存标签/正反收益误差全0；4aa1ca2完成P1/P2×3seeds。P1-bi复用P1 raw，无额外训练。P2 held regret0.001489609、STOP0.001413537；P2−P1-bi CI跨0，STRUCTURE_SIGNAL_FAIL。P2 fit rho0.9495、held rho−0.0333；75次选择中39次负收益。
- 独立审核确认Reverse聚合/15组bootstrap与原报告一致；P2−P0 NDCG也下降，CI完全负。原packing位置族外推和反向约70σ尺度限制保留；不是一般不可学定理。
- RISE：800bcd1导出既有B0全部129条seed-state候选分数；inner10×3seed的29次排序变化、0次最终选择变化，准确解释Future=Post regret。旧六指标和独立逐候选复算误差全0；FVD未通过原家族gate。
- D/S固定诊断：7245972，D-V/S-V raw epoch40；各4个cal窗口路由比较、2组accumulate2梯度分解，optimizer_steps=0，运行已完成。A100248477确认PENDING后取消，任务转已释放AutoDL GPU1 PID317797，现已退出。
- D的Value辅助导致同检测器梯度裁剪系数比为0.99856737/0.99999698，S为0.99998032/0.99998868；这些样本没有显示大的额外缩放。Value梯度在router外的实测范数见DS_DIAGNOSTIC_SUMMARY.json。不能仅凭旧四线约96%的普遍裁剪率，将mAP差归因于Value梯度。
- 固定raw40下，D/S各4个cal窗口中Value路由分别3个窗口loss低于Uniform；小样本loss不等于full mAP。该诊断运行已完成，最终结果交叉审核尚未做，按停止指令不再追加审核任务。
- Atlas原S收尾和另行授权interaction已由owner完成；interaction mAP差−0.023734pp、CI跨0，INCONCLUSIVE。B暂停，不新增查询；不把publication bank回流训练。

## 4090保留的课程

最后只读快照：2026-09-15T20:34:22+0800。四项均RUNNING。

|课程|当前epoch字段|successful updates|epoch40完整mAP|
|---|---:|---:|---:|
|wtr_d_v_s42|72|7148|64.260056%|
|wtr_d_u_s42|73|7297|64.334131%|
|wtr_s_v_s42|57|5690|64.061569%|
|wtr_s_u_s42|60|5959|64.245897%|

D40 V−U=−0.074076pp；S40 V−U=−0.184327pp。它们是完整训练的单轴组件消融；epoch80仍是primary。停止本任务后不主动监测、评估或续派作业，服务器原课程自身继续。

## 模型层级与下一步边界

诊断线：Atlas、Raw mini、T-local、407D R1/Reverse、离线G1/历史RISE、固定D/S诊断。完整组件消融：D/S现有80轮及其Uniform对照。正式论文候选：BMCR全轴T、Global-Task/TaskValue、独立S/D及全轴Graph/RISE，当前DESIGN/IMPLEMENTATION_PENDING。已证明最终模型：无。

旧局部pair/RISE gate只约束对应家族，不能否决具有不同选择空间和任务梯度的新全局模型。正式方案应在Standard768全轴生成完整K支持；S/D全合法token重轻分配与跨时间/层预算目标另行实现。现有S不是Raw ROI获取，D是可再进入的重轻路由而非单调early-exit。新Global需自身技术、任务、成本证据，本次没有启动任何新正式课程。

实施owner已与正式设计任务往返讨论，意见在FORMAL_MODEL_SCOPE.zh.md。S/D直接任务学习估计器尚未定稿；score-function需要区分有序抽样轨迹log-prob与无序子集概率，不能将集合概率简单当逐项概率乘积。

## 归档与未完成事项

工作树：C:/Users/skywalker/Documents/ChatGPT/H65/h65_clean_adatad/rfv_sprint_20260915，分支codex/wtr-rfv-20260915。已提交最新执行工具到72459727f61454194f0d84865f31a16ed4e10448；本次停止报告、图表脚本/证据同步仍在本地工作树。

本轮新代码/图表/日志尚未推送GitHub；上次公开RFV快照仍为6d1f924。按停止指令不再执行发布。已完成结果、原日志、台账、源码和审阅全部保存在本地，不改写历史source。

恢复时需用户明确指令。待做：D/S固定结果最终交叉审核、既有课程80 endpoint回收、正式Global配方与task梯度实现、独立S/D/Graph/RISE自身对照、新快照GitHub发布。不得自动重跑已完成R1、Reverse、RISE或Atlas。
