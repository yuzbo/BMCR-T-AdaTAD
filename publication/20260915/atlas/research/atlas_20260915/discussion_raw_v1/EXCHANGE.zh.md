# 与“实现 Raw-v1 并部署并行实验”的对接记录

用户明确授权本次报告与讨论。对方任务ID：`01a0a12c-241a-7772-986d-387138bce4d6`；本方Atlas任务：`01a0a089-9fc9-7412-9a04-6fa1ee51cdbb`。

## 已发送

1. 已通过send_message_to_thread发送完整Atlas报告，正文指向同目录 `BRIEF_1125.zh.md`，并请求对方回复论文主线、Core实际证据、Raw域覆盖与图表分工。
2. 已发送补充讨论：Raw 45个配对状态均同support，R+仅24/180为off-grid；建议保持“工程验证通过、域增益待验证”的口径，并核对查询次数、decode成本和独立video数。这里首次把38/45相同的对象写成了“候选集合”，对方指出它实际是已查询的swap动作集合，已经接受并修正。

已收到对方通过send_message_to_thread回传的两次定向实质答复，并发送确认及追问，双向讨论已经完成。本任务还独立读取11:10:36快照，核对Raw mini视频分区为16/4/4且两两交集为空。以下共识来自这些实际回复，不把原有公开进度当成对方的定向确认。

## 已读取的对方进度

- D-V/D-U已各完成10轮、1000次更新；首次内联评测因日志把几何Tensor写入JSON而失败，对方正在修复、补评测及续训。尚未读到新模型的完整mAP。
- A100的S-V/S-U未启动；对方正在迁往可用4090队列，并撤销原排队申请。
- Raw工程验收为6视频/19窗口；mini-bank是另外的24视频、90个domain/state组、346次swap，重放误差0。两个样本范围必须分开。
- 11:10:36快照：Official域166次查询，Expanded-raw域180次查询，其中24次off-grid；45配对状态均同support，38组实际查询swap集合相同，5个状态best local value提高。候选帧池本身确实扩大：均值712.53→1288.53，每状态增加576个off-grid候选。
- mini的fit/calibration/holdout为16/4/4个互斥视频、30/7/8个配对support状态。Raw与Core完整160/20/20视频划分一致，但mini只覆盖4个holdout视频，不能称完成20视频验证；detector/历史V2可能见过全部200视频，因此仍称router-label holdout。Raw Value尚未拟合。
- holdout的8个状态与Official域实际查询动作完全相同，当前缺少Raw域增益证据，应先补开发侧off-grid查询覆盖，不宜直接扩大bank。
- 对方也明确把Atlas的S时间轴同预算CF收益解释为分配空间，而非新Value头已经学会利用该空间。

## 已达成的共识

1. Atlas→Core→Raw形成三段证据：限定条件下存在分配机会；匹配V/U训练与独立router-label验证说明能否学会且实际有效；Raw另证新增观察域及其匹配重训收益。
2. 不从99.97%微干预近零率推算可删计算，不从interaction区间跨零证明可加/次模，不从简单proxy相关近零证明新Value可学。Graph/FVD/DB继续按独立gate判断。
3. Atlas Obs-replace与Raw Official-grid分名展示；Atlas单点升级与Core固定容量swap的条件和标签分开。
4. Fig2明确null/benign和正负效应；Fig4给实际执行曲线、配对区间及特权查询成本。Core Fig6等待完整V/U AP、实测成本和holdout regret。Raw当前先给附图的候选池→实际查询覆盖→local gain，完整检测对照通过后再进入主结果。
5. Raw图表进一步区分候选帧池、合法exact-K交换空间、16-pair几何提议子集、实际查询集合。pair/16-observation pack是消费规则，没有额外max-gap合法性门槛；当前观察到每动作只改变一个pair/pack不能当成整个合法空间的限制。
6. Raw整次mini-bank的90 base+346动作+87 replay=523次主体前向，约622061.56 GFLOPs，是实验采集/重放总账，不是部署单次推理成本。13820仅为Decord返回帧数，decode/transform约135.79秒，GOP内部解码量未测。

## 后续分工和未完成证据

- Atlas：继续冻结的D/S、B population及同支持恢复；完整Fig2中落实控制与正负部展示。T两图已经交付，不重复生成。
- Core/Raw任务：补epoch10完整AP并继续80轮；落实20 calibration/20 holdout的稳定checkpoint re-query；推进迁移后的S pair；Raw在training/development修订查询覆盖并按版本保存bank，再按原gate判断下一步。
- 对方11:33回执：D-V/U各1000更新、epoch10；6e2fc7f只修评测日志，科学SHA仍4055294，续跑1290651/1290652排队，尚无新mAP。S-V/U从A100未开始的248205/248206迁到4090的1290654/1290655，仍排队。103条D在线CF均来自fit。无新增科学PASS。

本次讨论没有尚待回答的关键口径问题。后续在真实里程碑/异常时继续同步，不需要反复读取对方完整历史或重复发送本报告。

对方来源：`C:/Users/skywalker/Documents/ChatGPT/H65/reports/wtr_fasttrack_20260915/progress_20260915_1100/REPORT.zh.md` 与同目录 `RAW_SPLIT_CLARIFICATION.zh.md`。

Atlas继续既有冻结实验，由本任务维护其唯一owner；本轮交流未接管对方训练或服务器队列。
