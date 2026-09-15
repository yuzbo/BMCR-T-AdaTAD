# 本轮跨任务讨论回执

2026-09-15。用户明确指定讨论对象：任务「实现 Raw-v1 并部署并行实验」，ID 01a0a12c-241a-7772-986d-387138bce4d6。

## 已完成往返

1. 本任务完整读取8bef34df用户附件、当前计划与关键T/S/D代码，向指定任务发送附件路径、用户最新原话及五项具体设计问题。
2. 本任务补充WTR专属执行分支的证据：D每pack配额；S按8个native时间分配quota；固定mod_layers=[4,6,8,10]；value_score无task梯度；BMCR选择与transport入口仍存在。
3. 向对方发送FORMAL_MODEL_DESIGN.zh.md草案，请其给出同意、不同意与实现缺口。
4. 对方返回OWNER_DISCUSSION.zh.md及直接消息，明确已读原用户附件与相关源代码，同意纠正旧门槛适用范围、恢复全局选帧与四层分类。
5. 本任务完整读owner回复，采纳准确部署与Raw边界，并保留S/D梯度估计器的不同倾向；未把讨论意见当作新模型验证结果。

## 已达成的原则

- 最多4交换是诊断原型限制，不是WTR定义。新Global模型不等待旧8/8 pair头成功。
- 当前D/S部署已经对合法token集合整体评分；局部CF是监督采样，不等于部署只换一个token。
- D/S80是完整TAD训练的组件消融；历史BMCR是历史完整方法；新Global/TSD/Raw是正式候选待实现；当前已证明最终模型为空。
- T首版Standard768候选、K384预算，继承rate→S0→condition→全轴S1，保留当前physical Cross/readout，关闭额外局部refiner。
- S/D首技术版可复用当前executor边界，最终研究需明确开放跨时间/层分配；不把固定配额永久化。
- Core的S token mask不等于raw ROI裁剪；D重/轻重入不等于单调early-exit；任务分数不等于已校准真实Value。
- 最小正式对照为Uniform384、BMCR机制全局task分配、同policy增加actual-Value，然后S/D单轴及联合；历史局部结果不能直接代替新模型消融。

## 未被实验解决的取舍

S/D task梯度：owner更倾向hard预算随机采样+score-function，以避免全heavy训练；本任务建议先验证hard-forward代理，保持当前确定硬执行的对应关系。owner表示该路线同样可作为可执行首版，条件是明确有偏估计、未选分支和完整训练成本。双方都未把任何估计器称为已经实现或更优。

Raw区域获取还缺空间廉价图、区域到物理patch/时间的映射、读取/解码成本和相应训练通路。BMCR transport需要的邻居RGB在Raw中不能视为免费。

## 计划同步责任与范围

本任务维护FORMAL_MODEL_DESIGN.zh.md与本回执。实施owner维护当前执行计划、实验队列及ledger。初次回复时同步未完成；随后已收到完成回执，并点验实际计划第5行、FORMAL_MODEL_SCOPE与ledger：旧局部gate只约束旧recipe，Formal Global T/S/D列DESIGN_PENDING/OWN_PROTOCOL_PENDING/NOT_STARTED，D/S80列完整组件课程，原诊断FAIL保留。

本次交流没有提交Global/S/D新GPU课程，也没有更改或取消运行中的训练；新的设计授权、实际实现、技术验证、完整训练和科学收益是不同状态。

owner特别说明：其初次回复依据用户原附件和代码，尚未逐行审阅本任务完整草案。因此“原则共识”不等于“全文逐项批准”。

## 交接中的最新状态

讨论后，实施owner转达其任务中的新用户指令：“先整理当前实现与工作进度，报告并停止任务”。owner已完成交接、分类与门槛作用域整理，停止主动推进并暂停RFV heartbeat；既有D/S训练保留。本任务已回执不再追加实现或实验工作。

已完整读取 C:/Users/skywalker/Documents/ChatGPT/H65/reports/rfv_sprint_20260915/STOP_HANDOFF.zh.md。交接记录最后只读快照为20:34:22，四条4090课程仍RUNNING；新Global课程未启动。新代码/图表/日志仅本地归档，没有进行本轮GitHub发布。

同步文件：reports/rfv_sprint_20260915/{EXPERIMENT_PLAN.zh.md,EXPERIMENT_QUEUE.json,EXECUTION_STATE.zh.md,FORMAL_MODEL_SCOPE.zh.md}，reports/wtr_fasttrack_20260915/FINAL_MODEL_LEDGER.csv及RFV副本。此次点验以实际计划作用域段、scope、ledger和停止交接为据，没有重复审核所有机器队列字段。新Global/S/D/Raw方案仍处于待实现状态；停止交接不改变已形成的设计共识。
