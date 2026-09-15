# RFV 首批代码交叉审核

候选 b611a41，2026-09-15。范围是 T valid mapping/zero init、共享 Standard bank、bounded local-CF、fixed-action replay、完整视频 smoke 路径，以及独立 Value/Graph/EMA/metrics primitives；不包含尚未完成的 G1 训练流程或 FVD 正式课程。4090、A100 既有环境分别运行13项CPU合同回归，全部通过。

独立只读审阅由 rfv_capture_review 与 rfv_value_review 执行。真实checkpoint/科学边界另由讨论任务给出完整意见，见工作区 reports/wtr_rfv_review_20260915/RFV_REVIEW.zh.md。代码正确的部分明确接受，不通过增加无依据补丁来“解决”错误发现。

两条审阅疑点经主代理点验原文后均撤销：

1. smoke metadata 缺 slurm_job_id：实际 metadata=dict(hardware,...)；h65/full/runtime.py:56 固定返回 slurm_job_id。该字段已经存在，不需要伪造固定ID。
2. transition 缺 sigmoid：h65/raw/value.py:38 本来就保留 transition_logits 的原值，只有 actionness 做 sigmoid。bank与407D共同描述子的语义一致，节点可以使用明确记录的cheap logits，不改为另一套特征定义。

主代理额外确认：完整视频smoke中的第一条no-GT推理需同时移除CharacterizationData放进metas的characterization_gt；已补该清理。正式evaluate本身使用只收集公共META_KEYS的dataset路径。

结论：上述代码/协议范围可以进入GPU技术运行；真实GPU smoke、actual bank、G1/RISE科学收益仍为WAITING，不能据CPU或审阅PASS进入Final=yes。正式科学版本和GPU回执将另记。
