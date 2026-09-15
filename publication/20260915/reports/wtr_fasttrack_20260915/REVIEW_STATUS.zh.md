# 实验交叉审阅登记

本文件记录用户要求的实现后交叉审核。计划登记、CPU通过、GPU能运行均不能替代科学语义审核。

|实验|实现版本|已知执行证据|独立审阅状态|
|---|---|---|---|
|Raw-v1 reader/bridge|1259d25|4090单完整训练视频3窗口，RGB/恢复/预测及官方AP缓存一致|由讨论任务01a0a0b2-3d36-76b1-839b-bdb98ae48c0e审阅中|
|Raw mini/full bank与Value|1259d25|已实现CLI；尚无已完成bank/ranking回执|审阅中，科学gate WAITING|
|D-V/D-U|578bab2|CPU精确配额/梯度通过；两次GPU更新、fresh reload、无GT推理通过；已启动epoch1候选课程|独立审阅中；结果不解锁后续课程|
|S-V/S-U|578bab2|共用D/S接口CPU通过；A100已提交|独立审阅中，GPU证据待回收|
|DS/TDS/T-V|578bab2|配置与接口已登记/实现|未作逐配置GPU验证，组合科学gate WAITING|
|Graph G0/G1、RISE-A/B、DB|计划v3|科学runner尚未完成|不得登记实现完成或PASS|

当前待讨论事项：Core在线Value监督现覆盖200训练视频，而Raw有160/20/20划分。需要锁定Core开发gate的out-of-fit范围；若新增独立router holdout，必须从初始化重启对应Value候选版本，不把已参与监督的视频重称未见holdout。在线即时监督记录与可持久重放的checkpoint fixed-action bank也必须分开。

## 578bab2审阅讨论与修正

独立初审报告：../wtr_code_review_20260915_578bab2/REVIEW.zh.md。讨论确认D/S执行、梯度与成本合同本身可用于固定容量原型，但尚未实现最终共享Scout粗上下文与联合预算模型，不将其称为最终完整架构。

已在修订代码中处理：Value只在160 fit视频取监督（detector继续200）；20 calibration/20 holdout只做稳定checkpoint re-query；TDS的D/S标签先执行当前T选择再保持共同support；Raw训练要求所有shard完成回执与完整group清单；计划评测点直接取config；Temporal descriptor传实际D/S容量；Raw/Standard收益排序统一为实际cls+loc，RMS只作回归conditioning。578bab2候选已保存并归档，未用来解锁科学gate。

每配置准入回执字段：scope=code_and_protocol_review、passed、config_id、science_sha、reviewer_thread_id、evidence。该PASS只覆盖代码与协议；GPU两更新/fresh reload由同一课程内的preflight独立执行；科学有效性仍由开发/正式评测决定。

审阅发现的问题在本文件补入出处、处理决定、修正版本和验证结果；如果代码正确，应明确写明，不制造额外检查。

## 当前准入结果

Core40552945ad5d56b7404f833cd86f998e8558e8b1的D-U/D-V/S-U/S-V均已获得独立code_and_protocol_review PASS；四条回执已由实现者复制到运行准入目录。Raw27d557ee3bd38a52aec8caaf16860a95957b44b6的bank/Value修正单独PASS。D pair已在新版本重新完成两次更新、fresh reload并训练；S pair仍需对应GPU回执。未把这些结论扩大为科学收益或T/DS/TDS/Graph/RISE全部通过。
