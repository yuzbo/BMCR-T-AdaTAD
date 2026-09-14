# ANet数据准备恢复与课程轮转

2026-09-14 21:21:40复查。**没有新增完整测试结果**：仍95次唯一全测（71历史＋24当前），现有最佳/同60轮结论不变，没有重画相同性能曲线。

## 数据中断已经恢复

旧ANet CPU step1289574.0随Uniform-S宿主的预定时间切片于20:57左右结束；sacct为CANCELLED、exit0:15。调度器的宿主continuations明确记录`planned_checkpoint_time_slice`，准备锁已释放。这是资源生命周期中断，不是数据编码或训练损失崩溃。

21:20:38在仍运行的自有PBD-S allocation1289576内恢复为**1289576.0**。继续1CPU、1worker、nice15、CPU affinity[0]、CUDA不可见、**新增GPU0**。权威`paper_20260913/research/paper/data_preparation_job.json`和实际Support/PBD-S运行目录下的`cpu_colocation.json`已更新；21:21新step RUNNING，PBD-S继续训练到5410更新（epoch55内）。

恢复起点为**14360/14752**成功视频，剩余**392**有本地raw/ZIP来源。已完成的archive扫描记录包含train4819、val2383成员，所需成员均有成功journal；恢复工具还检查对应成品存在且大小匹配。因此直接复用已完成扫描并继续raw补齐，避免每次宿主切片后再次解压43个归档分片。日志已经确认两个`reused_archive_coverage=true`和`scheduled=392`。

修复仅涉及数据续跑工具，版本`0625cace00bb6a290beffe93b4434702313a81e0`。图模型科学版本仍841307e，任何训练配方、编码参数、blocked清单、数据分片和成品均未改变。归档中仍有未完成必需成员时会走原处理路径，不依据缓存直接宣布数据READY。全量READY仍要求10024训练、4728验证；当前尚未达到。

[中断与源码审计](recovery_audit.json)、[真实启动回执](recovery_receipt.json)、[恢复后快照](after_recovery_snapshot.json)。后续PBD-S若再次切片，先核对当前step真实状态，再从同一成功journal迁入其他自有运行allocation。已修订的`start_anet_cpu_step.py`支持终止step后的迁移；旧“仅首次迁移”的限制已被本次实现覆盖。活跃step不得重复启动。

**21:27实质进展复验**：新step已成功生成11条新视频准备记录，当前14371/14752；仍RUNNING，最新成品为15fps、短边256，尚无READY。实际Support/PBD-S目录的CPU共享回执存在且作业号一致。此时才将“已启动”升级为“已恢复实际处理”，详见[进展原始值](progress_verification.json)和[恢复验证](recovery_verification.json)。

## 当前课程和队列

唯一controller3506502健康；410阶段仍为11 COMPLETED、32 WAITING_INLINE、357 WAITING，当前**1 RUNNING、9 PENDING**。84训练课程为**1运行、3已提交等待、80持久等待**，没有正式课程终点完成。

|课程|当前状态|已保存成功更新|
|---|---|---:|
|Full-V1 S/B|原断点等待续跑|3360 / 2916|
|Full-V2 S/B|两条均预定时间切片后等待续跑|7877 / 7575|
|Uniform Full S/B|两条均预定时间切片后等待续跑|7062 / 7218|
|PBD-style S|1289576运行，epoch55内|5410|
|PBD-style B|1289883等待自身GPU预检/课程|0正式更新|
|G-Context S / G-Repair S|新提交1290163 / 1290164，PENDING|未运行|
|G-Full S/B预检|1290083 / 1290082，PENDING|未运行|
|Native AdaTAD K384 S/B|1289969 / 1289968直接评测，PENDING|未运行|
|Native AdaTAD K768 S/B|1290153 / 1290143参考评测，PENDING|未运行|

断点包含optimizer/EMA/RNG与样本游标，续跑仍属于原seed42的80轮课程。资源轮转未按mAP筛除实验；本轮没有取消其他项目，也没有重启竞争控制器。Graph及原版降采样依然没有GPU通过或mAP，提交更多作业不等同于获得实验结果。

完整配置/阶段索引在[项目统一入口](../../../../README.md)继续更新；[20:39发布快照](../../../project_status_20260914/publication_snapshot.json)和所有历史结果保留。
