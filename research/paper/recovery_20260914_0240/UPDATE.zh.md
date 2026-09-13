**论文入口故障与ANet归档恢复**

2026-09-14 02:38收到协作任务的ANet失败报告，随后读取自有Slurm日志与权威回执完成定位。

论文1288635–1288642八个任务在第一次训练更新前都因`dict() got multiple values for keyword argument 'source_revision'`失败。`initialize_gpu()`已返回该键，训练与评测入口又通过展开关键字重复传入。修订a8930de将两处构造改为映射合并并显式覆盖版本，源码中的真实metadata表达式已分别验证。没有更改模型、seed、课程或初始化。完整旧日志和失败尝试已保存。

控制器456261已替换为1999751，仅将这八个已诊断任务退回原seed42课程，实际重提为1288730–1288737。InternVideo1-MQ已完整传输并通过1024通道/24层tensor检查，对应课程1288719排队。当前科学入口修复通过CPU表达式验证，但GPU两步更新仍须由重新排队的技术检查完成，不能声明训练成功。

ANet旧准备1288466在val归档读取中出现`zlib invalid distance code`、`tarfile.ReadError: invalid compressed data`，退出1:0。成功journal和成品保留。重建共享报告后，准备覆盖为training9062/10024、validation3639/4728，共12701/14752，仍INCOMPLETE；此前5151/2340仅为旧报告计数。

诊断1288738对14个val分片与发布者既有LFS内容值比较：13个匹配，唯独`v1-2_val.tar.gz.09`不匹配，尽管文件大小正确。此次内容校验用于确定需要替换的一个文件，避免重新下载整个约57GB的val归档；没有新增无用途的指纹制度。实际逐片结果在validation_shards.json。

已提交1288754 `paper-anet-repair`，同一allocation内从[原发布者](https://huggingface.co/datasets/YimuWang/ActivityNet)取得.09替换件，匹配原发布者内容值后，将损坏原件保存在downloads/corrupt_1288466，恢复原名分片，再从成功journal接续既有准备流程。所有43个原始分片及成功成品保留；替换验证前不覆盖原文件，不提前生成READY。02:56核查该修复作业为PENDING，替换和剩余准备尚未执行。

数据作业权威入口改为research/paper/data_preparation_job.json。旧失败、诊断和新修复编号都存入当前目录。自动跟进已更新这些状态，协作任务已收到新编号并保持只读；不重复启动旧作业或旧seed训练。
