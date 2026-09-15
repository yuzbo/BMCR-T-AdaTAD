**三个完整候选的同轮次结果与ANet CPU接续**

12:36:54快照已收集Full-V1、Full-V2、Uniform的S/B全部epoch10 EMA完整测试。均为seed42、THUMOS200训练/211测试/792窗口，属于80轮课程的中间checkpoint。PBD、Static和当前官方Dense复测尚未产生本次结果。

| 骨干 | 方法 | epoch10 mAP % | 全测试平均GFLOPs/窗 | 代表完整窗GFLOPs |
| --- | --- | ---: | ---: | ---: |
| S | Full-V1 | 63.2251 | 1169.595 | 1170.153 |
| S | Full-V2 | **64.2351** | 1229.323 | 1229.324 |
| S | Uniform | 63.7713 | **1146.344** | 1148.916 |
| B | Full-V1 | 68.1685 | **3988.331** | 3988.198 |
| B | Full-V2 | **68.3708** | 4014.029 | 4009.431 |
| B | Uniform | 68.3322 | 4094.316 | 4094.316 |

这里计算量包括完整模型，按既定矩阵/卷积2MAC口径；平均窗口成本与代表完整窗口成本分别展示。各策略选择的预算不同，表格是同epoch比较，不是严格等FLOPs的单因素消融。

![同epoch性能与成本](figures/epoch_010_comparison.png)

Full-V2相对V1在S/B上提高1.0100/0.2023pp，平均窗口成本增加5.1067%/0.6443%。这是支持继续跑完整候选的早期信号，不能单独归因为某一个结构修正。

Uniform-S同时比V1-S更准、更便宜；Full-V2-S相对Uniform-S多0.4638pp，但成本高7.2386%。Full-V2-B相对Uniform-B多0.0386pp、成本低1.9609%；这是数值上的改善，幅度很小，尚未进行成对显著性检验。当前不足以证明复杂学习选帧优于简单采样。

**路由使用情况限制了可以声称的机制结论**

Full-V2-S的792窗全部选择K384_D100_S75。Full-V2-B为K384_D100_S75共736窗、K384_D100_S100共52窗、K384_D100_S48共4窗。两者D=100%占比都是100%，在这轮推理中没有通过动态深度token跳过省计算。

因此，不能把V2的mAP改善直接称为“稀疏深度恢复已成功”，也不能据此认定三轴动态必要。当前收益包含更温和的推理预算选择和整个训练/结构组合的影响。既有固定计划、T/TD/TS/TDS独立课程、深度机制矩阵及状态误差分析继续执行，不能由该表替代。

相对历史同骨干dense参考，V2-S仍低约4.7775pp，V2-B低约2.7572pp。历史dense-S的69.0126%/2347.894G也仍同时优于当前V2-B的绝对精度/代表窗成本；当前官方Dense复测尚在队列，最终同口径结论须等真实回执。论文级全局Pareto、PBD/Static竞争以及80轮终点尚未得到确认。

**ANet已从GPU等待改为现有allocation内CPU执行**

Slurm23.11.10和真实只读step验证支持在本路线已分配作业内启动CPU-only step。Uniform-S 1289574有6 CPU/约93GB内存，核验时平均约4 CPU使用、内存峰值约55.6GiB。已于计算节点记录的12:46:06启动step **1289574.0**：1 CPU、单worker、nice15、CPU亲和性[40]、CUDA_VISIBLE_DEVICES为空，新增GPU为0。

新step取得原有preparation lock后，取消了本路线仍在PENDING的原准备请求1289684；未取消训练或其他项目作业。权威data_preparation_job.json现为phase=cpu_step_in_owned_training_allocation，job_id字符串1289574.0，allocation_job_id=1289574。原请求和新启动回执均保留。

12:53:16核验该step仍RUNNING，journal已出现真实MKV转码成功记录：v1-2/train/v_z8lxaUC1Shk.mkv输出15fps、342×256、1125帧、75秒的MP4。此时Uniform-S仍正常训练至epoch14、1363更新，损失有限。全量READY尚未产生。

数据工具保持原输出编码参数，仅将worker并发降为1；CPU共享会影响训练/解码墙钟时间，已写入Uniform-S运行目录的cpu_colocation.json。epoch10表格在该共享启动前已经完成。之后的耗时分析必须标明共享时段，不将共享带来的时间差解释为模型效率或科学收益。

CPU step依赖宿主allocation生命周期；宿主完成或计划切片可能终止它。后续应查询精确step的sacct状态和成功视频journal，必要时诊断后在新的自有allocation接续；不重跑已准备视频、不盲恢复旧GPU请求。仍以training10024/validation4728和READY标记作为完整ANet数据条件。

科学代码与80轮课程不变。12:36快照六条训练仍正常，V1 S/B为1869/1528更新，V2 S/B为1514/1309更新，Uniform S/B为1121/1002更新。PBD及Dense复测继续等待账户GPU配额；其他项目10张分配未改动。

本目录保存原始progress_snapshot.json、cpu_step_verification.json和同epoch图表数据；绘图入口tools/paper_epoch_comparison.py。CPU接续代码提交为新树19e02fa、旧owner d341db5，运行回执以data_preparation_job.json及paper/data_cpu_steps/为准。所有正式课程继续80轮，不进行epoch10成绩淘汰。
