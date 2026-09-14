**Epoch20完整比较：B取得更好的精度/成本组合，S的简单对照仍强**

14:11:49快照已收集六条课程全部epoch20 EMA全量测试。均为THUMOS200训练/211测试/792窗口、seed42、80轮LR课程；当前仅完成中间候选，不据此提前结束任何科学课程。

| 骨干 | 方法 | epoch20 mAP % | 全测试平均GFLOPs/窗 | 代表完整窗GFLOPs |
| --- | --- | ---: | ---: | ---: |
| S | Full-V1 | 63.4087 | **1147.253** | 1170.153 |
| S | Full-V2 | 64.4855 | 1229.323 | 1229.324 |
| S | Uniform | **64.5400** | 1226.478 | 1226.478 |
| B | Full-V1 | 68.1542 | 4094.365 | 4094.366 |
| B | Full-V2 | **68.5014** | **3940.848** | 4009.431 |
| B | Uniform | 68.0731 | 4094.316 | 4094.316 |

![同epoch20完整测试](figures/epoch_020_comparison.png)

同epoch20下，V2-B相对V1-B增加0.3471pp、平均成本下降3.7495%；相对Uniform-B增加0.4282pp、平均成本下降3.7483%。S则相反：Uniform-S比V2-S高0.0546pp、成本低0.232%，没有证据可以把这轮概括成“V2在两种骨干都赢”。以上为单seed描述，未作成对显著性结论。

最佳已测候选也必须保留：B的V1最佳仍为epoch10的68.1685%，Uniform最佳仍为epoch10的68.3322%，V2最佳为epoch20的68.5014%。所以相对各自10/20最佳，V2-B增益应报告0.3329/0.1692pp，不能只引用对较弱epoch20控制的差值。S三条当前最佳均在epoch20。

这些最佳B候选的平均成本分别为V1 3988.331G、Uniform 4094.316G、V2 3940.848G。因此V2-B在当前已测候选内同时优于这两个控制的精度与平均计算量；这仍是早期单seed结果，并非完整论文级全局Pareto结论。

本轮还收齐当前官方AdaTAD-B复测：71.1204080% mAP、8082.154709 GFLOPs/窗、211视频/792窗口。V2-B的平均成本较其减少约51.24%，mAP相差2.6190pp。当前官方S复测仍待运行，PBD/Static尚无完整成绩。

**B已实际使用稀疏深度，但三轴必要性仍未得到完整证明**

V2-B在792窗中有519窗选择K384_D75_S100（65.5303%），273窗选择K384_D100_S75。它已经从epoch10全部D100转为在多数窗口实际降低动态深度配额；精度提高同时平均成本下降，是值得继续完整课程的早期证据。

V2-S仍全部选择K384_D100_S75。两个骨干这轮都没有同时D<100和S<100的窗口，因此不能将结果直接解释为三轴同时稀疏已经必要，也不能单独归因于light residual、full-KV或同支持监督。独立T/TD/TS/TDS、固定计划、PBD/Static和机制矩阵仍需完成。

**已测量到S空间路由的计算开销抵消问题**

读取V2与Uniform各自同一代表完整窗口的profile，均为K384、D100，V2采用S75、Uniform采用S100。下面是矩阵/卷积2MAC口径的实际算子差值，不是全测试平均成本分解。

| 成本项变化（GFLOPs） | S | B |
| --- | ---: | ---: |
| 重FFN减少 | -45.2985 | -181.1939 |
| 额外attention路由QK | +47.1859 | +94.3718 |
| 轻FFN增加 | +0.9437 | +1.8874 |
| frame及其他增加 | +0.0146 | +0.0500 |
| 净变化 | **+2.8458** | **-84.8846** |

![实测路由开销分解](figures/routing_cost_decomposition.png)

S的额外QK成本已经超过重FFN节省。源码h65/paper/engine.py的attention前向先调用SDPA，随后为路由调用incoming_attention，确有额外QK计算；profile中的attention_routing_qk与这一执行路径对应。该结果解释了为什么名义S75不保证完整模型更省算。B的FFN节省是S的4倍，而QK增加为2倍，净省算因此不同。

明确的后续算子优化候选是复用同一次attention计算产生的路由统计，避免重复QK，并独立验证数值、梯度、显存及完整推理成本。它尚未替换当前固定课程的推理实现，不能在本轮图中扣除这笔真实开销。Carrier迁移建议仍是候选，其额外总成本也不能拿局部估算代替。

**资源轮转与技术恢复**

由于本路线只能实际使用6张GPU，而前六条课程均已取得epoch20完整结果，14:35:42按资源覆盖进行一次轮转：最早启动的Full-V1 S/B收到USR1，保存optimizer/EMA/RNG/cursor并按既有协议退出75。调度器识别为计划续跑，原seed42、80轮目标及所有已完成结果保留。此操作选择最早的两个allocation，不依据mAP淘汰模型。

PBD-S随后通过自身预检并进入正式训练。PBD-B在预检的同支持cosine统计中触发OOM：当时已分配22.74GiB，申请114MiB失败。原错误和尝试保留。已改为逐pack FP32统计及NMSE梯度重算，attention仍记录诊断但不保留其未用于目标函数的梯度图；同支持目标、pre/post-TIA损失定义与权重、batch和分辨率不变。

FP32/BF16的前向值、梯度、cosine/RMS、有效位置掩码及目标停止梯度检查已通过，见support_memory_cpu_tests.log；内存修正a149aa7，运行版5b88b3c。14:59:35更换单一控制器为1867620，优先重提PBD-B，并仅重排自有PENDING续跑/补充任务；其他RUNNING作业和ANet宿主不受此次重排影响。GPU复验状态以最新deployment及本目录后续回执为准。

15:03:42回执：PBD-B修正课程已提交1289883，等待GPU进行自身预检；此刻不能宣称已通过GPU内存验证。Full-V1 S/B的同课程续跑为1289884/1289885，官方S复测1289886、ANet-B技术预检1289888也在排队。PBD-S1289576正式运行，预检峰值12.724GiB。旧V1退出已被调度器识别为planned_checkpoint_time_slice并保留continuations，不属于科学失败。

15:34:47关闭本轮核查：PBD-S已887次正式更新；PBD-B仍为1289883 PENDING，GPU复验继续等待。V1 S/B保存的成功更新分别为3360/2916，其原状态续跑仍排队；ANet step持续RUNNING。见pbd_retry_final.json。后续直接核验新回执，不重复发送此次轮转信号或盲重提失败前的旧作业。

ANet step1289574.0仍RUNNING，日志已显示至少13498条准备记录，继续有新的MKV成功转码，尚未READY。CPU共享限制及宿主结束后按成功journal恢复的规则继续适用。PBD和所有原课程最终仍完成80轮；不存在按epoch20成绩取消路线。

原始数据：progress_snapshot.json、六个epoch020_completed.json、profile_sources.json、resource_rotation.json、rotation_followup.json、pbd_b_memory_retry.json、pbd_retry_followup.json及public_adatad_b_completed.json。固定epoch绘图与算子分解入口分别为tools/paper_epoch_comparison.py、tools/paper_routing_cost.py。更新官方B后为84条唯一完整测试：71历史、12个当前epoch测试、1个当前公开复测；更晚checkpoint另按真实回执更新。
