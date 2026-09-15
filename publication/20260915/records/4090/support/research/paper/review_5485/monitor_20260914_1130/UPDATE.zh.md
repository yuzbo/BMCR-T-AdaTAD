**新课程首个完整结果、BMCR80收尾和ANet准备修正**

新seed42课程已产生第一个完整测试结果。Full-V1-S在epoch10 EMA上为63.2251083% mAP，211个测试视频、792个窗口；代表完整窗口1170.152664 GFLOPs，全测试平均1169.595255 GFLOPs/窗，合计926319.441847 GFLOPs。推理延迟记录133.600 ms，属于回执中定义的测量范围，不能与其他范围的端到端时间混算。

这是80轮课程的中间点，不进行性能门槛淘汰。其代表窗计算量较历史同骨干dense参考2347.894039 GFLOPs约减半，但mAP仍低于该历史dense参考69.0125535%。Uniform、Full-V2、PBD尚无本次已收集的完整结果，不能据此断言复杂选点或三轴动态已获胜。

792个窗口的预算为K384_D75_S100共214窗，K384_D100_S48共499窗，K384_D100_S100共79窗。当前已测策略使用了时间压缩以及不同窗口上的深度或空间压缩，没有在这些窗口同时降低D和S。完整模型的这一个成绩不能代替独立T/TD/TS/TDS矩阵对三轴必要性的论证。

11:28权威训练快照：Full-V1 S/B分别1000/903更新，Full-V2 S/B分别799/622更新，Uniform S/B分别437/350更新，六个模型均通过各自预检并运行。PBD S/B和两个官方Dense复测为AssocGrpGRES排队。Static S/B在同一持久队列自动补位；不增加seed、不根据中间分数早停。

**历史BMCR80已经全部完成**

两条历史seed3407课程均为warm20 + joint60，各6000次joint更新、12个完整EMA候选测试。36个调度阶段全部完成，后续不再重训或重复收尾。

| 模型 | 最佳mAP % | 最佳总epoch | 80轮终点mAP % | 最佳checkpoint实测GFLOPs | 延迟ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| BMCR-S | 63.8372101 | 65 | 63.5533031 | 1210.345353 | 135.298 |
| BMCR-B | 68.3447613 | 80 | 68.3447613 | 4077.475688 | 155.174 |

FLOPs按完整模型矩阵/卷积2MAC计数，其他算子列出但不计入。延迟为输入已在GPU上的scout/router/backbone/adapter/detector/preNMS过程，不含解码与NMS。最佳S65和B80均有对应checkpoint的独立profile回执，不移用其他模型成本。

S从最佳65轮到终点回落0.2839pp；B从70轮到80轮继续提高0.2087pp。延长训练的收益取决于模型，须同时展示峰值、终点和全部候选。历史Cross-S仍为64.5742816%/1226.364694G；Cross-B为68.5792268%/4094.124761G，因此相对已完成BMCR80的优势分别为0.7371/0.2345pp，额外成本16.0193/16.6491G。不同历史课程不能当作严格单因素消融。

历史dense-S仍在绝对mAP与FLOPs上同时优于这两个B压缩参考。当前必须等待P0匹配课程形成新的竞争证据。H65/BMCR/Cross均为作者内部方法；历史H65-S的65.3857%精度保留，但无匹配成本，不给它虚构Pareto横坐标。

![BMCR完整轨迹与损失分布](../bmcr80_final/figures/bmcr80_trajectory.png)

![更新后的历史效率参考](../bmcr80_final/figures/bmcr80_efficiency.png)

**ANet缺失格式已修正并接续排队**

1289554成功修复损坏分片并完成此次扫描，但因覆盖仍不足而退出1。已准备training9062/10024、validation4312/4728，共13374/14752；无转换失败记录，1378个缺失ID不在此前只计mp4的archive_coverage中。

进一步读取真实原始目录发现6996个mp4、736个mkv、16个webm；补充ZIP有1341个mp4、169个mkv、6个webm。缺失列表中明确存在这些MKV/WebM文件。prepare_anet_videos和prepare_anet_archives原先均只接收mp4；已统一识别三种实际存在的视频格式，输出仍严格为15fps、short256的mp4。全部缺失是否均由此造成，以接续扫描和完整READY为准。

修正提交：新树4a89751、旧owner数据工具deee9f2。新准备作业1289684（paper-anet-all-formats）已提交，当前等待账户配额；复用原始43分片、补充ZIP、已有视频与成功journal，不重复下载、不改变blocked列表或测试协议。当前仍未READY，正式ANet实验继续等待数据。

汇总器也修正了一个真实重复计数：BMCR同一目录的completed.json和metrics.json曾同时计入。优先使用完整回执后，本次72条唯一完整测试为71条历史、1条当前seed42；所有mAP数值保持不变。

原始证据：本目录full_v1_s_epoch10_completed.json和progress_snapshot.json；../bmcr80_final/receipts/；../data_recovery/mixed_format_submission.json；../analysis/。图表为PNG/SVG，重绘入口tools/paper_bmcr80_final.py。新Full-V2、Uniform/PBD和更晚checkpoint成绩待真实完整回执产生后更新。
