# 同40轮对照已齐：Uniform-S领先，Full-V2-B保持小幅优势

2026-09-14 17:18:13远端快照。新收集Uniform Full S/B epoch40 EMA，均完整211测试视频/792窗口；五阈值AP的算术平均与平均mAP一致。所有正式课程继续80轮，以下不是终点性能。

|骨干|方法|epoch40 mAP %|全测试平均GFLOPs/窗|
|---|---|---:|---:|
|S|Full-V2|64.9949|1225.980|
|S|Uniform Full|**65.3369**|1226.478|
|B|Full-V2|**68.0518**|**3909.386**|
|B|Uniform Full|67.8189|4094.316|

S：Full-V2低0.3420pp，成本仅低0.0406%，基本相同。B：Full-V2高0.2329pp，平均成本低约4.52%。当前只能说B的完整方法相对该内部均匀对照有小幅优势；S仍需要改进。这里Uniform Full是完整框架内的对照，不是原版AdaTAD直接降采样。

![同40轮完整结果](figures/epoch_040_comparison.png)

## 最佳与最近一次结果分开

V2-B最佳仍为epoch20的68.5013700%/3940.847995G；Uniform-B最佳仍为epoch10的68.3322096%/4094.315930G。因此按各自已测最佳，V2-B为+0.1692pp、平均成本约−3.75%，不能用较低的Uniform40来放大最佳模型优势。V2-S与Uniform-S的当前最佳均为epoch40。

![完整测试里程碑曲线](figures/training_trajectory.png)

曲线只连接实测里程碑，不插值生成30轮或60/80轮成绩。Full-V1尚无40轮完整结果，其按原checkpoint续跑的课程仍保留。PBD-style-S只完成epoch10全测63.7666685%/1131.741212G，不能当作最终9层压缩成绩。

## 路由与全局比较

V2-S40有783/792窗D100/S100、9窗D100/S48；Uniform-S40全部D100/S100。V2-B40全部D75/S100，Uniform-B40全部D100/S100。此次仍没有证明每个窗口同时使用D/S稀疏的必要性。

V2-S40的首个完整profile窗口恰好使用S48，成本1181.420662G，不能将它冒充测试集通常成本。主比较采用实际全测试均值1225.979723G；代表固定窗只作为单独测量视角。

总体图保留历史已核验官方AdaTAD-S：69.0125535%/2347.894039G，它在已知固定窗口径下仍同时优于当前压缩B模型的精度和计算。该参考以空心菱形标记，仅放在固定窗面板；没有伪造其全测试平均GFLOPs，也没有把它计作本轮新复测。因此，当前同B骨干的内部优势仍不足以证明完整方法获得了跨骨干全局新前沿。

![当前最佳与历史官方S固定窗参照](../analysis/figures/fig1_complete_method_pareto.png)

## 队列和数据

- 唯一paper控制器2408086健康。374阶段：10COMPLETED、5RUNNING、5PENDING，其余WAITING/WAITING_INLINE；没有当前FAILED。
- Native AdaTAD直接K384：S1289969/B1289968仍因AssocGrpGRES排队，GPU预检及新mAP尚无。80轮适配课程和K768参考均已持久登记。
- PBD-B1289883仍排队，不能宣布此前OOM修正已通过GPU；V1续跑1289884/85同样等待配额。
- 5条运行训练保留；本轮没有取消、重排或重启GPU任务。Graph Machine候选仍仅为研究建议。
- ANet CPU step1289574.0仍RUNNING，日志已到至少13974个准备记录；完整READY未出现。preparation.json仍是此前INCOMPLETE汇总，不能据旧prepared字段否认journal进展，也不能据日志计数宣称完整数据就绪。

## 归档与绘图

当前汇总为89条唯一完整测试：71历史＋18当前。保留之前84条结果且mAP逐项不变，新增5条完整结果进入图表（包括此前16:29已报告的V2 S/B40及PBD-S10，以及本次Uniform S/B40）。

修正了本地报告输入拼接：初始current_snapshot仍只有22条旧BMCR metrics，现由`tools/paper_review_snapshot.py`使用全部24条completed回执及匹配profile统一来源，保留两侧80轮终点。绘图/汇总显式读取UTF-8，适配本项目实际中文元数据；这些仅是报告工具变更，没有改变模型计算或训练。

后续报告先运行`paper_review_snapshot.py --live-snapshot <最新capture> --output <complete_evidence.json>`，再交给`paper_review_analyze.py`。本目录analysis_manifest.json还保留历史官方S的fixed_window_references，更新时继续保留。新曲线入口为`tools/paper_training_trajectory.py`。所有新图已视觉检查。
