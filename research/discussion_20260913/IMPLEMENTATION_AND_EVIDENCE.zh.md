# 当前实现与实验结果导航

|范围|源码/配置|已有证据|
|---|---|---|
|作者H65/BMCR初期与修正基线|`h65/full`、`tools/full_*`、`configs/full`|`phase2_20260910`、`fidelity_20260911`及对应完成记录|
|BMCR总80轮修正课程|`tools/bmcr80_*`（实际入口见bmcr80计划）、`bmcr80_20260913`|本目录`live_snapshot.json.evidence.bmcr80`与`legacy_runtime/bmcr80_20260913`|
|旧clip/DS3|`h65/ds3`、`tools/ds3_*`、`ds3_20260912`|历史证据保留，全部未完成路线取消|
|第一代原轴恢复与三轴执行|`h65/frame`、`configs/frame`、`tools/frame_*`|47条有效FPW测试、旧训练/校准/诊断，见快照和`research/paper/figures`|
|当前完整模型组合|[PaperModel](../../h65/paper/model.py)、[NativeEncoder](../../h65/paper/encoder.py)、[TaskReadout](../../h65/paper/readout.py)|实现/CPU验证与GPU结果分开|
|当前恢复与官方decoder对照|[decoder](../../h65/paper/decoder.py)、[MAE初始化](../../h65/frame/mae_init.py)|自建Cross主模型与MAE预训练/随机对照；不能把当前性能归因给尚未跑的官方decoder|
|当前深度/空间实际计算|[engine](../../h65/paper/engine.py)、[encoder](../../h65/paper/encoder.py)|真实compact计算与全预算一致性CPU证据；最终精度尚待实验|
|当前真实任务收益与预算|[interventions](../../h65/paper/interventions.py)、[routing](../../h65/paper/routing.py)、[calibration](../../h65/paper/calibration.py)|有限15菜单与单次换帧，非任意token闭环补算|
|当前训练与自蒸馏|[objectives](../../h65/paper/objectives.py)、[training](../../h65/paper/training.py)、[train入口](../../tools/paper_train.py)|区分external teacher、shared-full student、EMA选模|
|当前部署与评测|[course](../../tools/paper_course.py)、[dispatch](../../tools/paper_dispatch.py)、[eval](../../tools/paper_eval.py)、[evaluation](../../h65/paper/evaluation.py)|52份seed42配置均已归档；提交与等待依赖见快照|

主证据入口：[全测试记录](../paper/figures/full_test_records.json)、[基线参照](../../fidelity_20260911/FINAL_COMPARISON.json)、[效率图数据](../paper/unit_focus_20260913/best_efficiency_data.json)、[三轴八格重算](../paper/story_20260913/factorial_recheck.json)、[本次完整快照](live_snapshot.json)。

|结果身份|mAP %|GFLOPs/完整768候选窗口|说明|
|---|---:|---:|---|
|公开官方AdaTAD-S，VideoMAE-S backbone|69.0126|2347.89|完整211视频/792窗口；公开方法内部复现参照|
|作者旧Cross-S EMA20|64.5743|1226.36|内部实验，不是公开论文方法|
|公开官方AdaTAD-B，VideoMAE-B backbone|71.1280|8082.15|完整同协议参照|
|作者旧Cross-B EMA10|68.5792|4094.12|内部实验；非最终paper模型|
|作者修正BMCR80-S已测峰值|63.8372|此处不填跨checkpoint成本|已测epoch65；训练完成不代表每个checkpoint已评测|
|作者修正BMCR80-B已测峰值|68.1361|此处不填跨checkpoint成本|已测epoch70|
|作者历史H65-S|65.3857|尚无配对核实成本|内部历史高分，不能列为外部公开竞争者|
|当前seed42最终paper|未产生|未产生|截止本次快照，不能把旧Cross成绩当作最终模型成绩|

以上GFLOPs按矩阵/卷积2MAC口径，包含对应完整模型执行；不是整视频总计算、训练总计算或纯骨干计算。mAP是THUMOS五阈值均值，不能与公开表中单独mAP@0.5直接拼接。全部未测值保持为空。

已有图：[主效率图及分布/八格/课程图](../paper/unit_focus_20260913/)、[既有结构和实验汇总图](../paper/figures/)。这些图是讨论和改进的输入，不要求外部模型接受其叙事或认定已具竞争力。

当前数据对应的开放问题、可证伪主张、图表设计和后续实验优先级，交由本次讨论prompt提出，不在本导航预设答案。
