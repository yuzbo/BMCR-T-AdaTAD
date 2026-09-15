# 实验进度与发现

记录时间：2026-09-15 11:10–11:33 +0800。数值来自本轮SSH读取的原始回执，快照保存在同目录4090.json、a100.json、atlas.json及temporal_*_summary.json。后续状态可能变化。

## 1. 当前真实进度

|路线|已完成|当前状态|
|---|---|---|
|Core D-V / D-U|各完成10轮、1000次更新；epoch010、latest及优化器/EMA/RNG已保存|首次内联评测日志报错已修复并提交续跑；新作业1290651/1290652均Priority排队|
|Core S-V / S-U|代码/协议独立审阅通过|A100长期未启动，原248205/248206取消后迁到4090；1290654/1290655 Priority排队，尚无训练成绩|
|Raw-v1|6开发视频/19窗口接口验收；24视频、346次真实swap mini-bank；诊断统计|已完成；尚未拟合Value头，未扩完整bank或跑publication Value评测|
|Atlas T|S/B各211视频/792窗口、五策略×六预算、官方AP及10000次video bootstrap|已出完整T结果和两张图|
|Atlas D/S与population|S population完整完成|11:25时S-D509/792，B population321/792；完整D/S与恢复仍进行中|
|旧参考收尾|V2-B、Uniform S/B均80轮和全量评测完成|保留终点，其他低价值旧课程已退役|
|Graph/RISE/DB等增强|计划与解锁规则已明确|科学runner和对应增益证据尚未完成，不能称已训练或有效|

Core不是一直运行到本次查询：D-V在04:51、D-U在04:55首次epoch10评测时停止。根因为evaluation.py把包含wtr_geometry Tensor的内部执行plan写入JSON。模型前向和已完成训练没有因此失效。修复改为记录已有的公开trace.plan，并单独保存evaluation_source_revision。

该日志修复已通过独立定点复审，补丁6e2fc7f78424e2485e8d33b079fac926eb0cf4a6；科学模型/配置/标签仍为40552945ad5d56b7404f833cd86f998e8558e8b1。保留1000次更新，从latest先补完整epoch10 EMA评测，再继续训练，不重做前10轮。排队中的恢复尚不等于完整评测已通过。目前没有D-V/D-U新mAP，更不能声称D-V优于D-U。

D-V已收集103条在线actual-CF标签，fit以外为0；其中53条正收益、50条负收益，未出现全零标签。它说明真实监督在执行，不证明out-of-fit排序/regret已改善。稳定checkpoint的calibration/holdout re-query及科学gate仍需补齐。

## 2. 最明确的科学发现：时间分配有可利用空间

Atlas在冻结官方AdaTAD、固定有限动作空间中，比较Uniform、dense Attention诊断及GT-assisted有限CF参考。以下均为同等最终执行成本，单位为mAP百分点。

|保留组数|S: CF−Uniform及95% CI|B: CF−Uniform及95% CI|
|---|---|---|
|6|+4.285 [3.051,5.855]|+2.144 [1.044,3.511]|
|8|+3.507 [2.337,4.692]|+1.392 [−0.058,2.594]|
|10|+2.951 [1.668,4.119]|+1.305 [0.055,2.404]|
|12|+2.028 [1.035,2.855]|+0.413 [−0.655,1.331]|

8组点：S为Uniform61.828、Attention62.381、CF65.335 mAP，执行1225.31 GFLOPs/窗口；B为65.801、64.301、67.193，执行4184.13 GFLOPs/窗口。

结论：S的时间分配空间信号稳定；B较弱且8/12组区间跨零。它支持继续研究任务价值分配，不证明廉价Value头已学会，也不是mAP数学上界。CF有GT和额外查询：base+12候选的神经查询成本S约10320.53、B约34919.11 GFLOPs/窗口，六预算共享排序；该开销未放入最终执行横轴。区间为逐预算视频bootstrap，不是同时置信带或多seed不确定性。

## 3. Raw发现：工程闭合，但域诊断覆盖不足

- 六视频19窗口的Raw/legacy RGB、恢复与预测一致，官方AP缓存一致；这是工程证据，开发视频mAP不作为正式性能成绩。
- mini-bank完整：24视频、90个domain/state组、346次swap，符合先300–500条的范围；重放最大误差0，收益非全零，总正收益比例约48%。
- 两域45个配对状态的初始support全部一致。Official候选数251–768、均值712.53；Expanded-raw为827–1344、均值1288.53，每状态增加576个off-grid候选，确实是扩域。
- 但实际R+的180次动作中，仅24次插入off-grid帧（13.3%）；38/45个配对状态的O/R+动作集完全相同，holdout更是8/8全部相同。
- fit的30个配对状态中，只有5个的R+最佳单步loss收益更好；calibration和holdout没有最佳收益差。不能将这个结果解释为Raw已优于Official，也不能据此否定更充分搜索的Raw域。
- 本次每个swap都只改变1个pair、1个pack，没有观测到大范围tubelet重排。结论限于当前局部动作生成规则。

本次bank共90次base、346次动作、87次重放，合计523次主体前向，记录约622061.56 GFLOPs（不冒充端到端部署成本）。24 episode读取共返回13820个解码帧，解码/变换约135.79秒；codec内部GOP解码量未测。现有缓存及重复O/R+查询的成本必须保留口径。

下一步应先在training/development补足实际off-grid动作覆盖、保持公共support与预注册查询预算，再判断扩bank和拟合Value；不直接把现有采样放大到6400。当前仅支持“输入可执行、标签可重放”，尚无Raw domain/learnability或完整检测增益。

## 4. 旧参考的80轮终点

均为211视频/792窗口，属于旧模型；不冒充新Core成绩。

|旧模型|80轮Avg-mAP|mAP@0.7|平均GFLOPs/窗口|
|---|---:|---:|---:|
|Uniform S|65.154|43.642|1226.48|
|Uniform B|67.769|46.542|4094.32|
|Full-V2 B|68.061|46.833|3909.50|

同80轮，旧V2-B比Uniform-B高0.291pp，执行计算低约4.51%；尚无多seed/配对CI支撑强显著性结论。旧曲线也未表现出“训练越久越好”：Uniform S的40轮65.337略高于80轮65.154；V2-B的20轮68.501高于80轮68.061。保留终点与全部里程碑，不按test峰值修改新方法。

## 5. 论文解释和接下来最有判别力的证据

Atlas S微干预在1%相对整窗loss容差下D/S近零率约99.97%，但benign参照也为99.94%。不能把近零率当可删除计算比例，也不能把高Gini直接解释成可学习性。Atlas现有O/D/S平均interaction区间跨零，尚不支持强非加性；这不证明全局可加，更不能直接决定Graph/FVD是否有效。

证据应连成三步：Atlas显示有限执行空间存在机会；Core用匹配V/U训练、稳定checkpoint的router-label holdout和完整AP证明是否可学/可部署；Raw另证新增观察域是否有收益。Atlas的O是Obs-replace，Raw的O是Official-grid，图表明确命名。两者D/S的干预条件不同，不混用标签或效应大小。

最先补齐：恢复后的D pair完整epoch10 AP；同checkpoint的calibration/holdout ranking与regret；S pair实际训练；Raw新增候选覆盖充分的小bank。Graph G0/G1、RISE-A/B、DB及后续组合继续按独立gate执行，不从现有技术PASS推断成功。

Atlas已完成的图位于C:/Users/skywalker/Documents/ChatGPT/H65/h65_clean_adatad/wtr_characterization_20260915/output/temporal/；本报告不将全套PDF或新Core/Raw完整性能图标为完成。
