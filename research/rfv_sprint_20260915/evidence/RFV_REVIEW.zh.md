# RFV Sprint：独立讨论、资产核验与协议意见

核验时间：2026-09-15 13:17 +0800。根据实现任务转达的最新用户授权，T Grounded Value、Graph-conditioned T Value、RISE-A/B 并行优先，D/S 课程保持；Graph/FVD 不未经验证接入既有长训。最终主指标冻结 epoch 80，test milestone 不作选模。本文只作审阅和方案讨论，没有改动或提交 Core/Raw/RFV 源码，也没有操作队列。

**意见：认可 RFV 的优先级调整，可以按下述收紧后的协议实现。现有证据支持投入 T 价值学习，但当前局部 T 动作空间、G1 与 RFV future-value 均未获得科学 PASS。** 两个只读代理分别核对资产和复用代码；主代理亲自读设计、打开关键 checkpoint 元数据，并负责以下判断。

## 1. 当前证据已经更新

本轮直接读回 Core 两条 `eval_010_ema/completed.json`，均完成 211 视频/792 窗口，model SHA 为4055294、evaluator SHA为6e2fc7f。因此此前评测日志修复的实际全量运行验收已通过。

|epoch 10 诊断|Avg-mAP|mAP@0.7|GFLOPs/完整窗口|
|---|---:|---:|---:|
|D-V|64.2512|43.0853|1150.7643|
|D-U|64.7122|43.2662|1148.1494|

这只是预注册的单 seed、epoch 10 test 快照，没有显示该点 D-V 优于 D-U；不据此改 RFV 参数、停止有效课程或替代 epoch 80 主指标。独立 router-label holdout 排序/regret 仍需另做。

Atlas 的 S 时间组分配已有稳定有限 headroom，B 的部分预算 CI 跨零。但 Atlas 改变的是时间组分配，带 GT 与额外查询，不能把其 mAP 增益转记为当前 ≤4-round/16-pair T router 的能力。

Raw mini-bank 已完成346次真实swap且重放误差0。其R+实际动作只有24/180为off-grid，holdout 4视频8状态的O/R+动作全部相同；这支持工程/标签重放，尚不足以证明扩域或可学习性。

## 2. 可立即复用的真实资产

全部完整路径和直接核验字段见 [ASSETS.json](C:/Users/skywalker/Documents/ChatGPT/H65/reports/wtr_rfv_review_20260915/ASSETS.json)。

### 真正的 V2-S 历史 checkpoint

远端目录：

```text
/data/run01/sczc063/yuzibo/h65_clean_adatad_20260910/support_review_20260914/research/paper/runs/review5485_full_v2_s_seed42/
```

|文件|实际更新数|checkpoint内部source_revision|可用范围|
|---|---:|---|---|
|epoch_020.pth|2000|db9c749c7bbbdcc1b5f1f14e10b1e1d0f087cacc|RISE-A固定动作重查询|
|epoch_040.pth|4000|同上|RFV冻结teacher基点、RISE-A|
|epoch_060.pth|6000|同上|RISE-A、RISE-B保留的未来actual target|
|epoch_080.pth|8000|0313f136d076552e84a459b91869ad59169f4c3b|存在；加入同一drift轨迹前须明确源码变化范围|

四份都含真实累计EMA与optimizer，EMA decay=0.99；`train_scout=true`、`train_backbone=false`。不能只读目录当前metadata并把同一个source填回所有历史checkpoint。`thumos_s_point_uniform_seed42` 是另一个Uniform控制，不能误称上述V2-S课程。

旧T router是294维输入→64→4输出（含旧分布参数），不是RFV的407→128→64→2收益头。其EMA可作为旧函数历史诊断，**不能改名为RFV新T/Graph头的true EMA**。

已经验证存在的重建资产：

```text
/data/run01/sczc063/yuzibo/wtr_raw_v1_20260915/assets/v2_s_epoch040_light.pth
/data/run01/sczc063/yuzibo/wtr_raw_v1_20260915/assets/v2_s_initial_source.pth
/data/run01/sczc063/yuzibo/wtr_raw_v1_20260915/assets/adatad_s_ema.pth
/data/run01/sczc063/yuzibo/h65_clean_adatad_20260910/fidelity_20260911/fidelity_20260911/runs/s_h65/epoch_40.pth
```

其中epoch40导出资产属于该特定checkpoint；不能直接把它当成20/60的完整learned state。使用既有重建方法与对应checkpoint自身的EMA/learned state。

### 已有 bank

- Raw：`/data/run01/sczc063/yuzibo/wtr_raw_v1_20260915/revision_27d557e/results/mini_bank/`，24视频、90组、346动作，完整160/20/20分区中的16/4/4子集。每动作407维descriptor、两个真实收益分量与pair/pack记录；顶层保存episode、support、proposal及preview时间元数据，**没有完整cheap节点特征**。可做原冻结V2-S40＋共同192-preview语义的Plain小头pilot；不能直接恢复Graph输入。
- 旧V2 calibration：上述V2目录的`calibration/actual_actions.jsonl`，969条旧协议记录，294维frame feature、196维context，包含真实actual及repair诊断。可参考实现与历史对照，不能直接当新RFV训练bank。
- Core D-V/D-U：各自`/data/run01/sczc063/yuzibo/wtr_fasttrack_20260915/research/paper/runs/wtr_d_{v,u}_s42/epoch_010.pth`真实存在，可做对应D策略的后续re-query；不是同一条T/Graph训练轨迹。

公开release是源码、图和实验记录的分发入口，明确未上传模型权重与视频。应从服务器资产重建模型，而不是从release推断存在可直接加载的新RFV权重。[公开快照](https://github.com/yuzbo/BMCR-T-AdaTAD/releases/tag/wtr-snapshot-20260915)

## 3. 先测当前T动作空间，不能借用Atlas结论

现`TemporalCoreRouter.refine`从Uniform K384开始，每轮最多16个几何提议，只选一个正收益swap，最多4轮。因而最终选集最多引入4个新帧，约为384的1.04%；重复撤回之前的swap时更少。它是有限局部交换refinement，而不是任意K384集合搜索。pair重排可能放大特征影响，但现Raw样本只观测到单pair/单pack变化，不能假设这种放大必然发生。

headroom需要在完全相同的episode、teacher、初始support、候选生成规则、4轮上限、16-pair上限和STOP下测量。每轮保存**决策前**的状态与全部实际提议，完整重执行候选的分类/定位损失；用实际收益选择最佳正动作，之后从新support继续下一轮。

这是“有限greedy actual-CF参考”，不是穷举4步最优oracle，也不是mAP上界。单步loss改善不保证全数据AP提升。记录实际base、候选与replay查询开销，和最终部署推理成本分开。

现`collect_temporal_action`会先运行整个refiner，再抽取额外单swap；它可提供局部单步标签，**不能充当4轮内每个pre-decision state的完整监督或headroom测量**。RFV的新collector应在每轮决策前采集，明确标签是“一次交换后执行固定D/S+recovery”的myopic value，还是包含剩余T策略的value；两者不得混用。

先在小的fit/calibration子集排除动作覆盖不足与近零标签问题，再冻结正式cohort和参数。失败仅否定所测有限动作空间/表示/学习策略，不直接否定所有Temporal MVC。

## 4. Standard bank与共同状态合同

RFV若采用Standard 768-preview，应重新查询当前Cross与完整下游的actual gain。Raw 192-preview与Standard的cheap/recovery context不同，初始Uniform support也不应未经核对就认为相同；**只重算descriptor、沿用Raw旧gain是不成立的**。

每条固定action至少绑定：video与物理episode/augmentation、完整有序S及valid、remove/insert物理frame ID、候选集合、合法plan、rollout轮次r、teacher/checkpoint身份、cheap版本、D/S/recovery/light版本以及明确continuation。保存原始cheap node features、时间/valid/membership/provenance和相关cheap logits；不只保存407维最终pair向量。

固定detector、cheap输入、support、action和下游策略时，G1更换T价值头不会自动改变该action的真实标签，四模型应复用同一bank。改变前端、teacher或实际下游策略时才需对应re-query。新策略走到新support，也需新状态上的真实标签，不能用旧状态ID掩盖。

T的Graph输入只能来自acquisition前的cheap证据。旧GraphRecovery的heavy anchors或Cross recovered输出不能前移给T；这会把后验证据带到选帧前。无需为T bank保存整套heavy中间张量。

## 5. G1四控制：参数匹配还不够

|控制|建议定义|必须固定|
|---|---|---|
|Plain-M|现有紧凑407维pair MLP基线|相同bank、loss、split、实际gain单位|
|Plain-L|参数量匹配的无边节点MLP/集合聚合，再做pair预测|获得与Graph相同的cheap节点、time/membership与pooling机会|
|Static Graph|预先定义的物理时间邻接；共同消息/读出模块|节点、degree、层数、宽度与训练预算|
|Dynamic Graph|邻接/边权随当前cheap状态学习|与Static相同信息和容量；只改变关系建模方式|

如果Plain-L只看到四个汇总向量，而Graph看到全节点集合，即使参数一样，也混入了信息保留差异。应把Plain-M作为紧凑控制，Plain-L作为更充分的信息/容量控制；Graph需相对强控制显示增量。

Graph参数量、节点适配与pooling、邻接搜索、消息传递和每轮重复开销均计入成本。degree=16不意味着找邻居本身只有16条边的成本。Static/Dynamic的具体差异要固定，不能同时换节点证据、消息模块与更新数。

G1首先在相同固定状态/动作集上比较rank/regret，再看每个头实际执行≤4轮闭环rollout后的效果；固定bank排名改善不能替代闭环策略验证。旧GraphFrameRouter的294/300维旧接口只能参考实现，不能直接换名当RFV G1。

## 6. RISE-A：固定动作的真实价值漂移

优先用已核验的同课程、同source的V2-S20/40/60。固定action identity与完整S，不让各checkpoint重新选择T来替换被测动作；在每个checkpoint上完整重执行baseline/changed、恢复同一loss normalizer、保存两个signed gain及查询成本。

报告每视频聚合的rank变化、符号变化、幅值变化及重放噪声。区分checkpoint时间e和rollout轮次r。历史V2中Scout、Adapter、recovery/head等会变化，因此测得的是所声明整条轨迹的value drift，不是“只由detector head变化引起”的因果结论。

实际drift必须重查询；对旧router输出做减法不等于actual drift。也不能把换了support/episode/continuation后的差异叫作同动作漂移。

## 7. RISE-B：四控制应把Q40视为Current

认可拟议的连续同家族RFV训练：先用θ20的fit标签得到Q20，再沿同一优化轨迹用θ40的fit标签得到Q40，EMA逐真实optimizer step累计；θ60的fit标签不进入训练。在同一固定detached s40及同候选集上评估，actual future target来自θ60对这些固定动作的重查询。

预测起点已经是40，建议四个互异控制命名为：

|控制|含义|
|---|---|
|Anchor Q20|更新前的历史函数；是陈旧基线|
|Current/Post Q40|当前最新函数，β=1的Post；主要无外推控制|
|true EMA40|同家族RFV函数从实际训练轨迹累计的EMA|
|Future β>1|在共同输入上对两个完整函数的输出作外推|

```text
zA = QA(s40)
zP = QP(s40)
zF = zP + (beta - 1) * (zP - zA)
beta = 1  =>  zF = zP
```

z须处于相同的实际cls/loc收益单位。不能先各自做不同尺度标准化后混加，也不能把函数输出外推悄悄换成参数张量外推。Future必须胜过Current/Post与true EMA，不能只胜Anchor后把正常适配收益归因于外推。

每个快照都从同一原始cheap state运行自己的完整adapter、Graph/context、head和normalization；不能把Q40已加工的Graph context交给Q20/EMA，也不必给T输入后续heavy状态。EMA需涵盖该函数的可变参数/缓冲及明确处理规则，不能取两个checkpoint平均；不能复用旧294维router的EMA当新RFV EMA。

### beta与“未来”的证据边界

如果beta用θ60的20个calibration视频标签选择，而20个holdout视频只评一次，这是允许的**离线跨视频forecast诊断**；但beta选择已经用了未来时点信息，不能称为严格时间前瞻。若要做严格前瞻声明，须事先固定beta，或在更早/独立时间窗口选定，然后锁定预测θ60。无需为首个离线诊断强行扩大实验矩阵，先把声明写准确。

fit-only限制也适用于归一化统计、模型选择和Post额外更新。holdout20不用于选beta、head容量或checkpoint。detector及历史V2见过这些视频，因此仍称router-label holdout，不是完全未见视频的检测泛化。

## 8. RFV准入与当前接受范围

主要排序指标使用包含STOP的actual cls+loc regret、top-1决策和rank correlation。NDCG若使用，需预定义非负relevance（例如实际gain的正部）及零信号处理；不更改signed训练标签。统计按video聚合，不把多个state/action当独立视频。

T当前动作空间有稳定机会、Plain可学，才说明该T接口值得进入后续课程；G1只有胜强Plain控制并考虑成本才解锁Graph-conditioned T；RISE-B只有在真实drift存在且Future胜Post/EMA后，才有资格进入FVD实验。外推需要两个函数预测的开销单记；不直接当免费部署增益。

主结果固定epoch80。现有`evaluation.py`仍有“milestone EMA peak”字样、旧配置有`best_full_test_mAP`，RFV新入口须按最新授权修正报告/选择语义；无需修改正在运行的Core训练配置。10/20/40/60仅诊断，不依据test选择RFV方法或beta。

**本次授予的是协议接受意见与资产可用性核验，没有授予RFV代码/GPU/科学PASS。** 新RFV工作树完成具体实验后，仍按精确提交与配置分别复核。D/S课程继续；Graph Recovery、GraphKV、DB和未经验证FVD不混入本轮T对照。
