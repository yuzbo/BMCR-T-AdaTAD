# 论文主文与附录图表规格

## 图表总原则

每图都绑定一个问题、实际实验ID、固定checkpoint和原始数据来源。未完成seed42不画结果点；旧3407标internal precursor/archived；公开作者reported数字与本框架retested分面。GFLOPs=完整encoder+TIA+scout+router+decoder+head的2MAC；score-QK/AV计入。校准、teacher训练额外成本另表。

主选择为best full-test mAP，同时必须给terminal与候选epoch集合。best点成本必须来自同权重与同route。bootstrap只针对固定checkpoint的视频抽样；不画跨seed标准差。不同epoch不是独立样本，跨配置箱线图不是同模型训练稳定性。

按投稿届author kit检查排版。参考双栏总宽6.875in/单栏3.28125in，矢量PDF与SVG＋足够分辨率PNG；图内字体在最终插入宽度可读，不通过极小字号塞满全部实验。所有颜色之外必须有线型/marker/文字区分，打印灰度仍可理解。本文模板以独立panel输出，由LaTeX拼合，避免字体被二次整体缩放。

## 主图M1：三轴问题不是相似度，而是任务干预

**M1a — 可描述相似性。** 同真实窗口展示相邻帧cosine、空间相似、层变化，只标“proxy”，不标“redundancy proven”。需：source video/窗口、冻结checkpoint、特征层、归一化定义。

**M1b — 同checkpoint八格。** x=TDS000..111，y=五阈值mAP%；S/B独立panel。旧J01 epoch5 EMA为现有数据，title明确archived seed3407。000是该checkpoint全容量而非官方dense。新full@40的对应八格待实际完成。

**M1c — 条件交互。** y=差分pp；分别TD|S0、DS|T0、DS|T1、三阶；真实AP重算后做成对视频bootstrap才画CI，目前旧摘要只有点估计。不能从多个负二阶推断三阶也负。

**M1d — 任务位置分层。** 起点、动作内部、终点、背景分别做相同数量/预算的计算删除和回写干预；y=完整AP变化或明确recall/loss，不能把不同群组AP简单相加。加中部对照以检验“仅保护边界足够”的反假设。

图注模板："Task damage under controlled computation interventions. Similarity maps are descriptive proxies; panel (b) measures the same checkpoint under eight execution policies. Differences do not estimate independently adapted models."

## 主图M2：已实现图与建议机制

M2a已实现：Input→Scout→budget15/frame1swap→single-frame K→packed encoder A-MoD+FFN-light+TIA→6/9/12 memory→original-time decoder→trainable head。

M2b放大一层：X_full；depth admits决定Q/K/V；空间heavy在admitted内；depth bypass是hold；space bypass是light；scatter后TIA。注明默认compact KV，fullKV是对照。

M2c建议改进：same-support reference使用完全相同selected RGB；target虚线到层6/9/12；depth-bypass light与fullKV开关分别标；最终external original teacher只到原轴回写；箭头stopgrad清楚。

真实/预测状态用填充/边框/标签，不依赖颜色；teacher虚线，不与推理实线混同。写清K/2及Q=384，不画192visible+192mask错误网格。已实现与proposed不能合成一个已验证网络图。

## 主图M3：完整计算—最佳性能前沿

M3a同骨干相对：x=完整模型cost/对应同协议dense；y=best five-tIoU mAP，或y=dense gap(pp)；S/B分开。external公开方法只能同框架重测后进入此图。

M3b跨骨干绝对：x=完整GFLOPs/window或实际dataset mean/window（同scope），y=best mAP。公开dense-S/B与内部Cross等一起显示，不能隐藏dense-S支配Cross-B。旧只有firstfullcost，则在独立历史图，不和dynamic dataset mean混画。

M3c终点：空心terminal与实心best对应同run，图注列eval候选；离散点不样条插值。

M3d训练成本辅图：x=累计实际训练GFLOPs或GPU小时（来自日志），y=相同更新/选模规则mAP；含teacher与PBD候选评估，不用optimizer steps冒充等计算。

图注模板："Best full-test performance and complete-model computation. Peak and terminal results use the stated selection policy. No error bars across training seeds are implied; each configuration is trained once with seed42."

## 主图M4：深度保性能的机制拆分

M4a默认compactKV、fullKV、depth-light、same-support、组合，比较相同query capacity和等完整MAC两种方式，不能只选有利口径。

M4b层6/9/12与TIA前后误差：x=original block ID；y=与same-support reference的normalized error/幅度；标skip/reentry；不要拿selected192和original384做逐index误差。

M4c静态drop/PBD-style/动态同成本前沿。删除block也删TIA或只跳贵算子分别标。

M4d课程轨迹：x=成功optimizer updates；y=mAP或层误差。dense-query教师成本另列；完整课程中的渐进阶段不是不同模型seed。

## 主图M5：初始化与解码的必要性

M5a MAE-random vs MAE-pretrained同架构；M5b freshCross vs MAE（明确参数/计算不完全相同）；M5c hotR03 vs freshCross，初始化成本单列；M5d input alignment改善是否反映到TAD而非仅loss。

x=updates（主）/train compute（辅）；y=完整mAP与feature/temporal error分panel。所有40轮模型与main前40同eval集合。旧TCN近Cross作为强简单控制。

## 主图M6：实际计算价值与router校准

M6a predicted gain vs actual_delta，cls/reg分开；散点每点一次真实动作，按video标samplecount。M6b repair_delta vs actual_delta：展示代理偏差，不称同一量。

M6c decile actual gain/negative acceptance；M6d菜单oracle regret与固定最强plan。M6e plan1训练的frame head迁移到不同D/S/K vs plan-aware模型。

OOF限定router-fitting；calibrated90%after不是独立coverage保证。预算方差、共用baseline相关性和每计划查询数列图注。

## 主图M7：任务案例和失败模式

每个案例同一video/窗口、完整timeline：真实frame strip、GT动作、旧/新proposals置信度、单帧选点、anchor两贡献帧连线、pair跨度、depth曝光、空间heavy图、native恢复误差。

覆盖改善/不变/退化以及短动作、重复事件、中部证据、遮挡/小主体。选例规则预先固定；无RGB资源不得生成替代的“真实帧”。GT仅绘图join，不供推理。

差异图显示miss→hit、duplicate→merged或相反、start/end偏差，不只挑attention视觉更好看的案例。置信度与时间标尺跨模型一致。

## 主图M8：完整实现成本与轴间预算

x=执行层/阶段；y=Q、KV、heavyFFN、lightFFN计数与MAC分panel；确保doublecount查询排序被显示。按有效/physical候选、full/partial/short窗口分别看。

菜单分布按action-density/gap/困难事件做离线分组。Latency/memory/E2E各自独立附图或表，不标route pass/fail门槛。图证实算子确实少执行，不证实任务语义可无损删除。

## 附录12组

A01：全部52旧已注册+新增有限配置状态矩阵，标pending/null/complete。
A02：40/80完整trajectory，EMA/online、peak/terminal，固定候选集；不能作为重复样本方差。
A03：same-checkpoint各budget曲线，calibrated terminal与uncalibrated best分开。
A04：完整逐tIoU、逐类AP与配对error decomposition；oracle误差修正增益不相加。
A05：selection gap/pair跨度/边界和动作内部覆盖；short/partial physical slots。
A06：每层age与重入误差；旧last_heavy_depth因末层dense退化的对照。
A07：attention得分vs任务收益、coverage/tile/uniform/random空间差异。
A08：源参数依赖、teacher查询依赖、推理执行依赖三个矩阵，no_external命名解释。
A09：训练总账含PBD候选评估/多阶段恢复/teacher/support/self查询与缓存。
A10：真实model/encoder/decode/H2D/NMS timing，均值/中位/p95/原始样本；不剔异常后宣传改善。
A11：ANet、InternVideo1-MQ、TadTR所有完整结果，输入轴/head/预训练明确；无结果留空。
A12：相同mask dense-mask/compact、fullKV、padding、decoder init与gradient路径技术测试。

## 主表6张（按结果和篇幅选4张入主文，其余附录）

T1外部公开方法对比：AdaTAD/PBD/其他TAD，同协议复测与作者reported分区。
T2内部演进：H65/BMCR/Cross/FPW/当前paper，seed、课程、init、head、best/terminal全列。
T3D/S保性能：static/PBD/dynamic/fullKV/light/support，最终完整MAC与最佳/终点。
T4解码与teacher依赖：random/MAE/freshCross/R03、同支持/原支持、多层memory/监督。
T5有限预算与axis-factorial：独立训练和samecheckpoint分区，实际训练曝光与cost。
T6数据/teacher/训练/系统成本透明表。

## 最低结果数据合同

```text
run_id, config_id, source_commit, role[external|internal|proposed], dataset,
video_id_set, backbone, encoder_pretraining, task_init, scout_init, head_init,
recovery_init, decoder_init, seed, epochs, successful_updates, schedule_epochs,
eval_epoch, eval_candidates, state[ema|online], selection_rule, terminal,
AP03, AP04, AP05, AP06, AP07, average_mAP_fraction,
complete_model_gflops, compute_scope, full_window_gflops, dataset_sum_gflops,
window_count, video_count, valid_candidates, physical_slots, plan_histogram,
query_KV_FFN_light_counts, TIA_counts, score_qk_macs,
external_teacher_queries, same_support_queries, shared_full_queries,
actual_action_queries, repair_queries, candidate_layer_queries,
train_seconds, model_timing_samples, decode_seconds, transfer_seconds, nms_seconds,
raw_result_path, raw_profile_path, raw_prediction_path, status
```

图程序应校验unit、role、seed和checkpoint来源，缺失字段时拒绝相关图而非填0。每图写 `*.sources.json`，保留原始row IDs、数据路径、聚合/过滤代码版本和标题/图注。
