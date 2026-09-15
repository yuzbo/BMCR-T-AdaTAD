# 当前进度、发现与问题

只读服务器快照：**2026-09-15T20:45:20+0800**。本次仅整理、审阅和发布；研究自动跟进仍暂停，未启动新训练或修改旧课程。此前停止交接保留为历史记录，此次发布请求只恢复整理/同步工作。

**已经证实存在有限计算分配机会，但当前局部Value学习未稳定泛化，完整论文模型尚未实现和验证。** 不能把“有机会”“执行可信”“可学”“完整任务有收益”合成一个PASS。

## 1. 最重要的已测结论

|实验|结果|适用范围|
|---|---|---|
|Atlas-S T 8组|CF−Uniform +3.507pp，95% CI[2.337,4.692]|冻结模型、GT辅助有限组参考，非部署router|
|Atlas-S D 8组|+6.688pp，CI[5.196,7.741]|该D分组空间有机会，不是当前D75头的上界|
|Atlas-S S 8组|+4.451pp；已登记中间预算CI均正|该S分组空间有机会，不是原单token近零率的反面定理|
|部署局部T空间|cal20/46窗 +0.528413pp，CI[0.082440,0.714687]|最多4交换/每轮16候选；不是全局384选集上界|
|Value-R1|18个head完成；within rho0.00540、regret0.001598842|LEARNABILITY_FAIL；同视频未见候选未恢复|
|Reverse P1/P1-bi/P2|6个head完成；P2 regret0.001489609，STOP0.001413537|STRUCTURE_SIGNAL_FAIL；P2−P1-bi CI跨0|
|历史RISE|真实20→60 drift rho0.7764、sign flip14.39%；Future未改善regret|B0 held30个seed-state最终选择0次改变|
|固定D/S梯度|Value梯度router外为0；4组额外clip影响最大约0.1433%|小样本未支持“大额外裁剪解释整体mAP差”|
|Atlas跨轴交互|Avg-mAP交互−0.023734pp，CI[−0.103427,0.132906]|INCONCLUSIVE；不证明全局可加或联合规划必要|
|Raw mini|实际扩候选576个/state，但R+查询仅24/180为off-grid|域查询覆盖不足，尚无验证侧增量或完整Raw模型|

Atlas T/D/S表中相同“8组”不表示三轴横向同成本。所有CF特权查询成本单列；不能将其mAP当learned结果。完整图与精确表见[图集](FIGURE_GALLERY.md)。

## 2. 当前D/S完整组件课程

四条旧science405课程继续自身训练，尚未得到本快照的80轮primary endpoint。它们是固定T/K384、固定调制层和配额的单轴组件消融，不是全局TSD正式模型。

|课程|epoch|Avg-mAP (%)|AP@0.7 (%)|完整窗GFLOPs|
|---|---:|---:|---:|---:|
|wtr_d_v_s42|10|64.251242|43.085294|1150.764264|
|wtr_d_v_s42|20|64.726113|42.940293|1150.764264|
|wtr_d_v_s42|40|64.260056|42.224910|1150.764264|
|wtr_d_v_s42|60|64.368787|43.096330|1150.764264|
|wtr_d_u_s42|10|64.712168|43.266152|1148.149378|
|wtr_d_u_s42|20|65.370542|43.657267|1148.149378|
|wtr_d_u_s42|40|64.334131|42.749081|1148.149378|
|wtr_d_u_s42|60|64.465185|42.978239|1148.149378|
|wtr_s_v_s42|10|64.292456|42.876101|1184.738127|
|wtr_s_v_s42|20|64.906950|43.370010|1184.738127|
|wtr_s_v_s42|40|64.061569|42.243762|1184.738127|
|wtr_s_u_s42|10|64.528340|42.958475|1182.123240|
|wtr_s_u_s42|20|65.167894|43.465236|1182.123240|
|wtr_s_u_s42|40|64.245897|42.261238|1182.123240|

D在epoch60：V64.368787%、U64.465185%，差−0.096397pp；S当前共有的epoch40：V64.061569%、U64.245897%，差−0.184327pp。单seed中途差异没有证明最终方法收益，不能按各自test峰值择优。

原日志/metadata仍保留历史best_full_test/EMA peak字符串，以保持原始记录；本报告展示全部已测milestone，不采用这些字段进行方法或checkpoint选择，80 endpoint永久保留。新正式配方需要明确自身selection协议。

原始训练日志、每个milestone完整metrics与source、成本都在[courses](courses/)。神经FLOPs与端到端/模型延迟分开：两条S课程曾共享节点，当前日志的耗时差不能直接作为受控延迟结论。

## 3. 局部Value失败的准确解释

R1和Reverse在已见数据上能拟合：Reverse P2 fit Spearman0.9495，held −0.0333。原8/8 split的fit200全部在tubelet局部位置5，held200在位置1/2，具有执行位置族外推。不能把这项诊断写成所有全局task-driven模型不可学，也不能唯一归因于packing。

P2−P1-bi regret = −0.000112806，10k视频配对CI[−0.000373990,+0.000108307]，seed方向−/+/+。P2−P0 NDCG为−0.0576263，CI[−0.118795,−0.0054698]。P2的75次最终选择中39次负收益，分母不是600个候选。反向约70σ但梯度有限、fit较好，不改称技术训练故障；cal上点估计改善也不改换primary。

6对真实交换、24次独立前向的重放/缓存标签/正反收益残差均0。反向407最大误差5.96e−8。初版同次replay自适应门的盲区已修正为独立界；800的实际零误差经独立复核，未重复干净GPU测量。4aa只修复CPU重算统计逐位相等的误拒，沿用原CUDA冻结buffers。

## 4. Graph、RISE和D/S诊断

旧G1 Static点估计优于matched Plain-L，但paired CI跨0且seed不稳。新v3 Graph primitive有技术实现，但没有被包装成新的科学PASS。GraphKV、GraphRecovery和Graph-conditioned Value是不同作用位置。

RISE导出129个seed-state的完整候选分数，独立复算公式、STOP、runner-up、margin及六项旧video指标均一致。held30行中29次候选排序改变、0次最终选择改变；17行满足位移小于top2 margin的充分不变条件。不是代码从不改选择：fit75行改变5次，cal24行改变1次。不能据held无变化就扩大β追结果。

固定raw40 D/S诊断已完成最终只读交叉核验：[DS_FIXED_FINAL_REVIEW.json](reviews/DS_FIXED_FINAL_REVIEW.json)。每轴4cal窗中3窗Value loss更低；均值V−U为D−0.0080223、S−0.0266177，并增加约2.6148864 GFLOPs。它们不是完整mAP，也不解释不同训练后参数的全部差异。2组×accumulate2中Value梯度只到router，加入Value后的检测器clip系数比接近1，没有支持大的额外缩放机制。

## 5. Atlas和Raw的边界

Atlas-S五项测量、90个预算配置、recovery与图表已齐；另行授权interaction也覆盖211视频/792窗、34416次真实前向。6336次no-op差0只说明该重放一致；“严格非零”不是统计显著比例。单token近零与分组预算机会并不矛盾，不能把99.97%近零率当可删比例。

Recovery的physical相对packed有约+3.107pp配对收益；Cross相对physical约+0.254pp且CI跨0，MD on/off接近0。它不能替GraphRecovery写成功。B的补充population/D/S/recovery仍暂停，不称全S/B矩阵完成。

Raw-v1已有先定Episode、前置preview、bounded proposal、任意合法frame-ID读取、tubelet/pack元数据和384→192→384→768恢复接口；6视频19窗技术验证及24视频mini已完成。当前R+不是全部raw域，时间hidden不是空间特征图；返回解码帧数也不是codec/GOP内部实际工作量。历史Raw mini看过完整router-holdout的4-video子集；各研究线的实际数据暴露记录必须保留。

## 6. 论文最终模型与尚存问题

正式目标是全域条件计算分配：Standard768或有成本Raw候选 → 全轴T选集 → 全栅格轻状态与可重入D/S重计算分配 → 物理恢复与TAD。K384是预算点；4-swap是旧诊断限制。空间token区域不等于Raw ROI获取，重/轻重入不等于单调early-exit。

当前主要缺口：

- Global-Task/TaskValue正式recipe与优化路径尚未接通；T已有BMCR/transport primitive，不能声称整套新模型已训练。
- S/D跨时间/pack/层预算与直接task路由梯度尚未实现；hard top-k本身不可微。
- Raw空间cheap图、区域读取/物理映射、实际解码成本与训练通路未闭合。
- Value、Graph、RISE对正式全局模型的独立增量尚无证据；旧诊断FAIL只约束旧家族。
- 新Global模型的完整80端点、匹配成本/延迟、多seed及B/ANet泛化尚待。

详细可执行规格和源代码差距见[Method与实现](METHOD_AND_IMPLEMENTATION.zh.md)。诊断、完整组件消融、正式候选、已证明最终模型分为四层；最后一层目前为空。
