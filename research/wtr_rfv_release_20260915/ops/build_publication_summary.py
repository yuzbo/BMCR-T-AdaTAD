"""Build human-facing publication indexes from saved, measured evidence."""
import json
from pathlib import Path
import shutil
from deploy_probe import LOCAL,OUT
HUB=LOCAL/'research/wtr_rfv_release_20260915'
REPO='https://github.com/yuzbo/BMCR-T-AdaTAD'
TAG='wtr-rfv-evidence-20260915'
snapshot=json.loads((HUB/'courses/SNAPSHOT.json').read_text())
manifest=json.loads((HUB/'SOURCE_MANIFEST.json').read_text())
rows=[]
summary={}
for name,row in snapshot['courses'].items():
    evaluated={int(epoch):m for epoch,m in row['metrics'].items()}
    for epoch,m in sorted(evaluated.items()):
        if (m['test_videos'],m['test_windows'])!=(211,792):raise ValueError('Incomplete full evaluation in publication table')
        rows.append(f"|{name}|{epoch}|{100*m['metrics']['average_mAP']:.6f}|{100*m['metrics']['mAP@0.7']:.6f}|{m['full_window_mean_gflops']:.6f}|")
    latest=max(evaluated)
    summary[name]=dict(training_epoch_field=row['last_training_record']['epoch'],
        successful_updates=row['last_training_record']['successful_updates'],latest_full_epoch=latest,
        latest_metrics=evaluated[latest]['metrics'],source_revision=row['source_revision'])
text=f"""# 当前进度、发现与问题

只读服务器快照：**{snapshot['recorded_at']}**。本次仅整理、审阅和发布；研究自动跟进仍暂停，未启动新训练或修改旧课程。此前停止交接保留为历史记录，此次发布请求只恢复整理/同步工作。

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
"""
text+='\n'.join(rows)
text+="""

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
"""
(HUB/'STATUS_AND_FINDINGS.zh.md').write_text(text,encoding='utf8')
(HUB/'COURSE_SUMMARY.json').write_text(json.dumps(dict(as_of=snapshot['recorded_at'],courses=summary),indent=2),encoding='utf8')
readme=f"""# WTR / RFV：研究证据与正式全域模型设计

**本次完整发布：2026-09-15。** 包含已实现代码、真实正负结果、训练日志、独立审阅、Atlas最终图集及正式模型设计。它是科研快照，**没有把诊断原型称为已验证的最终论文模型**。

|入口|内容|
|---|---|
|[当前进度、发现与问题](research/wtr_rfv_release_20260915/STATUS_AND_FINDINGS.zh.md)|Atlas/R1/Reverse/RISE/D-S/Raw的事实与边界|
|[论文Method设计与实现差距](research/wtr_rfv_release_20260915/METHOD_AND_IMPLEMENTATION.zh.md)|768/Raw自由选帧、空间token/区域、计算深度路由|
|[可视化图集](research/wtr_rfv_release_20260915/FIGURE_GALLERY.md)|最终Atlas8页、交互6页、轴内2页、RFV与课程曲线|
|[复现、来源与下载](research/wtr_rfv_release_20260915/REPRODUCE_AND_SOURCES.md)|各science SHA、源码包、数值/日志档案|
|[完整Release下载]({REPO}/releases/tag/{TAG})|源码、最终图集、完整重要数值和日志|
|[最终模型台账](research/rfv_sprint_20260915/FINAL_MODEL_LEDGER.csv)|研究层级与真实状态；所有Final仍为no|

## 当前最重要结论

- 冻结Atlas-S在有限分组空间有T/D/S分配机会；T8组CF−Uniform为+3.507pp，D8组+6.688pp，S8组+4.451pp。这是带GT和额外查询的参考，非廉价router成绩。
- 当前≤4-swap空间也有+0.528pp开发侧机会，但R1与反向一致性均未恢复未见候选泛化。
- 真实Value drift存在；历史RISE held30个seed-state最终选择未改变，Future未胜Post/EMA。
- D/S旧完整组件课程仍继续，D60 Value64.369%低于Uniform64.465%，S共同40点Value64.062%低于Uniform64.246%。快照为{snapshot['recorded_at']}，不是80 endpoint。
- 固定D/S诊断已完成交叉核验，小样本没有支持大的Value额外裁剪机制；Atlas交互没有稳定AP联合收益证据。

## 正式论文模型不是“Uniform加4次交换”

正式T应继承BMCR整轴rate→S0→条件重分配→完整exact-K支持，K384只作首预算点。S/D应在合法token状态上学习昂贵空间计算及可重入深度轨迹，逐步开放跨时间/pack/层预算。Raw自由帧/区域获取另需空间廉价图、真实读取与成本合同。

现有T采样/transport和物理Cross、D/S compact执行、Raw reader是可复用基础；全局正式recipe、S/D task路由梯度、跨组预算及Raw ROI仍有实现缺口。Policy logit与真实Value分开，Graph/RISE需自身完整任务证据。

|层级|对象|
|---|---|
|诊断/路线验证|Atlas、T-local、407D R1/Reverse、离线G1/历史RISE、Raw mini、固定D/S诊断|
|完整训练组件消融|旧D-V/U、S-V/U80轮及历史BMCR/H65|
|正式论文候选，待实现|Global-Task/TaskValue、全局S/D、全轴Graph/RISE、Raw获取|
|已证明最终模型|目前没有|

新Global家族不受旧局部pair gate一票否决；它必须通过自己的执行、梯度、完整训练和成本证据。当前研究自动跟进仍暂停，本次没有启动新实验。

![反向一致性与RISE](research/rfv_sprint_20260915/figures_reverse/rfv_reverse_rise_evidence.png)

主要源码：[物理执行与训练](h65/paper/) · [RFV诊断与学习](h65/rfv/) · [BMCR采样/transport](h65/transport.py)。
Atlas和Raw的精确独立源码身份见[来源清单](research/wtr_rfv_release_20260915/SOURCE_MANIFEST.json)，不要将整理提交回填为旧实验science SHA。

原始视频、官方/训练大权重、凭据及机器私有资源文件不随本发布上传；可复现入口、初始化来源、运行元数据、实际日志和重要数值保留。历史已公开档案仍在[原快照]({REPO}/releases/tag/wtr-snapshot-20260915)；[旧README原文](research/wtr_rfv_release_20260915/history/README.before_release.md)中的“当前”属于各历史日期。
"""
(LOCAL/'README.md').write_text(readme,encoding='utf8')
print(json.dumps(dict(status='BUILT',course_summary=summary),ensure_ascii=False))
