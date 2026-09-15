# WTR / RFV：研究证据与正式全域模型设计

**本次完整发布：2026-09-15。** 包含已实现代码、真实正负结果、训练日志、独立审阅、Atlas最终图集及正式模型设计。它是科研快照，**没有把诊断原型称为已验证的最终论文模型**。

|入口|内容|
|---|---|
|[当前进度、发现与问题](research/wtr_rfv_release_20260915/STATUS_AND_FINDINGS.zh.md)|Atlas/R1/Reverse/RISE/D-S/Raw的事实与边界|
|[论文Method设计与实现差距](research/wtr_rfv_release_20260915/METHOD_AND_IMPLEMENTATION.zh.md)|768/Raw自由选帧、空间token/区域、计算深度路由|
|[可视化图集](research/wtr_rfv_release_20260915/FIGURE_GALLERY.md)|最终Atlas8页、交互6页、轴内2页、RFV与课程曲线|
|[复现、来源与下载](research/wtr_rfv_release_20260915/REPRODUCE_AND_SOURCES.md)|各science SHA、源码包、数值/日志档案|
|[完整Release下载](https://github.com/yuzbo/BMCR-T-AdaTAD/releases/tag/wtr-rfv-evidence-20260915)|源码、最终图集、完整重要数值和日志|
|[最终模型台账](research/rfv_sprint_20260915/FINAL_MODEL_LEDGER.csv)|研究层级与真实状态；所有Final仍为no|

## 当前最重要结论

- 冻结Atlas-S在有限分组空间有T/D/S分配机会；T8组CF−Uniform为+3.507pp，D8组+6.688pp，S8组+4.451pp。这是带GT和额外查询的参考，非廉价router成绩。
- 当前≤4-swap空间也有+0.528pp开发侧机会，但R1与反向一致性均未恢复未见候选泛化。
- 真实Value drift存在；历史RISE held30个seed-state最终选择未改变，Future未胜Post/EMA。
- D/S旧完整组件课程仍继续，D60 Value64.369%低于Uniform64.465%，S共同40点Value64.062%低于Uniform64.246%。快照为2026-09-15T20:45:20+0800，不是80 endpoint。
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

原始视频、官方/训练大权重、凭据及机器私有资源文件不随本发布上传；可复现入口、初始化来源、运行元数据、实际日志和重要数值保留。历史已公开档案仍在[原快照](https://github.com/yuzbo/BMCR-T-AdaTAD/releases/tag/wtr-snapshot-20260915)；[旧README原文](research/wtr_rfv_release_20260915/history/README.before_release.md)中的“当前”属于各历史日期。
