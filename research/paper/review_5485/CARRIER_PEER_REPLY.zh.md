# Progressive Carrier-TIA：内部实现与Full-V2迁移短答

2026-09-14。仅代码/既有结果讨论，没有修改模型、运行新训练或取消作业。Full-V2部分以绘图/实验线程明确给出的当前定义为讨论前提；本回复没有独立复核该工作树的全部实现。**PVR、U-static/U-prog、E1均是本项目内部方法/消融，不是外部公开基线。**

**结论：认同“保护强anchors、少数晚层coupling、全部计费”的候选顺序；没有现成证据证明逐层carrier优于当前Full-V2末端Cross。** 实验线程确认Full-V2 epoch10测试均为D100，本回复不预设深度节省或三轴收益。新增carrier应标为待比较候选，不并入已实现主图或已验证结果。

## 1. 当前PVR真正实现的是什么

输入是768个采样RGB帧、160×160；这里的时间轴是输入采样轴，物理时间另由frame_inds/metas保留，不等同于原视频全部连续帧。

下表的零初始化指从识别初始化构建模型时的初值，不是训练后参数值。从已训练Full-V2迁移时，**不重置现有TIA、Up或其他旧权重**。

| 部件 | 确切实现/初始化 |
|---|---|
| C0及各层carrier | **[B,384,5,5,128]，保留空间grid，不是[B,384,128]向量序列。** MobileNetV3-Small ImageNet前缀features[:9]在80×80 RGB上输出48×5×5；每两个输入帧平均，对齐tubelet=2，再Linear48→128、两层DWConv1d(k3)+Linear残差。CNN可训练，BN running stats冻结、affine仍训练；这两个新增时间残差块没有额外激活 |
| P_l / Proxy | **12层各自独立，不共享**；每层LN128→Linear128→r→GELU，再bilinear 5×5→10×10。P没有专门的零初始化。B的r=192，S的r=96，不能把S也画成192 |
| heavy低维输入 | 原ViT attention+MLP之后，原TIA Down(D→r)+GELU；没有新加的heavy侧前置LN |
| Insert / TIA | 用native_ids将已计算real**硬写回**proxy：u=index_copy(proxy, real)。在完整native时间384、每个10×10空间cell上执行原DWConv(k3)→PWConv(1)，v=u+PW(DW(u)) |
| Q反馈到carrier | Q_l=LN(r)→Linear(r→128)，**仅末Linear的weight/bias置0**；先将v空间pool10×10→5×5，再C_l=C_{l-1}+Q_l(pool(v))。无第二个零门控 |
| Up反馈到heavy | Gather(v,native_ids)后复用原TIA Up(r→D)，再乘gamma加回heavy。原Up初始化0、gamma初始化1；Down截断正态std=.02/bias0，DW/PW按源码非零初始化 |
| final selected replacement | base=Linear128→D(mean_hw(LN(C_L)))；selected处直接index_copy最终heavy空间均值，unselected处保留base。是精确写回，不是软混合。之后按原linear/align_corners=False到detector_length；THUMOS为768，ANet为192。保留官方heavy.norm路径，不另加fc_norm |

关键源码：

- [CheapEncoder与P/Q、空间读写](C:/Users/skywalker/Documents/ChatGPT/refine-tad/V3/OpenTAD/opentad/models/bricks/bcr_carrier.py:21)，[CarrierLayer](C:/Users/skywalker/Documents/ChatGPT/refine-tad/V3/OpenTAD/opentad/models/bricks/bcr_carrier.py:65)。
- [12层独立ModuleList](C:/Users/skywalker/Documents/ChatGPT/refine-tad/V3/OpenTAD/opentad/models/backbones/progressive_carrier_vit.py:60)，[原轴索引、逐层执行和Up](C:/Users/skywalker/Documents/ChatGPT/refine-tad/V3/OpenTAD/opentad/models/backbones/progressive_carrier_vit.py:136)，[最终写回](C:/Users/skywalker/Documents/ChatGPT/refine-tad/V3/OpenTAD/opentad/models/backbones/progressive_carrier_vit.py:162)。
- [官方TIA初始化与算子](C:/Users/skywalker/Documents/ChatGPT/refine-tad/V3/OpenTAD/opentad/models/backbones/vit_adapter.py:19)。

梯度边界：ViT原权重冻结，但激活图保留，TIA及cheap/P/Q/输出可训练。固定选择mask时，重分支real、完整低维TIA、unselected proxy和Q可接收检测梯度；**被index_copy覆盖位置的proxy对该写回输出梯度为0，selected中间carrier节点也没有直接检测损失路径**，不能画成每个位置都受到相同检测监督。共享P/Q仍可从unselected位置学习。Q的末层零初始化使C初始为identity残差，但Q末层本身可从非零输入和最终读出获得梯度，不是整个carrier被冻结。

这个零梯度结论针对相应carrier/proxy节点，不等于selected RGB对cheap参数没有检测路径：cheap时间邻域和共享参数仍可贡献，视觉辅助也会监督selected C0。

原Router读取detach(C0)，这是选择器输入边界，**不是carrier融合路径detach**。[Router](C:/Users/skywalker/Documents/ChatGPT/refine-tad/V3/OpenTAD/opentad/models/bricks/bcr_router.py:104)。视觉辅助监督用已计算heavy的detach目标训练独立A_l(C0)/A_out，并不直接监督实际P_l(C_{l-1})；A/D在部署导出时移除。[视觉读出](C:/Users/skywalker/Documents/ChatGPT/refine-tad/V3/OpenTAD/opentad/models/bricks/bcr_supervision.py:47)、[训练/导出边界](C:/Users/skywalker/Documents/ChatGPT/refine-tad/V3/OpenTAD/opentad/models/detectors/bcr_actionformer.py:44)。

当前状态：这些PVR组件已有完整THUMOS单seed实现/训练和导出记录；相关旧实验已完成，目前没有本任务活跃GPU作业。**迁入Full-V2的carrier候选尚未实施或取得正式精度/总FLOPs**，不能把PVR已有部署当作Full-V2已部署。

## 2. 有什么可靠结果，缺什么结果

同为完整THUMOS200训练/211测试、seed42、60轮epoch59 EMA、K16的内部结构消融：

| 内部方法 | B平均mAP | S平均mAP |
|---|---:|---:|
| U-static | 56.1546 | 53.1331 |
| U-prog | 59.4518 | 55.6313 |

U-prog分别高约3.30/2.50点；B的carrier-only为54.4705，低于U-prog约4.98点。这支持原PVR协议中progressive更新和selected replacement具有单seed正向证据，**不证明对Full-V2的Cross更优，也不是多seed稳定性结论**。[B结果](C:/Users/skywalker/Documents/ChatGPT/refine-tad/V3/OpenTAD/reports/status_checks/verified_results.json:455)、[S结果](C:/Users/skywalker/Documents/ChatGPT/refine-tad/V3/OpenTAD/reports/status_checks/verified_results.json:332)。

E1在已有progressive架构内只比较base/heavy证据源，K16的heavy−base为B+1.4224、S+2.1051点，视频配对95%区间均正；这支持内部强证据恢复设计，仍不是“逐层carrier vs仅末端Cross”。[完整E1/E2终点和边界](C:/Users/skywalker/Documents/ChatGPT/refine-tad/V3/RESEARCH_STAGE_FINAL_RESULTS_20260913.md:5)。

**缺失：同一Full-V2骨干、同单帧support/物理时间映射、同实际深度/空间执行与预算、同训练配方/初始化下，Cross-only与late/full progressive carrier的完整对照。** 当前最强内部控制E1-heavy K20 B63.0069/S59.2391也仍未达到原64.026/62.127目标；不能用整体未达标反推每个部件无效，亦不能据局部增益预设迁移成功。

## 3. 保持单帧选择时怎样迁移

1. **先保护可信heavy anchors与当前Cross起点。** 当前PVR能硬替换，是因为完整16f块保留了原tubelet/native_id一一对应。Full-V2的不规则单帧pack中，一枚tubelet可能来自两个不相邻的原采样帧，不能直接按packed索引硬写原384轴。可先采用来源独立映射、残差强度控制或一致性约束；这是新迁移方案，不是PVR已经采用软门控。若Cross已有同等保护则先复用，不叠加同构模块。
2. **L6/9/12少数coupling是合理的首个候选。** 保持当前选帧、原backbone/Cross/检测器主路径，另维护固定原轴低维状态，注入已计算heavy的新信息。它不复刻全部12层PVR，增益需重新验证。对于已训练Full-V2，保留旧selected-support TIA，把新full-axis correction作为额外残差；新分支最后输出Linear置零、gate初值非零，不重置旧Up。若Cross新增投影读取完整非零C_L，零投影可以先学输出映射、再逐步训练上游Q；但若读取的是(C_L−C0)，同时Q末层和该输出映射都为零，则差分输入和读出两端均为零，会锁住此支路。不能笼统说“所有新映射都置零即可”。同RGB/support/旧权重下还应验证初始化输出一致；仅Q=0不能保证直接把旧TIA替换成全原轴TIA后heavy输出不变。
3. **显式定义tubelet和空间支持映射。** 保留每个packed tubelet对应的两个frame ID、各自物理时间及跨度；以这个支持定义scatter/splat及gather，不能把相同通道数当成坐标等价。中心时间插值或把一枚tubelet贡献分配到两个原native位置都改变了算子，应标清新定义。若保留空间carrier，heavy/light token须先按原patch坐标恢复到对应网格并携带有效性/来源mask；不能对不规则packed patch列表直接做5×5/10×10 resize。当前PVR的双线性resize成立的前提是两路来自同一增强RGB的规则网格。

用于主图：当前PVR画出**空间5×5、逐层独立P/Q、完整384时间TIA、selected gather与最终硬写回**。Full-V2新增L6/9/12路径用候选/消融标注；D100结果不作为深度稀疏收益证据。若迁移改为[B,384,128]向量carrier，应明确这是去掉空间grid的新变体，不能继续引用原PVR结构或算量。

## 4. 额外总FLOPs与风险

**迁移版本额外总网络FLOPs：MISSING（未实现/实测），不填伪精确值。** 以下仅为确定形状下的算式量级，不是Full-V2新profile：3个coupling点、T_original_native=384、carrier grid=5×5/128D、TIA grid=10×10、2 FLOPs/MAC，B r=192/D=768，S r=96/D=384。

| 新增主体项 | B GFLOPs | S GFLOPs |
|---|---:|---:|
| 三层P+Q线性部分：3×2×2×384×25×128×r | 2.8311552 | 1.4155776 |
| 三次额外完整原轴DW3+PW：3×2×384×100×(3r+r²) | 8.6261760 | 2.1897216 |
| 上述小计 | 11.4573312 | 3.6052992 |

该例保留现有selected-support TIA，因此没有扣旧TIA；若改变为替换方案，必须明确减去实际被移除的计算。**小计还不是总增量**：需加入/核实cheap全轴状态来源、所有LN/GELU、pool/resize、坐标读写/权重、门控、最终carrier读出以及新增feedback。若不复用既有Up而另外计算三次独立r→D Up，在T_packed_native=192、100空间cell的示例下还增加B16.9869312G/S4.2467328G；Down若重复也要计费。实际空间/深度支持改变时按真实shape重新计数，不沿用本例。

最终应报告C_new=C_current_FullV2+所有新增算子−确实移除的算子，并与各自官方全网络边界对齐。空间grid乘数、新的cheap encoder、dense时间TIA和重复Up都可能吃掉预算余量；不能仅因“128D”宣称便宜。还有三类主要风险：弱状态的任务语义不足；不规则tubelet/空间来源对齐失真；新增反馈改变已训练heavy与Cross分布。已有E1收益只支持优先检验强证据保护，不消除这些风险。

因此本次只支持小范围、单因素、同support的候选验证；不主张立即全12层迁移，不取消现有课程，也不预设超过Cross。
