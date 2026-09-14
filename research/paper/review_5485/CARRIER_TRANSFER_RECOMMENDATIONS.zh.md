# Carrier-TIA设计借鉴与Full-V2主图落实

2026-09-14。用户要求重绘当前网络主图，并与方法线程讨论、向实验线程报告迁移建议。本文件区分当前实现、对方实现、已有证据与新候选；本轮没有修改训练代码、取消作业或替换既有实验矩阵。

## 1. 当前主图已重新绘制

交付在 `analysis/figures/model_full_v2_cvpr.{pdf,svg,png}`，可复现代码为 `tools/draw_full_v2_cvpr.py`，中英文图注和源码对应见 `analysis/figures/model_full_v2_cvpr.caption.md`。

主干保持单向横排；Scout上下文有独立上方通道；K与D/S控制使用紫色虚线并接到相应模块端口；每条路径只有一个末端箭头。两个残差加法各自使用局部回路，放大图由面板(b)/(c)索引关联。训练监督单列，不画成推理时运行的carrier。PDF按7.1英寸双栏宽度导出，并使用Poppler渲染检查。

实验线程确认当前科学结构相对da26af3未变，运行模型版本为db9c749。当前主图正确保留：H65/BMCR单帧选择、前4层dense、原block ID [4,6,8,10]路由、最终block dense、packed-domain full-KV、depth-light和spatial-light、selected-support TIA、末端Cross恢复384原轴以及TAD768位置读出。当前使用既有R03 Cross初始化；官方VideoMAE decoder是另一组消融。

## 2. 已与方法线程确认的真实设计

方法线程：`01a084a0-88c4-7730-9c2b-e351aaef33c9`，名称“实现官方 OpenTAD V3 路线”。

- carrier为 `[B,384,5,5,128]`，保留完整原时间轴和5×5空间网格，不能简写后据此按纯时间向量估算成本。
- 12层各自有独立的 `P_l`：LayerNorm(128) → Linear(128, TIA_dim) → GELU；VideoMAE-B的TIA_dim为192，S为96。
- Proxy在低分辨率carrier上投影，再由5×5插值到10×10。重计算状态经官方TIA down投影后，在精确native_id处替换Proxy。
- TIA在原时间轴的10×10低维网格上执行；选中位置经原Up返回heavy状态；全轴结果先池化到5×5，再经Q回写carrier。
- 当前Q末Linear为零初始化。官方TIA Up零初始化、gamma为1；不能同时将新增输出映射和乘法gate都设为0，造成梯度锁死。训练后的旧Up不应被重新归零。
- 当前默认最终读出是精确native_id的硬 `index_copy` selected replacement。软门控、锚点一致性、只在L6/9/12耦合并保留Cross，都是我们提出的迁移候选，并非对方现实现。
- 对方明确：尚无“同support、同预算、同配方”的逐层carrier与我们的末端Cross已完成直接对照。其U-prog与U-static比较不能替代该证据。

已点验源码：

- `C:/Users/skywalker/Documents/ChatGPT/refine-tad/V3/OpenTAD/opentad/models/bricks/bcr_carrier.py:21`：全帧cheap CNN、相邻两帧聚合、carrier网格。
- 同文件`:65`：P/Q、空间resize/pool、插入真实状态、原轴TIA和carrier残差更新。
- `C:/Users/skywalker/Documents/ChatGPT/refine-tad/V3/OpenTAD/opentad/models/backbones/progressive_carrier_vit.py:60`：每层独立CarrierLayer。
- 同文件`:152`：heavy/TIA双向更新；`:163`：最终carrier预测与真实selected feature回写。

## 3. 最值得吸收的三个思想

### A. 可靠的已观测特征应受到保护

当前Cross对所有原轴查询输出“插值底座＋残差”，不会强制保留可靠的重计算anchor。对方的selected replacement提出了一个有价值的约束：恢复未观测位置时，不应无根据地覆盖已得到的有效证据。

迁移应先采用**锚点一致性或质量条件软门控**，而非照搬硬索引替换。H65/BMCR的不规则单帧打包可能把非相邻原始帧组成一个tubelet；一个anchor表示两个贡献帧及其时间跨度，不一定等于某个原轴query。需使用现有 `contributor_times / anchor_centers / query_centers / valid` 进行对齐。同理，深度/空间稀疏后的anchor也是近似状态，不能一律当作无误差真值。

建议检查恢复前后，在高质量anchor、较差anchor、缺失区、边界区上的状态误差和检测变化。若在anchor处做一致性，明确目标detach与质量权重；若做输出门控，保留Cross为缺失区和低质量anchor纠错的能力。

### B. 完整时间状态可随深度更新，而不只在末端补齐

当前Full-V2内部TIA只见选中时间支持；未选位置主要在末端解码。逐层carrier则让未选位置的低维状态持续变化，并接受重计算证据纠正。这是新的建模假设，不能由当前Cross成绩背书。

建议候选先保持原H65/BMCR单帧支持、物理时间元数据与Cross起点，在**L6/9/12**进行原轴低维状态更新和heavy correction，而非直接复制全部12层架构。该选择复用现有多层特征接口，是一种有明确成本边界的新完整模型候选。

要区分两种状态：`[B,T0,Cc]`仅时间carrier，和`[B,T0,5,5,Cc]`空间carrier。前者便宜但不能直接复用对方的空间网格TIA接口；后者才对应对方设计。二者不得用同一名称、同一FLOPs估算或同一效果解释混在一起。当前96D Scout已经池化，不能仅通过expand制造一个有空间信息的5×5carrier；若迁移空间carrier，需要明确的空间cheap特征来源并计入成本。

### C. 用真实计算纠正预测状态，保留清楚的残差接口

值得借鉴的是 `Proxy → real correction → temporal interaction → residual writeback` 的接口约束。它能够把“未观察到的时间位置”与“已观察但少算深度/空间的位置”都表述为有待纠正的状态，但二者的support和监督必须分开。

新增接口应在初始化时保留已训练Full-V2函数：保留原selected-support TIA，把新全轴纠错作为额外残差；仅新增残差映射的最后Linear为0，乘法gate保持非零。Cross的新carrier输入投影也采用同样边界。**直接把旧TIA改为全轴TIA，再仅把Q置零，并不保证旧函数不变**，因为已经训练过的旧Up仍可能输出不同结果。

本建议保护训练起点，不能保证最终mAP。整个模型仍需完整课程比较。

## 4. 已有内容与不能直接照搬的内容

- 低成本Scout辅助选择、路由输入stop-gradient、low/high维投影、残差更新等，在当前路线已有部分对应；不应作为新贡献重复命名。`h65/paper/routing.py:29`的budget context与`:102`的frame features已detach。
- 当前Full-V2的depth-light已处理选中支持上绕过token的状态更新；carrier进一步针对的是未选时间位置的状态演化，不能把两者的效果混称。
- 对方精确native_id基于连续原clip保持了tubelet对应；不能搬到不规则单帧重打包后仍用整数索引替换。无需恢复整clip选择，但需要显式物理时间的lift/gather和有效性处理。
- 保留原索引的注意力域不等于恢复了原时间语义；空间resize/pool也会带来信息损失。低维carrier不是零成本，也不是自动改善所有边界的保证。
- H65/BMCR/Cross与Carrier-TIA均属于用户项目内部路线，不列作独立公开论文方法。

## 5. 面向实验线程的实现建议

建议作为既有P0课程之外的有限候选，不替换正在运行的Full-V1/V2、Uniform、PBD/Static。正式完整课程仍为seed42、80epoch；不存在以早期mAP输赢停止的科学gate。

可并行准备三个完整答案，共用既有Full-V2参考：

1. **Anchor-consistent Cross**：只加可靠anchor保护；回答性能损失是否部分来自恢复器覆盖了有效证据。
2. **Sparse progressive carrier + Cross**：在L6/9/12建立原轴状态与新残差接口；回答中间时间状态维护是否值得其成本。
3. **Carrier + anchor consistency**：联合两项；回答状态演化与最终证据保护是否互补。

三者是建议中的新候选，未在本轮声称注册/提交。主线结果与这些候选不必存在成绩先后依赖；排队次序仅取决于资源。

代码边界可限定为：`geometry.py`增加基于实际贡献时间的lift/gather；`encoder.py/engine.py`接入少数层carrier纠错；`decoder.py`加入质量条件一致性/新carrier输入；`objectives.py/support_targets.py`区分选中support与原轴目标；`profile.py/evaluation.py`记录真实成本、carrier/gate使用与边界误差。不要把这些迁移改动悄悄加入当前Full-V2配置。

建议新增数据字段：carrier grid及更新层；P/Q投影、全轴TIA和cheap空间前端的完整计算；spatial resize/pool/gather/scatter的操作与访存耗时；不同gate的实际幅度；选中/缺失位置与边界区的状态误差；完整模型mAP和总GFLOPs。总计算必须包含所有已执行部分；不只报告heavy backbone，延迟/显存报告但不决定路线。

## 6. 当前证据尤其要求固定预算核验

实验线程2026-09-14 12:36快照：Full-V2-S/B epoch10完整211视频/792窗mAP分别64.2351/68.3708%。S全部选择K384_D100_S75；B全部D100、S比例不同。因此该结果没有展示动态深度跳过带来的省算，且完整候选成本不同，不能用来单独认定depth-light、full-KV或三轴动态有效。

现有固定计划、T/TD/TS/TDS和机制矩阵应继续。新carrier候选除动态预算整体结果外，也应在相同选帧/同一非平凡D/S预算上比较；训练支持和外部teacher历史匹配，完整实际GFLOPs可比。对非平凡深度预算的固定checkpoint诊断可以复用当前工具；独立训练比较仍需自己的完整课程。不能将D100、warm-start历史或不同支持造成的差异归为carrier收益。

来源：`monitor_20260914_1235/UPDATE.zh.md:3–28`。论文主图只展示结构，不将这些中间结果写成已证明的机理或最终论文结论。

## 7. 讨论与同步记录

- 已向方法线程发送源码/证据问题及三项迁移判断，并收到其对真实空间grid、独立P_l、零初始化及缺失直接对照的答复；已据源码点验。
- 已向实验线程询问现实现版本，并收到结构不变、当前预算D100和同轮次结果的澄清。
- 方法线程的完整短答将另存 `CARRIER_PEER_REPLY.zh.md`；最终建议以本文件和该回复共同留档。
