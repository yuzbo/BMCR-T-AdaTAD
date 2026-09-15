# Raw-v1进一步思考：输入接口基线与接受意见

日期：2026-09-15。本轮完整阅读并保存两份进一步评述，核对了Scout及几何辅助代码。只更新文档，没有修改数据管线、运行实验、训练或部署。

**判断：两份补充整体合理，实质提高了Raw-v1的可实施性。** 可以采用它们共同形成的输入接口基线，但需要限定Raw候选池的比较身份、tubelet约束的地位，以及共同preview四格实验能支持的结论。修订这些局部定义即可，不需要继续扩大主架构。

## 1. 原文记录

- [预览、有限proposal、tubelet兼容性与三坐标评述](C:/Users/skywalker/Documents/ChatGPT/H65/reports/wtr_raw_input_20260915/inputs/03_Preview_Proposal_Tubelet_Review.txt)：原附件 `d441b3b0-46c3-4d4c-a623-1c4f75c3c83e/pasted-text.txt`。
- [Episode冻结、Raw-Uniform与四格归因评述](C:/Users/skywalker/Documents/ChatGPT/H65/reports/wtr_raw_input_20260915/inputs/04_Episode_Factorial_Coordinates_Review.txt)：原附件 `a6a5d549-4f25-4dbe-b2c5-2e4aa6ac4af3/pasted-text.txt`。

此前[Raw输入报告](C:/Users/skywalker/Documents/ChatGPT/H65/reports/wtr_raw_input_20260915/RAW_INPUT.zh.md)及最初两份输入仍保留。本文件补充其接口，不将附件中“开始实现”的建议解释为本轮执行授权。

## 2. 本轮真正增加的完整性

|补充|为何需要|
|---|---|
|CheapPreviewReader|现Scout的64×64缩放发生在完整候选RGB已进入模型后；Raw需要把cheap读取前移|
|先确定episode|改变候选网格后重用旧random_trunc，即使seed相同也未必对应相同物理窗口|
|有界raw proposals|低密度preview提供期望收益线索，不支持零成本枚举并识别任意原始帧内容|
|显式TubeletBuilder|选帧与两帧tubelet/16观察pack的消费方式分开记录，便于归因|
|共同cheap scaffold|候选域比较不能同时混入不同preview内容、Scout权重或可见视野|
|三坐标独立投影|Scout辅助GT、heavy来源、detector GT不再通过一个长度比例互相推导|
|四格交互分析|除了两个最终分数，还能测候选域对Value增益的影响|

## 3. 可采用的Raw-v1默认链条

`EpisodeDescriptor → CheapPreviewReader → CheapPhysicalTimeline → BoundedRawProposalGenerator → Uniform/Value Allocator → RawSelection → HeavyRGBReader → TubeletBuilder → VideoMAE＋固定D/S → 同一物理query恢复/head`。

Raw作为输入获取扩展，继续保留WTR-Standard主证据线。首轮固定budget和局部D/S策略，Graph、动态预算、FVD、时间编码和micro-clip不同时新增。

## 4. Episode必须先于候选获取确定

先固定物理episode起止、可见视野、query时间、原始GT与裁剪后的GT/边界有效性，再分别构造Official-grid或Raw候选。选择新的grid不能反向改变本次crop/window。

测试使用固定完整window清单。训练则可以预先生成、按epoch/step重放共同的episode及augmentation序列；“冻结manifest”不意味着把每个视频永久锁死到同一个训练crop。保持原训练采样分布与跨epoch变化，同时使比较条件一一对应。

空间crop、resize参数与颜色/归一化约定也属于共同数据条件。相同随机种子只有在数据处理链一致时才可能产生相同结果，不能替代episode/augmentation身份记录。

## 5. 真正前置的cheap reader与Scout适配

代码核对：`h65/full/scout.py:20–24` 对传入的完整RGB展开并缩小；`h65/scout.py:38–48` 默认spatial_size=64，空间stem经AdaptiveAvgPool2d(1)再进入temporal网络。因此可以复用Scout架构与初始化，但现forward本身不能代替前置cheap reader。

`h65/scout.py:23–35` 的transition descriptors使用相邻logits、entropy和hidden差分。Preview cadence改变后，这些特征的含义与分布也会改变。旧checkpoint可作为初始化/直接迁移控制，不应未经验证就认作相同模型行为；但也不预先断言必须全量重训。

首轮因果比较建议采用共同preview内容与共同冻结Scout表示；若需要适配，先在训练/开发资料上形成一个共同适配版本，再用于四格。独立训练四个系统时，即使初始权重相同，最终cheap features也可能不同，应标为适配后整体效果。

Official-grid条件一旦使用新共同preview，就应命名为“共同preview下的官方网格控制”，不是原始官方AdaTAD或未经改变的H65。原始Standard reference另保留。

## 6. Raw候选池的定义与必要修订

可以采用cheap时刻加少量物理offset，再映射到合法原始帧。首轮offset可按固定几何规则定义，Value只学习分配身份；若以后学习offset本身，需要单独说明其监督，不能指望nearest-frame硬映射自动产生精确选择梯度。

候选经映射、边界处理和去重后，数量是 `Nunique≤Npreview×M`，并非恒等。实际独立frame IDs、原协议重复观察槽位、padding和有效预算计数分开记录，不悄悄补帧满足表面K。

**有限raw proposal不必然包含官方网格。** 如果要把对照解释为“在原域上增加off-grid选择”，可以在该诊断中令Raw池显式包含原有合法候选，再加入有界off-grid proposals；索引生成不要求先解码这些heavy RGB。

若采用完全独立的raw proposal池，也合理，但结论应叫“替换候选生成机制”，不能仅凭候选来自原视频就称为严格扩域。同一域中的Uniform和Value必须共用相同候选池；否则selector因素也混入了proposal变化。

候选池大小和评分/提议开销进入完整成本账本。更多候选或更自由时间位置不能被视为免费收益。

## 7. Raw-Uniform应定义在物理时间上

先在同一episode物理时间轴生成等距目标时刻，再按固定规则映射到该域的合法帧，并固定去重、覆盖与有效性处理。

“时间分位”应指物理区间的等分，而不是简单按非均匀proposal列表的序号取分位。后者可能继承proposal密度，不能再叫物理均匀。

在当前CFR/固定stride条件下，Raw-Uniform与Official-Uniform可能非常接近；这不是实验失败。应保存实际frame IDs、times、off-grid比例和span，先判断两者到底是否形成了不同观察。

## 8. TubeletBuilder：显式打包，不是隐式纠错器

接受TubeletBuilder作为正式接口，但首轮默认仍可采用**按时间排序的逐帧selection、现有有序两帧tubelet和16-observation packing，并记录真实span**。大span是否伤害任务是研究变量；定义接口不意味着必须先施加max-gap。

有序contributor pair、native tubelet ID、pack ID、时间中心/跨度和validity均明确保存。仅一个center不足以描述两个来源；帧顺序也不能被默认为可随意交换。

`|t₂i+1−t₂i|≤Δmax`是额外的合法集合约束，不是VideoMAE能够执行的数学必要条件。它可能排除有效远距组合或改变覆盖，应作为独立机制对照，不能只在Raw条件加入后把差值全部归因于off-grid。

Builder不能在选择后静默增加邻帧、重复/替换真实观察或改为local clip。任何修复如改变实际selection，都必须纳入动作定义、真实标签、预算与来源记录。

每次swap后，应按同一确定性规则重建相关pair/pack及映射。移除和插入会改变区间内的排序配对，不能只更新被替换的一个frame条目。完整重执行的T交换收益应包含这种真实影响。

Micro-clip、regular stride-4 clip、连续raw16帧仍是独立动作空间研究，不自动恢复任何旧课程，也不替代本轮逐帧Raw-v1。

## 9. 三套坐标与GT投影

显式保留 `Ipreview/τpreview`、`Iheavy/τheavy`、`Iquery/τquery`，以及episode offset、valid masks、空间变换和native tubelet/anchor contributor映射。

Scout action/boundary辅助target从物理GT单独投影到preview时间轴；heavy观察不需要一个等长GT栅格；最终TAD target投影到原detector query。原始边界与由crop产生的端点保持区分。

代码核对：`h65/paper/geometry.py:6–9` 的candidate_mask依赖RGB轴与detector轴整数比并repeat_interleave；`:12–15` 的feature_target_data将GT乘以该比例；`:38–41` 的scout_context还按相邻两项聚合。这些固定关系不能原样充当Raw三坐标转换，相关辅助路径需在实施时同步更换。

保存frame ID、秒单位时间、时间来源和normalized episode time。当前数据若使用已验证的dataset fps派生时间，可以继续声明该口径；扩展VFR时再使用并验证decoder PTS/time-base映射到benchmark时间原点，不把frame_id/fps永久视为所有视频的精确时间。

## 10. 四格实验能说明什么

令O/R表示official-grid/raw候选机制，U/V表示uniform/value，在共同episode与明确cheap条件下得到全数据mAP：

- `mAP(R,U)−mAP(O,U)`：uniform选择下的候选机制效应。
- `mAP(O,V)−mAP(O,U)`：官方网格内的Value增量。
- `Δinteraction=[mAP(R,V)−mAP(R,U)]−[mAP(O,V)−mAP(O,U)]`：所选协议和指标尺度上的候选机制×selector交互。

这是合理的因子分析，但正交互不自动证明唯一原因是off-grid evidence，也不等于组合最优headroom。需要候选池、preview、packing、训练状态和成本条件的对应；冻结同checkpoint的推理干预与独立匹配训练分别报告。

四个条件联合按video-cluster重采样，每次重新计算四组全数据AP及交互值；不以窗口为独立单位，也不简单拼接四个独立CI。相同heavy K只匹配部分成本，完整neural成本仍逐组报告，需要同成本主张时使用合法预算匹配，不用无用计算填账。

新的共同preview四格不取代原始Standard基线，也不允许把训练集pilot重标为全211视频/792窗口证据。

## 11. 时机和成本边界继续保留

Raw-v1控制的是：episode先确定、cheap先读取、heavy选择后读取。模型使用cheap上下文预测未heavy观察位置的期望收益，不承诺识别所有未观察信息。

低频/固定数量preview的盲区需测量，区域配额不能保证短动作无遗漏。神经计算、实际decoded frames/pixels、CPU decode、transfer、GPU latency与端到端时间分别报告。预解码-cache仅用于算法验证；按需reader与codec/GOP成本另行验证，不换算成未定义的“GFLOPs等价时间”。

同一Cencoder已经计入routed attention/FFN/TIA后，不再加第二份D/S成本。Graph、FVD、动态预算、temporal PE以及整视频THUMOS继续后置或独立比较。

## 12. 当前接受范围

**接受为Raw-v1接口基线：** 先固定episode；前置CheapPreviewReader；共同cheap scaffold；有限可复现raw proposal；逐帧selection；显式tubelet消费合同；三坐标和GT独立投影；全数据受控因子评测。

**修订后接受：** 候选数量写上界；Raw是否包含Official写清；Uniform按物理时间定义；共同preview下的Official控制正确命名；配对约束与时间编码保持独立变量。

**不作已证结论：** Raw必然更好、max-gap必需、旧Scout必须失效、正四格交互唯一来自off-grid，以及接口文档已经等于完整实现。

这一轮可以把研究接口收敛下来。实现时补具体参数和数据字段即可，不需要在此基础上继续叠加新的主模型。已有Standard证据线与Raw分支顺序不变。

本轮已保存两份新原文和接受意见，没有开始实现、训练或部署。
