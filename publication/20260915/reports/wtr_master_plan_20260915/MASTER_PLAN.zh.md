# Where to Refine：完整路线与最终方案

> 最新用户授权已转入双线实施：Batch-1A为Atlas/Core scientific gate，Batch-1B为Raw Evidence Acquisition Prototype。以[双线交接与硬规则](HANDOFF_DUAL_BATCH1_20260915.md)覆盖下文旧单线顺序。现有Atlas保持，Raw由独立实施任务负责；本文件以下仍保留原报告时间点的状态。

日期：2026-09-15。当前用户要求为“完整整理目标设计和最终方案，并推荐报告第一批模型实现和部署”。本文为推荐方案，不是实施或部署回执。本轮仅阅读、归档、核对代码版本及只读查询资源，没有修改模型/配置、运行新实验或提交作业。

首批具体工作见[FIRST_BATCH.zh.md](FIRST_BATCH.zh.md)。

## 1. 最终目标

**在给定证据、预算、执行历史和后续策略下，学习哪一次观察获取或内部重计算值得支付成本，同时维持TAD所需的物理时间定位表示。**

目标不是证明所有视频都冗余，也不是要求T/S/D、Graph、FVD全部同时有效。最终结果应体现：完整检测精度与实际计算之间的改善，超过强Uniform/Static控制，并有能够解释收益来源的实验。

统一价值定义为：

$$
\mathbf V^\pi_\theta(a\mid s)=\mathbb E[\boldsymbol\ell_\theta(s;\pi)-\boldsymbol\ell_\theta(s\oplus a;\pi)\mid s,a],\qquad
\boldsymbol\ell=[L_{cls},L_{loc}].
$$

训练用真实完整重执行测量收益；推理只使用可见证据预测收益，不读取GT、不执行反事实teacher。boundary、actionness、motion、uncertainty、Graph边权及重建误差均不是价值定义本身。

## 2. 当前实际进度

### 已验证的基础

- 独立Atlas工作树已存在，正式全数据测量已部署。协议覆盖THUMOS14的211测试视频、792窗口；训练/development与publication记录明确分开。
- Dense-S官方EMA严格加载499键；完整评测Avg-mAP为68.9767%，mAP@0.7为48.2622%，平均2347.8940 GFLOPs/窗口，处于既定复现技术容差内。
- S完整开发视频的技术验证、官方AP缓存复算、V2-S重构与同支持恢复预检已通过。
- Atlas尚不是新WTR训练：队列明确记录training_updates=0。其publication/test数据不能转为新Value Head训练标签。

### 本轮只读队列快照

截至**2026-09-15 01:47:56 +0800**：8个阶段COMPLETED、2个RUNNING、20个WAITING。GPU0运行allocation_s_T，GPU1运行baseline_b；不能把某一瞬间GPU利用率低当成可抢占的空闲资源。

回执：[队列原始快照](receipts/current_queue_snapshot.json)、[资产/GPU/版本快照](receipts/asset_gpu_revision_snapshot.txt)、[核对摘要](receipts/current_resource_snapshot.json)。这些是带时间的快照，不保证阅读本报告时状态仍相同。

### 代码版本

|对象|核对结果|
|---|---|
|研究模型基线|1955057508af5a5dfd59a98bddf49302bee5972c|
|Atlas本机分支|codex/wtr-characterization-20260915|
|Atlas本机HEAD|50a49e328e2a9dc0080d5233286a4d512bcb68af|
|远端CODE_REVISION|a50b84db1bcdafd86ceec4d9737156807f9daa9a|
|远端PLOT_REVISION|2c217ea8dddff02b4ff0b2dc0e9627f6c1314296|
|旧Graph工作树HEAD|1111e53da91e591e5ad702e1c4e499f905337841|

1955057是两个工作树HEAD的祖先。Atlas新增测量/统计/绘图代码，未改既有h65/paper执行代码。新工作建议从已提交的Atlas HEAD另建Raw工作树，保留1955057作为模型来源；各旧checkpoint与结果继续保留自己的source_revision，不能统一回填新SHA。

## 3. 最终系统结构

```text
Video / EpisodePublic
        ↓
Standard候选输入 或 Raw前置cheap读取
        ↓
Shared Coarse Evidence（时间＋必要的粗空间信息）
        ↓
Joint Capacity Planner：决定合法容量
        ↓
T：覆盖提议＋set-conditioned frame swaps
        ↓
所选RGB获取/gather → patch embedding → dense prefix
        ↓
逐层：Attention admission → attention执行
      → FFN allocation → FFN执行 → global TIA
        ↓
真实contributors与多层anchors
        ↓
Physical-time interpolation＋Cross multidepth recovery
        ↓
原检测栅格与TAD head
```

GT单独进入训练、反事实标签和评测路径。EpisodePublic不包含GT；包含GT的EpisodeTraining对象不能整体传入proposal/descriptor/planner后任意取字段。

### 计算动作

|层次|决策和监督|
|---|---|
|计划|在合法联合容量中选方案，标签来自完整plan执行，不用局部价值简单求和|
|T|固定K下的显式remove→insert收益；每轮更新集合，重建真实pair/pack后完整执行|
|D/A|决定heavy relational update；交换admission后，两分支分别由同一版本S策略在合法集合中继续执行|
|S/F|共同D集合与post-attention状态下交换heavy FFN slots，再执行后续策略|

默认Core使用嵌套FFN集合，但独立attention/FFN对照保留。Light Attention＋Heavy FFN并非非法；嵌套是受控架构选择，不能称数学必要条件。结构性整层删除、early exit与token的heavy更新路径分开。

## 4. 预算与训练规则

容量方案为 `m=(KT,{KA(l,g)},{KF(l,g)})`，packed组和native-time整数配额明确，保证实际计数与声明一致。每个token每层只执行一种FFN，不重复计费。

Core在给定成本上限下选择合法plan，局部头决定身份。成本覆盖Scout、controller、选择、patch/prefix、heavy/light、TIA、恢复与head。计算上界、平均算量、延迟、显存、解码成本分别报告。

训练目标按职责划分：TAD任务、原轴恢复、same-support、真实plan/交换价值、必要的Scout辅助目标；同输入heavy/light替代损失单独验证。它不能被same-support或重建损失替代，也不能代替真实价值标签。

控制支路默认从detached原始决策状态学习adapter/value/routing context，不靠hard index宣称精确反传。proposal首版可用固定几何规则，pair head学习交换；学习proposal偏移需要另定义监督。

更新light、局部策略或recovery会改变Vπ，应更新对应训练标签。相同输入、参数、随机态与策略版本需要记录；direct-effect固定mask诊断与策略条件总效应不能混称。

## 5. 模型家族与最终取舍

|版本|作用|是否为首批必须|
|---|---|---|
|WTR-Standard/Core|标准候选域下的任务价值分配与原轴恢复，作为主方法基础|共享接口与T价值原型先做，完整D/S学习后续|
|WTR-Raw|前移cheap读取，允许有界原始时间候选；其余执行受控|首批只做Raw-v1-S冻结检测器研究|
|WTR-DB|真实plan-response＋计算价格，在平均资源约束下选容量|否|
|WTR-FVD|完整routing函数快照的未来价值蒸馏|否|
|WTR-Graph|关系上下文或原轴Graph恢复的独立增益|否|
|WTR-Full研究配置|检验已经有独立依据的组件组合|否，不预设它一定获胜|
|系统/anytime扩展|真实按需解码、可增量执行及停止策略|否|

最终论文采用实测胜出配方，并明确其组成；不要求所有可选模块全开，不把旧PaperModel的泛化结果换名成最终WTR。

## 6. Raw输入的当前正式概念

Raw-v1先固定episode与输出栅格，再读取共同cheap preview；通过固定有界几何proposal产生候选，用Uniform或条件交换价值选帧，之后才读取heavy RGB。继续individual-frame acquisition，不默认micro-clip、max-gap或新时间PE。

Unseen Proposal Descriptor由preview插值/轻量聚合得到的上下文、物理几何、预览间隙、全局摘要、当前拟议选集及plan组成。此时S是已选/拟议frame IDs，不是已经获得其heavy features。“未heavy观察”也不等于该时刻从未作为低分辨率preview被看过。

候选身份明确分为：

- Official：O。
- Raw-Expand/Augmented：O与有界off-grid proposals的并集，保留原候选。
- Raw-Replace：独立几何proposal池，不保证包含O，后续再比较。

Expand上界是 `|R+|≤|O|+NP×M`，不能沿用Replace的 `|R|≤NP×M`。去重后的实际数量、重复观察与padding分别记录。

Preview、heavy和detector坐标各自独立；另显式保存native tubelet/anchor映射。当前Cross实际在原候选配对中心上恢复，再由readout映射到检测栅格：THUMOS首批KT384产生192个heavy anchors，Cross恢复到384个原配对位置，readout再映射到768个检测位置。不能只把decoder输出改成768就认为接口保持不变。ANet同样需保留其实际恢复轴与192检测轴之间的关系。

TubeletBuilder只做确定性排序、配对、打包及metadata；不静默新增/替换观察。记录pair span、N_changed_pairs及受影响pack，T交换收益包含真实重分组影响。

## 7. 对最新两份建议的整合与限定

新原文已保存为Raw输入目录的05/06文件。采纳其Unseen Descriptor、Expand/Replace、frozen/retrained分离、pair重组和preview覆盖诊断。

同时修正以下论证边界：

- Preview Blindness只作观测覆盖与失败关联分析，不叫“损失上界”。CF较好而Value较差，也可能涉及学习、proposal、表示或loss/AP错配，不能单独认定cheap sensing瓶颈。
- O-CF与R-CF是有限搜索参考。二者接近不能证明官方网格已经捕获全部机会；正差值也不构成mAP数学上界。
- 相对tubelet跨度要说明分母：官方candidate cadence与相同K的实际pair跨度不同。首批优先保存秒数及相同预算对照，不用未定义比值制造异常程度。
- 候选density研究在开发数据上确定规则；不能观察publication曲线后选择“最小近饱和池”再把它当独立结果。
- Expand和Replace的科学身份不同，Replace也不预定为最终系统赢家。

## 8. 完整研究路线

|阶段|目标|主要输出|
|---|---|---|
|P0 已部署Atlas|正式S/B必要性、coalition、TAD结构、有限分配空间与恢复|全数据原始记录、AP、视频聚类CI、正式图|
|P1 首批新模型|共享输入/价值接口＋Raw-v1-S冻结域实验＋一个Temporal Value小头|可执行原型、O/R有限域证据、冻结detector的Value结果|
|P2 Standard Core算子学习|attention/FFN损害解耦、light可替代性、真实D/S价值|Uniform＋算子适配强控制、nested/independent比较、S机制课程|
|P3 Raw适配验证|Frozen结果与独立训练后的结果分开|匹配重训四格；必要时再测时间编码、候选密度/Replace|
|P4 可选增强|DB、FVD、Graph各自回答独立问题|同成本增益、预测与任务双重证据，不做全笛卡尔积|
|P5 最终确认|冻结实证胜出配方|S/B完整80轮、关键对照额外seed、ANet与另一backbone/head|
|P6 系统扩展|算法成立后检验真实获取成本|on-demand decode；之后才考虑anytime、whole-video THUMOS等|

P1的T价值小头也是Core的可复用基础。把D/S完整学习后移，是为了利用正在产生的算子证据，并保持首轮Raw输入对照可归因；不是放弃Standard Core。

已登记的旧V2/Uniform/Native等课程仍由原owner处理。本轮没有重新查询整个Slurm队列，不能据旧快照断言其最新完成状态，也不重启或重复这些课程。

## 9. 论文证据和展示

论文逻辑继续为：问题定义→全数据必要性/集合交互→TAD结构→实测分配空间→原轴恢复→训练后WTR表现。对应Fig1–6的主叙事保持；Raw、算子和预算实验按其问题并入，不把教师研究占据整个结尾。

Frozen-intervention与matched-retrained分别标注。所有主性能使用完整dataset AP/Avg-mAP及实际成本；bootstrap按视频，并重算AP。四格交互由同一轮视频重采样联合计算。32视频只承担开发或技术职责，不取代211/792正式证据。

最终贡献须分别证明：存在可利用机会；cheap/状态恢复足以支持减少计算；学习策略确实超过强简单控制。Graph/FVD/DB、Raw输入和系统加速各自只声称已被对应实验支持的部分。

## 10. 当前资源与首批部署原则

推荐AutoDL服务器 `connect.nmb1.seetacloud.com:44909`，数据根 `/root/autodl-tmp/thumos14`，Python `/root/autodl-tmp/envs/opentad/bin/python`。两张RTX4080 SUPER实测各32760MiB，但当前已有Atlas任务占用。

远端存在S/B官方EMA、S/B V2 epoch40导出资产及S初始重构文件。S完整V2重构已通过预检；B相关输入存在不等于B恢复GPU验收完成。新模型首先用S。

新代码和输出放独立目录，资产只读复用。由现有owner安排后续资源或在Atlas阶段结束后明确交接；不依据瞬时低利用率抢卡，不起第二个竞争控制器，不修改正在测量的科学代码。具体首批任务、推荐参数及验收见FIRST_BATCH。

## 11. 原始材料与历史决策导航

- [最初研究、机制与全数据规划](../wtr_intake_20260914/README.zh.md)
- [Core设计及Vπ接受记录](../wtr_design_20260915/DESIGN_BASELINE.zh.md)
- [算子方案与Graph统一方案比较](../wtr_comparison_20260915/COMPARISON.zh.md)
- [动态预算修订](../wtr_dynamic_budget_20260915/DYNAMIC_BUDGET.zh.md)
- [Raw-v1接口记录](../wtr_raw_input_20260915/RAW_V1_CONTRACT.zh.md)
- [新Raw域与preview诊断原文](../wtr_raw_input_20260915/inputs/05_Raw_Domains_Pair_Repartition_and_Preview_Review.txt)
- [新Unseen Descriptor与两类四格原文](../wtr_raw_input_20260915/inputs/06_Unseen_Descriptor_and_Factorial_Implementation_Proposal.txt)
- [补存分层Core评述](inputs/hierarchical_core_review.txt)
- [原始wtr代码包](inputs/wtr_agents_1955057.zip)：提供局部组件与offline pilot，尚非完整Core或Raw实现；其CPU回执不等于GPU验收。

本报告冻结的是当前推荐路线与工作边界，没有将任何待实现模块、任务或方法效果标记为完成。
