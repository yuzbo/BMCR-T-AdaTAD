# BMCR-T → 条件计算TAD：固定源码复审与可证伪研究方案

**主审查快照：`239d098cd899936c35989fae6243c70259a85adb`**  
**证据截止：仓库所载2026-09-12 21:56（+08:00）快照。**  
**研究对象：VideoMAE-S/B；200训练视频，211测试视频/792窗口，seed 3407。**

## 阅读结论

目前证据支持“继续研究，但实质修订训练—推理合同与增量学习方式”，不支持宣布完整clip失败，也不支持宣布D1能够靠训练时长或蒸馏必然恢复官方精度。近期首先补做同配方修正BMCR；紧接着用已有D1 checkpoint分离填充器、路由和EMA三个因素，再做固定clip路由的C2混合图校准。中期更值得投入的是**保留可用全时间状态，稀疏执行昂贵更新**，而不是从无约束的新H0预测开始替换所有未重算位置。

BMCR最值得继承的不是全部旧模块，而是已经学到的任务表示、scout/边界先验、条件收益的建模问题以及高分checkpoint作为可回退锚点。新方法应沿着“同合同复用 → 校准增量 → 扩大条件执行”的链条发展。

本报告没有加载大权重、读取原始视频、执行本项目GPU前向或提交任何服务器作业。源码通过GitHub连接器按固定ref读取；指定关键模块已经逐项审阅，并补读上游VideoMAE Adapter和实际S配置。超大的递归树、机器快照及部分历史JSON返回存在截断，未声称逐字审完全部归档；关键状态由当前CONTEXT、最新快照可读部分及对应记录交叉核对。性能与GPU时延均为仓库回执，不是本次独立复测。附带的CPU脚本只验证代数、算量和反例，不等于运行项目源码。[^C01][^C02][^C03]

---

# （a）现状判断、纠错与科学问题

## A1. 可靠的成绩和执行事实

平均mAP均为tIoU 0.3/0.4/0.5/0.6/0.7的均值，不是单独@0.5。

| 骨干 | 官方AdaTAD | 旧H65-C，60轮终点 | 旧BMCR-T，60轮终点 | 修正H65 |
|---|---:|---:|---:|---:|
| S | 69.0126% | 62.8593% | 63.1563% | 63.4094%，60轮峰值/终点 |
| B | 71.1280% | 66.6128% | 67.3404% | 67.1328%，55轮峰值；60轮66.9026% |

旧BMCR的平均增益为S +0.2970pp、B +0.7276pp；S的@0.5、@0.7反而下降。“BMCR最好”仅适用于已完成的新模型B路线；S上修正H65更高，官方在两种骨干上都更高。旧权重来自15280e5旧配方，不能以当前LR/裁剪修复倒推旧训练已经正确。[^C04][^C05]

旧/修正H65与旧BMCR均从识别预训练出发，冻结非Adapter VideoMAE，训练Adapter、检测器和scout；warm也训练ASFormer。官方只复用提供的TAD权重测试。DS3则复用TAD教师并额外使用ImageNet MobileNetV3Small，初始化、监督与课程都不同，不能将最终差值只解释成训练轮数。[^S06][^C08]

修正H65的20warm+40joint训练、S/B各6000更新及全部16次中间完整测试有回执。历史65.385724%属于42dba3f的30+60轮global-TIA192课程，不能归入后来local-TIA8。较早TRAINING_COMPLETION的“测试尚未完成”被后续完整测试回执覆盖；同理，DS3 RESULTS.md的20:02状态早于21:56研究上下文。[^C03][^C06][^C07]

| S路线 | 已完成平均mAP | 真正含义 |
|---|---:|---|
| D768G | 69.0126% | 官方global-TIA384，48clips×12层 |
| D768L | 67.5344% | 同权重local-TIA8，仍48clips×12层，FLOPs不减 |
| Z16 | 43.2835% | 零新增训练，均匀16完整clips＋物理中心插值 |
| Z24 | 56.1065% | 零新增训练，均匀24完整clips＋插值 |
| Z36 | 62.5804% | 零新增训练，均匀36完整clips＋插值 |
| ZR24 | 53.0022% | 零新增训练，覆盖锚点＋固定种子随机24clips |
| T24A，epoch5 EMA | 50.6458% | D1第500次更新的早期自适应24clip＋H0填充 |

固定时点只保存至D1第15/80轮、1500/8000更新，第10轮完整测试仍在运行。第5轮原始比例0.5064583163849076，不是0.6458，更不是80轮终局。[^C01][^C02]

这些观测支持三个有限结论：local转换有可测代价；零训练clip删除对预算敏感；这个随机种子和锚点设置下均匀24优于随机24约3.1043pp。它们没有控制训练适配、teacher图、选择单位及检测表示，不能将完整clip从候选集中排除。预算曲线也不能证明所有损失都是信息不可恢复，或所有损失都是分布偏移。

## A2. 必须分清的四类问题

| 类别 | 源码/证据定位 | 判断 | 可证伪检查 |
|---|---|---|---|
| 已修复的实现错误 | `full/runtime.py`优化器分组；`full/data.py`裁剪合法端点 | 当前按参数身份确定ASFormer内部组，保留crop-created endpoint标记；旧权重不受追溯修复 | 新BMCR日志逐组核对LR与真实裁剪端点，不能复用旧尺度回执 |
| 预检覆盖缺口 | `tools/full_train.py::main`，`joint and not preflight`分支 | `--preflight`绕过真实corrected-warm及效用尺度加载，未覆盖正式joint初始化路径 | 从修正warm建立同一模型后再做无保存更新检查；比较真实父checkpoint与尺度元数据 |
| 控制命名/实现不完全一致 | `ds3/routes.py`嵌套深度选择 | 选择uniform且k12<k8时，深层子集落入随机控制路径；不能统一称均匀嵌套深度 | 固定有效clip列表，检查0/8/12各集合及选择模式；T24U的24/24本身不受此问题影响 |
| 模型合同变化 | `FormalH65`、`DenseTeacher`、上游`Adapter.forward` | rank384/global192、original768/global384、original768/local8是三种合同，不是宽松加载能够修复的shape问题 | 原合同全选等价测试；改变图的消融单列，不能要求其与官方等价 |
| 监督语义缺口 | `full/model.py::route`、`full/utility.py` | S0单交换标签用于全部分数修改后重解码S1，监督动作与执行动作不同 | 测S0单交换收益、完整S1收益及方向一致率，控制teacher/crop/预算 |
| 训练—推理失配 | `ds3/losses.py`与`ClipEngine.block/execute` | dense特征拟合不包含实际混合学生检测损失；MLP近似器训练输入也是dense状态 | 同预算同checkpoint测虚拟混合与真正compact一致；拟合误差下降是否伴随混合检测损失下降 |
| 测量/解释错误 | `full_eval.py`、`ds3_eval.py`、CONTEXT | FLOPs减半≠端到端加速；本轮官方不能与旧51.21ms混算；D1不是Z24 | 同封装、节点、窗口、缓存条件重复对照 |
| 待验证风险，不认定bug | `ds3/auxiliary.py`、`ds3/runtime.py` | H0含BN、部分窗填充、EMA同时平均浮点buffer，可能产生边缘分布/滞后 | online/EMA及BN统计隔离；短窗按有效观察分层，不擅自改掉历史训练 |

源码依据：[^S01][^S02][^S03][^S04][^S05][^S06][^S07][^S08][^S09][^S10][^S11][^S12][^S13][^S14][^S15][^S16][^S17][^S19]

当前已经正确的部分也要保留：硬RGB前向是所选真实RGB；离散预算及唯一性有显式处理；原坐标在NMS前回映；合法裁剪端点用于辅助边界监督；BMCR交换方向有符号且同时监督选中/未选候选；定位代价计入漏检；DS3冻结detector但保留其输入特征梯度；compact clip和稀疏heavy-MLP是真实压紧执行；计数器包含融合attention的QK/AV并检查未解析矩阵算子。已有CPU/4090预检是仓库记录，不是本次重复验证。[^S01][^S03][^S05][^S07][^S09][^S11][^S13][^S15][^S16][^C09]

## A3. 一个容易遗漏的关键：T24A没有测试全部DS3模块

`ds3/routes.py`中的T24A为k8=k12=24、空间保留率1。`DS3.native`实际执行24个完整12层clip，未重算位置由H0填充，使用相应clip效用分数。它不使用H8出口，不使用后四层稀疏MLP。因此，第5轮50.6458%只反映这一混合路线，不能证明H8或空间近似有效/无效。最先需要分开的是“选了什么”和“没选的怎么填”，不是立即加更多分支。[^S09][^S10][^S11]

## A4. 冗余不是相似度，而是条件可删除性

令z为可部署获得的廉价状态和元数据，S为已执行的观察/计算动作，a可以是新增一段观察、提高某区域分辨率、执行某时间token的下一块。定义

\[
u(a\mid S,z)=\mathbb E[\mathcal L_{task}(D(H_S),Y)-\mathcal L_{task}(D(H_{S\oplus a}),Y)\mid S,z].
\]

实际决策为在预算C(S)≤B下最小化任务风险，并约束有效覆盖、最长空洞和未决证据的不确定性。C应包含预览、数据取得、重建、路由、真实算子及系统开销；动作收益可以为负，不能先验假定“算更多必然更好”。这是本报告提出的建模，不是当前实现已经学习的目标。

相似像素可能隐含手部或器具的微小运动；低attention不等于不影响别的token；接近收敛的teacher绝对梯度可能很小，但大幅删除仍引起二阶损失；少数类和重复事件的召回不受平均重建误差保护。冗余必须相对于任务、已保留状态、检测器和允许的替代操作定义。

帧级效用通常不能相加成clip效用：两帧可能共同提供运动方向，单帧都无判别力；也可能是重复证据，第二帧收益下降；选帧还改变tubelet配对、rank度量、GT指派和NMS竞争。可加性只能作为需要用真实clip交换校准的近似。连续保留率的梯度正确，也不意味着hard采样代理就是无偏的离散动作梯度。

### 本次实际完成的代数/toy检查

标准库CPU脚本验证了恒预算sigmoid校准的隐式梯度，有限差分最大误差约1.49e-11；这只验证连续率校准公式，不验证hard sampler或RGB桥接代理。

另有三个反例：L(x)=(x−1)²在teacher x=1梯度为0，但删到x=0损失增1；g=(1,−1)、残差r=(1,1)时带符号内积0而绝对乘积和2；物理GT[0,10]与预测[0,5]的IoU=.5，经非均匀rank映射可变成1/6，即使端点逆映射精确。它们证伪“梯度/坐标回映必然足够”，不解释实际mAP损失的大小。

## A5. 目前能归因多少性能下降？

只能记录路径差值：69.0126→67.5344约−1.4781pp；再→Z24 56.1065约−11.4279pp。第一步是固定权重改TIA连接；第二步同时引入观察删除、插值、混合特征统计以及未适配检测器。不能把11.4279pp按主观比例拆成信息、几何和训练问题。[^C01][^C10]

还应纠正“global”的含义：上游TIA是在全时间栅格上做temporal depth-wise convolution和通道投影，并非一次任意时刻之间全连接的global self-attention；VideoMAE attention仍在各16候选clip内部。global384保留跨clip的时序邻接，local8把这些跨clip联系截断。其全局性是处理范围与跨clip连通性，不是所有时间点一次直接互看。[^S17][^S18]

区分以下可证伪预测：若teacher图是主因，固定路由/重建下global保留的收益应在跨clip动作与clip边界附近明显；若H0初始化/尺度是主因，固定24clip路由改回物理插值应改善早期混合检测；若路由是主因，同一个填充器下adaptive应稳定差于uniform且条件交换标签出现方向偏差；若信息不可观测性主导，即使教师监督充分、时间轴一致，完全未覆盖事件的召回仍难恢复；若GPU系统瓶颈主导，向量化路由及减少同步应改善时延而不改任何预测。上述都还需要真实数据证据。

## A6. BMCR收益监督：保留什么、重验什么

BMCR以EMA任务模型评估实际局部交换，定位分量包含分类约束匹配、漏检虚拟槽；标签为有符号损失变化，伙伴关系有明确语义。这比“只把绝对attention当重要性”更贴近当前任务。[^S01][^S03]

但应重新校验四件事。第一，定位miss penalty和置信度门槛导致饱和：交换前后都漏检时，严重缺失事件也可能给出零增量；第二，未匹配假阳性及整个类别的跨视频置信排序并不被局部匹配代价完整表达；第三，分类分量固定S0目标来隔离输入变化，但rank位置交换后GT指派也可能改变，需同时审计固定目标与重新指派的收益；第四，两个分量按尺度标准化再组合，并不保证对应相同单位的mAP收益。新尺度须从修正warm的训练数据重做，不继承旧权重的尺度。[^S03][^S04]

S0→S1的最大问题不是符号，而是动作不一致：局部“一出一进”的标签，不能直接担保全部候选分数修改后CDF整组重采样更好。应记录样本级u预测与实际ΔL、正负号一致、S1改动数量、漏检变化，必要时先限制为真正执行的少量交换，或者重新学习完整route refinement的收益。不要先修改所有旧模块再归因。

## A7. D1、C2、J3与EMA

| 名称 | 训练见到的状态 | 是否训练任务适配 | 推理关系 |
|---|---|---|---|
| Z0 | 无新增训练 | 无 | 直接改变观察/装配 |
| D1，当前 | 完整teacher前向，dense中间特征，辅助拟合 | 没有真实混合学生检测损失 | H0/H8/近似器及hard路由组合首次在部署图共同出现 |
| C2，建议 | 可以保留完整teacher/主干前向，但辅助图模拟缺失、出口、替代状态 | GT＋语义匹配KD对混合学生求梯度 | 把易失配部分提前暴露给训练；未必减少训练算量 |
| J3，建议 | 实际条件路径以及其改变后的中间状态 | Adapter/出口/必要detector联合适配 | 最直接匹配部署，代价是训练图与优化复杂度增加 |

“训练保留完整观察和主干计算”并不排斥C2。DyT恰好提供全量算子训练、masked中间状态、部署压紧执行的例子；CoLT5和CoDA则包含条件路径训练，不能归类为仅dense feature fitting。[^S11][^P08][^P07][^P14]

当前utility proxy在teacher特征处求梯度，再汇总绝对梯度×残差。它不是真正混合状态处的有符号边际，也不区分通道抵消；对于0→8→12还必须分别学习对应动作，而不能把0→12分数默认当0→8。后四层MLP即便逐层dense输入拟合良好，推理第10层看到的第9层替代状态也已不同。[^S09][^S11]

EMA=.999时，500更新后的初值参数系数=.6063789，1500更新后=.2229628，半衰期约692.8更新。不能把这些数字解释成输出或mAP组成。应读取同一checkpoint的online与EMA，用同一route/填充器比较；MobileNet BN浮点buffer也经过EMA，应单独报告统计滞后，不能把在线权重与EMA的另一路BN统计混用后仍称原EMA。是否需BN再校准只做独立标记控制，不改历史结果。低早期分数不能全部归因EMA。[^S12][^S14]

## A8. 哪种“性能保证”成立？

**可给的严格保证是合同内函数等价，不是未知数据上的mAP。** 同一权重、全选、同输入/归一化/位置/TIA/检测轴/后处理可以测试数值等价。固定uniform24＋原物理插值基线，加零输出残差，初始可以等于Z24；它不等于训练好的H65/BMCR。H8残差头恒等初始化只保证输出初始为H8状态，不会使第8层等于第12层。

更少观察的模型若遇到两个在全部已取得证据上相同、但未观察区域中动作不同的视频，无法仅凭这份证据同时恢复二者。全时间低成本状态可以减小这种歧义，但不能证明其保留了所有小动作证据。

在额外假设下可以建立局部稳定性：若混合特征误差有界、检测器局部Lipschitz且分类排序、GT匹配和NMS决策有足够margin，输出可保持稳定；没有这些margin，极小变化也可能跨tIoU或排序阈值。一个仅用于说明的保守界是：dense框恰为长度d的GT，学生两端各误差≤ε，则IoU≥(d−2ε)/(d+2ε)。这说明短动作容忍的绝对误差更小，但不能据此计算项目掉点比例。所有新路线的最终保精度承诺只能来自匹配协议下的真实测试。

---

# （b）跨领域机制图谱

以下是机制审阅，不是论文数量竞赛。原论文结论限定于其任务、训练设置和实现；“迁移/失败预测”是本报告推论。原文版本与作者实现边界见文末索引。

## B1. 被跳过的地方仍有状态：密集预测比分类更接近TAD

**DToP（ICCV2023，v2）**用辅助分割头使容易token提前输出，保留类别代表token供后续上下文使用，并恢复原位置上的密集结果。其有效路线包含剪枝后的适配，不是把分类token直接删除。原文还显示直接剪枝、只微调head、重新训练全部网络结果不同，不能把更长/更全面训练当成必然更优。[^P02]

迁移点是“退出位置仍有可用任务状态＋代表性上下文”，而不是“背景置信高就永远丢弃”。TAD的重复同类动作不能共享一个类别代表；边界、小动作和背景负证据必须按时间覆盖保留。空间token早退后需要scatter到TIA栅格；像素类别概率也不能直接成为TAD时序token的可靠性。

**TR-BERT（NAACL2021）**为QA等任务分配逐token深度，终止token仍保留状态与原位置；梯度/残差启发式服务于训练初始化，之后还有带计算惩罚的策略学习、任务训练及蒸馏。[^P06]

它与TAD的同构点是最终答案/事件跨度依赖原序列位置。差异是文本token已经是离散可读证据，RGB删除发生在语义提取之前；词不再更新不等于一帧从未被看见。可借鉴统一出口接口和原位置保留，不能借它证明视频重建器可恢复未观察运动。

## B2. 全token轻路径＋部分token重路径：最值得转移的结构

**CoLT5（v2，2023）**对所有token保留轻路径，分别选择heavy-query、key/value和FFN token，条件路径参与训练，并在长上下文任务中验证。query表示“谁需要信息”，key/value表示“谁提供信息”，不是同一个重要性分数。[^P07]

**CoDA（v2，2023）**把已预训练重模块与全token廉价adapter结合，冻结大部分主干并学习条件执行/adapter；路由权重参与梯度，但其条件图被显式适配。[^P08]

这两者为BMCR→DS3提供连续研究链：保留强表示的同合同重路径，让轻路径学其增量/替代，而非先破坏强状态后期待全量拟合恢复。TAD可将动作单元定义为clip、tubelet或原栅格上的额外块；必须保留时间轴、更新未重算位置的上下文，并分别校准查询价值与证据提供价值。条件图训练与新模块成本仍然存在。

**DyT（NeurIPS2024，v2）**尤其直接：所有token保留，adapter处理全体，dispatcher决定昂贵模块更新；训练仍全量计算但中间输出被掩码，部署才压紧。MLP dispatch在其比较中优于直接破坏attention交互；完整模型监督和自蒸馏会增加训练成本。其检测实验仍明显落后全量fine-tuning，是重要反例。[^P14]

本项目应借鉴的是masked-state task adaptation和全栅格保留，而非自动叠加MoE。VideoMAE-S的MLP占比与其图像ViT不同，本文已独立核算，不能照抄论文的节省比例。

## B3. 真正可行的dense-train/sparse-infer：条件相同，而不是口号相同

**QueryDet（CVPR2022）**在已有FPN特征上从低分辨率预测高分辨率查询位置，部署用稀疏卷积执行昂贵head，原anchor坐标保留。作者实现会扩展context邻域、gather既有特征、复制dense权重到稀疏卷积，并有稀疏结果densify步骤。[^P09][^I01]

这与“把未观察RGB交给新H0”不同：QueryDet没有凭空省去FPN之前的所有视觉观察。迁移时，应检查被选输出所需的receptive-field halo、归一化及位置是否与dense一致。有限halo不必然覆盖多层卷积全部依赖，不应把实现称为无条件精确等价。

**PointRend（CVPR2020）**让粗输出保留全域语义，在不确定边界读取已有细特征，使用点级网络细化；训练采样与推理递归细化并不完全相同。[^P10]

其启示是：允许训练与部署异构，但要保持条件预测函数及其输入分布足够一致。TAD可以保留全时间粗状态，查询边界附近和动作内部；不能把点级精化在已有特征上成功，变成少RGB采样的保证。训练只采最不确定位置还可能牺牲覆盖。

## B4. 删除前先聚合、删除后能恢复：点云与token重建

**RandLA-Net（CVPR2020）**随机下采样配合局部几何编码和特征聚合，解码端通过插值及skip恢复原点输出；采样是训练图的一部分。[^P04]

**3DSSD（CVPR2020）**结合几何和语义采样；仅几何覆盖可能漏掉前景，单独强调语义也可能增加假阳性。其检测结构保留点坐标和候选中心语义。[^P11]

迁移到TAD，代表点必须携带聚合范围、时间间隔、有效数量和边界不确定性；不能只携带中心。覆盖采样与任务采样应互补。视频还存在方向、速度和遮挡，且tubelet embedding依赖配对相位，不能把它当无序点集。多主体空间选择应保留多个证据区及背景上下文，不能只取最显著人体ROI。

**Expediting ViTs for Dense Prediction（NeurIPS2022）**先保存高分辨率关系、聚类降低中间token数，再重建密集状态，可在其任务上无需finetuning。[^P12]

它是“所有稀疏部署都必须重训”的反例，但不是信息恢复定理：关系来自删除之前已经计算的特征。对AdaTAD，不能到最后才恢复栅格，因为TIA在中间层已经需要完整的时空布局。应限制同tubelet内局部聚合并在每个TIA前恢复，或者修改TIA合同后重新适配；两者都须计重建成本。

## B5. 视频局部细化与事件驱动状态复用

**AdaFocus V2（CVPR2022，v2）**以全局低成本视觉通路指导局部细化，利用可微裁剪与辅助监督协调策略和表示；针对视频识别，不要求输出所有事件时间段。[^P01]

可借鉴辅助监督、策略输入梯度隔离和廉价视觉覆盖。视频级早退出口不能迁移成“已识别到动作即可停止整段TAD”：后续同类动作、其它主体或罕见事件仍需被发现。160×160既有缩放/裁剪合同也不自动背书任意逐时刻自由ROI。

**Eventful Transformers（ICCV2023）**用保存的状态判断哪些token需要更新，变化相对于该位置上次真正更新的状态累计，而不是只看相邻帧差；通过gather/update/scatter执行选择性计算。[^P13]

这给出与“删帧”不同的时间去冗余：不重复计算已可靠保存的状态，但仍可触发重新观测。迁移风险是摄像机运动、遮挡、窗口位置/clip相位变化、global-TIA上下文变化；这些会使缓存不再对应相同函数。初始建立缓存也要全算或以可验证方式近似，不能当免费前置条件。

## B6. TAD本身的反面证据：定位和连续性不能被类别正确掩盖

**ETAD（2022预印本）**用完整特征前向、抽样梯度重放及提案采样降低训练成本。其启示是训练图/梯度计算可以分离与重放；它并未证明RGB推理时删除一半观察仍保持TAD精度。[^P05]

**Temporal Corruption Robustness（CVPR2024，v1）**报告少量时间扰动可严重影响定位，动作中部扰动有时比边界更致命；通过FrameDrop与任务定位一致性训练提升鲁棒性，但协议、腐坏形式及指标不同于本项目稀疏推理。[^P15]

因此，边界先验应该是计算分配的一部分，而不是全部。应对动作内部、重复事件之间的分隔、长背景中的假阳性同样检查；只报告分类准确、边界加权MSE下降或proposal数量下降都不足以支持TAD性能保持。

### 机制收敛

可转移的共同原则不是“稀疏”，而是：**未重算位置仍有任务可用状态；坐标与支持范围明确；昂贵更新有条件价值；训练至少覆盖真正发生分布变化的状态；省算发生在实际被跳过的算子，而非只少反向传播。**

---

# （c）最多三条优先模型族

以下名称是本报告的工作性命名，不声称首次提出。三条路线共享问题定义，不要求同时完整训练，也不强制保留BMCR全部模块。

## C1. 模型族一：BMCR-Anchor——保留高质量合同的条件更新

**定位：最稳妥的成果继承线，先不改变时间轴和采样合同。**

训练图：

```text
修正warm20 → 同配方BMCR joint40 → 锁定父checkpoint
                          ↓
       原scout/rate/GT-rank/检测器保持原合同
                          ↓
同一已选K384 RGB → 全量强路径teacher状态（训练）
              └→ 掩码重更新＋完整旧状态 → rank检测损失
                         ↑        ↑
               条件收益校准   原父模型预测/特征监督
```

推理图：

```text
廉价scout → K384真实观察 → 24clips/native192
 → 每层保留完整状态
 → 选中位置执行重MLP/后续重块，未选位置走残差/轻更新
 → 每次TIA前scatter完整栅格，保持global192
 → rank384 detector → 物理坐标回映 → 原NMS
```

选择单位优先为已选观察上的空间tile/时间tubelet的重MLP更新，不立刻再次减少RGB观察；深度选择是某些位置少执行昂贵残差，而不删除全部状态。第一轮只探索晚层MLP作为接口/机制检查，其省算上限有限，不宣传大幅加速。结果允许时才扩展更多层或重块跳过。

teacher首先取同一父BMCR/H65合同内的checkpoint。官方global384预测可以作为额外真实时间teacher，但必须将rank输出回到物理轴后匹配；不能直接MSE native192与native384，也不直接替换detector768。旧B最佳checkpoint只作为独立旧配方锚点，修正H65与新修正BMCR另列。

初始化为原模型全heavy执行、新增残差末层零输出、路由明确强制all-heavy；这可以实现原合同下的初始数值等价。然后先校准替代状态，再降低重更新比例。零末层导致首步上游梯度为零可属于正常残差初始化，只要末层先获得梯度，后续上游梯度出现；不能把整个新分支每层都清零。

损失以原GT检测为主，加入父模型特征/预测约束和明确成本下的条件收益监督。若使用hard gate，必须有单独的监督/策略梯度/明确标注偏差的straight-through路径；TAD loss不会自动微分离散索引。初期冻结scout以隔离“省深算”因素，再考虑联合，避免采样集合和表示同时漂移。

已有：父模型、scout、rank映射、合法边界、任务损失、EMA收益机制。新增：全状态条件重更新、对应动作的收益校准、逐层执行计数。不能直接复用：DS3-L compact clip执行中的local独立假设。global192下不同clip状态仍会通过TIA交互。

**证伪条件：** 同一个K384与同一父checkpoint上，轻替代经校准仍在@0.6/.7、重复事件或短动作持续受损，而节省被路由开销抵消，应停止该压缩配置；不等价于BMCR本身无价值。代码基础：[^S01][^S02][^S03][^S07][^S17]

## C2. 模型族二：Grid-C2——原坐标clip重建锚点与任务校准

**定位：最小成本区分“完整clip不行”还是“填充/路由/适配不行”。**

训练图保留完整48clip frozen local teacher前向，缓存F8/F12和原时间元数据。固定uniform24，先不训练router：

```text
完整RGB → frozen local teacher → F8、F12（同一完整前向）
   ↓                               ↓ 按固定route索引
H0低成本视觉 → 零末层残差        已选F12
   └──────────→ 物理插值基线＋缺失位置残差
                              ↓
                    原native384 → detector768
                              ↓
                      GT检测＋匹配KD＋特征约束
```

对同一S定义Z型重建R_S(F_S)，采用

\[
\hat F=R_S(F_S)+(1-M_S)\odot q\odot A_\theta(z,R_S(F_S),\Delta t,\mathrm{support}),
\]

其中M_S为真实heavy tubelet位置，q为可选的可靠性系数；A末层为零，q不能与A同时用导致所有梯度长期为零的初始化。最简先令q=1，只学零初始化残差。选中heavy状态不被新H0覆盖；可靠性融合后续再根据混合检测收益校准。

推理只计算24个heavy clip＋H0/重建，绝不先全算48clips再mask。固定uniform24下，初始可严格复现Z24的装配行为；这个锚点的实测水平是56.1065%，不是高分BMCR。少观察条件下无法无代价继承BMCR原输出，必须坦率承认。BMCR在这里迁移的是训练过的scout/边界先验、真实时间预测teacher和重新校准的条件效用，不是盲拷贝rank detector。

C2校准使冻结detector仍对混合输入反传；heavy teacher分支detach，梯度进入重建器/出口。分类蒸馏匹配实际独立sigmoid输出语义，不能默认softmax互斥分类；回归通过可微pre-NMS预测、相同物理坐标和明确对应关系蒸馏，保留GT，不能只拟合teacher漏检。特征损失按有效tubelet、边界、动作内部/背景分层，防止多数背景主导。

重建成立后再开放router，利用训练时单次dense缓存构造“加入/移除/0→8/8→12”的真实混合反事实，主要重跑检测器评估ΔL。动作收益采用带符号、条件于当前重建和已选集的标签；同预算交换两端必须共同定义。scout先验只解决冷启动和覆盖，不当作新clip收益真值。H8需单独校准H8→F12残差，不把它当恒等精度出口。

local缓存成立的条件是：同视频窗口、同增强、同clip身份/位置、同有效掩码、teacher eval、无跨clip模块、数值精度可比。不同compact batch形状可能引起浮点差异，应分别FP32/AMP验证。global-TIA时交换一个clip会改变别处及后续层，不能只换最终缓存特征；稀疏MLP改变clip内部递归状态时也不能直接复用dense最终状态作为真实学生。

这条族内才做粒度辨别：先保留全量patch embedding与原clip位置，比较重更新单位为原tubelet、4/8/16候选观察对应的1/2/4/8 tubelet组，统一原检测轴和TIA。此控制检测的是**计算粒度**，不是少RGB观察。之后再进行真实RGB micro-clip选择，明确它额外改变attention支持、运动统计和预训练分布。

**证伪条件：** 同route、同teacher、同预算下，C2较等更新D1未改善混合任务误差，或明显恢复训练重建却不能改善完整测试定位/召回，应修订teacher/观察策略；若uniform重建稳定而adaptive下降，应先重做route utility，不把责任归给clip。代码基础：[^S09][^S10][^S11][^S12][^S13][^S14]

## C3. 模型族三：Full-State Sparse-Update——全时间轻状态与保留global-TIA

**定位：中期最有希望同时压缩时、空、深三维昂贵计算，但实现风险最高。**

起点用官方global384合同；所有768候选做patch embedding，必要时做共享浅前缀，形成完整native384×10×10栅格。保留每个位置的当前状态及有效性。重计算稀疏化，而不是把整段时间位置置空。

```text
训练：完整观察 → 强全局teacher（完整图，监督来源）
                └→ 具有masked/sparse状态的学生图
                     ↓ 每层GT/KD所需状态可追踪
推理：全时间patch/浅状态
 → 每层 [廉价路由 → 选择重query/FFN/tile → gather计算]
 → scatter回完整原栅格 → global-TIA384更新全部可用状态
 → native384空间池化 → 原detector768 → NMS
```

在每层，按原顺序执行attention残差、MLP残差、TIA；跳过昂贵残差的位置至少保存旧状态，TIA仍更新其上下文。更安全的attention版本是selected-Q读full-KV：保留证据提供者，不强迫“需要更新者＝提供信息者”。它只省部分Q/输出与QK/AV，不能按token保留率平方估算全attention；full-KV本身仍付费。

空间起步用规则tile/同tubelet局部聚合，保证多主体和小目标有基本覆盖；不同时引入任意ROI、可变形采样、merge、MoE、多个出口。若用merge，重建必须在TIA之前回到具有明确坐标的栅格；保留被merge状态的支持区域和权重，而非用同形状tensor伪装原位置状态。

时间稀疏是不同时间位置少执行heavy更新；空间稀疏是其内部tile或query少执行；深度稀疏是累计heavy层数不同。所有位置仍经历廉价时间上下文通路。这三者共享状态与实际成本表，不使用三个保留率的乘积。

全heavy时要求与官方同合同等价；新分支初始零残差、明确定义gate。先学习晚层替代，再渐进到更多层/更少heavy，保留全heavy预算样本作为锚。teacher使用官方global图；BMCR/H65预测与scout仅提供对齐到真实时间的先验或另一监督来源，不盲目混合不同表示。可以在训练保留完整teacher前向，但学生改变的global递归状态必须真实执行；C2只能缓存共享前缀，不能把全部尾层替代成固定dense缓存。这实质上接近J3的条件适配。

本族深度选择不是“冻结参数即跳算”，也不必须增加独立D8检测头：未重算位置持续保有同维状态，统一在最后接口交付。全时域D8出口、160→128规则降分辨率是低复杂度对照，用来判断复杂动态路由是否值得，不能提前声称有对应精度。

**证伪条件：** 如果在相同实测成本下，规则128或全时间D8经过同等适配比动态状态更新更稳、更快，应优先简单方案；若global保留无收益而状态维护开销明显，不为叙事强保global。代码基础：[^S09][^S17][^S18]；借鉴机制：[^P07][^P08][^P14]

### 相对已有机制，什么才可能形成研究贡献？

单独把DyT/CoDA换成VideoMAE、加蒸馏或加ROI都不构成已确立的新贡献。这里值得验证的是三件相互约束的设计：在真实时间轴上维护带来源/支持范围的任务状态；把原BMCR单帧交换扩展为与实际clip/深度/空间更新一致的条件收益；在完整teacher观察与稀疏学生状态之间明确可复用的计算边界。只有它们带来可归因的性能—成本改善、且简单静态控制不能解释时，才有理由形成方法创新主张。

## C4. 所有模型族共用的采样合同

| 选择单位 | 主要优势 | 主要风险/不可偷换项 |
|---|---|---|
| 单候选帧 | 时间覆盖细、预算灵活 | 原3D patch embedding依赖两候选配对；重复单帧/重排都会改运动分布 |
| 原tubelet（2候选） | 保留预训练配对与相位 | 单tubelet不能独立等价于16候选clip内attention上下文 |
| 4/8/16候选micro-clip | 覆盖与局部运动之间连续折中 | 长度变化改变attention支持/位置，不能只reshape当等价 |
| 重叠clip | 边界附近有双侧证据 | 重复计算；需去重、加权重建、偶数tubelet相位与真实成本 |
| 自适应长短段 | 根据不确定性分配观察 | 动态batch/shape开销、预算复杂，动作起止不能用测试GT决定 |
| 关键帧＋局部片段 | 发现与运动确认分工 | 关键帧scout漏掉事件时无法触发细化，需覆盖锚点 |
| 全时间轻状态＋稀疏深算 | 不直接制造时间空洞，保留上下文 | 廉价状态可能遗漏小证据，必须测量其信息上限与维护成本 |

选择发生在解码之前才可能省解码；在RGB/patch embedding前省相应视觉计算；在attention前省后续token交互；仅MLP前选择只省MLP，不能宣传少RGB/attention。当前已解码全窗口后选择，不自动产生解码收益。

原时间轴接口优先保留检测器均匀栅格：稀疏特征携带中心、有效支持区间、Δt、观察/估计标志，再重建到native384→768，沿原GT指派及回归。若坚持原生不规则序列，必须同时修改相对位置/邻接、TIA核、金字塔覆盖、点生成、GT尺度区间和回归单位；单加time embedding或deformable offset不够。rank检测是另一合同，可保留作为锚点，但不能把可逆端点映射称为完整度量等价。

对部分/短窗口，分别记录有效候选数、有效tubelet、部分有效clip及实际物理槽位。中心/支持范围不能由padding伪造；但改变padding也属于需验证的输入合同，不能在新实验中静默替换历史策略。短动作、边界、动作内部、重复事件间隔、长背景、遮挡、多主体和小动作证据均需覆盖。

---

# （d）最小判别实验顺序

## D0. 第一步：已授权的同配方修正BMCR

先复用S/B各自修正warm20，重新做训练集效用尺度审计，再joint40。保护旧BMCR和修正H65 checkpoint，不覆盖；不新增独立U384完整训练。预检补到实际warm/尺度加载分支。完整测试仍按用户规则每5轮选EMA峰值，并另报60轮终点。新的正确BMCR成绩在固定快照中不存在。[^C01][^S04]

判别问题是“同一修正配方下，BMCR条件收益相对H65是否稳定贡献”，而不是直接与不同初始化/80轮D1比训练效果。除平均mAP，重点看逐tIoU、S曾退化的@0.5/.7、miss penalty饱和、S0/S1收益一致性及真实路由耗时。此阶段不再加入KD、原轴重建、空间或深度新模块，避免污染因果比较。

## D1. 使用现有checkpoint分离初始化、路由与EMA；不先加训练

在同一已保存D1 checkpoint上，比较**uniform/adaptive × 物理插值/H0填充**四种装配；全部24完整clip、12层、空间比率1、local-TIA8、原detector768。插值与H0计算开销不相等，分别报告；观察/深算预算相等不等于FLOPs完全相等。

先在训练诊断窗口做online/EMA分离，再对有判别价值的固定策略做完整211测试，不通过结果调预算。`ds3/runtime.py`已有在线加载能力，但评测入口需新增显式state选项。评价H0特征尺度、范数、时间差分、边界/内部/背景误差，以及混合状态下分类校准与定位偏差。若在线改善而EMA滞后，可解释部分早期现象；不能省去填充与route控制后把全部差距归EMA。[^S12][^S13][^S14]

## D2. 一个短程C2配对试验，决定是否值得扩大

从同一D1辅助初始化复制两条支线，相同训练更新和采样增强：一条继续D1；一条固定uniform24并加入真实混合图GT检测监督。teacher保持local，先不加新global teacher、不训练router、不引入H8/MLP压缩。零残差重建作为可解释初始化另做epoch0等价检查。

这一配对唯一核心变化是混合任务适配。检测损失前向/反向与teacher查询额外成本单列，不声称相同更新数等于相同训练成本。先观察训练诊断上的混合loss、稀有/短动作误差；如有稳定改善再做完整测试和进一步预算。C2获益只证明它在该路线有效，不证明能够恢复全部11.4279pp。

## D3. 固定填充后才检验条件路由

以D2中较可靠的填充器固定状态，使用训练集dense local缓存做实际clip交换标签，校准0→12；需要深度动作时另校准0→8与8→12。比较uniform、既有scout边界/覆盖先验、条件收益route；首先在同一checkpoint推理控制，不新增三个完整训练。

用训练集oracle交换只估计“这个动作空间是否存在可获得的改善”，不把GT oracle部署成绩写成可实现结果。报告proxy/真实ΔL相关、符号一致、提升top-k命中率、收益regret及饱和漏检占比，另报最终mAP；ΔL不是AP。若oracle有收益但学得route没有，优先修监督/优化；若oracle也无收益，说明在当前表示/重建/预算下路由优化空间有限，而非所有clip选择都不成立。

## D4. 粒度与时间图只做必要的交叉控制

第一组在全量patch embedding后固定teacher权重、训练历史、原GT/检测轴、TIA设置、有效支持与成本，改变heavy-update分组为1/2/4/8 tubelet。它尽量隔离计算粒度；attention的支持集合若随单位变化仍不可完全分离，应显式报告。

第二组才是真实RGB micro-clip/完整clip选择，保持近似有效观察预算，再分别匹配FLOPs。观察数相等时短clip数量更多，attention二次项不同，所以两种匹配结果应分表；权重/训练历史相同的零训练控制和同等适配后的控制分别解释，不能把预训练分布冲击称为粒度的必然劣势。

global/local只在同一固定路由和重建设置下对比；全选global/local已有参考可复用。global条件路径不允许假借local缓存产生“等价”结果；需要真实更新全状态。先从少量固定训练诊断窗确认机制，再考虑完整课程，不开展全部维度笛卡尔积。

## D5. 空间/深度扩展由证据触发

优先比较低复杂度静态128、全时间D8和条件更新，而不是直接三维联合。保持相同teacher来源、训练更新和评测规则，同时披露初始化/新增参数/teacher查询差异。晚四层MLP只是可控小实验；如果它节省有限且无实测速度改善，不值得围绕它构建三维大模型。

扩大预算的依据是：真实混合状态上的任务误差改善、完整测试多阈值与困难事件结果可解释、真实算子计数与时延均支持目标，以及收益未依赖初始化/坐标/缓存泄漏。没有规定未经用户接受的1pp、160/40或哈希审批门槛。

## D6. 必须报告的误差和不确定性

主指标为平均及逐tIoU mAP；按训练分布预先定义短/中/长动作分组，补报告距离起止的边界误差、动作内部空洞、同类重复事件合并/漏检、背景误报、遮挡/多主体困难例，以及不同有效窗口长度。置信度检查包括TP/FP排序、GT匹配后的校准、类别间与不同出口间尺度；不要只看整体余弦相似。

选择机制可视化用同一窗口的真实时间轴：GT、scout先验、有效观察、heavy执行深度、空间选择、预测框、缺失/估计状态分层展示。再将实际删除/替代的ΔL与相似度、attention、绝对梯度并列，呈现“看似相似却不可删”的反例和“低收益可删”的例子。此图须来自未来真实数据，不用toy充当研究可视化证据。

对固定checkpoint可做按视频paired bootstrap重算完整AP差值，以反映测试样本不确定性；它不消除训练seed方差，也不能修复测试峰值选模偏差。792窗口不是792个独立视频。后续多种子是泛化建议，不把现有200训练改成未授权的160/40，也不以单seed +0.297pp宣称稳定显著优势。

---

# （e）系统测量、独立算量与成本边界

## E1. 本次独立核算结果

实际S配置为C=384、tubelet=2、patch=16、输入160→10×10空间栅格，每clip N=8×100=800，12层均有TIA，TIA瓶颈h=96。每clip每层矩阵MAC为：attention线性4NC²，QK/AV为2N²C，MLP为8NC²，TIA约N(2Ch+h²+3h)。忽略bias、activation、norm等，乘48clips×12层。[^S17][^S18]

| 分量 | 独立代数MAC（G） |
|---|---:|
| Patch embedding | 22.649241600 |
| Attention QKV/output线性 | 271.790899200 |
| Attention QK/AV | 283.115520000 |
| MLP | 543.581798400 |
| TIA | 38.353305600 |
| Backbone合计 | 1159.490764800 |
| Detector残余（用已记录全模型减骨干） | 14.456254464 |
| 全模型合计 | 1173.947019264 |

骨干分量由配置/算子独立推导，detector残余取仓库实测总量差额，不能把全部表格称独立运行算子计数。结果与记录吻合：TIA占全模型矩阵MAC约3.2670%；只在晚4层MLP保留75%/50%，理想最多节省3.8586%/7.7173%。当前宽32近似器与评分器也要计算，按同样矩阵口径净节省进一步约为3.5321%/7.3907%，仍未扣路由/内存/其它辅助模块。[^C11][^S09][^S12]

整个12层MLP占全模型矩阵MAC约46.30%，即便所有层MLP都只保留一半，理想上限也只是节省约23.15%；attention线性与QK/AV合计约47.27%。因此接近半算量的全时间路线，通常还须触及分辨率、attention或更大范围深度计算，不能指望TIA或最后四层MLP。

全时域D8加已知出口算子约67.7251%总FLOPs；160→128时空间token比r=.64，线性项按r、QK/AV按r²、detector固定，得58.8869%。这些都是结构估算，没有对应精度与时延实测。B需按C=768重新计算，不能直接复制S分项占比。

## E2. 已测时间的正确解释

Z24为本轮官方约50.6157% FLOPs；固定full窗口GPU模型均值/中位42.60/42.56ms，官方67.87/64.77ms。模型路径有实际节省，但全211视频评测1229.65秒与1236.41秒接近且缓存未控制，因此未证明端到端有意义加速。旧官方51.21ms不在同一个封装/节点计时口径中。[^C10]

测量应分为：已准备规范化clip和route的GPU骨干；从GPU原输入开始包含预览/路由/重建/detector的完整GPU模型；解码/预处理/传输；窗口NMS与视频聚合NMS；最后独立的AP计算/结果序列化。旧full_eval与DS3对末尾统计阶段的elapsed范围并不完全相同，不能直接拼接跨脚本e2e表。[^S05][^S13]

## E3. 实际执行审计，而不是保留率乘积

每个full/partial/short样本至少记录：有效候选、物理候选槽、有效tubelet/部分clip、每层执行clip数、Q/K/V数量、heavy FFN token数、surrogate token数、full-state/TIA位置数。48个物理clip不等于48个全部有效clip；尾窗固定物理执行不等于exact384有效观察。

2MAC口径纳入mm/bmm/addmm/conv及融合SDPA的QK和AV；softmax、norm、激活、位置插值、top-k、gather/scatter、mask、CPU/GPU同步、内存读写分开列。冻结参数不减少前向MAC；执行稀疏head不表示RGB骨干稀疏；route元数据与Python循环即使MAC极小也可能主导时延。主方法与官方保持相同计数/预热/同步/精度，并保留所有样本。

统计报告均值、中位、p95、标准差/区间、重复轮次和节点负载；现有20次定点样本足以暴露问题，但不足以代表全部窗口分布。端到端报告冷/热缓存及稳定重复，另外按实际测试窗口分布加权，不能将首个full窗口时延称为测试集平均。

## E4. 缓存和训练teacher成本

BF16一个完整S状态栅格384×100×384约28.125MiB，B约56.25MiB；空间池化后S native384约0.28125MiB，缓存F8和F12合计0.5625MiB/窗口。它们只覆盖所列张量，不包括QKV、四层MLP捕获、梯度和数据加载。若缓存四层MLP输入/输出，八份完整S栅格本身约225MiB/窗口，不能用池化缓存价格替代。

local固定teacher允许同一增强/窗口内多次交换复用F8/F12，收益主要是少做重复骨干标签查询，但每次反事实仍需装配和detector。GT梯度utility有detector反向成本，C2有学生检测器反向，J3还涉及改变状态后的重执行。完整teacher在所有训练批次的前向、额外global teacher、缓存构建/存储/读取均单列；相同epoch不等于相同成本。

跨窗口重用必须验证规范化RGB、空间裁剪、clip分组/phase、位置嵌入、valid mask、TIA上下文都匹配。global384受窗口上下文影响，不能按原帧ID直接复用最终token；改变训练增强或模型参数也使缓存陈旧。缓存首建、失效与回退算量都应计费。

---

# （f）Codex实施交接

以下为代码设计交接，不代表这些接口已经存在。另附CODEX_HANDOFF.zh.md可直接作为实施任务书；不运行服务器训练/评测作业。

## F1. 最小改动定位

| 文件/符号（固定主提交） | 最小改动 | 保留项 |
|---|---|---|
| `tools/full_train.py::main` | 让检查模式可进入真实corrected-warm＋utility-scales初始化；仅增加诊断，不改变旧配方 | warm20/joint40、seed、原EMA父链 |
| `h65/full/runtime.py` | 导出optimizer角色/LR及checkpoint合同描述 | 已修复参数身份分组 |
| `h65/full/data.py::LoadFramesWithBoundaryValidity` | 添加短窗/裁剪端点一致性测试 | 原数据增强与合法边界定义 |
| `h65/full/model.py::route/train_batch/raw_route/predictions` | 记录S0、S1、完整改动集；提供原时间teacher预测适配器 | 原rank384检测与NMS前回映 |
| `h65/full/scout.py` | 显式输出可迁移的action/boundary/coverage先验；后续向量化route相关循环 | 训练过的scout状态，不冷启动替代 |
| `h65/full/utility.py` | 输出signed分量、饱和miss计数、固定/重指派收益；新动作另建标签合同 | 原标签符号和旧实验可重现性 |
| `h65/transport.py`、`full/geometry.py` | 暴露物理中心/支持/间隔与mask；连续梯度、唯一性、逆映射测试 | 硬RGB前向、物理映射 |
| `h65/full/objectives.py` | 复用合法边界权重；将新损失放新配置而非篡改旧训练 | 已有课程与GT基础监督 |
| `h65/ds3/routes.py` | 分离route与fill policy；明确uniform嵌套深度；零分数显式均匀fallback；有效/物理计数 | Z16/24/36和ZR24原始可重放模式 |
| `h65/ds3/auxiliary.py` | 新增插值残差头和可校准出口；零末层仅用于新路线；记录BN语义 | 当前D1权重和H0实现单独保留 |
| `h65/ds3/model.py::DS3.native`、`ClipEngine` | 显式assemble_native；固定路由虚拟混合；cached-vs-compact路径；global另建engine | D768G/L/Z0全部基准 |
| `h65/ds3/losses.py` | 单独C2入口：混合GT检测、匹配KD、signed条件收益；不覆盖D1 | dense_aux原样留存 |
| `h65/ds3/runtime.py` | 保存/校验teacher图、轴、state、route/fill、预算元数据 | strict state加载 |
| `tools/ds3_train.py`、`ds3_eval.py` | 显式regime/state/fill-policy；eval支持online；阶段计时和误差输出 | 现有D1课程和测试峰值规则 |
| `tools/full_eval.py` | 同封装计时和实际执行计数；保持老结果独立 | QK/AV算量与未解析算子检查 |

所有新checkpoint必须注明表示合同，至少包括：`teacher_checkpoint`、`teacher_graph`、`native_axis`、`detector_axis`、`tia_scope`、`candidate_stride`、`tubelet_phase`、`padding_policy`、`route_unit`、`fill_policy`、`regime`、`state_key`、`parent_checkpoint`。形状相同或strict loading只是必要条件，不代表语义兼容。

## F2. 张量接口

输入为[B,1,3,768,H,W]与valid_candidates[B,768]；原clip RGB为[B×48,3,16,H,W]。S/B通道C为384/768；每clip token为[B×48,8×P,C]，160输入P=100。

原轴完整状态建议统一为[B,384,P,C]，池化后[B,C,384]，原detector输入[B,C,768]。H65/BMCR另一合同为[B,192,P,C]→[B,C,384] rank检测，禁止静默混用。

RoutePlan至少有clip/tubelet ID、物理中心、有效支持区间、valid_count、depth/heavy mask、spatial index、estimated/observed provenance。NativeState至少有features、valid、physical_time、observed、support以及confidence可选项。teacher cache包含可复用条件，不能只有tensor；计算账本记录实际调用与token数。

## F3. 必要一致性测试

全选local等于local，global等于global；不要要求global与local等价。均匀插值残差初值等于Z24；不许重命名为BMCR等价。局部cached/virtual与compact对同route在FP32/AMP下分别报告误差。禁用heavy位置的hook须确认该算子未执行，不能只查看mask。

覆盖K固定、索引唯一、nested深度集合、零logit fallback、有效前缀/部分clip、奇数候选tubelet、padding不计作观察；物理正逆映射和pre-NMS轴正确；测试rank IoU反例不应被误当成映射bug。

冻结teacher/detector参数与buffer直接张量相等，混合输入梯度非零，重建/出口更新可测；router梯度路径单独测试。零末层首步上游梯度为零不是失败条件，应检查末层第一步、上游后续步。KD不经过detach的NMS结果，student输出语义与teacher目标匹配。

utility检查交换方向、动作合同、miss饱和、分量尺度和S0/S1标签匹配。计数检查融合attention、full/partial/short及稀疏MLP真实token；未知矩阵算子应使profile失败，而非悄悄漏算。

## F4. 扩大与停止的研究依据

支持扩大：同checkpoint控制指向明确瓶颈，拟合与真实混合任务收益一致，实际执行符合声明，完整测试的多阈值和困难事件分析支持收益，成本在匹配封装下有意义。

支持改路线：oracle交换也缺少收益、被删除证据不可观测、强表示被轻状态持续破坏、只改善低阈值却恶化高阈值定位、复杂路由不如规则降分辨率/固定深度、或GPU节省被解码/同步/重建吞没。停止具体配置不意味着否定所有条件计算；保留可回退锚点，避免在失败表示上继续堆模块。

---

# 最终选择倾向

**先补同配方修正BMCR，再以现有D1做route/fill/EMA分离，随后优先做固定uniform24的C2残差重建与混合检测监督。** 原因是这三步分别解决“可靠成果锚点”“早期下降属于哪一部分”“dense拟合是否缺少部署状态适配”，每步都能给出独立、可证伪结论，而不必先建完整新网络。

S用于机制与系统审计，B保留已有BMCR优势作为迁移验证；不因S修正H65更高就抛弃BMCR条件收益，也不因B旧BMCR最高就把其旧LR配方当新基线。正式长期候选更倾向全时间轻状态＋global-TIA＋条件昂贵更新，因为它最直接保护视觉证据和时间几何；但只有在简单C2与静态分辨率/深度对照之后，才值得承担其工程与训练风险。

三维冗余不必通过三种独立模块同时删除。更合理的统一对象是“哪个时空位置，在哪一层，还值得一次昂贵更新”。不能承诺消除的风险包括：廉价观察丢失小动作、真实未观测信息不可恢复、teacher的局部/global偏差、硬路由的优化偏差、少数/重复事件定位敏感、单seed与测试峰值选择的泛化不确定性，以及4090上稀疏算子的系统开销。成功标准应由匹配协议下的性能—成本实测支持，而不是由结构叙事保证。

---

# 来源索引与阅读边界

主仓库源码引用均固定到239d098cd899936c35989fae6243c70259a85adb；下列方法论文按所列版本阅读。作者实现只核对明确标注的文件/README，不声称复现其完整训练或其论文原始commit。部分CVF PDF直接访问失败时使用作者arXiv全文；PDF图示在可访问时经截图核对，未把失败的截图当已读图证据。

[^C01]: `ds3_20260912/research_context_20260912/CONTEXT.zh.md`，固定主提交。当前实验上下文；完整关键叙述已读。 [源码/记录](https://github.com/yuzbo/BMCR-T-AdaTAD/blob/239d098cd899936c35989fae6243c70259a85adb/ds3_20260912/research_context_20260912/CONTEXT.zh.md)

[^C02]: `ds3_20260912/research_context_20260912/latest_snapshot.json`，固定主提交。机器最新快照；超大返回存在截断，以可读部分与CONTEXT交叉核对。 [源码/记录](https://github.com/yuzbo/BMCR-T-AdaTAD/blob/239d098cd899936c35989fae6243c70259a85adb/ds3_20260912/research_context_20260912/latest_snapshot.json)

[^C03]: `ds3_20260912/retrospective_20260912/REPORT.zh.md`，固定主提交。完整历史回顾；读取主报告并补读历史课程部分，未声称审完所有外链归档。 [源码/记录](https://github.com/yuzbo/BMCR-T-AdaTAD/blob/239d098cd899936c35989fae6243c70259a85adb/ds3_20260912/retrospective_20260912/REPORT.zh.md)

[^C04]: `phase2_20260910/comparison.json`，固定主提交。旧H65/BMCR与官方比较；读取关键结果。 [源码/记录](https://github.com/yuzbo/BMCR-T-AdaTAD/blob/239d098cd899936c35989fae6243c70259a85adb/phase2_20260910/comparison.json)

[^C05]: `fidelity_20260911/FINAL_COMPARISON.json`，固定主提交。修正H65最终比较与测试峰值规则；读取结果部分。 [源码/记录](https://github.com/yuzbo/BMCR-T-AdaTAD/blob/239d098cd899936c35989fae6243c70259a85adb/fidelity_20260911/FINAL_COMPARISON.json)

[^C06]: `fidelity_20260911/INTERMEDIATE_RESULTS.json`，固定主提交。中间完整测试记录；结果完成状态与曲线。 [源码/记录](https://github.com/yuzbo/BMCR-T-AdaTAD/blob/239d098cd899936c35989fae6243c70259a85adb/fidelity_20260911/INTERMEDIATE_RESULTS.json)

[^C07]: `fidelity_20260911/TRAINING_COMPLETION.json`，固定主提交。训练完成回执；不是本次权重验证。 [源码/记录](https://github.com/yuzbo/BMCR-T-AdaTAD/blob/239d098cd899936c35989fae6243c70259a85adb/fidelity_20260911/TRAINING_COMPLETION.json)

[^C08]: `ds3_20260912/PLAN.md`，固定主提交。D1计划、初始化和课程。 [源码/记录](https://github.com/yuzbo/BMCR-T-AdaTAD/blob/239d098cd899936c35989fae6243c70259a85adb/ds3_20260912/PLAN.md)

[^C09]: `ds3_20260912/IMPLEMENTATION.md`，固定主提交。实现说明及既有CPU/4090预检记录。 [源码/记录](https://github.com/yuzbo/BMCR-T-AdaTAD/blob/239d098cd899936c35989fae6243c70259a85adb/ds3_20260912/IMPLEMENTATION.md)

[^C10]: `ds3_20260912/RESULTS.md`，固定主提交。20:02较早结果；用于D768G/L/Z24及计时范围，不用于覆盖21:56新状态。 [源码/记录](https://github.com/yuzbo/BMCR-T-AdaTAD/blob/239d098cd899936c35989fae6243c70259a85adb/ds3_20260912/RESULTS.md)

[^C11]: `ds3_20260912/decisions_20260912/compute_estimates.json`，固定主提交。算子估算；本次对S骨干分量作独立代数核算。 [源码/记录](https://github.com/yuzbo/BMCR-T-AdaTAD/blob/239d098cd899936c35989fae6243c70259a85adb/ds3_20260912/decisions_20260912/compute_estimates.json)

[^S01]: `h65/full/model.py`，固定主提交。FormalH65；route、encode、train_batch、raw_route、predictions。 [源码/记录](https://github.com/yuzbo/BMCR-T-AdaTAD/blob/239d098cd899936c35989fae6243c70259a85adb/h65/full/model.py)

[^S02]: `h65/full/scout.py`，固定主提交。scout及条件效用头、伙伴关系与前向。 [源码/记录](https://github.com/yuzbo/BMCR-T-AdaTAD/blob/239d098cd899936c35989fae6243c70259a85adb/h65/full/scout.py)

[^S03]: `h65/full/utility.py`，固定主提交。反事实交换、分类/定位分量、漏检与尺度。 [源码/记录](https://github.com/yuzbo/BMCR-T-AdaTAD/blob/239d098cd899936c35989fae6243c70259a85adb/h65/full/utility.py)

[^S04]: `tools/full_train.py`，固定主提交。main中的joint/preflight初始化分支。 [源码/记录](https://github.com/yuzbo/BMCR-T-AdaTAD/blob/239d098cd899936c35989fae6243c70259a85adb/tools/full_train.py)

[^S05]: `tools/full_eval.py`，固定主提交。评测及矩阵/卷积算量与计时。 [源码/记录](https://github.com/yuzbo/BMCR-T-AdaTAD/blob/239d098cd899936c35989fae6243c70259a85adb/tools/full_eval.py)

[^S06]: `h65/full/runtime.py`，固定主提交。优化器分组、EMA、配置和加载。 [源码/记录](https://github.com/yuzbo/BMCR-T-AdaTAD/blob/239d098cd899936c35989fae6243c70259a85adb/h65/full/runtime.py)

[^S07]: `h65/transport.py`，固定主提交。率校准、sample_rates、硬RGB及梯度代理、tubelet恢复。 [源码/记录](https://github.com/yuzbo/BMCR-T-AdaTAD/blob/239d098cd899936c35989fae6243c70259a85adb/h65/transport.py)

[^S08]: `h65/full/geometry.py`，固定主提交。TrueTimeMap、选择/交换和rank映射。 [源码/记录](https://github.com/yuzbo/BMCR-T-AdaTAD/blob/239d098cd899936c35989fae6243c70259a85adb/h65/full/geometry.py)

[^S09]: `h65/ds3/model.py`，固定主提交。DenseTeacher、ClipEngine及DS3.native的密集/compact执行。 [源码/记录](https://github.com/yuzbo/BMCR-T-AdaTAD/blob/239d098cd899936c35989fae6243c70259a85adb/h65/ds3/model.py)

[^S10]: `h65/ds3/routes.py`，固定主提交。Policy、路由、嵌套深度、原时间元数据。 [源码/记录](https://github.com/yuzbo/BMCR-T-AdaTAD/blob/239d098cd899936c35989fae6243c70259a85adb/h65/ds3/routes.py)

[^S11]: `h65/ds3/losses.py`，固定主提交。dense辅助拟合及gradient-weighted residual proxy。 [源码/记录](https://github.com/yuzbo/BMCR-T-AdaTAD/blob/239d098cd899936c35989fae6243c70259a85adb/h65/ds3/losses.py)

[^S12]: `h65/ds3/auxiliary.py`，固定主提交。H0预览、ExitHead、低秩MLP近似与评分。 [源码/记录](https://github.com/yuzbo/BMCR-T-AdaTAD/blob/239d098cd899936c35989fae6243c70259a85adb/h65/ds3/auxiliary.py)

[^S13]: `tools/ds3_eval.py`，固定主提交。online/EMA入口、原轴检测、执行计数和计时。 [源码/记录](https://github.com/yuzbo/BMCR-T-AdaTAD/blob/239d098cd899936c35989fae6243c70259a85adb/tools/ds3_eval.py)

[^S14]: `h65/ds3/runtime.py`，固定主提交。辅助参数加载、EMA、课程。 [源码/记录](https://github.com/yuzbo/BMCR-T-AdaTAD/blob/239d098cd899936c35989fae6243c70259a85adb/h65/ds3/runtime.py)

[^S15]: `tools/ds3_train.py`，固定主提交。D1训练循环与8000更新。 [源码/记录](https://github.com/yuzbo/BMCR-T-AdaTAD/blob/239d098cd899936c35989fae6243c70259a85adb/tools/ds3_train.py)

[^S16]: `h65/full/data.py`，固定主提交。LoadFramesWithBoundaryValidity。 [源码/记录](https://github.com/yuzbo/BMCR-T-AdaTAD/blob/239d098cd899936c35989fae6243c70259a85adb/h65/full/data.py)

[^S17]: `upstream/opentad/models/backbones/vit_adapter.py`，固定主提交。读取1–550行；Adapter.forward、Block、PatchEmbed、VisionTransformerAdapter。 [源码/记录](https://github.com/yuzbo/BMCR-T-AdaTAD/blob/239d098cd899936c35989fae6243c70259a85adb/upstream/opentad/models/backbones/vit_adapter.py)

[^S18]: `upstream/configs/adatad/thumos/e2e_thumos_videomae_s_768x1_160_adapter.py`，固定主提交。实际S配置，12层都有TIA；native384→detector768。 [源码/记录](https://github.com/yuzbo/BMCR-T-AdaTAD/blob/239d098cd899936c35989fae6243c70259a85adb/upstream/configs/adatad/thumos/e2e_thumos_videomae_s_768x1_160_adapter.py)

[^S19]: `h65/full/objectives.py`，固定主提交。合法端点权重、warm/joint课程和辅助目标。 [源码/记录](https://github.com/yuzbo/BMCR-T-AdaTAD/blob/239d098cd899936c35989fae6243c70259a85adb/h65/full/objectives.py)

[^P01]: AdaFocus V2: End-to-End Training of Spatial Dynamic Networks for Video Recognition。CVPR2022；arXiv:2112.14238v2；读取原文方法和训练机制。 [原论文](https://arxiv.org/abs/2112.14238v2)

[^P02]: Dynamic Token Pruning in Plain Vision Transformers for Semantic Segmentation (DToP)。ICCV2023；arXiv:2308.01045v2；读取原文和训练比较。 [原论文](https://arxiv.org/abs/2308.01045v2)

[^P04]: RandLA-Net: Efficient Semantic Segmentation of Large-Scale Point Clouds。CVPR2020；作者arXiv:1911.11236全文；不将当前作者稿与所有CVPR补充细节视为逐字相同。 [原论文](https://arxiv.org/abs/1911.11236)

[^P05]: ETAD: Training Action Detection End to End on a Laptop。arXiv:2205.07134v2（2022-11-28）作者全文；重点读取梯度采样与推理成本边界。 [原论文](https://arxiv.org/abs/2205.07134v2)

[^P06]: TR-BERT: Dynamic Token Reduction for Accelerating BERT Inference。NAACL2021正式论文；读取方法、策略训练及任务设置。 [原论文](https://aclanthology.org/2021.naacl-main.463/)

[^P07]: CoLT5: Faster Long-Range Transformers with Conditional Computation。arXiv:2303.09752v2；读取light/heavy路径、路由与pretraining/finetuning条件。 [原论文](https://arxiv.org/abs/2303.09752v2)

[^P08]: Conditional Adapters: Parameter-efficient Transfer Learning with Fast Inference (CoDA)。arXiv:2304.04947v2；读取条件适配、已有预训练模型与稀疏训练条件。 [原论文](https://arxiv.org/abs/2304.04947v2)

[^P09]: QueryDet: Cascaded Sparse Query for Accelerating High-Resolution Small Object Detection。CVPR2022；作者arXiv:2103.09136全文；另核对作者qinfer.py。 [原论文](https://arxiv.org/abs/2103.09136)

[^P10]: PointRend: Image Segmentation as Rendering。CVPR2020；作者arXiv:1912.08193全文；读取训练点采样与推理细化。 [原论文](https://arxiv.org/abs/1912.08193)

[^P11]: 3DSSD: Point-based 3D Single Stage Object Detector。CVPR2020；arXiv:2002.10187v1；读取融合采样和原坐标候选。 [原论文](https://arxiv.org/abs/2002.10187v1)

[^P12]: Expediting Large-Scale Vision Transformer for Dense Prediction without Fine-tuning。NeurIPS2022；作者arXiv:2210.01035v1全文；聚类/重建机制。 [原论文](https://arxiv.org/abs/2210.01035v1)

[^P13]: Eventful Transformers: Leveraging Temporal Redundancy in Vision Transformers。ICCV2023；作者arXiv:2308.13494全文；事件更新、状态保存与计算选择。 [原论文](https://arxiv.org/abs/2308.13494)

[^P14]: Dynamic Tuning Towards Parameter and Inference Efficiency for ViT Adaptation (DyT)。NeurIPS2024；arXiv:2403.11808v2，2024-10-16；读取方法、MLP/attention对照、检测负面结果和训练开销。 [原论文](https://arxiv.org/abs/2403.11808v2)

[^P15]: Benchmarking the Robustness of Temporal Action Detection Models Against Temporal Corruptions。CVPR2024；arXiv:2403.20254v1，2024-03-29；读取FrameDrop、动作中心扰动和定位一致性。 [原论文](https://arxiv.org/abs/2403.20254v1)

[^I01]: QueryDet作者实现 `models/querydet/qinfer.py`，读取时blob `ac5b836c400066365279a1005bb14f737ec299a2`。检查 `_make_sparse_tensor`、`_make_spconv`、`run_qinfer`，非完整仓库复现。[所读blob](https://api.github.com/repos/Small-Object-Detection/QueryDet/git/blobs/ac5b836c400066365279a1005bb14f737ec299a2)

另核对的作者资料：DToP `README.md`，blob `f4bdd459123fb961b0c792084466cffd08e38520`，确认先训练base再以checkpoint剪枝适配；DyT `models/dynamic_adapter.py`，blob `76cd8d1c0ca59b58d46b59f3882066c32c2c4249`，确认 `_gumbel_sigmoid` 的straight-through和 `TokenSelect`。未将这些局部文件阅读称为作者所有训练/推理实现审核。

## 附件

`toy_audit.py`与`toy_results.json`给出本次标准库CPU代数检查。`CODEX_HANDOFF.zh.md`给出最小实现任务书。主报告中的新模型、实验预算收益与改进目标均是研究建议，没有新mAP、GPU时延或服务器作业结果。
