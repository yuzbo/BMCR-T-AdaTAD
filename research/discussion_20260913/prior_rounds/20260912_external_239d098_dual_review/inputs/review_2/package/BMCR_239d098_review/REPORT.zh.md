# BMCR-T / H65 / DS3：固定提交复审、条件计算机制与可证伪研究方案

**主审查提交：`239d098cd899936c35989fae6243c70259a85adb`**  
**实验事实截点：2026-09-12 21:56；研究报告日期：2026-09-12。**  
**范围：源码、公开记录、原论文及有限作者代码阅读；独立 CPU 数学检验。没有加载项目大权重、原视频，没有提交服务器作业，没有重做 GPU 或 211 视频测试。**

## 执行结论

最值得先完成的是已经授权的**同配方修正 BMCR**，而不是用新的弱初始化混合表示替换已有高质量状态。紧接着，用同一 D1 checkpoint 做“选点 × 缺失位置填充 × online/EMA”的小型诊断，并把 **Z24 插值作为精确可用初值**，验证一次真正面向混合学生输出的 C2 校准。长期架构倾向是：**保留全时间低成本视觉／空间状态和跨 clip 时间通路，把主要预算分配给 attention / FFN 的条件残差更新**。这同时保留 BMCR 的任务条件效用思想、H65 的边界先验和 DS3 的原坐标紧凑执行基础，但不强制保留所有旧模块。

现有证据**没有排除完整 clip**，也**没有证明 D1 延长或蒸馏足以恢复精度**。可执行的研究问题不是“clip 还是逐帧”，而是：在何种观察、状态、时间几何和训练图条件下，额外重计算的检测收益值得其成本？

---

# （a）现状判断：可靠事实、实现合同与未解决科学问题

## A1. 实验事实与版本账本

以下都是仓库记录中的结果，不是本次独立复测。[R1–R7]

| 骨干 | 官方提供 TAD 权重 | 旧 H65-C，60 轮 | 旧 BMCR-T，60 轮 | 修正 H65 | 修正 BMCR |
|---|---:|---:|---:|---:|---|
| S | 69.0126% | 62.8593% | 63.1563% | 63.4094%，60 轮峰值／终点 | 尚未开始，不能填写新成绩 |
| B | 71.1280% | 66.6128% | 67.3404% | 67.1328%，55 轮峰值；60 轮 66.9026% | 尚未开始，不能填写新成绩 |

平均 mAP 是 tIoU 0.3/0.4/0.5/0.6/0.7 的均值。旧 BMCR 相比同旧配方 H65 的平均增益为 S +0.2970pp、B +0.7276pp；S 的 @0.5 与 @0.7 反而下降。因此，“已完成新模型 B 路线中的最高分是旧 BMCR”成立；“BMCR 在所有骨干和阈值都最好”不成立。官方高于两者。修正 H65 的 LR 与真实裁剪端点同时改变，不能分摊二者的因果贡献。[R4–R7]

旧结果属于 `15280e5` 配方；当前修复不能倒推旧权重已修好。历史 H65-S 的 65.385724% 属于 `42dba3f`、30+60 轮、global-TIA192，不属于后来 `04c35a3` 的 local-TIA8。主审查没有把旧项目源码作为当前实现。纯净 OpenTAD 来源按上下文记录为 `346d09d19e2091372cec48172dbe40f7b28bdee6`；本次审查的是主提交内包含的上游文件，而非重新对整个上游仓库做逐文件独立哈希证明。[R1,R3]

| S 路线 | 已完成平均 mAP | 新增训练与解释 |
|---|---:|---|
| D768G | 69.0126% | 官方 global-TIA384，48 clips × 12 层 |
| D768L | 67.5344% | 同权重变 local-TIA8，仍全算；不是省算模型 |
| Z16 | 43.2835% | 零新增训练；均匀完整 clip + 物理中心插值 |
| Z24 | 56.1065% | 零新增训练；不是训练 24 轮，也不是整个检测器随机初始化 |
| Z36 | 62.5804% | 零新增训练；同类预算探针 |
| ZR24 | 53.0022% | 覆盖锚点 + 固定种子随机完整 clip 对照 |
| T24A，第 5 轮 EMA | 50.6458% | 500 次 D1 辅助更新后的早期结果，非 80 轮终局 |

最新快照记录 S D1 已保存 15/80 轮、1500/8000 更新，第 10 轮全测试仍在运行。`RESULTS.md` 和回顾中“尚无训练后成绩”等句子生成于更早时刻，应由 21:56 的 `latest_snapshot.json` 覆盖，而不能宣布记录互相造假。原始第 5 轮数值是 **0.5064583163849076**。[R1–R3,R9]

**预算曲线支持什么？** 在固定 local 权重、原 clip 单元、原检测轴和插值规则下，更多重观察改善了结果；均匀 24 优于此固定随机 24 约 3.1043pp，支持对覆盖空洞敏感的解释。它没有控制训练适配、全局上下文、补全模型、采样粒度、教师图，因此不能排除“经过适当训练的完整 clip 条件计算”。三个均匀预算的选点也不应未经核查便假定严格嵌套。

**可证伪预测：** 若问题主要是覆盖和接口校准，在相同选点下把稳定插值替换成校准后的残差补全，或使学生检测器见过该混合表示，应改善相关边界／召回误差；若始终不能改善，并且误差集中在完全未观察且无法从廉价状态辨认的短事件，则应提高观察覆盖或改变选择层级，而不是只延长训练。

## A2. 先明确三套不可混用的模型合同

候选时间步长为 4 个原视频帧。这里的“16 帧 clip”是 16 个候选观察，首末采样点跨 60 个原帧间隔，不是 16 张连续原视频帧。[C14,R8]

| 合同 | 重骨干输入 | native 特征 | 检测轴 | TIA |
|---|---|---|---|---|
| 官方 | 原 48×16 候选 | 384 tubelets | 插值至原 768 | global384 |
| 旧／修正 H65、旧 BMCR | 选中 K384 重新拼为 24×16 | 192 tubelets | selected-rank384，NMS 前回映 | global192 |
| DS3-L | 保留原 clip 身份，0/8/12 条件执行 | 拼回原 native384 | 原 detector768 | local8 |

`global-TIA` 不是跨全视频的全连接 temporal attention。当前实现把同一空间格点的时间序列接起来，经低维 **kernel-3 depthwise temporal convolution + pointwise convolution**；global 的关键是跨原 clip 接缝继续传播。attention 本身仍在每个 16 候选 clip 内执行。local8 改变卷积边界和跨层上下文传播，计算量却可以不变。[C15]

严格加载和相同张量形状都不足以证明上述三种合同相同。rank384 检测器见过不同密度的邻接、GT 长度、中心采样半径和金字塔回归范围；回映原坐标只修复输出端点的单位，不自动恢复骨干运动统计、原检测轴的目标指派或丢失语义。[C1,C4,C16]

## A3. 已经正确的部分与仍需验证的部分

### 确定性纠错已经在源码内，不应重复当成新发现

`h65/full/runtime.py::optimizer_for` 按真实 encoder/decoder `conv_out` 参数身份区分动作分类头，内部 attention/MLP 投影归 trunk LR；`h65/full/data.py::LoadFramesWithBoundaryValidity` 委托上游完成随机裁剪，再标记真实端点；`objectives.targets` 排除裁剪制造的端点。这些不是尚待修复的当前错误。[C3,C5,C6]

固定 K 采样处理有效前缀、短窗 `min(K,L)` 个独立观察和剩余物理填充；NMS 前回映正确；BMCR 漏检通过 dummy/capped 定位代价保留，不把未匹配 GT 丢出分母。DS3 确实压紧实际 clip 和后四层重 MLP token，而不是冻结参数后仍宣称跳算。计数器已经补上融合 attention 的 QK/AV，scope hook 返回 None，预检记录验证教师参数与 buffer 不变。这些都应保留。[C4,C7,C8,C10,C12,R8,R9]

另一个容易误读的命名是 joint 的 `teacher_rows`：它是 batch 中另一部分视频的均匀 companion 行，不是同一视频额外做 dense teacher 监督。warm 阶段也训练 ASFormer，不能把 warm 描述成只训练 CNN 或只训练检测头。[C1,C2,C6]

**本次没有发现足以解释全部掉点的、已独立复现的新增致命运行错误。** 下面主要是监督和模型合同之间的可定位缺口，而非把正常设计选择都列为 bug。

### BMCR：S0 的单交换监督不等于 S1 的全局重新解码效用

`FormalH65.route` 先从基础分数形成 S0；`FormalScout.condition` 利用候选、相反成员伙伴和集合上下文预测两个标准化效用分量；取均值加回基础分数后，再通过固定预算采样器形成 S1。`train_batch` 的真实检测输入可为 S1，但 `counterfactual_targets` 测的是 **S0 附近的一次具体交换**。[C1:45–57,71–118; C2; C7]

可靠之处是：真实重编码交换、方向定义明确、分类分量与定位分量分开、定位在原时间评估、无效交换跳过、只有真正互为伙伴时才能复用相反符号。仍不能推出 S1 一定降低任务损失：重新求容量阈值、平滑和系统采样可以同时移动多个选点，跨 cell 也受影响；伙伴会变，clip 重新组装又会改变 motion/tubelet 配对。一次交换的可加性和局部近似范围没有得到最终模型级验证。

分类交换代价冻结 S0 的目标指派，有利于避免“目标变了”污染标签，但不等同于实际 rank 训练时重新指派的分类损失；定位重做匹配，分母包括漏检，但 score threshold、cap 和 dummy 值会形成不连续和饱和区。分类/定位标准化尺度属于训练集数据分布，修正 warm 后必须重新审计；两分量等权平均是可检验的设定，不是自然单位一致性。[C7]

**最小新审计：** 对当前 EMA 分别测 `S0→一次交换`、`S0→实际S1`、`S1→一次交换` 的分类、定位和联合损失；记录移动数、物理位移、伙伴变化、符号正确率、效用排序与真实改善的相关性。检验交互项

\[
I(a,b\mid S)=\Delta U(a\cup b\mid S)-\Delta U(a\mid S)-\Delta U(b\mid S).
\]

若单交换预测好、实际 S1 经常变坏，应优先限制一次可改变的交换数量、重估条件效用或使用受限顺序决策，而不是先增加效用头容量。限制只根据推理可用分数/状态执行，不能推理时偷用 GT 确认改善。

### DS3：主要缺口是学生训练图，不是“有没有算完整教师”

D1 的监督是 dense teacher 特征拟合、边界加权和梯度加权残差代理。teacher 的 GT 检测损失对 teacher feature 求梯度，标签使用形如 `sum(abs(g * residual))` 的非负量；它不是有符号的一阶损失变化，也不是在实际混合学生状态处求得的导数。时空／层间误差相互抵消、放大和置信度排序变化都可能被漏掉。[C9]

进一步，当前 head0 预测 H0→最终深层的收益，head8 预测 H8→最终深层的收益；DAD 的第一步却是决定谁值得进入 8 层，而不是直接进入 12 层。此处 0→12 与 0→8 的动作效用需要分别定义。T24A 只使用 24 个完整 12 层 clip + H0，没有使用 H8 或稀疏 MLP；因此第 5 轮不能用来判 H8 或 MLP 近似失败。[C8,C9,C11]

空间 score 当前监督的是稠密输入上的 MLP 近似误差，不是条件检测收益。推理后续层接收已被近似改变的输入，D1 却在 dense 输入上独立训练这些近似器；这个递归分布差异即使单层拟合很好也可能累积。冻结的检测器仍会对新的混合特征尺度和边界分布敏感。局部 dense 67.5344% 是参照，不是严格上限；但拟合 local teacher 本身没有直接提供恢复 global 行为的监督。[C8–C10]

### 初始化和 EMA 应拆开，而不是给早期低分找单一原因

H0 的 MobileNetV3Small 视觉部分有 ImageNet 预训练，但投影、时间模块、输出和路由新初始化；它不是从 Z24 插值、训练好 H65/BMCR native 特征或官方特征恒等开始。H8 有残差形式，但不意味着初值等于最终深层输出；低秩近似器也不保证初值等价原 MLP。[C10]

EMA 0.999 在 500 次更新后保留初值参数系数 `0.999^500=0.6063789`；半衰期约 692.8 更新，1500 次后系数约 0.2230。它们是参数平均系数，不是输出或 mAP 的混合比例。早期更新还经过 LR warmup。应比较 online/EMA 的同选点补全误差、实际检测结果和 BN buffer 状态，再比较各自路由；不能把所有掉点归于 EMA。`load_aux(..., ema=False)` 已支持 online 读取，评估 CLI 扩展即可。[C13,C17]

零 logits 也不产生均匀 top-k：当前 48→24 的覆盖锚点为 `[0,9,19,28,38,47]`，其余 stable tie 依次取较小 ID，得到 `[0..20,28,38,47]`。这只是零分数条件下的确定性反例，**不是对当前已训练路由选点的测量**。最大相邻 clip-ID 间隔为 10，而对应均匀 24 为 3。应显式实现均匀先验或均匀路由阶段，不能靠零初始化暗示它。[C11; Toy]

## A4. “冗余”的统一任务定义与性能保证边界

令状态 \(s\) 包含当前可用观察、轻/重特征、物理时间、有效掩码、覆盖、边界不确定性与已花成本；动作 \(a=(\text{时间单元},\text{空间单元},\text{深度升级})\)。定义

\[
\Delta U(a\mid s)=\mathbb E[\ell_{TAD}(D(s),Y)-\ell_{TAD}(D(T_a(s)),Y)\mid s],\quad
 a^*\in\arg\max_a\frac{\Delta U(a\mid s)}{\Delta C(a\mid s)}.
\]

这是研究定义，而不是已实现算法。真实约束应包括预算、最大物理空洞、有效观察数、事件/边界覆盖和状态可观测性；各项权重需要训练集校准。mAP 涉及跨视频排序、阈值匹配与 NMS，不能直接把 batch 标量训练损失称为“mAP 边际收益”。

像素相似可以是两个动作发生前后的同一姿态；低 attention 的 token 可能提供其他 query 必需的 key/value；低绝对梯度可能来自 teacher 已正确拟合、局部平坦或饱和。独立 toy 中，`L(f)=(f²−1)²` 在 teacher `f=1` 梯度为零，但替换为 `f=0` 后损失增 1。因此低梯度不提供可删除性证明。[Toy]

帧效用不能一般地相加成 clip 效用：两帧联合才显示运动方向时存在互补；多帧反复显示同一物体时存在重复。只有按实际条件状态连续更新的边际值才可望远镜相加：\(U(S\cup\{a,b\})-U(S)=\Delta U(a|S)+\Delta U(b|S\cup a)\)。假设同一 S 下独立相加相当于忽略交互，不是被数据保证的事实。

**三种保证要区分。** 全预算、相同算子/位置/归一化/掩码/精度的恒等执行可通过严格单测建立；有限数据上可建立经验精度—成本前沿和不确定性；固定少观察对所有视频的性能无条件保持不能保证——两个视频在全部保留观察上一致、只在遗漏区间存在不同事件时，任何只看到保留信息的系统都无法区分。这个信息反例不证明 THUMOS 上实际损失有多大。

较强的局部稳定性论证还需要特征误差界、检测函数敏感度界，以及分类排序、tIoU、匹配和 NMS 决策都远离翻转边界。当前没有这些可用证书。即使原坐标回映可逆，非仿射变换下 IoU 也不保持；toy 中 1/3 可以变为 1/20，但不能将这个反例写成全部 mAP 损失的定量解释。[Toy]

---

# （b）跨领域机制图谱：同类问题，而非论文名称集合

## B1. 保留便宜状态，选择性增加证据

**视频空间动态计算——AdaFocus V2（CVPR 2022，arXiv v2）。** 原问题是视频识别中不必每帧整幅高分辨率重算；机制是低成本全局观察、可微局部 crop、辅助分类监督和抑制政策梯度干扰。动态训练稳定性是方法的一部分，不是把局部网络随机接入已有分类器就结束。[P1]

**对 TAD 的推导：** 小图全时域状态可判断动作开始、背景段与空间兴趣区；高分辨率局部算子负责手部、小物体或多主体的细节。但视频级分类得到一次高置信答案就退出，不等于 TAD 可以停止搜索后续同类动作。覆盖约束应按时间段和可能主体分配，而不是每个视频只保留一个最显著 crop。空间 ROI 路由漏掉小主体时，后续再好的检测器也没有该视觉证据。

**多速率视频——SlowFast（ICCV 2019）。** 快路径以低通道高时间频率保留变化，慢路径提供低频但较丰富的空间语义，并做跨路径融合。它是在该异构结构上训练的识别/空间动作任务模型，不是少帧 THUMOS TAD 无损的证据。[P2]

**对 TAD 的推导：** 比当前“两个低分辨率图像特征求均值”更值得检验的是顺序敏感的廉价 2/4 帧 temporal embedding，使全时间轻状态至少能表达局部运动方向。若 lightweight 状态本身对开始/结束不可辨认，深层路由再精巧也只能在盲区猜测。不能为了复制双路外形再加一个昂贵 ASFormer 和完整 MobileNet，而应核算复用 scout 与共用 stem 的取舍。

## B2. 密集输出并不要求每个位置等深计算，但需要明确“退出后留下什么”

**DToP（ICCV 2023，arXiv v1）。** 在语义分割中用辅助预测判断容易 token，保留这些位置的预测，并保留每类若干代表 token 作为后续上下文；未退出位置继续计算。作者固定代码 README 明确包含从 base checkpoint 的 pruning fine-tune 流程。[P3,A1]

**对 TAD 的推导：** 一个时间位置被认为“背景容易”不代表后面的短动作不依赖它提供的前状态；每类代表 token 也不保证保留多次同类事件。应保留时间位置、退出特征/预测、事件覆盖和低成本上下文更新，而不是删除后改变邻接。置信度须校准；边界附近“低熵但错误”是直接失败模式。

**TR-BERT（NAACL 2021）。** 在 QA 等任务中分配 token 深度，退出 token 的表示仍保留，便于原 token/span 输出；训练包含任务适配、路由学习和联合优化，而不是只训练 dense feature 近似器。[P4]

**对 TAD 的推导：** 这与“所有时间位置必须仍能产生边界/区间判断”同构；不同处是文本 token 已携带离散语义，视频 patch embedding 远未完成识别。可迁移的是多深度统一输出和被跳过位置的状态合同，不是直接将词级删除比率当作 RGB 删除预算。一个时间 token 是易分类背景，也可能是邻近 action start 的参照。

## B3. 全 token 轻路径与选择性重路径：最贴近本项目的条件计算机制

**CoLT5（EMNLP 2023，arXiv v3）。** 全 token 走轻量局部 attention/FFN，一部分 token 走重 FFN 和重 attention；query 与 key/value 路由分开。模型在条件计算结构上做约百万步 C4/UL2 预训练，再做任务适配，不能归类为纯 D1。[P5]

**对 TAD 的推导：** “需要被深度更新的位置”与“必须提供上下文的位置”不一定相同。可以保留全部 K/V 的轻状态，仅让重要 query 深算；或对不同空间/时间 token 分配 FFN。预训练资源相差悬殊，不能据此保证 200 视频上重新初始化路由和出口足够。合理迁移是保留已经学到的 VideoMAE/TAD 状态，再逐步学习重残差分配。

**Dynamic Tuning / DyT（NeurIPS 2024，arXiv 2403.11808v2）。** 训练时完整计算被路由的 block，再 mask 输出残差，并用 Gumbel-Sigmoid 代理训练路由；全 token 仍经过轻 adapter。论文明确区分训练全算与推理 gather/scatter。这证明“训练保留完整算量”与“学生见过缺失更新”不冲突。[P6]

**作者实现核查边界很重要：** 本次固定阅读 `d1744f0...` 的分割 `Block.forward` 和 `TokenSelect`：可见真实 hard/soft straight-through 门控、全 MLP 后乘 mask、残差及 adapter 相加。在这个读到的分割 Block 中，**没有看到 eval 自动切换为 compact MLP 的分支**；不能仅凭论文推理图和 mask 代码宣称该入口已实现实际压紧加速。这里只借鉴其学生状态训练机制，部署仍需本项目的真实输入计数验证。[A2]

**数学迁移条件：** token-wise MLP 在相同输入上先全算后 mask，与 gather→MLP→scatter 可等价；attention 若只 mask query 输出而训练仍使用全 K/V，推理却删除 K/V，就不是同一个函数。独立 toy 验证前者误差为 0，后者非零。真正的 C2 需要按算子逐个核对，不能用一个“稀疏 mask”概括所有条件执行。[Toy]

## B4. 合法的训练—推理异构反例：不必一律真实稀疏训练

**PointRend（CVPR 2020）。** 训练采用随机/不确定性点采样；推理逐级上采样并细化不确定位置。它保留粗预测，并从已有细粒度特征按坐标取信息，再由点头更新输出。[P7]

**QueryDet（CVPR 2022）。** 密集 backbone/FPN 提供状态，粗查询决定哪些高分辨率位置调用稀疏检测头；训练仍可使用密集检测分支。省算主要在高分辨率检测头，而不是在读取原图前删掉目标。[P8]

**对 TAD 的联合推导：** D1/C2/J3 的边界应由函数和可用状态决定。若输出头在两种执行下得到相同局部特征及足够卷积 halo，稀疏索引只是调度，未必需要重新学习所有主干；若依赖邻域被删、特征来源混杂或坐标改变，就需要校准。TAD 可以用“全时域粗边界 + 局部重细化”，但粗查询漏掉的事件不能由细化阶段凭空补出。因此应保留随机/均匀覆盖，并报告未被查询的 GT 比例。

**Expediting Large-Scale Vision Transformer for Dense Prediction without Fine-tuning（NeurIPS 2022，arXiv v1）。** 它在已有浅特征上做局部 token clustering，用少量聚类 token 计算中间层，再利用保存的高低分辨率关系重建密集特征，实证研究了无需 fine-tune 的场景。[P9]

**对 TAD 的推导：** 这是反对“任何稀疏推理都必须 J3”这一绝对命题的证据；但它先观察并编码了高分辨率输入，不能用于证明先删 RGB 也无损。重建必须保留位置关联和细粒度判别信息；把多个主体的相似背景 token 合成一个平均状态可能抹去小动作，TIA 前还必须恢复规则时空格。

## B5. 不规则几何：随机并非原罪，未建模的邻域丢失才是关键

**RandLA-Net（CVPR 2020，arXiv v3）。** 用低开销随机点采样，但配合显式相对几何的局部聚合、逐层扩大感受野及解码/跳接，恢复逐点语义；并非先随机扔掉大部分原始证据再要求输出自然保持。[P10]

**3DSSD（CVPR 2020）。** 几何最远点与特征导向采样具有不同盲区；融合采样及 candidate generation 为 3D 检测保留前景和位置支持，减少某些昂贵特征传播/后续阶段。[P11]

**对 TAD 的联合推导：** 物理时间上的均匀覆盖类似几何覆盖，可避免大洞，却不必最有任务效用；仅前景评分又会遗漏背景到动作的过渡。适宜的是“规则覆盖锚点 + 边界/事件先验 + 条件收益”，且在采样前保留局部运动聚合。点云用相对 XYZ，TAD 至少需要真实 Δt、有效长度、距上次重更新的 age、邻域的物理跨度。点数减少、随机覆盖成功和某个 3D 检测前景召回指标，都不能直接替代 TAD mAP 证据。

## B6. 用状态更新替代重复重算：与跨窗缓存的真正关联

**Deep Feature Flow（CVPR 2017）。** 关键帧重特征经当前图像估计的运动场传播到其他帧，再接密集预测；任务监督和传播误差有关。遮挡、新出现目标和错误运动都可能破坏旧状态。[P12]

**Eventful Transformers（ICCV 2023）。** 为 token 保存上次更新参考和输出 buffer，按累计变化选择更新，scatter 回原位置；进一步处理 QK/AV 的增量更新。论文的检测与视频识别有不同适配需求：识别中的 temporal 子网络会受新的空间特征分布影响，作者专门适配了该下游部分。[P13]

**对 TAD 的推导：** 缓存是一种带时间和上下文条件的状态，不是跨窗免费特征。VideoMAE 的 16 候选 clip 上下文、tubelet 相位、空间 crop、位置嵌入以及 global-TIA 的窗口邻居都必须一致，才可能精确复用。更稳健的新方向是在同一窗口/同一原坐标状态上积累“未执行残差”的误差估计，超过任务不确定性或 age 条件再更新。先实现 token-wise FFN 状态，不宜一开始保存巨大 attention 矩阵；额外存储与内存流量会抵消 FLOPs 收益。

## B7. TAD 训练效率不等于 RGB 推理效率

**ETAD（arXiv 2205.07134v2）。** 顺序完成视频片段前向、选择部分梯度反传，并对训练 proposals 采样，目标是降低端到端训练资源消耗。[P14]

**对本项目的限定：** 可借鉴完整前向与选择性反传的分离，帮助 4090 训练条件计算模块；不能把它当作少 RGB 推理仍保持精度的实验证据。完整 teacher 前向、输入梯度、辅助反向和查询检测头成本仍需分别计费。

## B8. 从机制图谱提炼的创新空间

不是“再引入一个 sparse 模块”，而是把四项通常被分开处理的合同共同建模：**观察覆盖、物理时间、持续可用状态、实际检测条件收益**。时间/空间/深度共用动作价值和成本接口，但不共用未经校准的分数；选点与提供上下文的职责分离；从已有强基线做可验证的增量替换。

这是一组尚待验证的组合假设，不声称轻重路径、token early exit、残差补全或条件效用本身首次提出。相对 BMCR，新增的是“实际 S1/混合状态的动作效用和有限步可信更新”；相对 DS3，新增的是“稳定基线残差初始化、C2 学生任务图及全局状态维护”；相对通用剪枝，新增的是 TAD 的事件覆盖与真实时间/边界合同。

---

# （c）最多三条优先模型族：先保住状态，再分配计算

## C0. 粒度不是“16 或 1”的二选一

| 单元/层级 | 保留的结构 | 可能节省 | 主要失配/风险 | 优先用途 |
|---|---|---|---|---|
| 任意单候选 RGB | 灵活覆盖、边界附近可密采 | patch embedding 及之后 | 原 tubelet 两帧配对改变，rank 时间不均匀；运动统计改变 | 已训练 BMCR 合同内继续审计，不以新 clip 实验直接否定 |
| 原始 2 候选 tubelet | Conv3D 相位和两观察顺序 | 选择后深层算子；若解码前选还涉及视频依赖 | 与其余 7 个 tubelet 的 attention 上下文可能缺失 | 首先在原格点轻状态上选择重更新 |
| 4/8/16 候选 micro-clip | 原始局部连续性逐步增强 | 中深层重计算 | 单元越短，上下文越弱；越长，背景陪算越多 | 用同一接口比较 16 与 8，再视证据扩展 |
| 重叠 clip | 跨单元边界的运动连续性 | 可提高质量，不一定省算 | 同一观察被重复嵌入/attention，不能只计唯一帧数 | 短动作/边界诊断，不默认主模型 |
| 自适应长短段 | 背景长段轻算、边界短段细算 | 重路径覆盖/深度 | 变长 batch 碎片化、位置信息、路由难度 | 条件效用已校准后再研究 |
| 关键帧 + 周围局部片段 | 姿态与变化联合 | 重片段数减少 | 单纯关键帧不足以区分运动和重复事件 | scout 边界/不确定性附近补片段 |
| 全时间低分辨率 + 局部高分辨率 | 完整时间可观测性和粗空间状态 | 重分辨率/重空间路径 | 小动作/多主体在低分辨率不可见；ROI 漏选 | 与现有 H0 对照，但用稳定残差融合 |
| 全时间 patch embedding + 稀疏深层更新 | 原 tubelet 相位、全时间空间状态 | attention/FFN/深度，**不省已做的 RGB 解码** | 浅状态与最终检测表示不同，需校准 | 本报告最看好的长期主干形态 |

在 RGB 前选，理论上可省后续所有算子，但真实压缩视频的随机访问仍可能解码依赖帧；当前项目已经把输入全量解码并搬到 GPU，不能据现有路由声称省去这些成本。在 patch embedding 后选，保留廉价可观测性但不节省 stem；在 attention 前选必须说明 query/K/V 分别是谁；在 FFN 前选通常能保持所有 attention 邻接，仅省 token-wise 计算。冻结训练参数不属于任何一种执行跳过。[C8,C12,C14,C15]

**时间几何必须贯穿而不是只在 NMS 前修复。** 最小接口带原 `frame_inds`、candidate 有效 mask、tubelet 相位与中心、距最近重观察的 Δt/age、当前 feature source。采用原 native384→detector768 可保持原金字塔、中心采样和回归单位；若坚持 irregular/rank 检测，则卷积/邻接要使用物理跨度，聚合可考虑 Voronoi 时间权重，GT 指派使用真实点位置与物理回归范围，不能仍把每个相邻 rank 当等长时间。增加 Δt embedding 是补充信息，不是恢复被删语义的魔法。

## C1. 模型族 I：BMCR/H65 强底座上的守合同渐进压缩

**科学假设。** 先保持已经学会的稀疏时序合同，只降低其内部冗余，比同时换采样单元、检测轴、teacher 与融合方式更容易保住实际成绩。该假设允许被同预算原格点路线证伪。

**底座。** 完成修正 BMCR 后，与旧 BMCR 和修正 H65 分别保留；不假设修正 BMCR 必胜。现有 B 的新模型最高分锚点是旧 BMCR 67.3404%，S 的修正 H65 为 63.4094%；旧权重明确带旧配方标签。复用对应 backbone/adapter、rank384 detector、scout 与条件效用头，不用宽松加载混入 original768 detector。[R4,R5]

**训练图。** 第一阶段仅将 `FormalScout.condition` 的逐 cell/候选循环张量化，保持伙伴规则、相反成员过滤、tie、集合均值和最终 S0/S1 完全一致。第二阶段在该底座的 K384/native192 空间格点上加入可选 late-FFN 或深层重残差路径；训练仍可全量算选中观察，但送给任务损失的学生状态必须见过被跳过的重更新。保留原 GT 检测损失、合法边界监督、已校准 BMCR 效用；跨来源 teacher 只在物理时间预测端蒸馏。

**推理图。** scout→S0→受控 S1→K384 重观察→原 clip/tubelet embedding→逐层选重 attention/FFN→scatter 全 native192 空间格→global-TIA192→原 rank384 detector→NMS 前原时间回映。先只压晚层 MLP，以最小合同变化检查可行性；再测试时间单元的深层重更新预算。无需强制保留一个证明无益且昂贵的旧 utility head，但撤换应有同 checkpoint/同预算证据。

**初始化与梯度。** 计算重门初始全开，因此该 K384 模型精确回到原锚点；新增残差末层可置零，不能把“零末层”误说成整个网络零参数。首步上游梯度为零是这种参数化的正常结果，末层更新后梯度可流入上游。路由任务梯度采用显式 surrogate/score gate 或反事实效用监督，不靠 hard selection 的离散索引自然反传。

**成本与不能复用的部分。** 若仅后四层 MLP 保留 75%/50%，省算上限有限，不能包装成三维大幅去冗余。改变深度/空间后的效用标签需重新校准；原逐候选效用不直接变成 clip/depth 价值。若要切换为原 detector768，此时已是合同迁移，应单独训练/评估，而非声称仍为“原锚点等价压缩”。

**可证伪预测。** 精确向量化应保持选点和预测不变；若结果变了，是实现不等价而非研究收益。训练 C2 后，晚层局部近似误差下降却检测指标仍恶化，说明任务敏感残差不是普通低秩误差可代表；此时不继续盲加低秩模块。

## C2. 模型族 II：原 clip 身份 + 稳定插值底座 + 残差补全与条件升级

**科学假设。** T24A 的早期下降部分来自弱 H0 绝对输出替换高质量缺失位置基线和路由未校准，而非完整 clip 单元本身不成立。这个模型族首先检验初始化和学生图，之后才比较 clip 与 micro-clip。

**初始函数。** 固定均匀 U24，按现有 Z24 从所选完整 clip 得到最终 native 特征，物理中心插值为 \(F_{base}\in\mathbb R^{B\times C\times384}\)。令

\[
F_{student}=F_{base}+(1-M_{heavy})\odot R_\phi(F_{base},H_{preview},\Delta t,\text{source},\text{valid}),\qquad R_\phi\text{末层}=0.
\]

保持所选重位置不变，其余位置只学习“比当前插值还缺什么”。函数初始等价不代表成本等价：额外preview/残差即使输出为零，若仍执行也必须计费。在相同精度/原模型模式下，初始化可精确等于 **Z24**；不等于已训练 BMCR，也不等于官方 dense。只看 24 clips 时不存在普遍的官方初始函数等价保证。额外 H0 预训练与 teacher 来源仍要披露。

**训练图：首选 C2，而不是直接复杂 J3。** 对每个训练 crop，冻结 local teacher 一次完整 48×12 前向，缓存每原 clip 的 H8/H12；采样 U24 或混合预算，以这些缓存构造与推理相同的最终混合状态，运行冻结检测器的真实 GT 损失，保留学生 feature 梯度。混合状态补全、分类/定位预测和效用标签都在原时间轴。preview/H0 可以从当前已训练辅助参数做残差重参数化，但必须验证初始函数；不能直接将旧绝对输出权重载入残差分支假称初始为零。

**为什么缓存可用？** local-TIA8、冻结 eval、原 clip 身份/空间变换/位置相位/填充一致、无随机 dropout 或 batch 耦合时，某个 clip 的最终输出不依赖其他 clip，dense 缓存子集与单独真实执行可数值一致。这样评估 `24clip→一次交换` 或 `8→12` 的 label 主要重装 feature 并重跑 detector。应直接测 FP32/BF16 最大误差、输出框差异与执行计数。[C8]

**为什么不能泛化？** global-TIA 会传播到其他 clip；晚层稀疏 MLP 使后续层输入改变；训练态随机深度/BN 或不一致预处理也破坏独立性。这些情况下缓存完整 dense 深特征不等于真实条件学生。缓存仅是合法 local 图的执行加速，不能作为 global 路线的廉价精确反事实 oracle。

**教师和损失。** local teacher 提供可比 native feature；官方 global teacher 提供真实时间预测端监督；BMCR/H65 提供真实时间预测、动作/边界/覆盖先验。保留 GT。分类蒸馏匹配当前头的 **20 类独立 sigmoid/Bernoulli**，不套互斥 softmax KL；回归匹配物理端点或同构金字塔点的左右距离，当前头不是离散分布回归，不凭空对回归 logits 做 categorical KL。[C16]

建议损失结构为

\[
\ell=\ell_{GT}^{mixed}+\lambda_f\ell_{feature}^{valid,boundary}+\lambda_c\ell_{BernoulliKD}+\lambda_b\ell_{boxKD}^{matched}+\lambda_u\ell_{conditional\ utility}.
\]

各权重在训练集校准，先保留少量必要项逐个加入。特征项兼顾原幅度与 cosine，避免仅归一化导致 detector 输入尺度漂移；难边界/少数类权重必须检查是否损伤整体 score 排序。teacher 张量 detach，学生通过 frozen head 的路径不能包 `no_grad`。

**路由和课程。** 先固定 U24，让补全真正不劣于自己初始化；再用训练过的 scout 提供候选边界/覆盖先验，训练新的 clip 交换效用和 0→8、8→12 升级效用。允许 16/24/36 等混合预算以增加状态支持，但每个对照记录重算与教师查询成本。路由先验显式用 U24，不用零 logits 代替。H8 的零残差可等于 H8 输入，**不能**等于 H12；其释放要经过实际混合检测校准。

**推理图。** 全时间便宜 preview→U24/受覆盖约束的 adaptive route→真正压紧 local clips→Z24-like 物理插值底座→缺失位置残差补全→原 detector768。随后可将选择单元改为原 tubelet 或 8 候选 micro-clip，但这是单独的粒度干预。没有推理 dense teacher，更没有先 dense 再 mask 冒充省算。

**可证伪预测。** 若固定 U24 的残差 C2 明显改善 H0(U24) 的边界/置信度而无需增加重观察，则初始化/学生图有因果证据；若插值+适配仍对未覆盖短事件无能为力，应提高廉价全时间观察能力或转模型族 III，不据此推断“逐帧必胜”。

## C3. 模型族 III：全时间空间状态 + 条件重残差，保留 global-TIA

**科学假设。** 保留原 tubelet 的浅视觉证据和完整空间格、持续更新全时间廉价通路，主要跳过昂贵 attention/FFN，比先删除整段 RGB 更适合 TAD 的密集事件与边界需求。

**状态与前向。** 原始输入经过完整 Conv3D patch embedding，形成 `[B,48,8,10,10,C]`。可先做 0 或 2 层完整重计算；以后每层分开执行 attention/FFN 与 TIA：

\[
\widetilde X_\ell=X_\ell+\operatorname{scatter}(H_\ell(\operatorname{gather}(X_\ell,A_\ell)))+L_\ell(X_\ell),\qquad
X_{\ell+1}=\operatorname{TIA}^{global384}_\ell(\widetilde X_\ell).
\]

这里 \(H\) 表示该层被允许的贵残差更新，\(L\) 可以先为零/已有便宜更新，再按证据添加；该式是设计示意，attention→FFN 的顺序与 norm/drop-path 必须按原 Block 精确展开，不能把两个残差简单并联改变基线。未被重算的位置保留已有状态，并继续经过 TIA；不是用随机 H0 最终输出直接覆盖。原TIA在空间池化前有非线性投影，均值与非线性一般不交换，因此不能仅靠池化native特征重现原TIA。最终仍 spatial mean→native384→原插值 detector768。[C15]

**选择粒度。** 初版按完整原 clip 分配重 attention/深度，按原 tubelet 内空间 token 分配 FFN；原格点支持以后细化为 micro-clip/query 级重计算。全 K/V、选 query 的 attention 可以让少数被更新位置读取完整当前上下文；若删 K/V，也应保留几何/语义代表并在训练中使用同一 attention 图，单独验证，不称为等价。

**空间压缩。** 最小版只用稀疏 FFN，不删 TIA 所需格点。规则 160→128 是更强且通常更易形成大矩阵的控制，但不是已测精度；高低分辨率同时使用时，先在原坐标重建公共 10×10 状态，再进入全局 TIA。局部 token merging 限在单 tubelet/空间邻域并保存成员位置与权重，避免跨起止边界或不同主体盲合并。ROI refinement 保留全图粗状态与覆盖，不能只算单个最显著前景。

**深度含义。** “退出 8 层”可重新定义为停止该位置后四层的**贵残差**，而廉价时间上下文仍更新。这比保存 H8 到末端完全不动更一致。不同深度通过同一 native 维度、相同原时间 mask 和校准过的尺度进入检测头；depth/source/age 是可选条件，不应无监督加法破坏原特征初始值。

**训练图。** 首先 full-budget identity 单测。之后 C2 虚拟重更新缺失：所有候选与主干算子仍可完整前向，但主学生状态按预算 mask 掉部分贵更新，并用混合状态的 GT 检测损失训练 adapter、轻残差和路由。固定 teacher 与学生变化状态可能需要两个前向流；不得把改变输入后的 global 学生后续层拿 dense teacher cache 直接替代。若任务必须进入真实稀疏图才能校准，则显式升级 J3，按阶段解冻 adapter/检测头，保留密集教师和 GT，而不是暗改 D1 定义。

**初始化限制。** 全重门与零新增残差保证官方全预算基线；在第一天就减少一半贵残差不再精确等价官方，必须承认。这与模型族 II 在小预算上精确回到 Z24 是两种不同的初值承诺。训练阶段可以同时算基线和学生作监督，但部署不能先算 dense 基线再用 mask 装作节省。

**成本预测。** S 全量 patch/TIA/原 detector 保留、每层只有一半 clips 做 attention+FFN，算子模型为官方约 **53.21%**；前两层全重、后十层半重约 **61.01%**，均未加新路由/重建，也不是精度或时延预测。它比单削后四层 MLP 更有预算空间，但对深层任务表示的扰动也更大。[Toy]

**可证伪预测。** 若全时间浅状态确实保留了 clip-drop 丢失的关键证据，同等实际 FLOPs 下应优先改善短动作/边界覆盖；若只改善 feature MSE，不改善 GT 召回/定位或置信度排序，说明浅状态不足、条件重更新未到任务关键部位，或 global 轻通路无法弥补深层不足。不能用“保留所有 patch”代替最终性能验证。

## C4. 三条路线如何构成连续研究链

强 BMCR/H65 保留作为实际成绩锚点与真实时间预测教师；其 scout/边界知识迁移为先验，不把 rank native 特征直接当 original native 特征。模型族 II 以稳定局部基线验证条件效用、补全和学生训练图，提供原时间接口与真实 compact 引擎。模型族 III 再将“是否观察”分解成“是否做便宜观察”与“是否值得继续重算”，用全局状态保护任务信息。

三个模型族不是同时开跑三套全部 S/B 全训练。每一步由判别实验决定是否继续，既允许已有 BMCR 继续胜出，也允许完整 clip 或更细原 tubelet 获得优势。

---

# （d）最小判别实验顺序

## D0. 先补同配方修正 BMCR，不夹带结构变化

复用修正 warm20 EMA，重新用训练集审计分类/定位效用尺度，再 joint40 和同一完整测试/EMA 峰值规则。共享 warm 只计一次训练成本。保留峰值与第 60 轮终点；旧 BMCR 与新修正 BMCR 分栏，不能把不同课程结果解释为纯 BMCR 增量。此项已授权但未开始，报告不给新分数。[C18,R1]

如果修正 BMCR 相比修正 H65 没有稳定增益，应优先审计 S0/S1 效用，而不是因为“BMCR 是锚点”就忽略证伪。官方只复用提供的 TAD 权重，不新增官方完整训练。

## D1. 用已保存 checkpoint 拆开 T24A 的混杂因素

用同一 local teacher、同一辅助 checkpoint、同一 test protocol，先构造四个推理控制：

| 选点 | 缺失位置插值 I | 缺失位置 H0 |
|---|---|---|
| 均匀 U24 | 现有 Z24 | T24U |
| 同一 adaptive A24 | 新 I(A24) | T24A |

四格均为 24 个完整 12 层 clip、spatial ratio=1。它们比较的是路由和重建，不是 H8 或 MLP。I(A24) 的路由仍可能需要 preview；I(U24) 不需要，实际成本应分开披露。记录相同的有效重观察、物理 padding、实际各层重 clips，避免短窗把名义 K 当真实有效 K。

再选择其中最有信息的固定选点，比较 online 与 EMA 的填充；随后固定填充分支，比较 online/EMA 的路由。不要为所有深度、空间和 teacher 做全乘积。检查 H0 在有效 native 点的幅度分布、cosine、时间差分、边界/前景/背景/少数类误差；误差落在高置信分类排序还是框端点；检查路由空洞、周期8相位偏差、首尾偏置及实际效用相关性。

**需要等待什么？** 80 轮终局及 60→80 的同轨迹收益必须等既定训练完成；第 10/15 轮只能按之后的真实完整回执更新。本轮可以定义和运行已有 checkpoint 的诊断，但不能预填其结果；online 优于 EMA 或反之均待测。

## D2. 一个固定路由实验识别初始化与 C2 的价值

先用 U24，保持重骨干/检测头冻结。比较现有 absolute H0 fitting 与零末层 residual completion；先验证 step0 精确 Z24，再在相同成功更新数下比较 D1 feature-only 与 C2 mixed GT loss。不得一边更换 global teacher、一边改变粒度、再把收益归初始化。teacher 查询次数、full forward 次数、输入梯度反向次数分别记录；“相同更新数”不是相同训练计算量。

其后仅给胜出配置加入一种真实时间预测监督：local feature teacher 不变，增加 global prediction teacher。这样才能判断补充全局任务行为是否有贡献。若 C2 已解决主要失配，才讨论 J3；若 C2 的虚拟与真实 compact 不一致，先改模拟图，不用无边界训练掩盖实现不等价。

## D3. 粒度实验：先固定语义接口，再承认不可分离的因素

建议先比较 **原 16 候选 clip 与 8 候选 micro-clip**，必要时再加原 tubelet。不立即扫所有 1/2/4/8/16、重叠率和长度。第一组在全量 patch embedding 的原格点状态上做“重更新分组”控制：固定 teacher 权重、已有训练历史、TIA、detector768、GT、有效候选和空间分辨率，改变贵更新决策的绑定单元。这样尽量不改变 tubelet 相位和浅观察。

第二组才研究真实 RGB 删除：保持原 tubelet 配对，在预算内选择连续 2/4/8/16 候选块。删除会改变 attention 的上下文、embedding 输入或重组方式，不能宣称此时只有“粒度”一个因素不同。

**匹配口径分两层。** 一组匹配独立有效 RGB/tubelet 观察，报告不同上下文导致的 FLOPs 差异；另一组匹配实际 2MAC FLOPs，报告观察数不同。空间/深度/片段上下文不同的模型一般不能同时固定观察数、FLOPs、语义、训练查询成本。被迫不同的变量写进表格，而非用名义保留率掩盖。

## D4. 一次全局状态验证，再只加一种空间或深度策略

先验证模型族 III 的 full-heavy 等价，再比较 fixed-half-heavy 的 C2 与真实 compact，观察 global 与 local 图对接缝/边界的差异。只有出现合理精度—成本趋势，才加入后四层 FFN75 或一种更早停止重更新策略；不同时把分辨率、深度、teacher 和路由全部变化。

S 适合先验证实现、预算和失配机制，B 用于确认赢家是否依然有效。S 失败不构成 B 必败的证明；但在 4090 上，应先区分“代码/训练图失败”与“容量不足假设”，再花 B 的完整训练成本。

## D5. 评价、错误分层与不确定性

主指标保持五阈值平均及逐阈值 mAP，辅以逐类 AP、短/中/长动作 recall、起止误差、同类重复事件合并/漏检、假阳性排序、pre/post-NMS 变化。现有 `duration_recall` 为 class-matched @0.5、每视频 top100、≤2s/2–8s/>8s，可继续使用，但不能称为完整边界评估。[C12]

覆盖报告至少区分：有效 RGB 数、物理槽位、原 tubelet 数、被完整重算的位置、估计状态的位置、最大时间空洞、每个 GT 动作内部和合法起止附近的重/轻覆盖。多主体、小物体、遮挡不是现有注释中都有的完备标签；可做事先定义的小型人工错误审查，不能伪造已有自动分层 GT。

独立两动作间隔很短时，低分辨率轻状态可能抹平“结束—背景—重启”；同时高分数 clip 的集中选择也可能把第二次动作遗漏。长背景应减少贵更新但仍保留动作新出现的廉价观察。真实短/部分窗口单列；全部填充区域不能当作易删背景样本，影响采样、空间均值和计时的程度需要报告。

单种子不能估计训练方差；对固定 checkpoint 可按视频做**成对 bootstrap 并重算整套 AP**，不能把视频 AP 简单平均当总体 AP。测试峰值选择的乐观偏差不会被 bootstrap 消除；应同时给终点及整条已测试轨迹。未来多种子或更严格泛化验证可以建议，但不将 160/40、1pp 或外部脚本/审批标记变成当前项目的强制门槛。

---

# （e）系统测量、算量核算与成本边界

## E1. 独立 MAC 核算

S 在 160×160、48 原 clips、每 clip `N=8×10×10=800` token、`C=384`、12 层；TIA 低维 `d=C/4=96`。每 clip 每层的主要 MAC：

\[
C_{attn}=4NC^2+2N^2C,\quad C_{FFN}=8NC^2,\quad
C_{TIA}=N(2Cd+d^2+3d).
\]

前者包含 Q/K/V 和输出投影及 QK、AV，不含 softmax 等逐元素运算；stem 为 `48 N × (3×2×16²) × C`。把原检测器记录的 projection/head 矩阵成本加回，得到下表；S/B 总量均与仓库记录一致。这是独立公式核算，不是本次 TorchDispatch/GPU profiling。[C14,C15,R4,Toy]

| MAC 组件 | S，GMAC | B，GMAC |
|---|---:|---:|
| patch embedding | 22.6492 | 45.2985 |
| attention 线性投影 | 271.7909 | 1087.1636 |
| attention QK/AV | 283.1155 | 566.2310 |
| FFN | 543.5818 | 2174.3272 |
| TIA | 38.3533 | 153.1478 |
| projection + head | 14.4563 | 14.9092 |
| **总计** | **1173.9470** | **4041.0774** |
| **2MAC GFLOPs** | **2347.8940** | **8082.1547** |

S 的 TIA 占总矩阵 MAC **3.2670%**；其余主成本集中在 attention/FFN。S 仅后四层 MLP 保留 75%/50%，在任何新增模块之前，最多省原总量 **3.8586%/7.7173%**；B 对应 **4.4838%/8.9676%**。D8 全时间在未加出口模块前为 S **67.7202%**，与含出口近似约 67.73% 的参考口径相容；不是精度结果。S 全时间 160→128 为 **58.8869%** 的算子估计；attention 二次项随 token 数平方变化，不能简单用分辨率面积比替代整个模型。[Toy]

三维保留率不能直接相乘：早退出位置没有后四层 MLP可再省；只有 active clips 进入空间稀疏 FFN；全时间 stem/TIA/head 和 preview/reconstruction 并不按相同比例缩小。正确形式是按每层实际动作求和：

\[
C=C_{decode/pre}+C_{stem}+\sum_\ell(C_{QKV,QK,AV,proj}(A_\ell)+C_{FFN}(M_\ell)+C_{TIA}(G_\ell)+C_{light,\ell})+C_{router}+C_{reconstruct}+C_{detector}+C_{NMS}.
\]

这里只对其中 matrix/conv 分量用 2MAC FLOPs；其余用操作计数、字节或时间报告，不冒充全部已计 FLOPs。

## E2. 当前已测结果能说到哪里

旧 H65/BMCR 的矩阵/卷积 FLOPs 约为官方 S 51.55%、B 50.45%，但指定窗口模型时延更高。因此它们提供准确率锚点和算量减少证据，**没有提供官方精度保持与推理加速同时成立的证据**。[R4,R5]

本轮 Z24：GPU 模型均值/中位约 42.60/42.56ms，官方为 67.87/64.77ms，按本轮均值比约 1.59×；全流程 1229.65s 与官方 1236.41s 接近，缓存和运行波动未控制。不能将该 1.59× 写成端到端加速，也不能混用旧官方 51.21ms。模型时延下降伴随很大的精度损失，并不是满足目标的部署成果。[R9,Toy]

现有输入全量解码、变换、传输；GPU 中只少算部分 clip 时，全流程可能仍由这些阶段或 NMS/调度主导。当前数据没有分摊各阶段耗时，不能凭总时延就断言某一个模块是全部瓶颈。`FormalScout.condition` 的 Python 循环、GPU 标量同步和 route metadata 是值得 profile 的候选，不是已测定因果解释。[C2,C12]

## E3. 下一轮需要的真实执行记录

记录每个样本/层的 patch 输入形状、attention 的 clip/query/key token 数、重 MLP token 数、轻 MLP/adapter 数、真正重算的 original clip ID、depth/source mask；用 hook 与声明的 RoutePlan 交叉验证。full、partial、short 使用同一视频窗口；同时报告固定窗口和真实分布加权窗口/视频结果。名义 24 clip 既不等于所有短窗 384 个有效观察，也不等于 24 个全有效 clip。

时延分别为：（1）已规范化且路由已准备的 GPU 骨干；（2）GPU 常驻原输入到模型输出，包含 preview/route/gather/scatter/reconstruction/head；（3）解码、CPU 预处理、H2D、窗口 NMS、视频 NMS/gather 和总体 elapsed。组件诊断可用 CUDA events 加同步 CPU timers，但异步重叠下不能把各组件时长简单相加当 e2e。

保留全部样本的均值、中位、p95、标准差/范围，跨多次进程/窗口重复。现有计时仅 20 次，p95 对个别慢样本敏感，应同时注明样本数与实际分位数计算方式；不删除“慢异常值”后宣称加速。固定功耗/频率条件、precision、batch、节点与其他占用；冷热 filesystem cache 分开。导出算子之外的 routing/reconstruction bytes/allocations，检查小 GEMM 与 gather 成本是否吞掉理论收益。[C12,C19]

## E4. teacher、缓存与训练总成本

D1 每更新有完整 48×12 teacher 前向、用于标签的 teacher detector 输入梯度及辅助反向；它节省训练参数和部分梯度存储，不是省掉完整 teacher 计算。C2 local 缓存建立成本、每次交换重跑 detector 次数、缓存存储/读取应单列；global C2 可能需要额外学生流，不能只计 optimizer updates。

单个 S/B 全原格点 BF16 状态约 **28.125/56.25 MiB**（batch1、48×800×C）；保存12层仅这些状态约337.5/675 MiB，还未包括训练 graph、Q/K/V、MLP hidden、梯度、教师与学生双流。native384 池化特征仅0.28125/0.5625 MiB，适合 local 最终混合缓存，但不能替代 global TIA 所需完整空间状态。[Toy]

跨窗口缓存只有在原帧、tubelet phase、clip 上下文、空间 crop、位置、padding、归一化与相关时间邻域满足一致性时才可精确复用。最容易复用的是确定预处理后的低层局部算子；global 中深特征通常依赖窗口上下文。近似缓存应带年龄/失配检测、首窗建立、重置、存储与命中率成本。不能默认前一个窗口的结果在新窗口免费且等价。

---

# （f）给 Codex 的最小实施交接

## F1. 文件定位与变更原则

独立交接见同目录 `CODEX_HANDOFF.zh.md`。所有现有模型、输出和辅助 D1 recipe 保留，不覆写旧 checkpoint，不把新训练课程混入 `ds3_ltia8_d1_80_v1`。以下是**建议改动，不是本次已经提交的 patch**。

| 文件/接口 | 最小任务 | 应保留的合同 |
|---|---|---|
| `h65/full/scout.py::condition` | 精确张量化伙伴/上下文计算；增加可选诊断 | 候选/伙伴定义、stable tie、均值及梯度一致 |
| `h65/full/model.py::route/train_batch` | 记录 S0/S1 差异；后续支持受限交换 | rank384、global192、NMS前回映不变 |
| `h65/full/utility.py::counterfactual_targets` | 新增实际 S1 与 clip/depth 条件审计接口 | 老 label 与尺度保留并版本化；漏检不丢 |
| `h65/ds3/auxiliary.py` | 新增零末层残差补全，不替换原 H0 定义 | 初值精确 Z24；旧 aux checkpoint 仍可读取 |
| `h65/ds3/model.py::native/ClipEngine` | 插值/H0/残差切换、固定 route 输入、local cached mosaic | original clip ID、tubelet phase、native384/orig768 |
| 新 `h65/ds3/global_engine.py` | split贵算子/TIA，pack/scatter全空间格 | 不能仅修改 temporal_size；full-heavy恒等 |
| `h65/ds3/losses.py` | C2混合GT与正确语义KD、条件动作效用 | 冻结head仍保留学生输入梯度；teacher detach |
| `h65/ds3/routes.py` | 显式uniform prior、有效预算/覆盖、动作标签 | 零logits不冒充uniform；source/depth可审计 |
| `h65/ds3/runtime.py`、训练工具 | 独立C2/J3 recipe、EMA/online元信息 | 已有D1更新、课程、状态不得重解释 |
| `tools/ds3_eval.py` / `full_eval.py` | 四格控制、online/EMA选项、分阶段计时 | 211视频/792窗、相同NMS、保留rawtiming |

## F2. 关键数据结构

`TimeGrid` 保存 candidate 原视频帧索引 `[B,768]` 与 valid mask；native centers `[B,384]`、tubelet有效性、原 clip ID/phase 和单位。`NativeState` 为 `[B,C,384]` 加 source/depth/estimated mask；global 引擎内部另有 `[B,48,8,Hg,Wg,C]` 空间状态。`RoutePlan` 是每层 actual active clips、query/K/V/MLP索引与真实计数，不能用单个depth向量表达所有将来的非单调重更新。

checkpoint metadata 显式标记 `rank384_global192`、`original768_local8`、`original768_global384`、S/B通道、GT/预测时间单位、teacher来源与pretraining来源。相同tensor shape不自动允许跨合同加载。

## F3. 必须先过的一致性测试

full-budget 原图/新图预测一致；现有T24A不执行H8/稀疏MLP；local cached mosaic与真实compact对应；global缓存反例被测试拒绝；残差零末层初始精确Z24；均匀路由明确且不依赖top-k零分tie；完整/部分/短窗有效与物理计数正确，含odd tubelet、无效clip、极短前缀；合法裁剪端点不进入边界标签；教师/检测器参数buffer不变但学生feature梯度非零；hard路由梯度路径明确；KD为sigmoid独立类别、物理端点匹配；向量化伙伴与旧实现同值/同tie/同梯度；hook实际输入计数吻合声明；NMS前回映保留。

## F4. 什么证据支持扩大预算或改路线

没有人为新增统一“必须≤1pp”的门槛。扩大实验的依据是：初始函数与执行图验证可信；同 checkpoint 诊断能解释关键误差；实际损失/效用预测与真实混合收益有关；在平均及严格 tIoU、短动作/边界/重复事件上不存在被平均分掩盖的明显退化；实际GPU和全流程成本趋势值得继续；收益相对测量波动和测试峰值偏差仍有说服力。

改变路线的依据也具体：S0标签好但S1差→改决策组合/效用条件；U24补全失败且主要是不可观测短事件→增强轻观察或更细原tubelet/全时间状态；dense fitting好而mixed差→C2/接口校准；C2虚拟与compact不等价→修算子图；理论FLOPs降而时延不降→压紧批量/向量化/减少新模块，而不是继续加路由网络。

## 最终选择倾向

**近期：修正 BMCR 完成对照 + 同 checkpoint 四格诊断 + U24 残差 C2。长期：全时间原 tubelet/空间状态与廉价 global-TIA 保留，条件执行贵 attention/FFN 和深层残差。**

原因不是预设 clip 失败，而是当前最强成果已经有可用任务表示；最大掉点发生在多个合同同时改变之后，最便宜的下一步应先拆开这些变化。保护 TAD 的三个对象——视觉证据、时间几何、任务表示——比追求三个名义保留率相乘更重要。短动作本身不可观测、低分辨率丢小物体、路由漏事件、global状态不充分、教师偏差和小数据优化风险仍不能承诺消除；精度保持与端到端加速必须由真实新结果共同证明。


---

# 附录 A：固定源码与记录索引

下列所有 C/R 条目均固定到主提交；范围为本次阅读和定位使用的文件区段，正文以符号限定相关行为。GitHub 工具返回的 JSON 内容位于工具引用 L2；这里另给出真实文件行号入口，避免混淆工具行与源码行。

- **[R1]** [ds3_20260912/research_context_20260912/CONTEXT.zh.md](https://github.com/yuzbo/BMCR-T-AdaTAD/blob/239d098cd899936c35989fae6243c70259a85adb/ds3_20260912/research_context_20260912/CONTEXT.zh.md) — 当前上下文、版本与已授权/未完成事项。
- **[R2]** [ds3_20260912/research_context_20260912/latest_snapshot.json](https://github.com/yuzbo/BMCR-T-AdaTAD/blob/239d098cd899936c35989fae6243c70259a85adb/ds3_20260912/research_context_20260912/latest_snapshot.json) — 21:56 机器快照：metrics、progress.json、running evaluation；最新状态来源。
- **[R3]** [ds3_20260912/retrospective_20260912/REPORT.zh.md](https://github.com/yuzbo/BMCR-T-AdaTAD/blob/239d098cd899936c35989fae6243c70259a85adb/ds3_20260912/retrospective_20260912/REPORT.zh.md) — 历史课程、出处、模型合同、旧记录勘误。
- **[R4]** [phase2_20260910/comparison.json](https://github.com/yuzbo/BMCR-T-AdaTAD/blob/239d098cd899936c35989fae6243c70259a85adb/phase2_20260910/comparison.json) — 旧官方/H65/BMCR 完整测试、矩阵算量和旧时延。
- **[R5]** [fidelity_20260911/FINAL_COMPARISON.json](https://github.com/yuzbo/BMCR-T-AdaTAD/blob/239d098cd899936c35989fae6243c70259a85adb/fidelity_20260911/FINAL_COMPARISON.json) — 修正 H65 峰值/终点、测试峰值选模披露。
- **[R6]** [fidelity_20260911/INTERMEDIATE_RESULTS.json](https://github.com/yuzbo/BMCR-T-AdaTAD/blob/239d098cd899936c35989fae6243c70259a85adb/fidelity_20260911/INTERMEDIATE_RESULTS.json) — 全部16个中间点全测试。
- **[R7]** [fidelity_20260911/TRAINING_COMPLETION.json](https://github.com/yuzbo/BMCR-T-AdaTAD/blob/239d098cd899936c35989fae6243c70259a85adb/fidelity_20260911/TRAINING_COMPLETION.json) — 修正训练更新/课程完成；时间早于后续评估，不代表缺测试。
- **[R8a]** [ds3_20260912/PLAN.md](https://github.com/yuzbo/BMCR-T-AdaTAD/blob/239d098cd899936c35989fae6243c70259a85adb/ds3_20260912/PLAN.md) — D1/Z0计划、课程、初始化、计时范围。
- **[R8b]** [ds3_20260912/IMPLEMENTATION.md](https://github.com/yuzbo/BMCR-T-AdaTAD/blob/239d098cd899936c35989fae6243c70259a85adb/ds3_20260912/IMPLEMENTATION.md) — 执行入口、预检、测试峰值与策略矩阵；正文R8包含此项和R8a。
- **[R9]** [ds3_20260912/RESULTS.md](https://github.com/yuzbo/BMCR-T-AdaTAD/blob/239d098cd899936c35989fae6243c70259a85adb/ds3_20260912/RESULTS.md) — 20:02参考结果与计时范围；进行状态服从R2。

| 编号 | 文件与真实行号入口 | 本次审查符号/内容 |
|---|---|---|
| C1 | [h65/full/model.py#L40-L150](https://github.com/yuzbo/BMCR-T-AdaTAD/blob/239d098cd899936c35989fae6243c70259a85adb/h65/full/model.py#L40-L150) | FormalH65.route (45–57), encode, train_batch, predictions, raw_route |
| C2 | [h65/full/scout.py#L1](https://github.com/yuzbo/BMCR-T-AdaTAD/blob/239d098cd899936c35989fae6243c70259a85adb/h65/full/scout.py#L1) | FormalScout.forward / condition；候选/伙伴/集合特征、初始化和重放 |
| C3 | [h65/full/runtime.py#L90-L140](https://github.com/yuzbo/BMCR-T-AdaTAD/blob/239d098cd899936c35989fae6243c70259a85adb/h65/full/runtime.py#L90-L140) | optimizer_for、EMA；参数身份LR、EMA更新 |
| C4a | [h65/full/geometry.py#L1](https://github.com/yuzbo/BMCR-T-AdaTAD/blob/239d098cd899936c35989fae6243c70259a85adb/h65/full/geometry.py#L1) | TrueTimeMap、maps_for、exchange；正文C4包括geometry与transport |
| C4b | [h65/transport.py#L1](https://github.com/yuzbo/BMCR-T-AdaTAD/blob/239d098cd899936c35989fae6243c70259a85adb/h65/transport.py#L1) | sample_rates、gather_with_transport、固定容量sigmoid与连续代理 |
| C5 | [h65/full/data.py#L8-L40](https://github.com/yuzbo/BMCR-T-AdaTAD/blob/239d098cd899936c35989fae6243c70259a85adb/h65/full/data.py#L8-L40) | LoadFramesWithBoundaryValidity.random_trunc / __call__ |
| C6 | [h65/full/objectives.py#L1](https://github.com/yuzbo/BMCR-T-AdaTAD/blob/239d098cd899936c35989fae6243c70259a85adb/h65/full/objectives.py#L1) | 课程weights、targets、auxiliary_losses；真实边界与覆盖 |
| C7 | [h65/full/utility.py#L1](https://github.com/yuzbo/BMCR-T-AdaTAD/blob/239d098cd899936c35989fae6243c70259a85adb/h65/full/utility.py#L1) | localization_cost、分类代价、counterfactual_targets；真实交换标签 |
| C8 | [h65/ds3/model.py#L1-L242](https://github.com/yuzbo/BMCR-T-AdaTAD/blob/239d098cd899936c35989fae6243c70259a85adb/h65/ds3/model.py#L1-L242) | DenseTeacher、ClipEngine、DS3.native / predictions；local compact和原坐标 |
| C9 | [h65/ds3/losses.py#L1](https://github.com/yuzbo/BMCR-T-AdaTAD/blob/239d098cd899936c35989fae6243c70259a85adb/h65/ds3/losses.py#L1) | D1 dense特征损失、teacher input gradient、非负残差utility proxy |
| C10 | [h65/ds3/auxiliary.py#L1](https://github.com/yuzbo/BMCR-T-AdaTAD/blob/239d098cd899936c35989fae6243c70259a85adb/h65/ds3/auxiliary.py#L1) | H0 preview、H8残差出口、late MLP surrogate和spatial scorer |
| C11 | [h65/ds3/routes.py#L1](https://github.com/yuzbo/BMCR-T-AdaTAD/blob/239d098cd899936c35989fae6243c70259a85adb/h65/ds3/routes.py#L1) | policy定义、覆盖锚点、stable top-k、native_geometry |
| C12 | [tools/ds3_eval.py#L1](https://github.com/yuzbo/BMCR-T-AdaTAD/blob/239d098cd899936c35989fae6243c70259a85adb/tools/ds3_eval.py#L1) | 评估、实际执行计数、timing、duration_recall、EMA策略 |
| C13 | [h65/ds3/runtime.py#L1](https://github.com/yuzbo/BMCR-T-AdaTAD/blob/239d098cd899936c35989fae6243c70259a85adb/h65/ds3/runtime.py#L1) | load_aux(ema=False)、AuxEMA、学习率课程 |
| C14 | [upstream/configs/adatad/thumos/e2e_thumos_videomae_s_768x1_160_adapter.py#L1](https://github.com/yuzbo/BMCR-T-AdaTAD/blob/239d098cd899936c35989fae6243c70259a85adb/upstream/configs/adatad/thumos/e2e_thumos_videomae_s_768x1_160_adapter.py#L1) | S骨干配置、原时间轴、候选stride4、160空间分辨率 |
| C15 | [upstream/opentad/models/backbones/vit_adapter.py#L19-L75](https://github.com/yuzbo/BMCR-T-AdaTAD/blob/239d098cd899936c35989fae6243c70259a85adb/upstream/opentad/models/backbones/vit_adapter.py#L19-L75) | Adapter；另参同文件Block.forward约272–295和patch/position/forward约387–490 |
| C16 | [upstream/opentad/models/dense_heads/anchor_free_head.py#L100-L310](https://github.com/yuzbo/BMCR-T-AdaTAD/blob/239d098cd899936c35989fae6243c70259a85adb/upstream/opentad/models/dense_heads/anchor_free_head.py#L100-L310) | forward_train/test、get_refined_proposals、sigmoid、losses、prepare_targets |
| C17 | [tools/ds3_train.py#L1](https://github.com/yuzbo/BMCR-T-AdaTAD/blob/239d098cd899936c35989fae6243c70259a85adb/tools/ds3_train.py#L1) | 80轮D1、pilot包含在主轨迹、保存aux/EMA |
| C18 | [tools/full_train.py#L1](https://github.com/yuzbo/BMCR-T-AdaTAD/blob/239d098cd899936c35989fae6243c70259a85adb/tools/full_train.py#L1) | 修正warm父权重、audit scales读取、joint课程与检查点 |
| C19 | [tools/full_eval.py#L1](https://github.com/yuzbo/BMCR-T-AdaTAD/blob/239d098cd899936c35989fae6243c70259a85adb/tools/full_eval.py#L1) | OfficialRuntime、ArithmeticCounter、矩阵2MAC含融合QK/AV、全测试 |

**读取边界。** 关键指定源码与记录均通过固定 ref 读取；不是默认分支。公开仓库没有本项目大权重和原始视频，本次没有访问它们，也没有确认其磁盘存在性。仓库预检、训练与测试成绩作为记录证据引用；本文独立完成的实验只有 Toy 条目。不同论文的作者代码阅读深度不相同，下列明确标注，不能把“链接可达”当“完整源码复现”。

# 附录 B：原论文、版本与作者实现证据

**[P1] [AdaFocus V2: End-to-End Training of Spatial Dynamic Networks for Video Recognition](https://arxiv.org/abs/2112.14238v2)** — CVPR 2022；arXiv 2112.14238v2，2022-04-12。主方法/训练稳定性章节。原CVF PDF访问受限，转读作者arXiv全文。作者实现入口由论文给出LeapLabTHU/AdaFocusV2，本次不声称完整审计该实现。

**[P2] [SlowFast Networks for Video Recognition](https://arxiv.org/abs/1812.03982)** — ICCV 2019 论文。主方法与双速率/横向融合；本次使用arXiv全文核查机制，不声称本项目TAD复测或作者全代码审计。

**[P3] [Dynamic Token Pruning in Plain Vision Transformers for Semantic Segmentation](https://arxiv.org/abs/2308.01045v1)** — ICCV 2023；arXiv 2308.01045v1。主方法/辅助出口/上下文保留/训练叙述；另核查A1。部分表格截图失败，正文没有据未核查图表抄录数值。

**[P4] [TR-BERT: Dynamic Token Reduction for Accelerating BERT Inference](https://aclanthology.org/2021.naacl-main.463/)** — NAACL 2021正式论文，2021.naacl-main.463。全文训练与QA token状态，方法图截图；作者仓库thunlp/TR-BERT为实现入口，本次未做完整源码审计。

**[P5] [CoLT5: Faster Long-Range Transformers with Conditional Computation](https://arxiv.org/abs/2303.09752v3)** — EMNLP 2023；arXiv 2303.09752v3，2023-10-24。主方法、routing及C4/UL2预训练条件；作者实现入口google/flaxformer，未固定读取其全实现。

**[P6] [Dynamic Tuning Towards Parameter and Inference Efficiency for ViT Adaptation](https://arxiv.org/abs/2403.11808v2)** — NeurIPS 2024；arXiv 2403.11808v2，2024-10-16。式(5)/(6)、dispatcher、变体及训练loss；另核查A2。部分arXiv截图失败，主要依据可读方法正文和实际代码，不引用未核验图表精度。

**[P7] [PointRend: Image Segmentation as Rendering](https://arxiv.org/abs/1912.08193)** — CVPR 2020论文。点特征、非迭代训练采样与迭代推理；作者实现为detectron2/projects/PointRend，未完整审计。

**[P8] [QueryDet: Cascaded Sparse Query for Accelerating High-Resolution Small Object Detection](https://arxiv.org/abs/2103.09136)** — CVPR 2022论文。主方法/训练与稀疏head图，方法页截图成功；作者实现入口Small-Object-Detection/QueryDet，未完整审计。

**[P9] [Expediting Large-Scale Vision Transformer for Dense Prediction without Fine-tuning](https://arxiv.org/abs/2210.01035v1)** — NeurIPS 2022 camera-ready；arXiv 2210.01035v1。局部clustering/reconstruction、无finetune机制；不声称完整运行作者实现或成功读取全部代码清单截图。

**[P10] [RandLA-Net: Efficient Semantic Segmentation of Large-Scale Point Clouds](https://arxiv.org/abs/1911.11236v3)** — CVPR 2020；arXiv 1911.11236v3，2020-05-01。局部聚合、随机采样、encoder/decoder与方法图；作者实现入口QingyongHu/RandLA-Net，未完整审计。

**[P11] [3DSSD: Point-based 3D Single Stage Object Detector](https://arxiv.org/abs/2002.10187)** — CVPR 2020论文。融合采样/candidate generation；方法图截图核查。

**[P12] [Deep Feature Flow for Video Recognition](https://arxiv.org/abs/1611.07715)** — CVPR 2017论文。关键帧/运动传播/任务适配机制；不将其特征传播等同于本项目clip缓存。

**[P13] [Eventful Transformers: Leveraging Temporal Redundancy in Vision Transformers](https://arxiv.org/abs/2308.13494)** — ICCV 2023论文；arXiv 2308.13494。gate/buffer、QK/AV增量、检测与识别适配差异、存储限制；gate/buffer截图成功。作者实现入口WISION-Lab/eventful-transformer，未完整审计。

**[P14] [ETAD: Training Action Detection End to End on a Laptop](https://arxiv.org/abs/2205.07134v2)** — arXiv 2205.07134v2，2022-11-28。顺序完整前向、梯度和proposal训练采样，明确不是RGB推理少观察证据。

**[A1] DToP 作者实现已固定阅读部分。** 提交 `2baefffc879622b6514504dff44cf04d7e0e7504`（2023-09-28），[README](https://github.com/zbwxp/Dynamic-Token-Pruning/blob/2baefffc879622b6514504dff44cf04d7e0e7504/README.md)。核查base训练、prune fine-tune和eval入口；没有将README成绩当本项目验证，也没有声称全库审计。

**[A2] DyT 作者实现已固定阅读部分。** 提交 `d1744f0b9366f79ad9b78f586e479af34e81807a`（2024-12-30），[分割backbone：Block.forward](https://github.com/NUS-HPC-AI-Lab/Dynamic-Tuning/blob/d1744f0b9366f79ad9b78f586e479af34e81807a/dense_tasks/Segmentation/backbone/segmentation_vision_transformer_IN21K.py#L224-L300) 和 [dynamic_adapter：TokenSelect/_gumbel_sigmoid](https://github.com/NUS-HPC-AI-Lab/Dynamic-Tuning/blob/d1744f0b9366f79ad9b78f586e479af34e81807a/models/dynamic_adapter.py#L26-L76)。读取范围分别1–330、1–153；可确认学生残差mask与ST代理，可确认所读分割Block没有compact eval分支；不外推作者其他性能测试入口是否压紧。

**文献时效边界。** 本报告不是截至2026-09-12的穷尽性SOTA排行榜。引用的是已读取、与机制有直接关系的原论文与版本；尚未对全部相关2025–2026工作做新颖性排他检索，任何“首次”声明仍需额外研究。

# 附录 C：CPU机制检验与复现

[Toy] 本目录 `mechanism_checks.py` 与 `mechanism_checks.json`。Python + NumPy，固定随机种子3407；运行 `python mechanism_checks.py --output mechanism_checks.json`。没有加载仓库代码、权重、视频或GPU。

断言检查：S/B矩阵MAC公式与记录一致；固定容量sigmoid的隐式梯度与有限差分一致；逐token运算dense-mask与gather-scatter一致；删K/V后的attention不一致；global与分clip的temporal卷积不一致；零残差保持可用基线；并输出EMA系数、零logit路由、效用非可加反例、teacher低梯度反例、rank-IoU反例。所有断言在本次CPU运行通过。它们只检查小型机制，不是TAD精度/速度实验，也不是主仓库pytest通过的替代品。
