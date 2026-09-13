# BMCR-T 独立源码审查与改进研究

## 一、结论与证据范围

**实现不是一个“选帧其实没有少算”或“评测时间轴完全错了”的空壳，但现有证据不支持“有效保持官方精度并实现加速”。最值得优先验证的方向，是让已经计算的全时序粗特征进入检测器，并把检测监督放回真实时间网格，而不是继续增加采样 MLP 的复杂度。**

审查对象严格限定为 `yuzbo/BMCR-T-AdaTAD@15280e5dc29e3df18d085aaba21067e04107ce84`。研究任务文档来自另一提交 `8a8529093f8a96b228e138c272de0f9df999bda8`，仅用于确定审查要求，不替换模型、分析和结果版本。当前执行链是 `tools/full_train.py → h65/full/model.py::FormalH65`，不是早期原型 `h65/model.py`，也不是历史 90 轮、局部 TIA 的 H65 路线。[^1][^2]

| 核心问题 | 判断 | 证据边界 |
|---|---|---|
| 当前实现是否正确？ | 核心预算、硬选帧、代理梯度接线、ASFormer 训练、EMA、交换符号、时间回映和完整评测链基本成立。存在一个可确定的填充语义偏差，以及几个与任务直接相关的设计风险。 | 没有发现足以单独解释全部精度差距的致命错误；源码可证明某些计算/梯度行为，不能证明这些行为造成了多少 mAP 损失。 |
| 是否保持性能并减少计算？ | 矩阵/卷积 MAC 确实减少约一半；精度没有保持，严格定位指标也下降；指定窗口计时反而更慢。 | 已报告指标与实现相互一致，但没有独立加载权重、运行 211 视频或重算压缩预测的 AP。指定窗口不是全测试集平均时延。 |
| 优先如何提高上限？ | 先补同日程 U384 均匀基线，再检验“真实时间密集网格 + 已有粗特征残差”。 | 这是可证伪的研究建议，不是已验证的新模型，也不预测 mAP。自蒸馏和规则 clip 集合采样作为两条独立备选。 |

材料覆盖关键训练、路由、骨干、检测头、数据变换、窗口合并、NMS、AP 计算源码，以及提交内总表、训练汇总、S 的原始 profile 和 S/B 的交换审计摘要。大权重和原始视频没有载入；压缩预测没有独立解压重评；初始化张量、全部原始日志和训练成功次数没有逐条独立复核。因此，下文区分“源码确认”“提交内实验记录”“数学推导”与“待验证假说”，不把作者的验证记录或 CPU 测试通过当成完整正确性证明。附带脚本只进行指标算术和小型几何反例检查，不启动模型训练或推理。[^3][^10][^11][^12]

## 二、真实调用链审查

### 2.1 输入、预算与真正少执行的位置

原始候选输入为 `B×1×3×768×160×160`。数据管线的候选间隔是原视频 4 帧，不是 768 个连续原始帧。固定预算为 384 个物理输入位置：有效候选长度为 L 时，真实不同位置数是 `min(384,L)`，不足部分填充，不可能从 L<384 的窗口得到 384 个不同真实帧。[^4][^19]

正常完整窗口中，`sample_rates` 将校准后的逐位置质量限制在 [0,1]，总质量为 K，再以累计分布的系统阈值解码有序硬索引；均匀分支使用包含端点的等距位置。前向 `gather_with_transport` 取的是实际 RGB 帧，梯度桥的数值贡献恒为零，不是把两张图像混合成“虚拟帧”。VideoMAE 实际执行从 `48×16` 个输入帧片段降为 `24×16`；每个 clip 的 tubelet/注意力计算仍执行，骨干冻结本身不被计作跳算。TIA 仍跨同一视频的选中序列进行全局时间交互。[^3][^4][^17]

该实现因而满足“减少高成本骨干执行”的核心要求，但并没有减少解码到候选输入之前的全部工作：CNN/ASFormer 仍读取全 768 个候选，报告计时也没有包含视频读取、解码或 NMS。

### 2.2 各模块的正确性判断

| 环节 | 源码确认的行为 | 不应扩大为的结论 |
|---|---|---|
| 固定 K | 完整窗口硬选 K 个有序不同位置，短窗口保留所有可用位置并物理填充。 | 不能把短窗口的物理 K 当成 K 个不同观测，也不能把冻结参数视为省 MAC。 |
| 代理梯度 | 密度校准和累计分布提供连续代理；RGB 局部斜率 stop-gradient，前向保持硬帧。 | 这是有偏任务梯度估计，不是离散选帧真实导数或无偏性能梯度；非零梯度不代表梯度方向有用。 |
| ASFormer | 辅助动作监督训练 CNN、ASFormer 编码器/解码器；策略分支通过同随机状态重放编码器，将策略反馈与 CNN stem 解耦。 | 不能说 ASFormer 冻结，也不能说采用了原论文完整动作分割训练配方。 |
| uniform companion | batch2 中一行使用均匀采样，另一行使用学习采样；有效策略行作梯度尺度补偿。 | 两行通常是不同视频/窗口，不是同视频互补视图，也不是检测自蒸馏。 |
| EMA | 同结构模型 eval、禁用参数梯度；按成功优化器更新进行 EMA；末轮 EMA 用于主结果。 | EMA 的存在不意味着已经蒸馏检测输出。当前主要提供交换效用测量。 |
| 交换符号 | 已选位置学删除损害，未选位置学插入收益；互反伙伴才生成反号标签。 | 不能因同一交换出现一正一负就认定符号错误，也不能给非互反伙伴强行反号。 |
| 集合条件 | 输入包含成员身份、伙伴特征、选中集合均值、邻近间隔、动作概率、预算比例。 | 当前不是独立逐帧 MLP；“加集合条件/加非线性”本身已不新。 |
| 检测监督 | GT 被映射到 selected-rank，再由 ActionFormer 的分类/中心采样/距离回归/DIoU 监督。 | rank 上损失不等价于真实时间损失。 |
| 推理 | 不以 GT 选帧；proposal 在 NMS 前回到原候选时间，再转换为视频秒数。 | 可逆坐标回映不等于恢复被丢弃的视觉证据。 |
| 完整评测 | 全视频窗口结果合并，再逐类 SoftNMS；AP 按分数排序、GT 单次匹配；计算五阈值类别均值和总均值。 | 不能以窗口子集或匹配上的 GT 子集代替完整测试，也不能从代码断言独立重现实验已经完成。 |

相关实现集中在 `h65/full/model.py`、`scout.py`、`runtime.py`、`utility.py`、`geometry.py`，以及 `tools/full_train.py`、`tools/full_eval.py`；上游 AP 实现位于 `upstream/opentad/evaluations/mAP.py`，窗口合并位于 `upstream/opentad/cores/test_engine.py`。[^3][^5][^6][^7][^8][^9][^10][^20]

校准密度的梯度满足预算守恒。若 `r_i=σ((z_i+b)/τ)`，隐变量 b 使 `Σr_i=K`，令 `a_i=r_i(1-r_i)/τ`，则

\[
\frac{\partial r_i}{\partial z_j}
=a_i\left(\delta_{ij}-\frac{a_j}{\sum_l a_l}\right).
\]

这解释了为什么不能同时“增加所有位置”的采样质量，但没有消除硬离散决策的梯度偏差。RGB 斜率主要反映邻帧像素变化；相机运动、压缩噪声和动作语义变化并不等价。[^4]

### 2.3 交换效用：符号正确，目标仍有限

对于当前集合 S、删除成员 r、插入非成员 a，令 `S'=S\{r}∪{a}`。当前定义等价于

\[
 y_r = \mathcal L(S')-\mathcal L(S),\qquad
 y_a = \mathcal L(S)-\mathcal L(S').
\]

正的 `y_r` 表示保留 r 有益；正的 `y_a` 表示插入 a 有益，二者都可以正向提高相应位置的保留倾向。训练目标按训练来源审计拟合的分类/定位尺度标准化；预测头本来学习的就是标准化单位，推理没有漏除一次尺度。[^7]

分类交换损失固定基线的正负样本分配，定位交换损失在真实时间重新匹配。后者采用唯一匹配、漏检 dummy 惩罚并对全部 GT 归一，不是只对容易匹配的实例求均值。这些是可以确认的优点。然而，分类分配冻结与定位重新匹配是两种不同的估计目标；分数阈值和代价截断也会产生局部平坦区域。等权混合两个标准化通道是设计选择，不是已经校准为 mAP 最优的效用。[^7]

## 三、实际偏差与优先疑点

以下源码链接全部固定到审查提交。严重性按与本任务的相关性排列，而不是按是否容易构造边角反例排列。

### 3.1 可确定的实现偏差：重复填充被改成零 RGB

**位置：** [`h65/transport.py:96–98`](https://github.com/yuzbo/BMCR-T-AdaTAD/blob/15280e5dc29e3df18d085aaba21067e04107ce84/h65/transport.py#L96-L98) 与 [`h65/full/model.py:65–70`](https://github.com/yuzbo/BMCR-T-AdaTAD/blob/15280e5dc29e3df18d085aaba21067e04107ce84/h65/full/model.py#L65-L70)。

**触发：** L<384。采样器先用最后有效索引填充，但 `encode` 第 66 行随后把无效位置的 RGB 乘成零。骨干归一化和 tubelet 卷积发生在之后，而骨干内部没有收到对应的时间有效性 mask。公开 profile 的 253 有效候选窗口就是可触发的实际输入类型，不是虚构空输入。

**影响：** 重复边界帧上下文被替换为黑图像上下文；若有效长度为奇数，最后一个真实帧和一个零帧还会处于同一二帧 tubelet 中。最终输出 mask 无法逆转此前卷积、注意力和全局 TIA 的混合。可以确定特征可能改变；不能据此声称全部 5.86/3.79 点损失都由它造成。

**最小修复：** 取消骨干前这一次 RGB 清零，保留重复帧填充和最终输出有效性 mask，使行为更接近上游 edge padding；以固定权重单独评估短窗口/尾窗差异。若研究意图就是引入黑帧上下文，则应明确标为一个模型设计变更，而不是声称完全保持上游填充语义。固定权重比较仅用于判断敏感性；新实验的各对照组必须使用一致的训练/推理填充规则。更彻底的内部时间 mask 需要改 tubelet/attention/TIA，不能冒充一行修复。

### 3.2 确定存在的优化几何差异，而非“回映反了”

**位置：** [`h65/full/model.py:84–87`](https://github.com/yuzbo/BMCR-T-AdaTAD/blob/15280e5dc29e3df18d085aaba21067e04107ce84/h65/full/model.py#L84-L87)、[`h65/full/geometry.py:7–31`](https://github.com/yuzbo/BMCR-T-AdaTAD/blob/15280e5dc29e3df18d085aaba21067e04107ce84/h65/full/geometry.py#L7-L31)，以及上游 ActionFormer head 的 GT 分配与 DIoU。

**触发：** 所选时间间隔不恒定。GT 和预测均能正确往返映射，但非线性单调变换不保持长度比、IoU、中心采样半径和回归范围的物理意义。

**反例：** 选中位置 `[0,1,2,3,8,9,10,11]`，真实 GT 为 `[3,9]`、预测为 `[2,10]`。真实 IoU 为 6/8=0.75，映射后是 `[3,5]` 和 `[2,6]`，rank IoU 为 2/4=0.50；反向映射仍完全正确。

**影响：** 检测器优化的是依赖采样密度的局部时间尺度；严格定位评价则在秒数上进行。这是确定的目标不一致，但其 mAP 因果效应尚待消融。

**最小研究性修正：** 若保持 rank head，先增加一项真实坐标回归损失，并保持其他目标不变作诊断；完整解决方案是把特征对齐到真实时间网格后直接训练检测头。仅换 DIoU 不能同时修正 rank 上的正样本中心、回归范围和特征邻接关系。

### 3.3 单交换监督与整集合重解码不匹配

**位置：** [`h65/full/scout.py:63–98`](https://github.com/yuzbo/BMCR-T-AdaTAD/blob/15280e5dc29e3df18d085aaba21067e04107ce84/h65/full/scout.py#L63-L98)、[`h65/full/model.py:45–57`](https://github.com/yuzbo/BMCR-T-AdaTAD/blob/15280e5dc29e3df18d085aaba21067e04107ce84/h65/full/model.py#L45-L57)、`h65/full/utility.py` 的交换标签生成。

**触发：** 条件头在临时集合 S₀ 上预测某个 16-slot cell 内伙伴交换的收益，但这些分数随后全部加到密度 logits，再全局执行一次累计分布解码。最终 S₁ 可能同时改变许多位置，也不一定执行被标注的那一对交换。

**影响：** “这个局部交换有益”不推出“同时按这些修正重解码出的新集合有益”；集合均值、伙伴、间隔已经过时。模型学习到的是局部条件效用，却被作为全局密度修正场使用。

**分类：** 设计取舍，不是符号 bug。**最小修正：** 增加一个只执行一次精确可行交换的 trust-region 对照，每次交换后重算条件；或者将训练目标改为最终 S₁ 相对 S₀ 的集合级效用。两者都必须单独记录教师成本，不能默认推理时反复跑教师验证。

### 3.4 条件路由的计算量很小，不代表执行便宜

**位置：** [`h65/full/scout.py:70–79`](https://github.com/yuzbo/BMCR-T-AdaTAD/blob/15280e5dc29e3df18d085aaba21067e04107ce84/h65/full/scout.py#L70-L79)。每行 48 个 cell、两个成员方向，包含小张量布尔索引、argmin 和 scatter 写回。

S-BMCR 原始 profile 记录 96 次 argmin、192 次 index_put，以及大量索引/切片操作。这些大多不进入矩阵/卷积 MAC 统计，却可能引发内核启动和主机—设备同步开销。BMCR 相对 H65-C 只多约 0.0176 GMAC，但指定窗口中位时延多约 31–35 ms；不同作业条件和缺少分段计时意味着不能把这段差值全部归给该循环。[^11]

**最小实现优化：** 用 `B×48×16×16` 的静态 cell 距离表、成员 mask 和一次批量 argmin 生成伙伴；保留平局时最早索引、不存在伙伴时的 mask、所有有效长度和原有排序语义。先要求伙伴、索引和预测数值等价，再进行同卡成对计时。不要把移除某个循环先验地称为“实现加速”。

### 3.5 扩展真实时间蒸馏时必修的接口陷阱

**位置：** [`h65/full/geometry.py:19–25`](https://github.com/yuzbo/BMCR-T-AdaTAD/blob/15280e5dc29e3df18d085aaba21067e04107ce84/h65/full/geometry.py#L19-L25)。`interpolate` 将待映射的 `value` 也 detach。

**现状判断：** 当前用它变换 GT、无梯度教师预测或测试预测，符合现用途，不是现训练漏梯度 bug。**扩展触发：** 用同一函数把 student 预测边界转到真实时间后计算可微定位蒸馏，会直接切断回归梯度。**最小修改：** 新增单独的可微值映射，只 detach 坐标节点与索引，不 detach student 边界值；加输入梯度测试，不改变现有 GT/评测接口。训练用边界映射还应避免先硬 clamp 越界预测而令梯度饱和，可采用端段外推或直接真实时间解码；合法视频范围裁剪留到评测后处理。

另外，该映射在窗口端点 clamp，并不是对窗口外任意预测的全局可逆映射。跨窗口长实例的截断应单独统计，不应把“可逆”表述为无条件成立。

## 四、性能保持、成本与 S/B 差异

### 4.1 先核实“论文所称”

可核实的基础论文是 Liu 等人的 **End-to-End Temporal Action Detection with 1B Parameters Across 1000 Frames（AdaTAD，CVPR 2024）**。本报告核对的是 arXiv `2311.17241v2`；论文的 VideoMAE-S/B、768 帧、160 分辨率表列平均 mAP 为 **68.8/71.5**。固定上游 model-zoo README 列 **69.03/71.14**；本仓库指定检查点完整测试为 **69.01/71.13**。三组数字来源不同，不能混为“同一个论文结果”。[^15][^16]

BMCR-T 是当前研究提案名称；没有找到与其实现和实验条件对应、可核实的已发表论文。AdaTAD 的贡献主要是通过 TIA 和冻结大骨干降低端到端训练的内存瓶颈，不能作为 BMCR-T 采用 K384 后应保精度、应加速的证据。当前比较也不是只替换采样器的单变量消融：新模型从识别预训练起步、20+40 轮训练；官方仅复用已有 TAD 权重，其 checkpoint 的 epoch 字段不能还原原始训练总长度或完整选模协议。[^9][^12][^15][^16]

### 4.2 提交内结果的独立算术核对

平均 mAP 是 tIoU 0.3、0.4、0.5、0.6、0.7 的平均，表中均为百分数；mAP 差值为百分点。

| 骨干/方法 | 平均 mAP | mAP@0.5 | mAP@0.7 | GMAC | 指定窗口平均/中位时延 ms | 指定口径推理峰值 GiB |
|---|---:|---:|---:|---:|---:|---:|
| S 官方 | 69.01 | 72.33 | 48.24 | 1173.95 | 51.21 / 51.17 | 0.935 |
| S H65-C | 62.86 | 66.33 | 41.31 | 605.16 | 92.84 / 86.67 | 0.861 |
| S BMCR-T | 63.16 | 65.91 | 41.00 | 605.17 | 121.91 / 121.59 | 0.861 |
| B 官方 | 71.13 | 74.93 | 49.62 | 4041.08 | 118.31 / 118.17 | 1.508 |
| B H65-C | 66.61 | 69.93 | 44.31 | 2038.72 | 123.08 / 123.02 | 1.123 |
| B BMCR-T | 67.34 | 70.67 | 45.35 | 2038.74 | 153.75 / 153.74 | 1.123 |

数值来自固定提交的 `comparison.json`。附带 `arithmetic_checks.json` 保留五阈值数据和派生差值，不能替代从原始预测重算 AP。[^11]

**精度：** BMCR-T 相对官方 S/B 分别下降 **5.86/3.79 点**；不支持通常意义下的性能保持。相对 H65-C 仅提升 **0.30/0.73 点**，且仅有单次训练，不能宣称稳定或显著。

**严格定位：** @0.7 分别下降 **7.24/4.27 点**。S-BMCR 相对 S-H65-C 的 @0.5、@0.7 还分别下降约 0.42、0.30 点。低/高 tIoU 的 AP 都混合分类排序、召回和定位因素，不能把某个阈值变化直接等同于纯分类或纯回归质量变化。

**MAC：** 分别减少 **48.45%/49.55%**。有真实输入形状和融合注意力 QK、AV 计数支持。这里统计矩阵/卷积 MAC；按一次乘加计两次浮点运算时可换算为 2×MAC，但仍不包含全部逐元素、归一化、softmax、索引、数据搬运和控制流成本。

**实际时延：** 指定窗口中位数是官方的 **2.376 倍/1.301 倍**，即更慢。条件为 RTX4090、batch1、输入已在 GPU、BF16 骨干/scout 与 FP32 检测器，5 次预热、20 次同步计时；不含解码和 NMS。没有全 792 窗口的延迟分布，也没有独立测得的端到端吞吐和能耗，不应扩展结论。

**显存：** 当前口径推理峰值下降约 **7.92%/25.50%**，值得保留，但并非减半，也不是所有显存指标：allocated/reserved、模型和输入常驻、训练激活、进程外显存不能混用。

**训练成本：** S/B 的 BMCR 概念完整课程分别约 **6.05/6.52 GPU 小时**，报告峰值约 **7.72/13.59 GiB**；共享预热只计一次的六个训练阶段合计 **21.22 GPU 小时**。这些不含审计、评测、排队与预检。没有同协议官方训练计时/显存，不能声称相对官方的训练成本下降。[^11][^12]

### 4.3 为什么 S 损失更多：可检验的解释顺序

**首先是时间表征失配，而不只是参数量。** 从候选步长 4 变为均匀 K384，局部平均步长近似变成原视频 8 帧；学习采样再引入可变间隔。冻结 VideoMAE 的 tubelet、位置编码和 TIA 都不直接知道真实 Δt。S/B 对这个分布变化的敏感性可以不同，但必须以同日程 U384 和规则 clip 对照分离“少观测”与“改局部节奏”。[^17][^19]

**其次是检测几何与证据完整性。** rank 轴压缩改变中心采样、物理回归范围和同类相邻实例之间的距离；CNN/ASFormer 虽看到了全时序，却没有直接给检测器补证据。严格 tIoU 的下降与这一解释相容，但尚不是因果证明。验证应测真实时间边界误差、短/重复动作召回及边界附近观测缺口，而不是只看全局平均 mAP。

**再次是优化与路由目标。** S/B 的 adapter 学习率本来不同；相同成功步数不代表同样接近各自收敛状态。固定的双通道标准化、等权混合和一次全局重解码可能分别影响两个骨干，需要最终 BMCR 条件头在未参与训练的开发视频上重新测量排序、符号和集合 regret。不能用旧 H65 代理的审计替代最终 BMCR 的验证。[^8][^14]

这里有两个应保留的反证：旧代理的定位 Spearman，S 为 -0.028，B 反而更低，为 -0.203；温启动审计中重复计数的 matched-GT 为 S 的 2654/2683 与 B 的 2653/2683，几乎一样。因此，现有证据**不支持**“S 一定因为效用代理更差”或“S 因更少 GT 能匹配上而损失更多”这种直接归因。这些计数也不是独立实例数，更不是检测召回率。[^14]

对时延，S 的骨干更便宜，而两者付出的全时序 scout 和路由开销近似固定，所以 S 更容易被附加开销吞没；这是结构性成本解释。不能把这个解释偷换为 S 精度下降的因果结论。

## 五、跨领域机制：迁移什么、不迁移什么

下列近邻先例指机制上最直接相关的已核实工作，不声称穷尽各领域最新文献。论文证据任务、实际删算位置和本项目的迁移假说必须分开；其他任务的精度/速度收益不作为 BMCR-T 的预期收益。

### A. 粗细融合、门控、记忆和边界条件

**SlowFast → 低成本全时间支路，而非孤立“重要帧”。** SlowFast 用高时间分辨率的轻量 Fast 支路补充低帧率、宽通道 Slow 支路，并有 AVA 动作检测验证；但 AVA 的人物框动作检测不等于这里的时间起止定位。可借鉴侧向残差连接，让 `96×768` 的粗时序特征补到稀疏重特征，而不是只用于决定删谁。落点是 `FormalScout` 输出与 `FormalH65.encode` 后接口。失败方式是粗支路对背景运动敏感，或两个分支时钟没对齐；“双路融合”本身已是成熟机制。[^21]

**AdaFocus / AdaFocusV2 → 廉价全局观察 + 昂贵局部处理。** 前者以轻量全局特征指导局部处理；V2 用可微空间采样改善训练复杂性。这些主要是视频识别证据，不保证短动作边界。它们支持把“观察全局”和“高成本精看”拆开，而不支持直接删除 TAD 的密集时间状态。可借鉴 coarse-to-fine routing 与辅助监督，但连续裁剪图像的可微性不能直接视为离散取真实帧的可微性。近邻先例包括后续统一空间/时间/样本条件计算路线；本项目创新必须落在真实时间定位和硬预算接口上。[^22]

**TALLFormer → 用未重算的位置状态补全长序列。** 它直接在 TAD 上验证长记忆，训练时只重算部分片段，并读取/更新预先生成的特征 bank。官方说明明确包含全训练集初始化和磁盘就地更新成本。迁移前提是缓存的时间戳、增强视图和参数版本可追踪；当前全局 TIA 使特征依赖整段选中上下文，不能把不同窗口的 post-TIA 特征无条件当作同一个缓存条目。适合训练成本研究或分离局部编码器后的缓存，不等价于首次推理也只看一半视频。[^23]

**ETAD 是重要反例。** 采样参与反向传播的片段可以省训练激活/反向成本，但若全片段前向仍计算，就不是推理时跳过 RGB 重骨干。这一界限在讨论“记忆补偿”和“少算”时必须保留。[^24]

**DyFADet、BRN、DiGIT → 直接面向 TAD 的边界/尺度机制。** DyFADet 提供动态聚合与多尺度检测头；BRN 针对时间金字塔中的边界损失；DiGIT 使用多膨胀门控编码和中心/邻接的顺序可变形采样。它们支持边界处保留邻域和跨尺度证据，但多为检测器或预提取特征层上的改进，不证明原始 RGB 可少算。落点是本项目真实网格特征之后的 projection/head，而不是声称其本身解决选帧。失败方式包括跨实例平滑、错误边界门控以及昂贵的全局跨注意力。**门控 + 多尺度 + 邻域边界注意力不能作为本项目的独立新颖性主张。**[^25][^26][^27]

因此优先顺序是：零初始化线性粗残差 → 带间隔/可观测性输入的标量门控 → 必要时有限邻域跨注意力。全局 cross-attention、跨窗口记忆和额外大分支不应同时上马。粗输入还可拆成 CNN 的局部高频变化与 ASFormer 的上下文特征；是否拼接两者应单独消融，不能假设语义更平滑的一支必然最适合边界。

### B. 自模型蒸馏：对象、可靠性与真实训练账单

**BYOT → 深层教师、浅层学生。** 深层输出和特征 stop-gradient，监督同模型浅层分支；证据主要是图像分类。作者当前代码入口指向期刊扩展版本，不能冒称与 ICCV 2019 代码逐字一致。可在 VideoMAE 的一个中间层增加短期辅助检测头，减少单独教师重前向；但若推理仍执行全部 12 层，就没有深度删算收益。对 TAD，浅层可能更适合局部边界，深层语义教师未必应强迫其特征完全相同。[^36]

**data2vec 2.0 → EMA 上下文教师与多掩码成本摊销。** 教师生成停止梯度的上下文目标，学生编码可见内容，使用轻量重建模块；官方代码还检查目标/预测方差，防止坍缩。可迁移“同一教师结果服务多个学生视图”和 masked-position 的语义补偿，但原证据不是实例定位。多学生视图本身增加学生重前向/反向，不能只报教师被摊薄而隐去总训练成本。[^37]

**SoftTeacher → 坐标变换与分任务置信度。** 这是有额外未标注数据的半监督目标检测，而不是本任务同数据自蒸馏的成功证明。值得移植的是 teacher/student 几何对齐、teacher 的 no-grad、分类与框回归分别处理可靠性。官方代码先按图像身份对齐视图，再变换教师框。这里应先对齐视频、窗口和真实时间，再筛掉错误教师框；不能只因分类分数高，就相信边界足够准。[^38]

**Localization Distillation → 定位不是两个分类 logits。** LD 蒸馏边界的概率分布并选择有价值区域；当前 ActionFormer 是两端距离标量，不能直接对两个坐标做 KL，冒充 LD。最小迁移是对齐后的真实端点 L1/DIoU；增加分布式边界头属于另一项架构变更，要单独证明。[^39]

**负结果：更强教师不一定是更好教师。** Cho 与 Hariharan 的研究说明教师准确率、容量与可蒸馏性不是单调关系。这要求先测本项目教师在开发集的正确分类、定位质量和覆盖，而不是默认把同模型 EMA、完整预算教师或 B 教师叠加起来必然有益。[^40]

本报告建议优先只蒸馏一种可靠信号，再逐项增加：同模型 EMA 的分类一致性；然后真实时间定位；必要时边界或实例关系。不同预算 teacher 必须能正确处理不同 clip 数和全局 TIA 长度；不能把 768 帧直接传给写死 384 帧的骨干包装。若采用固定 B→S，则是**跨模型蒸馏**，不是 S 的自蒸馏；当前审查证据未确认官方检查点存在一个可直接归因到本轮结果的 B→S 蒸馏训练协议。

### C. 非均匀位置、非线性决策、集合交互与梯度估计

当前已有四种非线性：内容描述符 MLP、sigmoid 预算校准、累计分布硬采样、集合条件 utility MLP。因此，增加 GELU、增加一层或再次称其为“非线性采样”不足以形成方法贡献。真正欠缺的是**决策与最终集合的关系，以及真实时间对特征与损失的条件作用**。[^4][^5]

| 机制 | 可以解决的问题 | 前提、代价和失败方式 |
|---|---|---|
| 非均匀位置 | 把固定 K 分配到较有用区域。 | 不自动改善表示；高置信动作核心过采样会丢边界与背景对照。 |
| 单调连续时间变换 | 保证顺序，并提供可微的采样密度表示。 | 当前 CDF 已属此类；取整、去重、最小间隔仍是离散约束。连续坐标不等于真实帧。 |
| Δt/支持区间感知 | 让 TIA 或融合模块知道相邻 token 对应多少真实时间。 | 编码时间不会补回缺帧；对冻结骨干宜先在 adapter/融合端加入小模块，不重训整个骨干。 |
| 覆盖与多样性 | 避免预算全部落在相似动作核心或单个实例。 | 特征多样性可能优先场景切换而非动作；硬最大间隔和全局覆盖可作安全约束。 |
| 边界/不确定性 | 把精算分配给起止、模糊转场和短实例。 | 不确定背景可能占满预算；必须混入覆盖位置，保留边界两侧，而非只选峰值。 |
| 规则 clip、非均匀中心 | 保留 pretrained clip 内的局部时间节奏，改变 clip 之间的预算分布。 | 可能漏掉短于 clip 的动作；跨 clip 的全局 TIA 仍需要间隔信息。 |
| 迭代集合修正 | 使每次效用预测与实际执行的交换一致。 | 若每次都重跑大骨干则失去效率；推理只能进行少量廉价打分更新，不用 GT 或教师验证。 |

一个可检验的集合目标是

\[
\min_{S:\,|S|=K}\;
\mathbb E[\mathcal L_{\mathrm{det}}^{\mathrm{true}}(S)]
+\lambda_{\mathrm{cov}}\sum_t w_t\min_{i\in S}|t-i|
+\lambda_{\mathrm{red}}\!\sum_{i<j,\,i,j\in S} R(C_i,C_j,|i-j|),
\]

同时限制最大间隔、端点覆盖或分区最低预算。`w_t` 在训练时可以受 GT 边界监督，在推理时必须完全由输入预测与固定覆盖规则给出。边界附近的相似帧可能是有价值的双侧证据，所以冗余项不能无条件排斥它们。这个目标不自动具有次模贪心保证，也不是凭公式即可证明的创新。

梯度估计应列出实测方差和一致性：当前 ST 便宜但有偏；REINFORCE 对所定义随机策略及其真实测量奖励可给无偏估计，但高方差、需要 baseline 与重复样本；连续松弛/软 top-k 易训练，却可能训练软混合、测试硬集合。若所谓“选择”发生在全帧大骨干之后，它只能减少下游计算，不能归入原始帧骨干减算。TR-BERT 的随机策略训练是相关先例，但不能把其 NLP 加速幅度移植到视频。[^31]

### D. 检测、分割、span 和点云的坐标保留方式

| 先例与证据任务 | 原始机制与坐标处理 | 本项目的迁移、实际少算位置与边界 |
|---|---|---|
| **PointRend，实例/语义分割** | 从粗预测中选不确定点，结合精细特征；保持图像/框坐标，并逐步细化。官方代码先插值 logits 再算不确定性，而不是先算不确定性再插值。 | 可做时间边界局部细化；落点 `fusion.py` 或边界 head。主要省 point head，不省已经执行的密集图像骨干。边界细化无法救回被粗分支完全漏掉的实例。[^28] |
| **Sparse DETR，2D 检测** | 预测将被 decoder 使用的 encoder query；只精更新选中 query，然后 scatter 回全位置状态，其余保留旧特征。 | 最贴近“有粗状态、少量深更新”的结构。落点 projection 或深层 token 更新；减少选中 query 的 attention/FFN，而不是自动省全部 RGB 编码。坐标和未选位置状态不能一并丢弃。[^29] |
| **CoLT5，长文本 QA/摘要等** | 全 token 轻分支加路由后的重分支，保存完整序列位置。 | 支持全时间低成本状态；不是 NER/事件边界的直接证据。作者官方实现未在本轮确认，不能用第三方 CoLT5-attention 当成官方代码或稳定性证明。[^30] |
| **TR-BERT，含抽取式 span QA** | token 可在不同层停止更新，保留退出层表征；深层仅处理剩余 token。原论文明确恢复各 token 的最终状态供预测。 | 保留原 token/时间索引，将“停止精算”与“删除位置”分离。可借鉴浅深特征融合，不把重编号后的相邻位置当作原始相邻位置；已退出边界若表征不足仍会错。[^31] |
| **Sequence-Labeling Early-Exit，NER 等序列标注** | 用邻域窗口共同决定 token 退出，并用 self-sampling 缓解训练/推理差异。 | 比只看单点置信度更适合起止边界。落点小型边界/退出头；减少高层更新，但提前退出的错误实体边界可能不可逆。官方代码有独立的自采样训练阶段，不能隐去成本。[^32] |
| **IA-SSD，3D 目标检测** | 类别/中心感知采样和投票，xyz 与特征一同保留，下游预测真实空间中心和框。 | 可借鉴任务驱动、实例覆盖采样；真正减少后续点集聚合。不能照搬“只留前景中心”：TAD 起止需要边界外背景与双侧证据。[^33] |
| **RandLA-Net，大规模点云语义分割** | 廉价随机采样配合局部特征聚合。 | 反证“复杂学习采样必需”；对应 U384+融合。随机/均匀自身仍会漏小目标，需要补偿与完整坐标监督。[^34] |
| **SampleNet，分类/重建/配准** | 用输入点邻域的软投影近似可微采样，推理再匹配到原始点；配准还要求两点云的采样一致性。 | 可借鉴跨视图一致性和软/硬差距控制；原论文不是 3D 检测证据。RGB 混帧不等价于选真实帧；最终必须 snap、去重、排序并严格满足 K。[^35] |

对于文本事件/span，边界之外的语境也可能决定类型或关系；对于点云配准，两侧一致性比单侧显著性更重要；对于 TAD，实例的真实时间间隔和外部上下文同样不能由“选中序号”替代。这是迁移假说，不是已经在该仓库验证的收益。

## 六、主方案：真实时间网格上的粗细残差融合

### 6.1 为什么把它列为主方案

它直接针对两个已确认的结构事实：全时序 CNN/ASFormer 信息已经算过但没有直接输入检测器；检测监督却放在采样后的 rank 轴。相比继续训练更复杂的选帧器，这个方案先检验是否缺少可供检测器使用的证据。优先从同日程 U384 开始，采样固定时也可成立；只有学习路由在开发集超过均匀采样才保留 BMCR 路由。

### 6.2 张量、真实坐标与融合位置

令原候选网格 `t=0,…,T−1`，T=768；选择 `S={s₀,…,sₖ₋₁}`，K=384。记骨干通道 D 为 S 的 384 或 B 的 768。

已有粗分支产生

\[
C\in\mathbb R^{B\times96\times768}.
\]

VideoMAE 的二帧 tubelet 经空间池化、全局 TIA 后，取**最终时间上采样之前**的特征

\[
F\in\mathbb R^{B\times D\times192}.
\]

当前 `Interpolate` 会把 192 上采样成 384；新接口应返回原 tubelet 表征及其有效性，而不是先把它当作等间距 384 帧特征，再无条件缩放两倍。正常 tubelet 的真实候选时间为

\[
u_j=(s_{2j}+s_{2j+1})/2.
\]

对于短窗口的半有效 tubelet，只对有效成员求中心，并携带有效比例；没有有效成员的 tubelet 不参与插值。先修正零 RGB 填充，再保留 padding mask，不能让重复填充成为新的真实观测。

按 `u_j` 将 F 插值到真实候选网格：

\[
\widetilde F_t=\mathcal A(F,u,t),\quad
Z_t=\widetilde F_t+g_t\,W_c C_t,
\]

\[
g_t=\sigma\!\left(\mathrm{MLP}_{102\to32\to1}
\left[C_t,\,d_L,d_R,\Delta t,\,q,\,b,\,h\right]\right).
\]

六个标量分别表征两侧观测距离、局部时间间隔、有效观测比例、粗边界置信度、粗预测熵，均在真实时间和有效 mask 下计算并归一化。`W_c:96→D` 零初始化，先训练常数门控对照，再启用可学习门控；线性插值只补网格，不被称为恢复了缺失观测。

`Z` 的形状是 `B×D×768`，送入现有 projection，输出 512 通道时间金字塔与 20 类 ActionFormer head。设置 detector 的 `max_seq_len=768`，使用原候选 masks 和原 GT；移除这条分支的 GT→rank 与预测→true 变换。最终视频秒数仍按 `候选时间×4 + 窗口起始帧 + offset` 除 fps 计算，不能重复乘两次稀疏采样率。[^18][^19]

### 6.3 梯度路径与目标

初始只使用

\[
\mathcal L_{\rm main}
=\mathcal L_{\rm cls}^{\rm true}
+\lambda_r\mathcal L_{\rm DIoU}^{\rm true}
+\mathcal L_{\rm existing\ scout\ auxiliaries}.
\]

检测梯度进入 projection/head、TIA、`W_c` 和门控。粗特征接口使用编码器重放后的表征，但不沿用专门控制路由的 `adapt_scale` 来意外关闭融合路径；初始保持检测梯度不回流 CNN stem，CNN 继续接受既有辅助动作监督，避免一开始同时改变太多优化路径。是否允许检测梯度进入 CNN、是否拼接 CNN 与 ASFormer 特征，分别消融。

若保留学习采样，其梯度仍可通过现有零前向 RGB 桥进入策略；插值的硬坐标节点 detach，梯度只经过特征值。初始关闭交换损失和 KD；融合有效后，才在同一个真实网格下重新评估 utility。此时交换前后的检测 anchor 网格与 GT 分配天然一致，能减少当前 rank 漂移对分类效用测量的干扰。

### 6.4 训练与推理伪代码

```text
# 所有坐标均为原候选时间；此处为设计伪代码，不是已运行实现。
TRAIN_STEP(batch, successful_step):
    optimizer.zero_grad(set_to_none=True)
    C, scout_aux, route_state = scout.forward_with_fusion_features(batch.rgb, batch.mask)
    S = uniform_K384(batch.mask)                   # 第一阶段主对照
    # 学习路由只有经开发集验证后替换上一行；测试不能读取 GT。
    X = hard_gather_with_existing_ST_bridge(batch.rgb, S)
    X = edge_pad_without_zeroing_valid_context(X, S.valid)
    F, tube_valid = backbone.forward_before_temporal_upsample(X)  # B,D,192
    u, quality = true_tubelet_centers(S, tube_valid)
    F_dense, gap = irregular_align_values(F, stop_grad(u), original_grid)
    Z = F_dense + gate(C, gap, quality) * zero_init_projection(C)
    pred = detector(Z, original_mask=batch.mask)                  # B,D,768 input
    loss = true_grid_detection_loss(pred, batch.gt) + existing_aux_loss(scout_aux)
    backward(loss)
    if finite_loss_and_gradients:
        optimizer.step(); scheduler.step(); EMA.update()
        successful_step += 1

INFER(video_window):
    C, _, route_state = scout(video_window.rgb, video_window.mask)
    S = selected_deployment_policy(route_state, K=384)
    F, valid = backbone_192_tokens(hard_gather_and_edge_pad(video_window.rgb, S))
    u, quality = true_tubelet_centers(S, valid)
    Z = align_to_original_grid(F, u) + gate(C, gaps(u), quality) * project(C)
    boxes, scores = detector(Z, original_mask=video_window.mask)
    return convert_original_candidate_coordinates_to_video_seconds(boxes, scores)
# 视频所有窗口完成后，再执行原有逐类别 SoftNMS 与完整评测。
```

### 6.5 真实成本与修改文件

保留 BMCR 路由时，从 384 恢复到 768 检测网格，按现有 profile 分量估算，projection/head 增量为 S 的 7.5804 GMAC、B 的 7.8069 GMAC；`96→D` 粗投影约 0.0283/0.0566 GMAC，示例标量门控约 0.0025 GMAC。总计约 **612.78/2046.60 GMAC**，相对官方仍约减少 **47.80%/49.35%**。这是按现有算子形状进行的设计估算，**不是新模型实测**；插值、归一化、访存和更大检测器激活不能忽略。

推理仍仅一次 K384 重骨干，不保留教师。完整网格会增加训练显存；`B=1` 的 FP32 对齐特征本身约 1.125/2.25 MiB，但这不能代表金字塔、中间 attention 和反向图的总增量。若最终使用均匀策略且仍有粗融合，scout 为融合所需不能删除；若均匀且无融合，则部署版本可以移除无用 scout，需单独给出精简版成本。

| 文件 | 变更 |
|---|---|
| `h65/full/scout.py` | 暴露未受 routing adapt 梯度缩放误控的 `coarse_features`；不重复运行一次 CNN。 |
| `h65/full/model.py` | 区分骨干 tubelet 输出与检测网格；修填充；引入融合分支；真实 GT 和 masks；同步修改 raw_route/teacher 接口。 |
| 新增 `h65/full/fusion.py` | tubelet 时间中心、有效比例、稀疏到真实网格插值、粗投影与门控。 |
| `h65/full/geometry.py` | 保留旧行为用于基线；新增明确区分“值可微/坐标元数据”的映射。 |
| `h65/full/runtime.py` | 注册融合参数优化器组和 EMA 状态；checkpoint 记录架构版本。 |
| `tools/full_train.py` | 统一成功更新日程与各项预算计数；独立控制融合、路由、KD。 |
| `tools/full_eval.py` | 根据架构决定是否需要逆映射，防止二次回映；新增分段计时，不更改完整测试集范围。 |
| 新增小型测试 | `identity/uniform/irregular/short/odd` 时间映射、hard gather 前向、有效性、梯度、K384 实际输入形状。 |

首次实现不应修改整个上游 backbone 或重训官方 AdaTAD；本地包装器和新增模块足以验证主要假说。

## 七、备选一：不改推理结构的互补视图 EMA 自蒸馏

这条路线主要改**训练信号**，而不是扩展推理特征网格。student 为 S 或 B 本身；teacher 为同一架构 EMA，按成功更新

\[
\bar\theta_{n+1}=0.999\bar\theta_n+0.001\theta_{n+1}
\]

更新。teacher 始终 eval、no-grad；推理只保留一个 student/其 EMA 权重，不保留双模型或蒸馏头。

同一个视频窗口使用 student 的学习视图与 teacher 的 K384 均匀或覆盖互补视图。空间裁剪先保持相同，确保主要区别是时间采样。这里不能复用当前 batch 中“另一行”的 uniform companion 当作教师目标，因为它通常是另一段内容。较大预算 K768 的同模型教师另列实验，必须先将骨干包装改为正确支持 24/48 个 clips 和相应全局 TIA 长度，且计入约更大的重前向成本；不能默认该教师因看得更多就一定更可靠。

先只加分类一致性。在共同真实时间/共同 GT 对象上建立匹配后，对每类温度化 sigmoid 概率 `p=σ(z/τ)` 使用 Bernoulli KL，而不是把 20 类当作互斥的 softmax：

\[
\mathcal L_{\rm KD,cls}
=\frac{\tau^2}{\sum_i w_i+\epsilon}
\sum_i w_i\,D_{\rm KL}\!\left(
\mathrm{Ber}(\mathrm{sg}[p_i^T])\;\|\;\mathrm{Ber}(p_i^S)
\right).
\]

`w_i` 为 stop-gradient 的可靠性权重：有效时间、教师与 GT 类别一致、合理置信度，以及教师边界质量；已知正 GT 区域不能因为教师漏检就被当成负样本蒸馏。不同 rank 或不同 FPN 索引不构成天然对应，必须按真实时间和对象匹配。教师的后处理阈值不得直接拿测试集调。

分类项有效后，才独立加入定位：

\[
\mathcal L_{\rm KD,loc}
=\frac{1}{\sum_j\omega_j+\epsilon}
\sum_j\omega_j\,
\frac{|\hat a^S_j-\mathrm{sg}[\hat a^T_j]|+
      |\hat b^S_j-\mathrm{sg}[\hat b^T_j]|}
{\max(b^{GT}_j-a^{GT}_j,\epsilon)}.
\]

预测端点先进入真实时间，必须使用第三节所述“值可微”的新映射，不能调用现有会 detach student 值的接口。再增加边界分布或实例关系蒸馏属于独立变体：边界头额外参数、关系配对 O(G²) 和正样本筛选都应收费。深浅层 BYOT 则可在同一视图减少教师重前向，但不自动减少推理层数。

```text
TRAIN_STEP:
    optimizer.zero_grad(set_to_none=True)
    S_student = policy(window)
    student = forward_student_with_raw_logits_and_boxes(window, S_student)
    loss = GT_loss(student)
    if teacher_budget_meter.allows_this_event(successful_step):
        with no_grad(), teacher.eval():
            S_teacher = uniform_or_complementary_K384(window.mask)
            target = teacher.forward_raw(window, S_teacher)
        pairs, weights = match_in_original_time(student, target, GT)
        loss += lambda_cls * Bernoulli_KL(student, stop_grad(target), pairs, weights)
        # 定位项仅在独立消融阶段启用，并保留 student 端点梯度。
    backward_and_update_if_finite(loss)
    EMA.update_only_after_successful_optimizer_step()

INFER:
    return one_model_normal_K384_inference(window)  # 无额外教师前向
```

**修改文件：** 新增 `h65/full/distillation.py`；`model.py` 返回训练用原始 logits/boxes；`geometry.py` 提供可微值映射；`full_train.py` 管理教师预算和事件；`runtime.py` 保证新参数、EMA 和恢复状态完整。推理主干、K 和检测网格可以保持当前结构。

**成本：** 增加的是周期性 teacher 前向及目标匹配，不增加 teacher 反向。当前 BMCR 在激活反馈的区间，每 8 个 batch2 更新最多进行一份基线与两份交换前向，故额外前向上界为 `3/(8×2)=0.1875` 个 K384 重前向/学生样本；不是总训练时间只增加 18.75%。按当前课程、忽略预检，满足条件的事件最多 416 次，全 joint 平均上界约 0.156 F/学生样本。实际交换去重/不可行情况会降低次数，应以记录的已执行重前向与 MAC 为准。

蒸馏实验应替换或复用这份预算，不是在原三次前向上继续叠教师。若使用更低成本的一次教师方案，应诚实报告不同训练成本的 Pareto 点；“等教师预算”比较则由总前向 MAC 配额控制频率。额外学生视图也计入重前向、反向和数据曝光，不能用“同成功步数”隐藏。

**失败与停止：** 教师边界质量不足、对短动作高置信错误，或蒸馏让 @0.7/真实端点误差恶化，就停止定位蒸馏；若只有增加额外 GT 视图也能获得同样收益，应将收益归于额外数据曝光而非 KD。EMA 教师与 student 同偏差、互补视图无新增可用信息时，自蒸馏可能完全不增益。

## 八、备选二：规则 clip 内采样、非均匀 clip 中心与受限集合修正

这条路线主要改**观测几何和离散动作空间**，不是加训练蒸馏，也不是恢复完整 768 高维特征。

把 768 个候选划成 48 个规则 16 帧 clip，每个 clip 内保持原候选步长 4 个原视频帧。选择 24 个 clip，仍严格执行 K384 个真实帧的高成本处理。首先测试均匀选 24 个 clip，与逐帧均匀选 384 帧对照，分离局部预训练节奏和全局稀疏程度；之后才引入内容决策。

粗 clip 表征是 `B×48×96`，由当前粗特征池化并附上边界/不确定性统计。打分既考虑任务收益，也考虑全局覆盖、重复实例、多样性和边界两侧；只选前景置信度最高的 clip 不是充分目标。保持真实 clip 索引、帧索引和时间戳；短视频按有效候选处理，填充不计为新观测。

\[
\min_{\mathcal C:\,|\mathcal C|=24}
\mathbb E\mathcal L^{\rm true}_{det}(S(\mathcal C))
+\lambda_c\,\mathrm{CoverageGap}(\mathcal C)
+\lambda_d\,\mathrm{Redundancy}(\mathcal C),
\]

并限制允许的最大空档。与当前不同，条件效用针对**一个 clip 换一个 clip**；不能直接沿用单帧交换标签或尺度。

先使用 K384 的规则 clip 几何对照；若局部节奏确实改善，可把稀疏检测头改为真实时间 anchor：输出仍在稀疏网格上，但每层携带对应真实时间和有效支持区间，正样本分配/回归目标使用真实坐标，边界直接解码为 `真实中心±预测距离×尺度`，不再依赖事后非线性逆映射。其实现可新增 `time_aware_head.py`，而不恢复完整高维密集特征。各金字塔层的 anchor 时间、物理回归范围和 mask 都必须显式更新；仅给位置编码加一个 Δt 不能冒充完成这项修改。

集合推理最多执行一到两轮廉价 trust-region 修正：选择一个允许的成员/非成员 clip 对，交换后重新计算条件，并保留覆盖约束。**推理不运行教师或 GT 检测损失来验证交换。** 训练时周期性教师可测真实交换收益，按备选一相同的总预算规则收费。

```text
TRAIN/INFER ROUTING:
    clip_features = pool_existing_coarse_features_into_48_clips()
    C0 = coverage_preserving_select_24(clip_features)
    for r in range(pre_registered_small_number_of_refinements):
        candidate_pair, predicted_gain = conditional_clip_swap_score(Cr)
        if feasible_and_above_dev_fixed_threshold(candidate_pair, predicted_gain):
            Cr = exact_one_clip_swap(Cr, candidate_pair)
    S = chronological_concat_of_selected_regular_clips(Cr)  # K=384
    run_heavy_backbone_once(S)
    detect_with_explicit_original_timestamps()
TRAIN ONLY:
    if teacher_budget_allows:
        measure baseline and exact proposed clip swap; train the corresponding gain
    optimize GT detection loss; update EMA after successful optimizer update
```

**修改文件：** 新增 `clip_sampler.py` 和必要时的 `time_aware_head.py`；更新 `scout.py` 的 48-cell 聚合、`utility.py` 的 clip 交换对象、`model.py` 的打包与时间元数据。骨干仍执行 24 个规则 clip；采样打分规模从 768 个候选降为 48 个 clip，但实际时延仍须测量。

**真实成本：** heavy backbone 的物理形状可与当前 K384 相同；时间元数据与低维打分是新增开销；若稀疏 head 张量长度不变，其主要矩阵成本接近当前检测头，但不能在未 profile 时保证完全相同。全局 TIA 跨稀疏 clip 的间隔失配仍存在，可后续在 adapter 中加入小型时间间隔调制；这与局部 clip 保真是两个独立因素。

**失败与停止：** 短动作/相邻重复动作可能被整个 clip 决策遗漏；若规则 clip 控制组已伤害这些实例，就不应继续训练复杂 clip 策略。优先退回逐帧 U384 或更细 packet，并如实记录重新组合 16 帧预训练 clip 的代价。若每次修正都需要大骨干验证才能可靠，也应停止该推理策略。

## 九、最小可解释消融与实验协议

### 9.1 数据、初始化和更新数

开发集必须从原训练来源的 200 视频中按视频隔离产生，例如预注册分层 160/40 划分；覆盖类别、短实例与重复动作，固定随机种子和清单。仓库的 `cfg.val` 实际指向最终测试来源，不能直接拿来调超参数。已经在全部 200 视频上训练过的 warm checkpoint 也不能充当这个 40 视频开发集的未见数据起点。[^13][^14]

所有开发实验使用同一 K400 识别预训练、空间分辨率、增强分布、batch2 和相同数据抽样流，以 **2000 warm + 4000 joint 成功更新**为主日程。160 视频时每遍不再是 100 个 batch，所以应按成功更新数和预先固定采样流控制，不以“同样 20+40 epoch”掩盖更新量改变。数值失败、跳过的更新、额外 teacher/student 前向分别记录。

只有前向语义与参数完全相同的 warm 阶段才共享 checkpoint。本方案固定采用：U/H/B 共享功能一致的均匀 warm；R/F/G 在各自预注册的检测网格与融合结构上，分别从相同 K400 初始化执行 2000 次 warm，再执行 4000 次 joint。不同结构额外发生的 warm 成本如实记账，不在 joint 中途临时切换网格或按成绩改日程。最终配置锁定后，再用全部 200 视频按规定课程训练终端 EMA，并完整评估 211 测试视频/792 窗口。

官方现有 S/B checkpoint 只作为参照，不重新完整训练官方 AdaTAD；因此本研究可以回答哪些新组件改进了同协议 K384 系统，却不能消除“与官方检查点训练历史不同”的因果边界。

### 9.2 按阶段展开，而非全排列

| 阶段/编号 | 实验 | 唯一主要问题 |
|---|---|---|
| 0 | 固定权重短窗 padding 检查、坐标单测、伙伴向量化等价性与分段 profile | 先排实现偏差和运行开销；不训练新模型、不用测试 AP 选超参数。 |
| I-U | **同日程 U384**，路由强制均匀，保留研究所需相同辅助训练 | 当前学习路由是否真的优于简单少帧？不能用仅 warm20 作为它的替代品。 |
| I-H/I-B | H65-C / BMCR-T 同协议开发集对照 | 对齐更新数和预训练后，代理替换是否稳定有效？ |
| II-R | U384，仅恢复真实 768 检测网格；无粗残差 | 是否只是网格、anchor 密度与几何改变带来收益？ |
| II-F | II-R + 已有粗特征的简单残差/常数门控 | 全时序粗证据有没有独立价值？这是主方案最小核心。 |
| II-G | II-F + 间隔/边界条件门控 | 自适应门控是否优于更简单融合？只有 II-F 有效再做。 |
| III-KD | 最佳简单基线 + 一种 KD 信号；教师总预算匹配 | 是否来自知识迁移，而非额外计算/额外视图？分类与定位不同时首次启用。 |
| III-C | 均匀 24 规则 clip，然后可选受限学习交换 | 局部时间节奏是否比逐帧非均匀更重要？先几何、后学习。 |

这不是要求一次训练完所有行。最低决策链是 U→R→F；只有有收益才进行 G 或 KD；clip 是替代分支。对现有 H/B 的复现用于建立可比较的开发基线，不需要重新训练官方模型。额外诊断可比较“粗特征只在选中位置融合”与密集融合、ASFormer 单支与 CNN+ASFormer，但不列为第一轮必跑组合。

KD 对照至少包括相同教师预算下“算教师但不蒸馏”，以及在确实增加学生视图时的“相同视图只用 GT 监督”。否则更多重计算或更多视频曝光会混入蒸馏收益。零教师预算的 U/R/F 单列成本曲线，不用人为浪费教师计算才称公平。

### 9.3 评价指标和诊断

主评价仍是五阈值平均 mAP 与完整测试。开发阶段额外记录 @0.7、真时间端点误差（秒数及 GT 持续时间归一）、短/重复动作召回、每类 AP、每实例最近观测距离、边界两侧覆盖、clip 内/间 Δt 分布与高置信误检。

对最终 BMCR 条件头，重新在隔离的开发视频上测：正负号准确率、Spearman、收益幅度校准、实际所选交换的 regret、`|S₁△S₀|`、伙伴是否变化、局部预计收益与最终集合收益是否一致。温启动旧 H65 的低相关只能否定那个代理，不能验证新 BMCR。

对 S/B 成因，使用同一批窗口和相同选择索引作交叉控制：先比较均匀与规则 clip 的时间失配敏感性，再比较路由带来的增量。只统计高 tIoU mAP 还不够；在正确类别和相近置信度下比较端点误差，避免把排序变化误认为回归变化。

有资源时最终候选至少三种子；分别报告种子间差异和视频级配对 bootstrap 的 AP 差异区间。两种不确定性不能相互替代，也不能把 bootstrap 当成额外独立训练。

### 9.4 性能、时延与显存的统一测量

用同一 GPU、软件版本、精度、输入 residency、batch 和预热方案成对交替测量。分段记录 `decode/transfer → CNN → ASFormer → density/CDF → condition → gather → backbone → projection/head → time mapping → NMS`，而不是只看一个总 MAC。

至少给全部窗口的 p50/p95、均值、完整/尾窗/短窗分组，以及整视频端到端吞吐。分别记录 CUDA allocated 峰值、reserved 峰值、增量激活与训练峰值；冷启动与缓存推理分开。任何缓存都必须计初始化、失效和存储成本；当前全局 TIA 下不能默认跨窗口重用 post-TIA 特征完全等价。

### 9.5 停止与改道条件

**学习路由未胜均匀：** 若 BMCR 在同协议开发集上不稳定胜过 U384，或最终集合收益经常与局部预测相反，就停止增加条件头容量，优先 U384+融合；不把细调更多 gain 当成默认下一步。

**融合只赢在网格：** 若 R 与 F 接近，粗特征没带来独立收益，就保留更简单 R；若 F 有益而 G 无益，使用常数/简单残差，不把门控复杂度包装成贡献。若梯度进入 CNN 导致精度或稳定性下降，则维持原辅助监督路径。

**蒸馏伤害定位：** 若 teacher 的有效覆盖不足，或新增定位 KD 使 @0.7、短实例与端点误差恶化，就保留 GT 或仅分类 KD；不通过加更多关系损失掩盖失败。

**规则 clip 漏实例：** 均匀 clip 控制组即损害短/重复动作时，停止复杂 clip 策略，退回更细 packet 或逐帧基线；保留这一负结果。

**MAC 省而实测不快：** 完成静态形状向量化和热点定位后，若路由仍吞掉大部分节省，则部署采用更简单采样器。精度/MAC 改善可单独报告，但不再称“加速方案”。

未来“性能保持”容忍度必须事先在开发阶段规定，例如平均与 @0.7 各允许至多某个固定百分点，并同时规定真实时延目标；不能看完最终测试再选择容忍度。本报告不为当前结果倒设一个宽松阈值，也不提供未验证的预期 mAP。

## 十、创新性边界与最终建议

已有机制包括多速率双流、粗细残差、稀疏深更新、EMA/深浅自蒸馏、CDF 采样、窗口置信度、边界细化和任务驱动点采样。把这些名称叠加不是充分创新。比较有价值、仍待证明的切入点是：**在严格真实 RGB K 预算下，用可观测性和实际时间间隔约束稀疏重计算，把既有廉价全时间状态交给定位器，并证明它改善了真实边界，而不只是改变检测网格或训练账单。**

近期时间算子设计也在继续发展。2026 年 LiquidTAD 的 v2 明确将其描述为液态动力学启发的并行松弛先验，而不是复现完整 Liquid Neural Network 动力学。因此不能仅凭“Liquid/continuous-time”名称，把它当成本项目已经解决真实非均匀时间戳或硬 RGB 跳算的证据。该预印本的全部实现和结果未在本轮复现，不用它支持本项目收益。[^41]

推荐优先级为：**修正/隔离填充语义 → 完成 U384 同日程对照 → R/F 分解验证真实网格与粗证据 → 成功后再做门控或一种自蒸馏 → 若局部时间失配明显，再走规则 clip 备选。** 任何阶段都允许得到“均匀采样更好、简单残差足够、当前 BMCR 主张不成立”的结论。

## 来源与核验入口

以下本项目链接均固定到审查提交；任务文档例外单独标明。外部官方代码仅用于相应机制核对，未对外部整库进行版本审计。BYOT 的作者代码当前指向期刊扩展版；CoLT5 的作者官方代码及 BRN 的官方代码未在本轮确认。对外部代码的深入检查主要集中在 ASFormer、PointRend、Sparse DETR、SoftTeacher、data2vec、作者版自蒸馏、IA-SSD 与 SampleNet；TALLFormer 核对了官方记忆库训练说明，其他论文/仓库链接不代表完整复现。

[^1]: [研究任务全文（任务提交 8a8529…）](https://github.com/yuzbo/BMCR-T-AdaTAD/blob/8a8529093f8a96b228e138c272de0f9df999bda8/docs/EXTERNAL_REVIEW_PROMPT.md)。
[^2]: [固定审查提交 REVIEW_MAP](https://github.com/yuzbo/BMCR-T-AdaTAD/blob/15280e5dc29e3df18d085aaba21067e04107ce84/docs/REVIEW_MAP.md)。
[^3]: [h65/full/model.py](https://github.com/yuzbo/BMCR-T-AdaTAD/blob/15280e5dc29e3df18d085aaba21067e04107ce84/h65/full/model.py)。
[^4]: [h65/transport.py](https://github.com/yuzbo/BMCR-T-AdaTAD/blob/15280e5dc29e3df18d085aaba21067e04107ce84/h65/transport.py)。
[^5]: [h65/full/scout.py](https://github.com/yuzbo/BMCR-T-AdaTAD/blob/15280e5dc29e3df18d085aaba21067e04107ce84/h65/full/scout.py)；[h65/scout.py](https://github.com/yuzbo/BMCR-T-AdaTAD/blob/15280e5dc29e3df18d085aaba21067e04107ce84/h65/scout.py)；[采用的 ASFormer 源码](https://github.com/yuzbo/BMCR-T-AdaTAD/blob/15280e5dc29e3df18d085aaba21067e04107ce84/references/ASFormer/model.py#L210-L365)。
[^6]: [h65/full/geometry.py](https://github.com/yuzbo/BMCR-T-AdaTAD/blob/15280e5dc29e3df18d085aaba21067e04107ce84/h65/full/geometry.py)。
[^7]: [h65/full/utility.py](https://github.com/yuzbo/BMCR-T-AdaTAD/blob/15280e5dc29e3df18d085aaba21067e04107ce84/h65/full/utility.py)。
[^8]: [h65/full/runtime.py](https://github.com/yuzbo/BMCR-T-AdaTAD/blob/15280e5dc29e3df18d085aaba21067e04107ce84/h65/full/runtime.py)。
[^9]: [tools/full_train.py](https://github.com/yuzbo/BMCR-T-AdaTAD/blob/15280e5dc29e3df18d085aaba21067e04107ce84/tools/full_train.py)；[课程和辅助损失](https://github.com/yuzbo/BMCR-T-AdaTAD/blob/15280e5dc29e3df18d085aaba21067e04107ce84/h65/full/objectives.py)。
[^10]: [tools/full_eval.py](https://github.com/yuzbo/BMCR-T-AdaTAD/blob/15280e5dc29e3df18d085aaba21067e04107ce84/tools/full_eval.py)。
[^11]: [comparison.json](https://github.com/yuzbo/BMCR-T-AdaTAD/blob/15280e5dc29e3df18d085aaba21067e04107ce84/phase2_20260910/comparison.json)；[S-BMCR 原始 profile](https://github.com/yuzbo/BMCR-T-AdaTAD/blob/15280e5dc29e3df18d085aaba21067e04107ce84/phase2_20260910/runs/s_bmcr_test/profile.json)。
[^12]: [TRAINING_SUMMARY.json](https://github.com/yuzbo/BMCR-T-AdaTAD/blob/15280e5dc29e3df18d085aaba21067e04107ce84/phase2_20260910/TRAINING_SUMMARY.json)；[实验报告](https://github.com/yuzbo/BMCR-T-AdaTAD/blob/15280e5dc29e3df18d085aaba21067e04107ce84/phase2_20260910/EXPERIMENT_REPORT.md)。
[^13]: [PROTOCOL.md](https://github.com/yuzbo/BMCR-T-AdaTAD/blob/15280e5dc29e3df18d085aaba21067e04107ce84/docs/PROTOCOL.md)。
[^14]: [full_audit.py](https://github.com/yuzbo/BMCR-T-AdaTAD/blob/15280e5dc29e3df18d085aaba21067e04107ce84/tools/full_audit.py)；[S 审计摘要](https://github.com/yuzbo/BMCR-T-AdaTAD/blob/15280e5dc29e3df18d085aaba21067e04107ce84/phase2_20260910/runs/s_audit/completed.json)；[B 审计摘要](https://github.com/yuzbo/BMCR-T-AdaTAD/blob/15280e5dc29e3df18d085aaba21067e04107ce84/phase2_20260910/runs/b_audit/completed.json)。
[^15]: Liu, Zhang, Zhao, Ghanem. **End-to-End Temporal Action Detection with 1B Parameters Across 1000 Frames**, CVPR 2024；[arXiv 2311.17241v2](https://arxiv.org/html/2311.17241v2)。
[^16]: [固定上游 AdaTAD model-zoo README](https://github.com/yuzbo/BMCR-T-AdaTAD/blob/15280e5dc29e3df18d085aaba21067e04107ce84/upstream/configs/adatad/README.md)；[S 配置](https://github.com/yuzbo/BMCR-T-AdaTAD/blob/15280e5dc29e3df18d085aaba21067e04107ce84/upstream/configs/adatad/thumos/e2e_thumos_videomae_s_768x1_160_adapter.py)；[B 配置](https://github.com/yuzbo/BMCR-T-AdaTAD/blob/15280e5dc29e3df18d085aaba21067e04107ce84/upstream/configs/adatad/thumos/e2e_thumos_videomae_b_768x1_160_adapter.py)。
[^17]: [vit_adapter.py](https://github.com/yuzbo/BMCR-T-AdaTAD/blob/15280e5dc29e3df18d085aaba21067e04107ce84/upstream/opentad/models/backbones/vit_adapter.py)；[backbone_wrapper.py](https://github.com/yuzbo/BMCR-T-AdaTAD/blob/15280e5dc29e3df18d085aaba21067e04107ce84/upstream/opentad/models/backbones/backbone_wrapper.py)。
[^18]: [ActionFormer 配置](https://github.com/yuzbo/BMCR-T-AdaTAD/blob/15280e5dc29e3df18d085aaba21067e04107ce84/upstream/configs/_base_/models/actionformer.py)；[actionformer_head.py](https://github.com/yuzbo/BMCR-T-AdaTAD/blob/15280e5dc29e3df18d085aaba21067e04107ce84/upstream/opentad/models/dense_heads/actionformer_head.py)；[anchor_free_head.py](https://github.com/yuzbo/BMCR-T-AdaTAD/blob/15280e5dc29e3df18d085aaba21067e04107ce84/upstream/opentad/models/dense_heads/anchor_free_head.py)。
[^19]: [数据变换](https://github.com/yuzbo/BMCR-T-AdaTAD/blob/15280e5dc29e3df18d085aaba21067e04107ce84/upstream/opentad/datasets/transforms/end_to_end.py)；[THUMOS 数据配置](https://github.com/yuzbo/BMCR-T-AdaTAD/blob/15280e5dc29e3df18d085aaba21067e04107ce84/upstream/configs/_base_/datasets/thumos-14/e2e_train_trunc_test_sw_256x224x224.py)；[视频秒数变换](https://github.com/yuzbo/BMCR-T-AdaTAD/blob/15280e5dc29e3df18d085aaba21067e04107ce84/upstream/opentad/models/utils/post_processing/utils.py)。
[^20]: [窗口合并与 NMS](https://github.com/yuzbo/BMCR-T-AdaTAD/blob/15280e5dc29e3df18d085aaba21067e04107ce84/upstream/opentad/cores/test_engine.py#L90-L140)；[AP 实现](https://github.com/yuzbo/BMCR-T-AdaTAD/blob/15280e5dc29e3df18d085aaba21067e04107ce84/upstream/opentad/evaluations/mAP.py#L294-L348)。
[^21]: Feichtenhofer et al. **SlowFast Networks for Video Recognition**, ICCV 2019；[论文](https://arxiv.org/abs/1812.03982)；[作者代码](https://github.com/facebookresearch/SlowFast)。
[^22]: **AdaFocus**, [arXiv 2105.03245](https://arxiv.org/abs/2105.03245)；**AdaFocusV2**, [arXiv 2112.14238](https://arxiv.org/abs/2112.14238)；[AdaFocus 作者代码](https://github.com/blackfeather-wang/AdaFocus)；[AdaFocusV2 作者代码](https://github.com/LeapLabTHU/AdaFocusV2)。
[^23]: Cheng, Bertasius. **TALLFormer: Temporal Action Localization with a Long-memory Transformer**, ECCV 2022；[论文](https://arxiv.org/abs/2204.01680)；[官方记忆库初始化/更新说明](https://github.com/klauscc/TALLFormer/blob/main/README.md#b-init-the-memory-bank)。
[^24]: Liu et al. **ETAD: Training Action Detection End to End on a Laptop**；[arXiv 2205.07134v2，2022](https://arxiv.org/abs/2205.07134v2)；[作者代码](https://github.com/sming256/ETAD)。
[^25]: **DyFADet: Dynamic Feature Aggregation for Temporal Action Detection**, ECCV 2024；[论文版本](https://arxiv.org/html/2407.03197v1)；[作者代码](https://github.com/yangle15/DyFADet-pytorch)。
[^26]: **DiGIT**, CVPR 2025；[论文版本](https://arxiv.org/html/2505.05711v1)；[作者代码](https://github.com/Dotori-HJ/DiGIT)。
[^27]: Kim et al. **Boundary-Recovering Network for Temporal Action Detection**；[arXiv 2408.09354v1](https://arxiv.org/html/2408.09354v1)。引用此预印本版本，不推定未核实的最终出版版本或代码。
[^28]: Kirillov et al. **PointRend: Image Segmentation as Rendering**, CVPR 2020；[论文](https://arxiv.org/abs/1912.08193)；[官方 point_features.py](https://github.com/facebookresearch/detectron2/blob/main/projects/PointRend/point_rend/point_features.py#L40-L200)。
[^29]: Roh et al. **Sparse DETR: Efficient End-to-End Object Detection with Learnable Sparsity**, ICLR 2022；[论文](https://arxiv.org/abs/2111.14330)；[官方 gather/scatter encoder](https://github.com/kakaobrain/sparse-detr/blob/main/models/deformable_transformer.py#L394-L471)。
[^30]: Ainslie et al. **CoLT5: Faster Long-Range Transformers with Conditional Computation**, EMNLP 2023；[ACL 原文入口](https://aclanthology.org/2023.emnlp-main.309/)。本轮未确认作者官方实现，不将第三方实现作为官方证据。
[^31]: Ye et al. **TR-BERT: Dynamic Token Reduction for Accelerating BERT Inference**, NAACL 2021；[论文及 PDF](https://aclanthology.org/2021.naacl-main.463/)；[作者代码](https://github.com/thunlp/TR-BERT)；[核对的分类编码器实现（外部固定提交）](https://github.com/thunlp/TR-BERT/blob/bfac1d9e6704be482f99b20c2c571b0759300bd0/transformers/src/transformers/modeling_autobert.py)。抽取式 QA 的状态保留结论依据原论文方法和任务说明，不把分类文件当作 QA 专用实现。
[^32]: Li et al. **Accelerating BERT Inference for Sequence Labeling via Early-Exit**, ACL 2021；[论文](https://aclanthology.org/2021.acl-long.16/)；[作者代码/两阶段说明](https://github.com/LeeSureman/Sequence-Labeling-Early-Exit)。
[^33]: Zhang et al. **Not All Points Are Equal: Learning Highly Efficient Point-based Detectors for 3D LiDAR Point Clouds（IA-SSD）**, CVPR 2022；[论文](https://arxiv.org/abs/2203.11139)；[实际 IASSD backbone](https://github.com/yifanzhang713/IA-SSD/blob/main/pcdet/models/backbones_3d/IASSD_backbone.py)。
[^34]: Hu et al. **RandLA-Net: Efficient Semantic Segmentation of Large-Scale Point Clouds**, CVPR 2020；[论文](https://arxiv.org/abs/1911.11236)；[作者代码](https://github.com/QingyongHu/RandLA-Net)。
[^35]: Lang, Manor, Avidan. **SampleNet: Differentiable Point Cloud Sampling**, CVPR 2020；[论文](https://arxiv.org/abs/1912.03663)；[作者代码](https://github.com/itailang/SampleNet)；[registration/src/samplenet.py](https://github.com/itailang/SampleNet/blob/master/registration/src/samplenet.py)。
[^36]: Zhang et al. **Be Your Own Teacher: Improve the Performance of Convolutional Neural Networks via Self Distillation**, ICCV 2019；[论文](https://arxiv.org/abs/1905.08094)；[原作者入口](https://github.com/ArchipLab-LinfengZhang/pytorch-self-distillation)；[当前转向的期刊扩展版 train.py](https://github.com/ArchipLab-LinfengZhang/pytorch-self-distillation-final/blob/master/train.py#L95-L165)。
[^37]: Baevski et al. **Efficient Self-supervised Learning with Contextualized Target Representations（data2vec 2.0）**, ICML 2023；[PMLR 原文](https://proceedings.mlr.press/v202/baevski23a.html)；[官方 data2vec2.py](https://github.com/facebookresearch/fairseq/blob/main/examples/data2vec/models/data2vec2.py)。
[^38]: Xu et al. **End-to-End Semi-Supervised Object Detection with Soft Teacher**, ICCV 2021；[论文](https://arxiv.org/abs/2106.09018)；[官方 teacher/student 与框变换](https://github.com/microsoft/SoftTeacher/blob/main/ssod/models/soft_teacher.py#L25-L100)。
[^39]: Zheng et al. **Localization Distillation for Dense Object Detection**, CVPR 2022；[CVF 原文](https://openaccess.thecvf.com/content/CVPR2022/html/Zheng_Localization_Distillation_for_Dense_Object_Detection_CVPR_2022_paper.html)；[作者代码](https://github.com/HikariTJU/LD)。
[^40]: Cho, Hariharan. **On the Efficacy of Knowledge Distillation**, ICCV 2019；[论文](https://arxiv.org/abs/1910.01348)。
[^41]: Sun et al. **LiquidTAD: Efficient Temporal Action Detection via Parallel Liquid-Inspired Temporal Relaxation**, 2026 年预印本；[arXiv 2604.18274v2](https://arxiv.org/abs/2604.18274v2)。采用 v2 对方法定位的说明，不把 v1 的措辞无条件当作当前版本；不引用未独立验证的本项目收益或实现。
