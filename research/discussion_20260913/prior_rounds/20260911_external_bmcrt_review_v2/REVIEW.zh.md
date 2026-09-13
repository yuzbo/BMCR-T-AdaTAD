# 第二份BMCRT新建议：逐项复审与主线取舍

日期：2026-09-11。本报告评价用户新提供的`BMCRT_review_15280e5.zip`和完整粘贴答复；与上一份文件名为`BMCR_T_review_15280e5.zip`的附件区分保存。固定审查源码仍为`15280e5dc29e3df18d085aaba21067e04107ce84`。

**不能完全照单接受。** 总体性能判断、原时间几何和粗信息利用方向基本成立；新增填充观察有价值，但取消RGB清零不能直接列为确定性修复。新报告也遗漏了上一轮已经实证确认的学习率分组错误。其“必须U384、必须160/40”是研究设计建议，不能覆盖用户刚明确的全200训练、跳过均匀基线要求。

## 1. 完整归档和本轮检查

- [原始完整回复](inputs/pasted-text.txt)：347行。
- [原始ZIP](inputs/BMCRT_review_15280e5.zip)：原样保存。
- 包内[完整Markdown报告](inputs/package/BMCRT_independent_review.md)573行、68,764字节；[原始HTML报告](inputs/package/BMCRT_independent_review.html)333,352字节；[checks.py](inputs/package/checks.py)及[原始算术结果](inputs/package/arithmetic_checks.json)均保留。
- 主代理完整阅读Markdown和回复及脚本，未改写原件；HTML作为原始呈现版本保留。脚本只在`checks_rerun/`副本运行，输出与附件JSON相同，六组AP和GMAC转录与真实记录一致。[验证记录](VERIFICATION.json)
- 本轮检查是源码/记录/数学/文献复核，不是重新加载大权重或完整重评AP。上一轮实际scout/优化器CPU核验继续有效，见[先前实际参数记录](../20260911_external_bmcr_review/actual_scout.json)。

## 2. 五项最重要取舍

### 2.1 学习率错误仍必须优先修正，新报告未提及不等于它不存在

上一轮通过实际`FormalScout`与真实`optimizer_for`、六个阶段运行metadata确认：`.conv_out.`错误匹配四个attention内部输出投影，共18,816个参数；真正两个动作分类头仅388个参数。warm/joint阶段被错配投影的base LR都是预定trunk LR的两倍。

新报告对runtime给了源码引用，却未指出此问题，属于重要覆盖缺口。它没有宣称发现所有错误，因此不能把遗漏解释为反证；但后续建议的优先级需要补上这一项。旧分数仍是旧配方真实结果，不因此作废。[固定代码](https://github.com/yuzbo/BMCR-T-AdaTAD/blob/15280e5dc29e3df18d085aaba21067e04107ce84/h65/full/runtime.py#L101)

### 2.2 填充差异成立，直接取消清零的“最小修复”不直接接受

现代码确实先gather最后有效帧作为无效索引填充，再在`encode`里将无效RGB乘为零；随后才做均值/方差归一化。因此它是raw黑RGB，归一化后是常数负值，不是normalized feature零。奇数有效长度时，最后一个有效帧与无效RGB可进入同一tubelet；输出mask不能撤销前面的混合。这些事实和toy例子正确。

但原项目`phase2_20260910/PLAN.md:29`明确规定“Zero invalid packed RGB slots”。历史H65固定提交的`ActionFormer._duca_gather_raw`也在gather后乘slot mask，`gather_selected_observations`默认`pad_value=0.0`并显式mask填充。这是既定设计合同，不是仅凭当前行为就能确认的无意bug。[历史gather](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/04c35a3b76897e6c1569eeede41ed3aecaf7f854/opentad/models/detectors/actionformer.py#L332)、[历史选取函数](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/04c35a3b76897e6c1569eeede41ed3aecaf7f854/opentad/models/duca/acquisition.py#L3618)

纯净OpenTAD的数据读取确实使用edge padding，这是另一个可比较的输入约定。可以把zero和edge作为固定权重短窗敏感性诊断或明确的新输入变体，但不能据此预言mAP改善。为了隔离已经授权的三个主线因素，初始三版本保持同一现有填充规则，避免把填充变化混入“仅纠正LR”的第一阶段。

### 2.3 粗特征梯度建议与上一报告不同，两种都不是唯一正确答案

第一份报告建议保存原始encoder hidden，让融合检测梯度进入CNN和encoder；本份建议使用encoder重放路径，初始不让该梯度进入CNN。两种前向数值可以相同，反向路径不同。保留policy隔离的较小改动方案是：暴露**未受routing adapt_scale控制的重放encoder特征**，为融合单独设置梯度尺度，CNN继续由动作辅助任务训练。

这个选择可以用于三阶段主线的初始普通残差，不需要再跑一次CNN或encoder。若以后比较raw/CNN-live路径，应单列优化路径变化，不能把效果都归给融合结构。零初始化投影时首步上游融合梯度为零是正常现象，验证非零梯度要在投影非零后进行。

### 2.4 不恢复U384和160/40为部署门槛

它们有明确的科学用途：U回答学习采样是否优于同预算均匀；未见开发集支撑调参泛化判断。但用户最新要求是全部200训练、211终点测试，并跳过均匀基线。本轮不把附件建议当成更高权限的指令。

因此三阶段保留BMCR学习采样和其训练课程；20轮均匀预热是模型课程，不是新增U384完整基线。固定全部方案和超参数，不用测试集选轮次或修改阈值。没有U对照时不宣称“学习采样胜过均匀”，但这不阻塞修正配方、原时间检测和粗细融合的阶段比较。

### 2.5 “算教师但不蒸馏”不列为必须的完整训练组

报告§9.2提出KD对照至少包含“算教师但不蒸馏”。若teacher严格eval/no_grad、输出不进入loss且随机性/状态被隔离，额外空跑教师不改变学生学习，只增加耗时。算法效果可以由no-KD与KD对照、匹配额外学生视图和完整成本记录说明；系统开销可短程单独测量，无需为了名义同算力再完整空跑一遍教师。

若teacher前向产生状态或随机流副作用，应明确隔离和控制，而不是把副作用作为保留无效计算的理由。本轮主线也未授权新增KD实验。

## 3. 主体建议逐项评价

原报告行号指573行Markdown；粘贴答复中的同义条目合并评价。以下45项覆盖全部主要技术建议，文献变化另见附录。

| ID | 原报告行/事项 | 评价与落实边界 |
|---|---|---|
| 01 |7–15 固定版本与证据范围|接受。源码审查、记录换算、toy例子与独立AP重评必须区分。|
| 02 |21–25 候选stride/K/heavy clips|接受。候选stride4；K384是RGB物理输入，native192是tubelet特征数，不能再把RGB预算改成192。|
| 03 |31–32 预算与代理梯度|接受。隐式校准的求导可以正确，而整个hard-set/RGB桥仍是代理；称“有偏”应明确针对离散任务梯度，不把校准导数也说成错的。|
| 04 |33 ASFormer训练|接受。动作任务训练stem与encoder/decoder，policy重放隔离stem；不是完整动作分割配方复现。|
| 05 |34 companion|接受。不同视频的均匀行不是同视频teacher视图。|
| 06 |35 EMA|接受。当前是效用教师，不是检测输出KD。|
| 07 |36、55–64 效用符号/尺度|接受。保留/插入符号、互反条件和标准化单位自洽，推理不缺一次再除尺度。双通道等权仍待验证。|
| 08 |38–40 时间与AP|接受。GT在rank、预测回原轴后视频聚合；不能将匹配GT子集当全测试。|
| 09 |44–51 预算Jacobian|接受公式。总量约束的导数不证明离散集合增益有效。|
| 10 |70–78 zero/edge差异|行为接受，取消清零不直接接受。见第2.2节，既定合同和源代码证据必须保留。|
| 11 |84–90 IoU反例|接受。真实IoU0.75、rank0.50且往返正确；证明度量不等价，不证明实际损失比例。|
| 12 |90 “完整解决方案”|修订措辞。原网格可处理一组监督/锚点问题，但不补回视觉信息，也不修复内部tubelet/TIA间隔。|
| 13 |96–100 S0/S1与精确交换|接受诊断，交换/集合级监督为备选。不能保证限制为一两次swap更强，当前三阶段不同时更换决策算法。|
| 14 |104–108 路由热点/向量化|接受。实际profile确有96次argmin、192次index_put；保持tie-break、无伙伴和排序语义后才比较时间，不按MAC给循环分摊毫秒。|
| 15 |112–116 detach/clamp|接受。现有用途正常；可微学生框不能直接用现接口；clamp区间外也不能无条件宣称可逆。外推策略是未来训练设计。|
| 16 |122 三套AdaTAD数字|接受并已查原论文。论文68.8/71.5、固定model zoo69.03/71.14、重评69.01/71.13是不同来源；当前对照继续用重评值。|
| 17 |128–149 分项性能裁决|接受。精度/严格定位未保持，矩阵工作下降，指定窗口更慢，allocated并未减半。|
| 18 |151 训练成本|接受。不同VJP/teacher开销意味着同更新数不等于同训练算量；无同配方官方训练成本，不能宣称训练更便宜。|
| 19 |155–161 S/B解释|接受候选原因，不接受因果排序已定。标题“首先是时间表征失配”应理解为诊断优先级，不是已证实首要原因；遗漏LR错误需补充。|
| 20 |163 S固定附加成本|接受结构性算术解释，时间仍待分解。不能用固定scout MAC推导固定毫秒，更不能解释精度下降。|
| 21 |171–181 粗细/近邻TAD机制|接受机制借鉴。TALLFormer/ETAD训练效率不等于推理RGB跳算；DyFADet/DiGIT/BRN也不能自动处理硬选帧后的真实间隔。|
| 22 |185–195 KD对象与边界头|接受。ActionFormer当前回归两个距离标量，不可直接把它们做KL称作分布式Localization Distillation；新增分布头是额外架构变更。|
| 23 |199–220 非线性/覆盖目标|接受为研究假设。已有非线性和集合条件不能改名为创新；相似但不同的重复动作不能被无条件去重。|
| 24 |222 REINFORCE|接受限定。对定义清楚的随机策略与真实奖励可作无偏策略梯度估计；其目标与当前确定性代理不同，有限样本方差和成本必须计入。|
| 25 |226–237 跨域原坐标保留|接受。恢复token身份/廉价状态不等于恢复没算过的深层信息；分类、span、3D与TAD的证据不能互换。|
| 26 |243 U起步且胜U才保留BMCR|科研用途接受，本轮不采用门槛。用户已跳过U；原时间和融合版本保留BMCR，不能据附件重新改变范围。|
| 27 |255–267 native输出与锚|接受。K仍384、24clips，去掉最后192→384插值才能取得native192。partial pair须记录支持；这与取消zero padding不是同一问题。|
| 28 |271–283 线性粗残差与gate|接受最小线性残差候选，gate暂不叠加。96→D零初始化与此前96→96→D+LN是不同方案，成本不同正常。|
| 29 |283 原网格GT与秒数|接受。使用原masks/GT和max_seq_len768，移除该分支的额外rank映射；仍用官方metadata换秒，避免重复乘稀疏比例。|
| 30 |296 粗梯度重放路径|有条件接受，见2.3。必须独立于routing adapt_scale；与上一报告raw-hidden/CNN-live方案明确区分。|
| 31 |298 先关闭交换损失和学习路由|不用于当前主线。用户要连续BMCR三阶段；首版保持同一BMCR效用设置，只改变配方、检测网格、粗残差，避免额外目标变化。|
| 32 |302–350 伪代码与接口|接受为设计，不是已运行代码。融合参数、optimizer、EMA、native mask、teacher raw_route和严格加载须一起实现。|
| 33 |335–337 计算/内存估算|算术接受。新gate102→32→1与单投影对应612.783919104/2046.603890688GMAC；1.125/2.25MiB仅单张B1 FP32对齐特征，不是峰值增量。|
| 34 |354–374 同模型分类KD|有条件接受为后续备选。相同窗口空间视图、真实时间/GT匹配、Bernoulli KL和可靠性detach；不使用另一batch行代替对应teacher。|
| 35 |376–415 定位KD/预算|接受为备选。学生端点要可微、GT不能被漏检teacher覆盖；0.1875是激活区间前向样本上界，整个joint上界0.156，不是18.75%训练时间。|
| 36 |421–436 clip采样与clip效用|备选方向接受，本轮不部署。新clip交换需要新标签/缩放；真实稀疏head需显式改全部层锚点与分配，不是加Δt即可。|
| 37 |421、423 全部窗口exact384|必须补合同。L503末clip仅7有效帧，若23完整clip＋末clip只375有效观察；短视频说明不能代替L≥384部分尾窗的预算定义。|
| 38 |465–471 160/40与独立warm|科研逻辑接受，本轮不采用160/40。全200训练下20+40仍6000更新；每种结构使用自己的新warm，不能接续错误旧warm或声称模型未见训练子集。|
| 39 |477–486 U/R/F矩阵|原则上能分解机制，但不是当前执行清单。当前采用修正BMCR→原时间BMCR→原时间BMCR+粗残差；不宣称完成了U对照。|
| 40 |488 “算教师不蒸馏”|不作为必须完整训练组。见2.5；保留实际teacher成本测量和额外学生视图的GT控制。|
| 41 |492–498 分层和不确定性|接受诊断原则。全200模型的训练来源检查应标in-sample；@0.7混合排序/召回/定位，不单独等于回归能力。多种子和bootstrap不互替。|
| 42 |502–504 全流水线profile|接受。按真实部署范围和样本组成测量，同卡交错、窗口分组、GPU/CPU边界明确，缓存成本不隐去。|
| 43 |508–518 停止条件|接受作为事先约定的证据原则，不自动中止用户已授权三阶段，也不恢复被跳过实验为前置门槛；测试结果不用来临时改配置。|
| 44 |524 LiquidTAD定位|接受v2限定。原文明确是并行液态启发松弛先验，不复现完整LNN动力学；没有据此证明本项目非均匀真实时间戳或RGB少算已解决。|
| 45 |526 最终优先级|整合后采用：先修已确认LR；维持统一padding；原时间检测→普通粗残差。U/160/40、KD/clip和门控不因本报告而重新加入当前主线。|

## 4. 新增或变化的文献核验

旧报告已核验的PointRend、Sparse DETR、CoLT5、SoftTeacher、SampleNet、BYOT、data2vec及KD负证据继续适用，见[前次文献复核](../20260911_external_bmcr_review/LITERATURE_RECHECK.zh.md)。这次重点补充：

| 来源 | 核验结论及primary链接 |
|---|---|
| AdaTAD论文数值 | [v2原文表2/4/11](https://arxiv.org/html/2311.17241v2)明确S/B为68.8/71.5；表5与表11对应768、160²。与model zoo及重评不同的辨析成立。 |
| SlowFast | [原论文](https://arxiv.org/abs/1812.03982)含AVA人物框动作检测；与THUMOS起止区间TAD不同。双速率融合有先例，不能当新方法独有。 |
| AdaFocusV2 | [原文](https://arxiv.org/abs/2112.14238)支持可微patch选择与视频识别效率；不等于离散真实帧选择无偏，也没有TAD边界结论。 |
| TALLFormer | [论文](https://arxiv.org/abs/2204.01680)、[作者记忆库说明](https://github.com/klauscc/TALLFormer#b-init-the-memory-bank)支持训练期部分重算/记忆补充。首次新视频推理不能假设缓存免费，当前global-TIA上下文也不能直接跨窗复用。 |
| ETAD | [固定v2](https://arxiv.org/abs/2205.07134v2)的顺序前向、选择性反向与训练proposal采样是训练效率机制，不是RGB推理只算部分的证据。 |
| DyFADet | [原文](https://arxiv.org/abs/2407.03197)动态kernel/感受野聚合与多尺度head为直接TAD先例；未证明原生接收不规则时间戳。 |
| DiGIT | [原文](https://arxiv.org/abs/2505.05711)支持多膨胀门控encoder和中心/邻接区域decoder；标准特征时间轴上的可变形采样不等于RGB前稀疏计算。 |
| BRN | [原文](https://arxiv.org/abs/2408.09354v1)处理金字塔边界消失，跨尺度插值/交换有意义；不是对缺失RGB视觉证据的恢复保证。 |
| TR-BERT | [原论文§3.1](https://aclanthology.org/2021.naacl-main.463.pdf)明确被跳过token的当前层表示被当作其最终表示；作者代码含QA任务。此结论有原文支持，不必因搜索不到restore一词而拒绝。 |
| RandLA-Net | [论文](https://arxiv.org/abs/1911.11236)、[作者仓库](https://github.com/QingyongHu/RandLA-Net)支持随机采样＋局部聚合的大规模点云分割。它反对复杂采样“必需”的通则，但不能据它判定本项目均匀一定更好。 |
| Localization Distillation | [论文](https://openaccess.thecvf.com/content/CVPR2022/html/Zheng_Localization_Distillation_for_Dense_Object_Detection_CVPR_2022_paper.html)、[作者代码](https://github.com/HikariTJU/LD)的bbox分布与valuable region不能直接套到两个距离标量上。 |
| BYOT代码版本 | [作者当前扩展版](https://github.com/ArchipLab-LinfengZhang/pytorch-self-distillation-final/blob/master/train.py)注明conference与final feature-distillation差异，并detach深层目标；报告对版本边界的说明正确。 |
| data2vec方差检查 | [作者data2vec2.py](https://github.com/facebookresearch/fairseq/blob/main/examples/data2vec/models/data2vec2.py)确有target/pred variance检查及异常退出；这不意味着TAD应机械照搬预训练方差阈值。 |
| LiquidTAD | [v2](https://arxiv.org/abs/2604.18274v2)明确并行、非递归、liquid-inspired relaxation，而非完整LNN；对真实不规则时间戳、任意连续时间与硬RGB预算的适用仍需实证。 |

没有逐行重审所有外部仓库，也未独立跑它们的实验。报告“41项来源”包括本项目源码/配置/记录，并非41篇论文。

## 5. 对当前已授权主线的影响

主线目标不被这份文档替换：使用200训练、211测试、S/B两骨干，跳过独立均匀基线，三个版本各完整20+40课程。首先修已确认LR分组；第二阶段只调整native特征到原时间检测；第三阶段增加零初始化普通粗残差。采样仍使用BMCR；KD、clip、gate和padding切换不同时混入。

本轮新复审的原件、计算证据和意见均独立保存。旧发布提交及旧实验结果保持可追溯。该文件记录复审结论与后续取舍，不将报告中的伪代码、开发划分或实验门槛自动转化为操作指令。
