**下一批模型推进任务书：先补性能上限与公平对照，同时完成Pro核心缺口**

状态：本次形成的具体实施计划；新增head/full-backbone/多层融合/校准等配置尚未实现或注册，不是Slurm提交回执。既有BMCR80与71配置矩阵继续执行。原Pro已授权的未完成工作仍需要落实，不能只因GPU排队就停止可独立完成的代码与数据分析。

主代理负责实现、取舍和最终验证；子代理按当前AGENTS约定只做默认角色的独立读取/核验。下面各组可以并行实现和准备，不以其他组mAP胜出作为启动条件。真实GPU技术预检、各自所需资产/检查点仍须满足。保持H65/BMCR候选帧主线，原DS3/T24/U24 clip选择不恢复；A-MoD主线仍交替层、首末dense、attention排序。

**A. 首要性能组：可训练学生检测头**

目标是回答冻结头是否限制恢复后的TAD适配，而不是预设解冻必涨分。

1. 把当前兼作student readout的OriginalTeacher拆清：external teacher的encoder/head始终冻结、eval；另建独立student readout，初始化自同一官方原轴head。readout包含projection/neck/rpn_head，保持native384→detector768合同。
2. student GT loss、最终预测、学生侧输出KD与真实动作收益使用student readout；teacher feature、teacher侧logits/offsets继续来自external teacher。不得通过修改teacher.requires_grad把教师也一起训练。
3. 将新head参数和真实需要保存的训练状态纳入optimizer、resume、EMA；效用成对查询时固定normalizer/随机状态。新模式使用新recipe与checkpoint，不对旧配置宽松加载后冒充同一实验。
4. 并行设置四格：A0 decoder-only；A1 decoder+head；A2 decoder+Adapter但head冻结；A3 decoder+Adapter+head，先固定选帧器。A3作为主要性能候选；另设A4 decoder+完整VideoMAE+head，检验完整骨干适配的上限及训练成本。A2沿用现有D00的对照思想，但初始化/课程必须与这一组匹配，不能直接拿不同起点的D00比较。
5. 这些组需要相同初始化、K、数据/增强、teacher、成功更新和选模范围。最清楚的起点是同一个已完成R03 EMA20，再让所有组进行相同长度的新适配；A0也必须获得相同额外更新，不能拿新组多训练后的结果与原20轮直接作因果比较。额外训练明确记账，不覆盖旧结果。
6. 分组LR明确记录，预训练head/Adapter/骨干使用独立组；具体LR是新实验配方，不沿用warm optimizer。先采用一致的新20轮适配课程比较结构，不自动把所有新分支都改成80轮。BMCR既有80课程继续原样完成测量。
7. 关键验证是：学生head确实获得非零梯度并更新；external teacher参数/buffer保持固定；EMA与resume可恢复学生状态；原frozen配置输出不变；两次真实任务更新后完整预测有限。扩大训练不要求等待A0/A1的性能结论。

“完整训练”中的selector端到端更新另外列组：当前FrozenAnchor.select和scout context存在no_grad/detach，不能只requires_grad=True就宣称选帧器参与训练。若增加selector适配，明确采用既有transport硬前向代理或真实有符号动作监督，并配匹配的冻结selector控制。

**B. 公平归因组：选择、联合课程与self-KD**

1. 复用相同强锚点和decoder初始化，补“训练选择learned/uniform × 推理选择learned/uniform”四格；只新增缺少的uniform适配组，不重训历史warm。若释放Adapter/head，两种选择都释放同一组参数。
2. 在K384/320/256按完整实际GFLOPs作比较，保留scout/decoder必需成本；多个训练seed与成对视频区间分开。已有同checkpoint推理控制不能冒充独立训练控制。
3. 新建与J01相同K/D/S混合课程的utility分支，再检验frame router增量。现有固定K384的J02保留原身份，不静默改其配置或据J02−J01归因utility。
4. self-KD控制保持相同full分支GT、学生更新与teacher查询，单独移除一致性项；不能让“有self-KD”同时多一个接受GT的full训练分支，再把全部收益归于蒸馏。

**C. 恢复与压缩组：强初始化、上下文与多层信息**

1. 从现成R03恢复器初始化稀疏适配的独立控制，与现有从零残差恢复器启动的D/S分支匹配额外更新，分清初始状态与路由算法收益。
2. 保持用户指定A-MoD排序与首末dense，比较selected-KV与full-KV，容量50%和12.5%分别匹配。12.5%-S EMA5当前51.2194%不能当20轮结论，也没有证据支持把更强剪枝设为默认。
3. 加入少量多层anchor读出，例如6/9/12层；空间池化与物理来源要在每层保持一致。每个decoder block可学习融合多层memory，初始选择末层或零增量，避免一开始破坏已学恢复器；记录额外投影/池化/存储成本。
4. P01现有全时域浅stem与R06官方decoder初始化继续独立推进，用真实结果检验“多一份廉价观察”与“预训练恢复先验”的价值，不混成一组。
5. 静态drop/PBD必须有强匹配控制。现有自有PBD式选择不写成官方复现；若要主张优于PBD，需要落实其跨深度对齐思想和对应训练预算，并在本数据/指标口径实测。

CrossMAE的inter-block attention利用多个encoder层的特征供decoder读取，是多层anchor扩展的依据；它证明的是其预训练任务中的结果，不保证我们的TAD增益。[CrossMAE原文](https://arxiv.org/html/2401.14391v2)

DyT的动态适配和完整分支蒸馏适合参考真实稀疏学生训练与self-KD对照。[DyT原文](https://arxiv.org/html/2403.11808v2)

PBD直接研究TAD逐步删block与跨深度对齐，应作为强深度压缩参照。[PBD原文](https://arxiv.org/abs/2503.16916)

Expediting ViT用聚类后的低分辨率计算再恢复高分辨率特征，为保留浅空间状态、减少后续重计算提供参照；该机制与完全不观察RGB不同。[Expediting ViT原文](https://arxiv.org/abs/2210.01035)

**D. 补全Pro的真实三维效用**

1. 在相同学生前缀状态上定义depth容量升级、space token/tile升级，实际执行受影响后缀；保留global TIA与后续状态反馈。保存真实任务loss差、实际算子差与route变化，不用teacher feature替换冒充动作执行。
2. frame/depth/space都分别记录repair_delta与action_delta。固定K frame swap可按收益比较；新增计算动作记录增量cost，不能对近零增量cost机械相除。
3. 用训练视频分组OOF产生校准评估，最后仍使用全部200训练视频；实现signed收益校准及可检验的不确定性。当前router EMA scale不算完成这一项。
4. 第一版保留A-MoD attention排序，研究是否可由任务收益选择“给哪个层/时间区域多少容量”；把分配容量与token内部排序分开。统一预算分配是新独立方法，不能把已有固定K/D/S称为已完成的统一分配器。
5. 以相同实际FLOPs比较动态分配、固定比例、静态层/分辨率、单维压缩，并统计短动作/高tIoU退化。不以时延否决路线，时延仍完整报告。

**E. 分析、数据与部署交付**

1. 补真实视频时轴案例、pair/gap覆盖、层间漂移、真实空间heavy叠加、feature误差—检测损失关联和OOF风险图。样例按固定规则含改进与失败；未测数据留空，不生成伪视频图。
2. 保持每次完整测试、有效观察/物理槽位、矩阵/卷积FLOPs和训练/teacher成本记录；补同卡交错与完整系统分段测量。报告图与投稿排版分开，不称当前所有图已达到最终CVPR排版。
3. G02先准备真正的数据集配置/类别及时间合同、训练/评测入口和teacher清单，再接完整ANet数据。现有THUMOS入口写死200/211/792，不能只改路径就称第二数据集已支持；CUHK分数不能充当TAD teacher。
4. 调度保持FPW最多8个活动/排队、最多6长训练及账户16限制。在后续释放槽位时轮换ready训练与评测，避免所有新训练都排在整批旧评测之后；不取消正在运行作业、不重复已完成训练、不等待跨路线mAP准入。
5. 已实现部分继续跑，缺代码的部分并行实现；每个新组明确code_ready、preflight、submitted、running、train_complete、full_eval_complete，使用实际回执填状态。新head等配置在完成实现和预检前不得写入“已提交”成绩表。

仍可参考的旧指令：BMCR80的RESEARCH_NEXT保留同合同蒸馏、强路径任务约束、真实S0/S1交换与同课程比较；239d098两份交接中的坐标合同、真实混合学生GT、global TIA完整状态与反事实语义继续适用。原T24/U24/D1的clip选择执行命令已由用户取消，不因读取旧交接而恢复。
