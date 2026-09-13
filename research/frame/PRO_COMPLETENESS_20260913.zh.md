**Pro完整性复核、冻结检测头的理由与实现缺口（2026-09-13）**

结论：没有完全实现Pro全部要求，也没有把全部实验提交到Slurm。已实现的第一代模型矩阵已部署给控制器，但“配置已注册”“代码已实现”“已向Slurm提交”“训练完成”“完整测试完成”是五种不同状态。此前用“全面部署”概括整个Pro包过宽，应按下面的实际范围纠正。

本次完整阅读原始粘贴、Pro REPORT、AGENTS_TASKS、COMMANDS、27项experiments.json和FIGURES，结合当前源码及远端回执复核。原包只提供CPU原型的执行界限已由用户后续明确GPU授权覆盖。用户另外明确改了四项：BMCR总80轮；深度主线改为间隔层A-MoD、首末dense、attention评分；多阶段独立并行，不等待别的路线mAP；总FLOPs与最佳mAP主导，延迟不作淘汰条件。这些属于接受的任务变更，不计为漏实现。

**冻结官方原轴TAD头为什么合理，以及为什么现在应补联合训练**

Pro原报告5.3明确写“官方teacher的参数/最终检测器参数冻结，但学生feature到GT loss必须有autograd”（原包REPORT.zh.md L99）；原始回复L238则说检测器参数可以冻结。它的合理用途是第一版恢复实验：保持检测器不变，让R01/R02/R03比较回答恢复特征是否更有用，减少换头、换坐标与恢复器一起变化的混淆。当前真实GT梯度会经过冻结头回到学生，并没有因为冻结头而关闭任务学习。

这不是最终性能方案必须永远冻结检测头的理由。当前所有FPW配置确实都冻结原轴检测头，D/S/J虽然训练Adapter，也未解冻head；当前没有decoder+head、decoder+Adapter+head或full-backbone训练模式。这是需要补的性能上限对照，而不是已有实验已经证明全训练无效。

BMCR80本身的scout、Adapter和检测头都参与训练；冻结限制针对新的FPW分支。头参数是否训练不改变其推理计算图，联合适配有机会在不增加结构性推理FLOPs的情况下提高性能，但现有结果不能证明冻结头就是全部精度缺口的原因。

正确实现应保留完全冻结的external teacher，另建可训练student readout，包含projection/neck/rpn_head及原轴输入转换。学生GT loss、预测和动作收益走student head；teacher特征与teacher输出KD仍走独立external teacher。不能直接解冻当前OriginalTeacher里的head，因为它同时承担教师和学生读出角色。新head的参数与影响训练的状态必须纳入optimizer、checkpoint/EMA，counterfactual查询不能改变用于比较的loss normalizer。

**实际部署范围**

14:56:56完整快照中，FPW有71配置、518阶段：11/66条训练完成、37次完整测试完成、1个评测运行、7个评测PENDING；55条训练均未提交。448个WAITING阶段中，104个材料已齐可提交（55训练、47评测、2测量），279个缺自己的checkpoint/资产文件，65个CPU分析等待对应完整评测。没有跨路线mAP胜负准入。当前8个FPW在队上限及实际GPU不足限制继续提交，但这不能解释尚未完成的算法/分析代码。

15:15:10增量快照：A-MoD12.5%-S EMA5已完整211/792完成，51.2194%/871.9542GFLOPs/126.05ms；FPW变为63完成、1运行、7排队、447等待，完整评测38次。Cross-B10 1288253运行；在队的其余FPW是1288254、1288258、1288287、1288290、1288292、1288299、1288364。两种A-MoD容量的20轮训练均已完成，但12.5%当前只有5轮全测，不能作20轮终局结论。

BMCR80：两条80轮训练完成，17/24次候选全测完成，B65 1288294已提交；80终点与后续profile还未完成。控制器FPW3944214、BMCR2187096健康。当前audit_s/audit_b分别1288043/1288044，均COMPLETED；1288036/1288037是保留的早期失败历史，不是当前失败，不能据此认定技术依赖被阻塞。

指定OpenTAD V3任务15:04只读回报：旧路线尚未达到原联合目标，当前自有活跃GPU作业为空，DST正式训练0且HOLD，没有可再取消的相关作业。ANet原始分片43/43已下载，164.49GiB；成品仍7491（train5151/10024、val2340/4728），缺7261，且没有已核实匹配ANet的TAD teacher。这不是完整ANet READY。

**Pro 27项逐项映射**

|Pro ID|当前代码与实验覆盖|尚缺或解释边界|
|---|---|---|
|B01|修正warm复用、尺度审计、joint60/总80、全测与分布工具|用户把原joint40改为60；终点评测未完；如要归因BMCR优于H65，另需匹配80课程对照|
|C01|native/旧路径核验、condition精确向量化、B全792同选点与同mAP|已有实际证据；系统同卡收益仍另测|
|R01|插值+original头及旧rank头原轴控制，S/B完整评测|已完成主要零训练对照|
|R02|TCN残差恢复器，S训练20/5轮全测，B配置注册|B及后续完整测试未完成|
|R03|来源/物理时间Cross，S/B训练20；S全20、B早期全测|缺最终B结果与多种子；当前仅最后层anchor memory|
|R04|去直接检测GT loss的feature-only，S训练20/5轮全测|仍有GT边界加权；不能称无GT；B与后期全测未完成|
|R05|输出Bernoulli/距离KD与时间差分分别实现、注册|尚未训练；更充分的等查询成本归因未完成|
|R06|官方S/B完整decoder权重严格加载，同结构random/pretrained，真实预检|正式训练未开始；不能宣称预训练收益|
|R07|decoder来源/scout条件分别消融、注册|尚未训练；共同物理插值与选帧scout保留|
|U01|frame同K真实重编码swap与repair proxy并列记录；4组诊断完成|已有的是frame动作，不含真实depth/spatial动作|
|U02|有符号frame swap MLP与真实标签、简单尺度归一化；离线评分诊断|没有OOF校准、预测不确定性、repair→actual偏差校准；正式训练待提交|
|U03|local与global最近伙伴交换配置|没有完整的覆盖约束全局候选机制；无正式结果|
|U04|同R03 checkpoint的uniform/random推理控制已完成；router-off注册|原Pro本来也不含独立uniform完整训练；为强化必要性论证仍建议补匹配适配|
|D01|保首末层static8/9、自有PBD式逐次删层实现|不是官方PBD跨深度对齐复现；未适配static/部分浅出口控制未齐，部分原拓扑已按用户改为A-MoD|
|D02|按用户改为交替A-MoD50/12.5，attention路由、QKV/FFN真压紧、完整TIA|已完成若干训练/早期全测；不是原包8/12晚层门的逐字实现|
|D03|dense-mask训练和compact训练/推理控制，FP32/真实GPU检查|该支线正式训练未启动；BF16误差历史保留|
|S01|128分辨率真实输入路径、训练配置|等待提交|
|S02|token/tile heavy-light FFN；token版S/B训练完成|tile和后期评测待完成；无真实空间升级效用标签|
|S03|selected-Q/full-KV的真实投影与attention路径|等待正式训练/测试|
|J01|混合K384/768和D/S、共享full学生、同checkpoint八格；S/B训练完成|EMA20八格未全测；缺同训练成本的self-KD归因对照|
|J02|固定K384、D.5/S.48，加frame ActionRouter的联合配置|尚非统一gain/cost驱动三维预算；课程也不同于J01，不能称单因素utility对照|
|K01|K320/256恢复与低K混合联合训练配置|尚未提交/测量，未形成新低K前沿|
|M01|每次完整eval的实际FLOPs、20次模型计时、full/partial/short与E2E；同卡交错CLI注册|同卡交错尚未运行；完整decode/H2D/NMS分段与冷缓存测量未齐|
|A01|五阈值、DETAD式分类/漏检、逐视频bootstrap；R03-S分析完成|不是全部DETAD oracle；feature/gap/边界损失关联及全部图表未齐|
|G01|R03 S/B两个额外种子已注册|尚未提交；不能覆盖其他将来胜出路线或证明完整管线seed稳定性|
|G02|共享ANet资源与状态核对|无完整G02数据配置/训练评测stage，teacher未核实，数据未完整准备；不是仅等待一个普通GPU作业|
|P01|额外全时域80分辨率原配对stem，S/B预检|正式训练未开始；不等于已经让重骨干保持原tubelet相位|

**跨实验的核心代码缺口**

1. 可训练student head及完整backbone/selector适配模式尚未实现；当前更多是恢复器与PEFT研究。
2. 真实depth/spatial动作从相同前缀状态重执行后缀并测任务收益尚未实现。
3. OOF有符号效用校准、不确定性估计、风险—计算量策略尚未实现；当前EMA scale只是尺度归一化。
4. 统一三维gain/实际增量cost分配器尚未实现；J01是混合预算训练，J02是固定门率加frame swap。
5. J01/J02课程匹配、self-KD等训练成本消融、匹配uniform适配对照未齐。
6. 多层anchor融合/多深度目标仍是可参照的扩展，当前decoder仅用末层memory。

**图表与CLI覆盖**

生产audit/train/eval/profile/plan/intervene/analyze均已有实际argparse入口和dry-run；dispatcher默认emit-only、不提交。CLI功能已经实现，但原包的示例配置名称/参数不是当前生产命令的逐字副本，应使用当前工具help与实际JSON配置。

F01已有实测前沿；F02已有与代码对应的结构图；F04已有frame干预数值/相关性；F06已有真实算子计数。它们尚不等于完成全部论文panel与排版。F03真实视频时轴案例、F05完整feature—任务误差关联、S03深层状态漂移、S04真实空间heavy叠加、S09不确定性—风险图尚未形成完整实现/交付。S05有taxonomy/漏检，不是全部oracle。S10多种子/第二数据集无结果。必须保留未测状态，不能用通用绘图函数或注册ID代替实际数据。

当前的新结果图与报告图也不是已经完成CVPR单栏/双栏定稿。本次已将新增结果图中GT消融面板改为点图显示小差值，并保存可复现脚本plot_new_results.py，避免截断柱轴夸大差异；图形表达调整不改变已测数值，已视觉核对。

本次审计的下一步执行说明见[NEXT_MODEL_INSTRUCTIONS_20260913.zh.md](NEXT_MODEL_INSTRUCTIONS_20260913.zh.md)。这份任务书列出待实现/待注册工作，不是这些新实验已经提交的回执。

证据：[14:57完整远端状态](pro_audit_live_20260913.json)、[15:15增量](pro_audit_delta_20260913.json)、[结构源码复核](MODEL_ARCHITECTURE_20260913.zh.md)、[原Pro报告](../../../reviews/20260913_fpw_tad/inputs/package/tad3d_agents/REPORT.zh.md)、[原任务书](../../../reviews/20260913_fpw_tad/inputs/package/tad3d_agents/AGENTS_TASKS.zh.md)、[原27项计划](../../../reviews/20260913_fpw_tad/inputs/package/tad3d_agents/plans/experiments.json)。
