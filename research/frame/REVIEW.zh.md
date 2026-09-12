**当前判断：保留强锚点，继续原时间轴恢复；按A-MoD落实深度主线，独立并行验证三维收益。**

本包的实验建议已转为本项目的生产接口和调度计划。用户后续授权覆盖了原包的“仅CPU原型”限制，并明确取消性能串行准入与延迟淘汰条件；BMCR总80课程保留，所有原16候选clip选择继续取消。这里区分实现、实际提交、运行和完成，不把计划表当成测得成绩。

**07:10新增学习式结果：** R03-S第5轮EMA完整211/792为64.117164% mAP、1226.364694GFLOPs、平均107.6494ms；比R01零训练+0.2926pp、比固定H65-S60+0.7078pp。A-MoD50-S第5轮为61.136502%、1027.004182GFLOPs、127.8399ms，相对同轮R03计算量−16.26%、mAP−2.9807pp。两条20轮训练已经完成，10/15/20轮成绩未据此推断。恢复器出现初步正向信号，A-MoD的最终性能—计算权衡仍待后续里程碑；延迟变慢不作为淘汰理由。R06-S/B、P01-S/B和J01-B五项新模块真实2更新/EMA预检全部通过。

已完成的R01全211视频/792窗口结果：

| Backbone | 固定强锚点 mAP | 原时间插值+官方头 mAP | 变化 pp | 完整模型 GFLOPs | 平均模型 ms |
|---|---:|---:|---:|---:|---:|
| S | 修正H65-S60：63.4094 | 63.8245 | +0.4152 | 1225.471 | 101.557 |
| B | 旧BMCR-B60：67.3404 | 67.2721 | -0.0684 | 4093.089 | 171.160 |

这两组没有训练恢复器。S的正向观察支持继续研究原时间轴接口；B基本保留强锚点性能。R01同时改变了时间接口和检测头，因此不能把变化单独归于物理坐标，也不能声称MAE已经学会补帧。新注册的同锚点head原时间轴控制用于拆开这一混杂。旧锚点的FLOPs分别1210.310和4077.476G，R01并非更低计算量下的严格支配。官方全观察基准S69.0126%、B71.1280%仍更高。

生产实现及逐项对应：

| 原包编号 | 当前实现/实验 | 要回答的问题与解释边界 |
|---|---|---|
| B01 | 既有BMCR80：修正warm20复用+joint60 | 与旧60课程LR轨迹不同；中途低分不能直接判无效，也不重启warm |
| C01 | native提取恒等；精确向量化condition控制 | 已验证native与旧插值路径；向量化全测试另逐窗口检查同选点，profile不含检查开销 |
| R01 | physical-time插值、官方头及锚点头控制 | 确定零训练底座，拆开head/时间接口变化 |
| R02/R03 | 参数规模接近的TCN与cross-query decoder | 复杂恢复器是否超过便宜恢复器；cross版本无query自注意力 |
| R04 | feature-only对照 | 特征相似是否足以支撑检测，GT是否必要 |
| R05 | Bernoulli输出/距离KD、差分分别训练 | 检测校准与边界高频信息是否带来额外收益 |
| R06 | 官方4层decoder块/投影/norm/mask初始化 vs同结构随机 | 不把RGB decoder直接叫TAD latent decoder；严格加载后才进入新模块预检 |
| R07 | 分别去decoder显式来源/时间元信息、去scout，保留共同物理插值底座 | 测显式描述符的额外价值，不声称移除了一切时间信息 |
| U01 | 冻结模型的repair oracle与真实同K RGB交换 | 保存两种真实loss变化与实际action FLOPs；proxy相关性不能预设 |
| U02 | 有符号actual-utility学习；actionness/absgrad/error诊断 | 后两种oracle使用训练GT/外部teacher，仅作离线比较，禁止作为部署输入 |
| U03 | 同cell与跨cell伙伴，交换数/查询数固定 | 这里的16cell是帧交换局部先验，不是原clip选择路线 |
| U04 | 同checkpoint uniform/random/anchor与router-off | 与独立训练方法比较分开；训练中与审计中保留S0→S1实际变化 |
| D00/D01 | 同配方dense Adapter、保首末层static8/9、PBD式逐次删层 | 区分额外适配和删层收益；PBD为训练数据上按任务/feature损害选择的TAD变体 |
| D02 | A-MoD50/12.5、uniform score与full-KV控制 | 真实切QKV和FFN；不乘路由分数；末层dense，全globalTIA保留并计成本 |
| D03 | compact训练与dense-mask训练 | FP32数学等价已经核验，BF16路径差异保留为实测事实 |
| S01/S02/S03 | 128分辨率；token/tile重轻FFN；selected-Q/full-KV | 分别测试简单空间缩减、结构化执行和attention成本；延迟只报告 |
| J01 | 混合K/深度/空间训练，同checkpoint 2×2×2推理 | 不用8份独立训练混淆交互；预算必须被训练覆盖 |
| J02 | 联合恢复/计算配置加真实效用 | 最终按实际总FLOPs匹配单维/静态前沿，不能把更多计算叫协同 |
| K01 | K384/320/256恢复适配，另有低K联合三维适配 | 与同checkpoint预算扫描分开标注，关注短事件不可恢复区间 |
| M01 | 每次全测的完整profile；另同GPU交错四模型 | FLOPs包括scout/路由额外QK/encoder/TIA/decoder/head，保留20次计时，不清共享缓存 |
| A01 | 五阈值、DETAD分类/漏检、视频cluster bootstrap | 分类/定位/重复/混淆/背景与漏检分开；不相加解释重叠oracle |
| G01 | R03恢复器S/B预先增加两个种子 | 不能把R03的稳定性外推到尚未胜出的其他路线；实际胜出路线需对应复测 |
| G02 | 共享ANet现成资源与准备清单 | 完整数据/匹配TAD教师尚不足；该资源缺口不阻塞THUMOS实验 |
| P01 | 独立的全时间轴原配对stem80浅观察支线 | 显式支付额外全帧浅观察和投影成本；不冒充只观察K帧或相位无损 |

A-MoD遵循前层attention的incoming列均值路由，12层中第2/4/6/8/10层为MoD，首末dense，不增加可训练深度router，也不把分数乘到输出。本项目额外保留完整globalTIA，是面向TAD的适配，不能宣称与图像分类版逐操作相同。[原论文](https://arxiv.org/html/2412.20875v1)。

官方VideoMAE decoder原本预测RGB patch。这里仅复用严格匹配的Transformer块等权重，换成原时间轴latent残差目标，先冻结TAD头让任务梯度回到学生，再用外部dense native特征辅助。S/B完整预训练文件和56项映射均已保存。[官方实现](https://github.com/MCG-NJU/VideoMAE/blob/5cefe18cecab25e3cce8de88fad0b42e6ce858a7/modeling_pretrain.py)、[官方权重表](https://github.com/MCG-NJU/VideoMAE/blob/main/MODEL_ZOO.md)。

训练使用全部200视频，新增模块20epoch/2000真实更新，batch1累积2，EMA.99；BMCR80仍原.999。记录teacher-native获取次数、训练墙钟时间、loss/grad/LR分布；这些forward token计数不冒充包含反向、shared-full、干预的训练总FLOPs。激活重算只改变训练执行，单列说明。

首要科学问题是：在相同实际计算预算下，完整原时间状态是否比缺失/压缩时间状态更利于边界定位；哪些被预测的位置值得花真实计算修正；以及attention重要性是否真的与TAD边界收益一致。R01提供了继续研究的依据，学习式恢复、A-MoD、空间和联合的性能结论仍等待各自完整测试。所有负面结果保留在图表与表格中，不按延迟删路线。

绝对计算量还必须跨backbone比较：当前官方S的2347.894GFLOPs/69.0126%同时优于R01-B的4093.089GFLOPs/67.2721%。不能只画B组内部前沿掩盖这一事实。新增K320/K256联合训练用于直接检验低预算恢复与深度/空间压缩能否共同进入全模型前沿；它们不等待高K路线的成绩才启动。

统计实现复用OpenTAD的数据导入/去重规则并先复现保存的官方AP，随后按211个视频共同抽样、固定原始score tie顺序。置信区间条件于已选checkpoint，不能替代多种子或校正测试集选模。错误分类依据[DETAD官方实现](https://github.com/HumamAlwassel/DETAD/blob/master/src/action_detector_diagnosis.py)，不冒充其全部敏感性/修复oracle实验。

证据纠错：早期子代理`live_results_0455.json`中的BMCR-B40/B45误读了其他历史目录（job1286931/1286970），不能用于本轮曲线。已沿本轮精确路径重新读取；本轮B40实际job1288045、mAP65.5803%，B45当时未完成。权威轻量快照为归档的`live_verified_0505.json`。原始输入包和原回复始终保持不变。
