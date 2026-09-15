# TAD三轴条件计算的固定证据复审与论文研究方案

## 总体判断

**推荐论文中心：稀疏观察和稀疏更新之后，如何保持时序定位所需的状态，并把完整模型计算分配给真正能降低任务损伤的动作。** 不是预设三个轴均有大量无损冗余，也不是把MAE、attention gate和蒸馏并列成三个贡献。

当前证据最强的是“单帧稀疏＋原时间轴恢复具有可运行的准确率—计算折中”；最强的负证据是深度和空间压缩在旧J01同checkpoint上存在明显联合损伤。**新seed42 PaperModel尚无完整结果，不能判其成功或失败。** 必须把旧Cross、旧FPW和新PaperModel分开。[C01–C04,C25–C30]

建议并行开展三个有科学区分力的模型族：**P：PBD式静态渐进压缩控制；D：同选帧支持监督＋上下文保留的动态更新；R：无R03依赖的恢复器初始化与采样兼容。** 各族模型独立启动，只允许代码正确性、资源、数据和同课程checkpoint依赖；不要求另一族先取得更高mAP。所有完整配置均只运行seed42一次。

### 证据边界与阅读范围

主审查提交固定为 `5485c3cb8dfbd823acbb177a0b51292dadc44360`。源码判断以本包 `SOURCE_MAP.md` 的文件和符号为准。已读统一入口、有效决定、实现导航、52课程表，以及PaperModel/encoder/engine/decoder/readout/objectives/routing/interventions/calibration/training/geometry和相关配置、计划生成器、训练入口；读取了旧frame核心和多轮交接主体。

**未能完整读取巨大 `live_snapshot.json`：文件读取返回空正文，blob接口返回过大的截断base64。** 状态判断由同提交的较小EXPERIMENTS、DEPLOYMENT、结果JSON和故事复算文件交叉支撑，不声称完整解析机器快照。`full_test_records.json`为分段读取；部分长交接尾部截断。未逐一打开全部历史ZIP/DOCX/XLSX及每份原始回复；已有图的源数据/图注已读，但本次未逐张视觉打开仓库图像二进制。详见 `ACCESS_MANIFEST.json`。

本次未加载私有大权重、原视频或访问远端结果绝对路径，未运行训练、部署、取消或全测试命令。提供的数值复算和绘图仅重算公开记录，不能代替模型复测。官方VideoMAE权重结构通过代码与项目导入器核查，**没有独立打开权重张量**。

## 1. 当前事实、历史冲突与建议映射

### 1.1 五种状态不能合并

“已设计、已实现、已注册、已提交、已完成完整测试”是五种不同证据。2026-09-13 22:50课程表为52个seed42配置；六个正式课程PENDING，另有两个ANet预检PENDING，余项WAITING。部署文件较早段落的60配置/233阶段和图表报告的三种子说法属于旧状态。最新52配置/217阶段、单seed42覆盖它们。旧55条未完成FPW被退役不是55次算法或技术失败。[C02,C04,C21,C29,C30]

BMCR80两条训练已经完成，但“已训练到80”不等于每个候选或终点都已评测。已测峰值S63.8372%在65轮、B68.1361%在70轮；不把它们写成80轮终点。[C03,C27]

### 1.2 已完成结果应如何使用

|对象|最佳已测平均mAP|完整768候选窗口GFLOPs|论文身份与边界|
|---|---:|---:|---|
|官方AdaTAD-S|69.0126|2347.8940|公开方法在本项目中的参考复测|
|旧Cross-S EMA20|64.5743|1226.3647|作者内部前身；不是新seed42 PaperModel|
|官方AdaTAD-B|71.1280|8082.1547|同上|
|旧Cross-B EMA10|68.5792|4094.1248|内部前身；EMA20终点68.3819|
|修正BMCR80-S已测峰值|63.8372|此表不跨checkpoint配成本|内部控制，65轮|
|修正BMCR80-B已测峰值|68.1361|同上|内部控制，70轮|
|当前PaperModel seed42|未测|未测|不能填预测收益|

旧Cross-S/B分别使用对应官方约52.23%/50.66%计算，仍低4.4383/2.5488pp。历史H65-S65.3857高于Cross-S但配对成本未核实。更重要的是，**官方S同时比旧Cross-B准确、且绝对FLOPs更低**；只展示各自骨干归一化图会掩盖这一事实。[C25,C27]

恢复器的必要性也尚未被强证据证明：旧TCN-S峰值64.532与Cross-S64.574只差约0.042pp；Cross-S相对插值约+0.7497pp，但仓库给出的200次成对视频bootstrap区间约[-0.1252,1.5218]pp包含0。这是仓库诊断记录，不是本次独立复算或跨训练种子证据。[C28,C29]

### 1.3 历轮要求到当前代码的映射

|原始建议/要求|当前实现|实验地位|支持的主张|
|---|---|---|---|
|DS3完整clip、H0/H8、D1等|`h65/ds3`历史保留|取消|仅负结果/演进材料，不重启|
|修正warm＋BMCR joint40|后续作者改为总80；BMCR80训练完成|部分候选全测|配方纠错与内部控制，不是独立外部方法|
|native提取与原坐标回写|`h65/frame/geometry.py`、PaperDecoder|旧R01/R03已有全测|原轴读出可运行；并非全面保精度|
|冻结head做恢复诊断|旧FrameModel符合该目标|旧FPW结果|隔离读出变量；不是最终必须冻结|
|可训练独立student head|`TaskReadout`已实现|新课程排队|不再把旧冻结缺口归给新模型|
|真实frame/depth/spatial干预|`collect_action`重执行完整当前student|代码已实现，最终新结果未有|区分actual与repair是正确基础|
|gain/cost统一预算|BudgetRouter选择15项菜单|已实现、尚待完整验证|有限菜单决策，不是任意token最优规划|
|OOF与不确定性校准|`calibrate`视频分组router拟合|已实现|仅router拟合OOF，非整模型独立泛化保证|
|多深度恢复|6/9/12等层融合memory|已实现|多层输入，不等于多层监督|
|shared-full self-KD＋GT|`objectives`已实现|新课程排队|已有真正共享分支自蒸馏，不再是纯D1|
|selected-Q/full-KV|默认非此模式；已有full_kv对照|排队|须测上下文保留与增加成本的权衡|
|官方MAE decoder|导入器＋同架构random配置已实现|官方初始化课程PENDING|初始化对照，不能解释旧Cross成绩|
|完整uniform训练|新S/B80课程已注册|WAITING|纠正旧阶段只有推理uniform的不足|
|多种子、时延准入门槛|被最新决定替代|不执行|单seed42；计算与最佳mAP主导|

这是一份按已读主体做的追溯映射，不冒称审完全部历史归档。更完整的逐项CSV随包提供。[C02,C31–C35]

## 2. 论文中心与可证伪主张

TAD不同于视频分类之处，不是“视频更长”这么简单。它必须保留多个事件、起止边界、重复同类事件之间的间隔、置信度排序及高tIoU下的几何精度；足以支持视频标签的少数片段，未必支持每个事件的定位。稀疏状态还会进入时间金字塔、目标指派和NMS，局部误差可以导致离散输出翻转。[L01,L07]

建议统一定义：在状态s、预算B和任务分布D下，动作a的可替代性由**不执行/近似执行该动作后的条件任务损伤**刻画，而不是像素相似度。

\[
\Delta(a\mid s)=\mathbb E[\ell_{TAD}(s,Y)-\ell_{TAD}(T_a(s),Y)\mid s],\qquad C(T_a)-C(s).
\]

候选主张H1：坐标/来源感知的回写比同成本简单插值/TCN更能恢复定位。反证：匹配初始化、监督和更新后优势消失；或者只改善feature loss。

H2：深度/空间损伤与状态及上下文缺失有关；同选帧支持的中间监督和轻更新能减少该损伤。反证：等成本静态drop始终更优；状态对齐改善但TAD不改善。

H3：真实干预训练的有限预算路由优于最强固定菜单及attention/误差代理。反证：OOF排序差、oracle regret高、部署预算分布退化为单一计划且不优于该固定计划。

三项同时成立才适合“任务状态驱动的三轴条件计算”主线。仅H1成立时，收缩为“单帧稀疏的原轴表征恢复”；H1+静态压缩胜出时，收缩为“可恢复稀疏表示与渐进压缩”，不强行宣称动态路由必要。

## 3. 三轴冗余的测量：从相似度到干预

需区分五个概念：表征相似、任务条件冗余、由现有信息可预测、由便宜算子可替代、真实观察不可恢复。相同feature可能编码不同时间支持；低注意力位置可能为另一事件提供上下文；低梯度可能处于饱和区而有限删除损伤很大。任何一项相似度图都不能完成可省略性的证明。

旧J01八格为同epoch5 EMA上的部署干预，数据如下。T=1指K384，D=1指路由层容量.5，S=1指空间重FFN.48；0为768/1/1。[C26]

|TDS|S平均mAP|B平均mAP|
|---|---:|---:|
|000|67.2196|70.2037|
|001|66.8259|68.7396|
|010|65.1197|67.0604|
|011|64.0353|65.1751|
|100|64.0653|68.1111|
|101|63.3173|65.6399|
|110|60.6001|63.1962|
|111|59.4416|60.5032|

\(I_{TD\mid S=0}=m_{110}-m_{100}-m_{010}+m_{000}\) 为-1.3653/-1.7716pp；三阶差分为+0.2802/+0.1994pp。不能说所有交互都负，不能把000视作官方dense，也不能把该图当成八个独立适配模型的性能上限。

补充两种干预：**固定外生route的单算子干预**测直接机制；**允许路由重新响应的policy干预**测整个系统反应。二者估计对象不同，均应保留。关闭某轴会改变其余轴的得分和排序时，不能称完全单因素局部干预。

按GT离线分组：真实起止邻域、动作内部、长背景、短动作、同类重复事件；困难样本加遮挡、多主体和小物体人工审查。分组用于评估，不进入推理路由。短动作长度用秒和候选数同时报告；所有阈值预先固定，背景与动作重叠、多标签按显式规则计数。记录完整平均/逐tIoU mAP、class-matched召回、起止偏差、最大观察空洞和层内最大未更新空洞。

## 4. 深度：最需要先解开的故障机制

### 4.1 当前A-MoD实际做什么

在12层模型的交替层使用前层attention得分，首末层dense；D=.5是这些层的token准入率，不是删去一半网络深度。默认准入token同时构成Q/K/V，未准入token保留旧残差状态，再进入全grid TIA。下一dense层可重新进入，所以它也不是不可逆early exit。额外attention排序所需QK统计同样有算量。[C07,C08]

当前空间light只覆盖深度已准入但FFN未重算的位置。**深度未准入位置没有同样的可学习light残差**。用no_light结果去解释所有深度陈旧问题会误判。

A-MoD原论文主要提供分类迁移证据，其DETR附录中A-MoD.5为38.6 mAP，标准MoD.5为39.6，dense为39.9；所以attention排名并非密集定位保性能定律。此数字只作跨任务反例，不与THUMOS直接比较。[L02]

### 4.2 PBD到底值得借鉴什么

PBD从已训练TAD模型出发，评估删去候选block后的训练集指标，逐个删除并恢复，保留与原层编号的中间feature对齐，再结合分类/定位监督。官方实现还涉及LoRA/norm/head训练、LoRA合并、阶段优化器重建以及可选dynamic teacher；不能概括为“删三层后延长训练”。[L01,O01–O04]

其静态全block删除与当前token级交替准入本质不同：静态网络无token间深度参差；动态网络中token处于不同状态，且Q/K/V集合随层改变。PBD不能证明我们的token路由会保精度，但其**渐进改变计算图＋对齐保留层状态**有迁移价值。

必须同时比较：未经适配静态drop（仅技术负对照）、适配静态drop、PBD-faithful复现、在本项目架构中的PBD-style同预算改编、A-MoD、full-KV动态更新。PBD官方分类KD的softmax语义不能原封不动移入独立sigmoid point head而宣称等价。

### 4.3 最有价值的新监督：同选帧支持教师

设S_K为本次实际选帧。建立
\[
X_\ell^{ref}=E_\ell(X_{S_K};D=1,S=1),\quad
X_\ell^{sparse}=E_\ell(X_{S_K};D<1,S<1).
\]

两者拥有同一输入、tubelet配对、空间位置和global-TIA(K/2)，因而可以做中间层/残差对齐。原768输入官方teacher只用于原轴最终目标。**不要直接把selected192中间state逐index回归到original384。** 这个分解把时间观察失配和D/S计算失配分开了。

教师可为冻结的强初始锚点，或明确stop-gradient的shared-support full分支；后者须保留GT/外部锚定以防共同漂移。其额外前向与激活必须记账。缓存只有在同state、同增强、同位置、无随机差异时才能复用；global TIA下不能独立缓存每个clip并称真实反事实。

### 4.4 判别实验

首先用现有`full_kv`和默认compact-KV比较“上下文被删除”的影响；冻结同route，按同token容量和等完整MAC分别报告。其次新增depth-excluded light更新，比较hold、轻残差、有/无support-matched层监督。若full-KV改善而light无效，优先保context；若full-KV仍掉而light/对齐改善，支持状态漂移解释；两者均无效则检验路由排序与可恢复上限。

在层6/9/12、TIA前后分别记录同源状态误差、feature范数、相关性、任务梯度加权误差、跳过后重入时误差。oracle注入teacher同支持state只能作为损伤上界诊断，不是部署收益。

### 4.5 课程与并行

提出预先固定的渐进D/S日程，或逐次PBD评估—删层—恢复，均属于**一个seed42完整课程**。同课程阶段自然依赖前一checkpoint，模型族之间同时启动。总预算同时报告optimizer updates、训练样本、teacher/干预查询、逐阶段耗时和所有删层候选评估成本。PBD论文中的性能while条件不应直接成为本项目多路线启动门槛。

## 5. 空间：保留定位信息而不继续盲目删token

### 5.1 当前空间门的含义与失效风险

当前排序来自attention，不是从真实空间计算收益学习的token policy；BudgetRouter能选择整体空间比例，但不能证明每一个空间token的准入是任务最优。空间FFN门位于depth-admitted集合内部；当某个时间slice没有depth token获准时，空间配额本身不能救回它。[C08,C12]

高attention可能偏向大主体、背景纹理或已识别动作，而漏掉边界瞬间的手部、工具、新进入主体。当前完整grid仍在，胜过直接删除所有未选位置；但保留一个旧状态不意味着它仍足够准确。最终时间decoder看到的是空间池化结果，不能逆转已经发生的错误attention传播和TIA邻接。因此空间损伤应在**每层重新进入全grid之前**处理，不应全部寄托在最后native384恢复。

### 5.2 相关机制的适用条件

EVAD在AVA等时空动作检测保留关键帧完整token，并对非关键帧稀疏，随后用context refinement支持目标检测。它有关键帧/actor-centric先验，本项目无每个动作的推理GT关键帧，不能照搬。[L07]

DToP允许easy tokens退出但保留它们的密集预测，并保留类别代表性上下文。它提示“输出可退出，context未必可删除”，并不证明当前attention top-k足够。[L08]

Expediting ViT通过局部聚类、低token昂贵处理、重建回原grid保留下游接口；成立前提包括已经拥有完整浅层信息和显式assignment。Point-M2AE等多尺度方法也强调坐标与skip，而不是仅用零mask猜未知几何。[L09,L13]

### 5.3 推荐两条空间机制控制

首选S1：每个有效时间slice至少保留一部分空间覆盖，再在余额上按attention/任务预测分配；所有未重算token保留light residual；为条件位置做same-support的FFN输出或block输出监督。覆盖来自几何和scout，不用推理GT。空间最低配额只能保证网格覆盖，不能保证“短动作一定看见”。

备选S2：**merge—heavy—unmerge residual**。在同一时间slice内局部合并相似token，保存assignment A与原浅状态X，执行缩短序列的昂贵更新，再做
\[
X'=X+\operatorname{Lift}_{A}(\Delta X_{merged}).
\]
在每次TIA前恢复原坐标；不能把合并后的token序列直接reshape为原空间格。时间上不同主体的运动轨迹不应因为feature相似就跨帧混合。merge assignment是状态的一部分，预算、重建、读写成本全部计入。

还应区分两种恢复目标：**same-input算子目标**用冻结heavy FFN在当前student输入上产生target，检验light是否近似了该算子；**same-support轨迹目标**用同S_K完整网络产生中间state，检验误差是否沿层累积。二者不是同一种蒸馏；只在dense轨迹输入上拟合FFN，并不能证明推理时student输入上的替代正确。当前Paper默认最终GT/feature训练也不能被说成已包含逐层FFN拟合。先在诊断中同时记录两种误差，按其与TAD损伤的联系决定是否新增对应loss。

若压缩K/V，合并后的attention质量和数量权重也变化；只有在同簇key/value近似一致等条件下，按簇大小校正注意力才近似合理。没有这些条件时就是待学习的近似，不是严格等价。

S1/S2与均匀、固定seed随机、静态空间块、128分辨率做等完整MAC比较。主要评价仍是最佳完整TAD性能；时延/显存作为报告维度，不淘汰算法。

## 6. 官方VideoMAE decoder：能复用，但不能错认它学到的任务

### 6.1 官方合同

核查版本 `MCG-NJU/VideoMAE@5cefe18cecab25e3cce8de88fad0b42e6ce858a7`。官方先对完整clip做patch embedding，再只将visible tokens送入encoder；encoder末端有LayerNorm。decoder接收encoder-to-decoder线性投影、mask token和规则flattened时空位置编码。S为384→192/3heads，B为768→384/6heads；常见预训练脚本4decoder blocks，类默认8层不能代替实际checkpoint配置。输出头预测1536维RGB tubelet patch，非TAD feature。[O05,O06]

常见224输入对应8×14×14 tokens；本项目160输入为8×10×10，再空间池化成temporal latent。官方tube mask沿时间复制空间mask，不等于非均匀删完整时间观察。官方K400预训练16帧stride4与候选stride的局部名义相近，也不消除全窗口重排和TAD微调带来的分布变化。

### 6.2 当前已实现的迁移

`read_pretraining`核对连续decoder层、projection维度与关键权重；`MAELatentDecoder`复制projection、mask token、blocks和norm，替换RGB head与位置编码。输入是**anchors A＋所有原轴queries Q**，K384时为192+384=576，而非384个原grid内“192真实＋192缺失”。这恰当地避免了anchor并非原网格子集的问题，但也进一步改变预训练交互图。[C18]

此处复用是weight initialization，不是“官方decoder零训练恢复TAD”的事实。input projector原先处理归一化的spatiotemporal encoder tokens，当前处理已池化、任务微调、非均匀时序contextual anchors；形状相同不足以保证统计兼容。`heads=width//64`只适用于已确认的官方S/B等规格，应在导入manifest显式核验，而非普遍猜测。

### 6.3 公平比较应怎样做

两条现有MAE配置保持同结构/宽度/层数/头数、同head/scout/encoder初始化、相同训练增强和40轮80日程，只改官方权重或随机，这是正确初始化比较。**不能拿带已训练R03的Cross主线直接当相同起点对照**。需新增fresh Cross（不加载R03），让它与fresh MAE共享任务模型资产，再报告hot-start Cross的额外恢复课程成本。

最小三组：M-random、M-official、Cross-fresh。可增第四组M-official＋输入统计适配（LN/小仿射仅作用decoder输入，不破坏base feature尺度）。初始化收益主要看完整TAD峰值与终点，辅以固定更新和固定训练总计算下的收敛、feature/边界误差。不同decoder参数量与计算差异必须列出。

无需保留RGB head，也不应把预测RGB重新送满骨干。官方decoder迁移**可以取消对R03恢复器权重的依赖**，但不会自动取消H65/BMCR task encoder/scout和官方TAD head/teacher的依赖。没有独立读取大权重前，不宣称复制比例、权重CRC或实际加载成功。

## 7. 采样、空间mask与深度容量的共同设计

### 7.1 三种缺失不能混成一个mask

输入观察缺失改变可观测信息；空间token/FFN缺失改变局部处理；深度跳过改变状态演化。在当前图中还存在selected帧重新配tubelet和global-TIA(K/2)的原时间距离失配。decoder需要知道真实源时间与处理历史，而不是单一0/1“observed”。

当前geometry已正确区分两帧贡献有效性、pair中心/跨度与原query；对ANet重复物理时间，Paper geometry会合并相等中心用于插值、保留原contextual memory，这是已正确部分。[C16,C17]

### 7.2 建议的重建友好单帧策略

保持单candidate选择，不引入整clip选择。允许对选择集合增加**覆盖—跨度约束**：总K固定，限制最大物理空洞；对过大selected pair跨度加软惩罚；边界候选由scout提供，不用GT。它不是强制原pair同时选中，也不预设越连续越好。

重建友好与任务友好可能冲突：选容易预测的背景会使MSE低，却漏掉短动作；只选高频边界可能丢失动作内部类别证据。建议比较当前selector、覆盖约束selector和任务收益校准selector，保持同teacher/decoder/预算，分解最大gap、动作内部覆盖、起止支持与任务误差。

### 7.3 课程不是“训练从未见过缺失”

当前已经mixed sparse student，不是纯D1。引擎在activation checkpoint重算时保存各pass的TIA时间轴，这项多预算正确性处理应保留，不应重列为当前bug。[C08] 建议把预算覆盖从“15项平均轮转”改为**阶段平衡覆盖**：完整状态占固定份额、当前目标容量占固定份额，其余预算均匀探索；预先固定epoch配比。比较维持均匀轮转与渐进收紧两种完整课程，同更新/teacher查询报告；若含更多dense查询，也明确其训练代价。

预训练decoder应在当前选帧/深度/空间输入分布上适配，而不是只接受标准.9空间tube mask后直接部署。mask课程可以保留局部连续的帧片段作为增强，但基本决策仍是单candidate。

### 7.4 崩塌的判别树

同K fullD/fullS teacher仍失败：先看观察不可辨识/时间语义及任务初始化。相同S_K完整D/S较好、稀疏很差：看Q/K/V缺失、状态漂移、routing。层内oracle对齐有效、最终decoder无效：说明损伤传播在解码前，需中间恢复。层内误差低但TAD差：看输出尺度、排序、GT分配及重建目标。训练指标不断下降但测试不改善：看200视频泛化/模型复杂度与测试选模，不直接推出再训练一定恢复。

## 8. 原长latent与自蒸馏的有机组合

当前默认 `task=1, feature=1, full_gt=.25, self_feature=.1, action=.1`；输出KD/时间差分并非全部默认启用。shared-full使用同学生参数、K768/全容量分支并有GT，self feature target stop-gradient；EMA.99用于评估，不等于EMA target teacher。[C11,C15,C20,C22]

因此当前已经构成共享分支自蒸馏，但仅最终回写表示的自蒸馏；6/9/12融合不自动等于deep self-supervision。还需补same-support中间目标，以消除与外部original384的支持错配。

不同损失的分工：pixel目标关心外观细节，第一优先级低；latent Huber/cosine约束尺度及方向但可能均值化；关系KD保护相对结构但不能单独保证幅度；时间差分保护变化但权重过大会放大噪声或牺牲类别；分类KD影响排序，定位KD影响端点；GT监督保证teacher漏检不被固化。

point head为独立sigmoid类别和左右距离，使用Bernoulli语义与同轴距离/端点KD；TadTR的query需要匹配，不把pointwise同index KD强套进去。先测GT+latent，再加一项目标，以等查询课程验证必要性。

防漂移：记录冻结教师、共享full分支和学生的三方任务loss/AP轨迹；冻结外部teacher；sharedfull有GT锚定；EMA-target可作独立选项但不能同时放开教师和学生全部自由追逐；输出回归监控方差、边界斜率、重复事件间谷值与置信度翻转。不要把自蒸馏的低loss当独立泛化证据。

## 9. 当前路由目标代表什么，以及还缺什么

当前BudgetRouter已有15项真实完整模型cost菜单，以scout统计预测分类/定位收益及方差，在预算下选风险校正分最高项。FrameRouter最多一次局部换帧，接受预测保守收益为正的交换。它们分别是**窗口级菜单选择**和**单交换决策**，不是每层每token均直接学习真实任务价值。[C12]

`collect_action`确实重算当前student、TIA、decoder和head，并把repair_delta另外保存；这一点已修复历轮最重要的概念错误。actual_delta仍是训练loss差，不是mAP边际。由于AP全数据排序与NMS的非局部性，仅追求loss下降可能仍不改善最终mAP。[C13]

重要剩余失配：frame动作标签主要在plan1（K384,D1,S1）生成，但部署会在多种D/S/K上调用，FrameRouter特征没有显式D/S预算。建议把plan和曝光历史加入条件，并在不同计划下抽样交换；保持总干预查询数，比较plan-aware与现状。不能只增加查询数量后归因于更好建模。

BudgetRouter只读scout统计，在昂贵encoder/decoder前决策；目前不读取本次实际解码误差。因此“解码后不确定处再次请求计算”的闭环是新设计，不能画成当前网络。

校准应报告符号准确率、Spearman/Kendall、分位gain、oracle regret、接受交换的负收益率、预算利用率、每类事件风险。OOF仅路由拟合按视频分离，学生表征已见全部训练集；90%覆盖经同组残差缩放后的数值不是独立coverage保证。单seed42仍可做分组拟合和固定模型的视频bootstrap，它们不是新增多seed完整训练。[C14]

若动态菜单多数时间总选同一项，应与该项独立训练、以及该项来自同一supernet的固定推理都比较。动态不胜出时保留静态最强模型，不为论文强保router。

## 10. 已实现网络与建议最终网络

### 10.1 已实现（PaperModel）

输入[B,1,3,768,H,W] → 全时间scout[B,768,96] → 15项budget＋H65/BMCR初选/最多一次交换 → K个单帧按16打包 → native K/2空间状态 → 交替A-MoD、准入内FFN heavy/light、完整TIA(K/2) → 6/9/12融合或末层anchors → Cross/MAE/TCN → original native384 → 独立可训练point/TadTR readout →检测。[C06–C12]

默认更新：Adapter、scout、decoder、light、frame/budget router及student head；非Adapter主干冻结，另有full_finetune。外部teacher仅训练/诊断；sharedfull增加训练前向；推理只执行选中的学生计划。THUMOS point head虽独立，却从官方TAD head初始化。

### 10.2 建议候选D（首选科学主线）

保留现有frame路径和原轴回写；新增**same-support层目标、full-KV对照、depth排除位置light状态**；有限菜单保留，但frame utility加入plan条件，门率课程预先收紧。最终方法是否保留全部部件由独立消融决定，不要求并行课程相互等mAP。

### 10.3 建议候选P（简单且强的替代）

固定同一单帧selector与回写器；PBD式逐层评分/静态压缩，匹配恢复训练和same-support层对齐；可加静态或轻量空间FFN。若它以更少复杂度达到更好完整FLOPs—mAP前沿，论文应接受该结果。动态不是必要终点。

### 10.4 建议候选R（初始化与采样兼容）

不加载R03，保留相同任务encoder/scout/head，比较fresh Cross、同架构MAE random/pretrained、partial pretraining+input alignment；固定单帧K及D/S合同。再加覆盖/跨度约束采样对照，回答官方预训练知识是否有用，而非笼统“MAE能补帧”。

### 10.5 依赖消融必须精确定义

|名称|应移除什么|不能宣称什么|
|---|---|---|
|no_external_loss|external feature/output损失|不是teacher-free；现有teacher和repair查询仍可能存在|
|no_teacher_queries|训练期间外部teacher前向/repair诊断|不代表head/encoder/R03未继承其知识|
|no_recovery_init|R03恢复权重|仍有任务encoder/scout与official head|
|random_task_head|任务head权重|仍有其他task assets|
|recognition_only_init|encoder仅公共识别预训练，scout/head/decoder无TAD权重|必须重新完整任务学习，不能与热启动同课程成本混称公平|
|public_MAE_decoder_init|加载官方预训练decoder|不是任务teacher，也不消除其余任务资产|

`learned_state`只保存可学习和指定buffer，部署加载仍需冻结资产；“推理不用teacher网络前向”与“无需任何teacher来源参数”是两回事。建议输出合并部署权重包与资产清单，但不在论文中混淆二者。[C06,C10,C23]

## 11. 多领域机制图谱、反例与创新边界

|问题|原领域/方法|可借鉴机制|迁移障碍与必要对照|
|---|---|---|---|
|mask输入如何预测结构化输出|MAE/VideoMAE、CrossMAE、MultiMAE|坐标/任务query、轻decoder、latent memory|多数是预训练工具，不是下游推理自动恢复；fresh/random同结构对照|
|像素恢复是否必要|MVD、data2vec、V-JEPA|contextual teacher latent、stop-gradient预测|大规模预训练/教师不同；不证明少RGB TAD不掉点|
|层内状态不足如何适配|PBD-TAD、DyT、OFA|逐步收紧容量、同支持状态对齐、shared-full监督|动态跳token比静态网络复杂；训练成本与监督要匹配|
|谁应更新，谁应提供context|CoLT5、TR-BERT、DToP|Q/KV分离、逐token深度、保留早退出状态|分类/span输出不等于多事件检测；需TAD独立实证|
|稀疏后怎样回写密集接口|Expediting ViT、点云多尺度恢复|assignment、坐标、skip与unmerge|原本具有完整浅观察，不能凭空恢复被删事件|
|视频状态可复用到何时|DFF、Eventful Transformers|特征传播、缓存与delta更新|遮挡/新事件/窗口位置变化；要验证同上下文缓存条件|
|局部误差是否值得精算|目标导向误差估计、主动感知|以任务后果而非误差幅度衡量额外计算|mAP非平滑，低置信度不自动保证错误被发现|
|动态时空视频分类怎样呈现|Uni-AdaFocus|同模型多预算曲线、独立训练模型与推理曲线分开|其ActivityNet mAP是视频识别，不是本项目tIoU定位mAP|

尤其需要反对“边界永远最重要”的预设。CVPR2024的TAD temporal corruption benchmark发现，在其测试设置中动作**中部**受损可能造成最大下降，并提出FrameDrop与Temporal-Robust Consistency。稀疏采样不等于黑帧腐蚀，但该反例足以说明应同时保护类别内部证据和起止信息，不能只做boundary heatmap论证。[L19]

对当前创新的判断：原轴恢复、attention路由、多层memory、自蒸馏、三轴动态本身都有先例。最有机会形成独立贡献的是：**把不规则选帧的支持错配与D/S计算状态错配分开建模；用真实干预校准有限预算决策；证明状态一致性设计确实缓解三轴交互损伤。** 这仍是待验证贡献，不是已证明独创或首次。

若简化TCN＋PBD比复杂Cross＋动态全菜单更好，论文应选择简化模型，保留复杂模型作为反证。替代方案胜出并不消除研究价值，但会改变贡献表述。

## 12. 现有证据是否足以形成完整论文

目前足够写出：问题动机、当前网络、真实先导结果、三轴交互的负证据、明确的可证伪假设。**不足以写成已实现三轴性能保持的完整结果论文。** 缺口不是只有mAP，还包括同课程恢复器/路由必要性、正式PBD与静态控制、seed42完整课程、泛化和三维联合成本证据。

不应把旧Cross-B相对旧BMCR80的+0.4431pp写成纯decoder因果收益；不应把旧同epoch5的learned-vs-uniform约+.2pp写成强路由必要性；不应把新有可训练head的架构图当成已恢复这些差距的证据。

单seed42允许：描述每个指定训练轨迹的峰值/终点、同checkpoint干预、固定模型成对视频bootstrap、按视频分组的router拟合诊断。**不允许**跨训练随机种子稳定、随机训练误差条、多个epoch当独立重复样本。每个配置的10/20/40/60/80分布只是trajectory distribution；不同配置的分数箱线图也不是训练方差估计。

测试峰值遵循当前作者规则，但须披露预登记候选集合和选择次数；40轮消融与full前40以相同10/20/40候选比較，再各报40终点。80全程最佳与40消融最佳可以作总资源产品比较，不能称严格消融。校准terminal checkpoint曲线不可贴上best-EMA标签。

性能“保证”只在有限条件下成立：同权重/同图/同状态/全容量数值一致；或固定数据条件上的经验损伤。当前全容量仍经过继承task权重与decoder/readout，不必等于官方提供checkpoint。未知短事件的不可观测性、mAP排序翻转和校准分布外失效仍无法承诺消除。

## 13. 公平公开对比

**外部同框架核心**：提供的官方AdaTAD复测；PBD-faithful及本框架PBD-style；经过同等适配的静态block drop、静态depth/空间比例与128输入；uniform/random单帧方案＋相同恢复/teacher课程。uniform是科学控制而非公开论文名。H65/BMCR/Cross全部标作者内部前身。[L01,L17]

**机制对比**：A-MoD/标准learned-MoD式门、DyT-style MLP dispatch、DToP式confidence exit、EVAD-inspired空间保留、误差幅度与真实action-utility。迁移后明确“adaptation”，不得沿用原任务mAP声称复现。

**公开TAD场景表**：AdaTAD、ViT-TAD、TALLFormer、Re²TAL、ETAD及任务相关的新方法；只使用核实的任务、数据、预训练、输入和head信息。ViT-TAD提供跨snippet上下文证据；TALLFormer/Re²TAL/ETAD多针对训练memory或梯度计算，不能写成少RGB推理直接竞争。TadTR是head/generalization，不是视频编码器。[L16–L18]

PBD论文THUMOS主要比较@0.5，不能把其70.47写为本项目五阈值均值；论文213条测试描述与当前211ID要核对实际loader。公开不同backbone分辨率或离线特征成本不能直接并入本项目完整encoder+scout+decoder+head的Pareto。

当前InternVideo分支是**InternVideo1-MQ**，不是前文讨论过的InternVideo2；图注必须用实际资产，不套用InternVideo2 tubelet_size=1结论。若今后增加另一编码器，它是新配置而非当前已实现泛化。

## 14. 网络图与三轴问题图

详见 `FIGURES.zh.md` 的逐panel字段、步骤和图注。本包提供可转绘Mermaid草图。

主网络图应有四个区：全时间scout和预算；单帧选样与计算打包；层内dense-grid sparse-update；原轴回写与可训练head。训练专用external teacher和shared-full分支用虚线、stop-gradient标签。实线梯度区分TAD→decoder→Adapter/scout和hard选择索引，不把不存在的hard top-k梯度画出。

标注关键尺寸：候选[B,768]；scout[B,768,96]；选帧K；重encoder空间状态[B,K/2,10,10,C]；anchor[B,K/2,C]；queries[B,384,d]；readout[B,C,768]。ANet与InternVideo/TadTR的不同长度/通道单列图注，不硬套THUMOS。

三轴动机图首先展示真实预算干预下的AP损伤/恢复，而不是三张相似度热图。可附attention或cosine作为描述，但箭头到“可省略”必须经过真实干预证据。GT动作/边界仅用于离线绘图，图注说明路由没有读取GT。

## 15. 结果图的可信竞争力

借鉴Uni-AdaFocus区分独立训练模型与同checkpoint可调预算曲线的表达，但不复制其分类任务数值、也不假装已视觉审阅每张原图。其Fig.9图注支持“同一模型内改变预算不重训”的曲线语义，而不同分辨率模型是不同曲线。[L14]

主结果双视角：同骨干归一化FLOPs—最佳mAP；跨骨干绝对完整GFLOPs—最佳mAP。旧Cross-B被官方S支配的事实必须可见。每个最佳点的FLOPs取**同一个checkpoint、同一路由配置**，不能拼接最小计算和另一权重最佳分数。

动态策略至少报告全测试实际窗口平均成本、固定完整窗口成本和分位数。旧记录只有固定第一full窗口，则在独立历史图显示，不和全数据均值连线；缺失成本标null，不能画0。整视频总GFLOPs不可把单窗平均错误标为video成本。

最佳与终点可用填充/空心符号和短连线；种子保持42，不加假标准差。预算曲线连接同checkpoint同校准来源的测试点，不用平滑样条制造未测前沿。每配置训练里程碑图展示trajectory而非独立样本分布。

真实案例至少包含改善、无变化和失败；选择规则先固定，例如短动作/重复事件分层内最大改善与最大退化各一个，再随机固定若干。无原视频不得合成“真实案例”；本包仅提供历史数值示例，不提供虚构空间mask图。

## 16. 后续少量主路线、并行任务与论文框架

### 16.1 既有52课程怎样处理

保留现有full/uniform/dense S/B80；保留40轮消融、MAE两种初始化、full-KV、lightoff、TCN/interp、训练head及多深度memory控制；按技术资产到位并行。ANet、InternVideo1-MQ、TadTR只因数据/权重/对应head实现依赖等待，不等THUMOS胜负。

现有no_action、fixed_plan、独立axes配置需改**解释**或增匹配控制，不在已提交recipe下静默修改。no_action同时关frame refinement与budget router；fixed_plan又改训练计划覆盖；axes关闭后多项菜单坍缩为相同实际计划，采样频率可能变化，因此需保存实际plan曝光分布。

### 16.2 新增有限模型族

并行新增的完整模型上限先按本包 `EXPERIMENTS.json` 的明确行数，不生成无限笛卡尔积：

- P族：静态drop适配与PBD-style＋same-support层对齐，S/B各一组核心配对。
- D族：现有full-KV作为控制；新增depth-light、same-support层监督、二者组合与渐进课程的匹配组，先S主机制、B一组预登记迁移。
- R族：fresh Cross、现有MAE random/pretrained、partial input alignment；另做计划条件FrameRouter和真实完整成本的固定菜单控制。

这些是研究规格，不是已经部署。更大encoder、更复杂merge/闭环主动感知作为替代源码分支，不能不受限地追加全训练。

### 16.3 推荐论文结构与摘要草案

候选标题：**Support-Consistent Predictive Computation for Temporal Action Detection**。中文定位：“面向时序定位的支持一致预测式计算”。若动态贡献不成立，标题和贡献去掉动态/统一最优暗示。

摘要方法草案（没有填入未测结果）：

> 未裁剪视频的时序动作检测需要同时维护事件语义、时间边界与置信度排序，因此减少观察和减少网络更新会造成相互依赖的任务损伤。本文研究在完整模型计算预算下，如何从不规则单帧观察维持可用的原时间轴表示。我们区分观察支持变化和层内计算状态变化：坐标及来源感知的解码器将稀疏anchors写回检测时间轴；同选帧支持监督和上下文保留更新对齐深度及空间稀疏状态；有限预算路由通过真实重执行的分类/定位收益训练。实验将在匹配初始化、课程与成本的控制下，检验恢复、状态适配和动态分配各自的必要性，并报告最佳与终点性能、三轴交互及真实执行成本。

结果段只能在完整测量后填：数据集/骨干、完整GFLOPs、最佳mAP及dense差、终点、seed42、选模规则。若只有某一骨干有效，摘要不能写一致跨骨干；若静态胜出，按结果收缩。

引言逻辑：任务状态敏感性 → 稀疏选择与层压缩已有工作 → 当前旧八格展示条件耦合 → 支持一致性假设 → 三项可检验贡献。Method中用一张图一个统一任务目标；Experiments中公开竞争、内部机制、任务错误、依赖消融四层分开。

### 16.4 最终结论

当前最值得的新增研究，不是再把容量从.5压到.125，也不是认为官方decoder权重一定解决问题，而是检验：**保留上下文、补充深度被排除位置的可用状态，并用同选帧支持的完整路径监督，是否比现有最终原轴拟合更能保护TAD。** 这项主线能同时回应PBD、A-MoD和MAE迁移的挑战。

论文尚不能承诺三轴性能保持。可以确定的是实验规格、最小因果比较和证据产生方式；最终方法应从并行完成的有限候选中按完整FLOPs—最佳TAD性能选择，同时报告终点、代价和负结果。时延不是路线门槛，单seed42不是统计稳定保证，两者均应忠实写入论文。

---

## 引用索引

C编号对应同包 `SOURCE_MAP.md` 中固定提交文件。L/O编号对应 `REFERENCES.md` 的原论文版本和官方代码。它们区分已读正文、原论文摘要和源码访问范围；不存在把项目内建议当外部事实的引用。
