# WTR 最新科研规划吸收记录与首轮实验建议

> 后续修订：当前优先级见同目录 `CURRENT_PLAN.zh.md`。32视频probe仅保留为工程验收；正式工作先做完整数据集characterization和allocation headroom，本文的首批5条S训练后移。本文作为上一轮核对记录保留。

记录日期：2026-09-14。范围：完整阅读三份 Markdown、四份补充文本，核对本机代码、配置及已保存运行记录，形成下一轮建议。本轮没有修改科研代码或配置，没有应用补丁、创建实验分支、启动训练、登记或部署作业，也没有连接服务器。

**主要判断：第一轮要验证，在固定观察支持、原轴恢复和真实计算预算下，局部 heavy update 的任务收益能否被可靠测量、预测，并用于超过均匀或周期分配。** 最新补充要求把 Hold / Light / Periodic / Value 的状态更新研究提前；原时间轴恢复是另一核心支柱。Graph 和未来教师均须独立通过对照，不能作为预设有效的组成部分。

## 1. 本轮输入与指令边界

七份原文已经逐字复制到本报告旁的 `inputs/`，保留全部建议、公式、命令、工单和图注。复制不改变原件；材料中提到的部署命令、代理工单及旧授权均作为研究材料记录，**不构成本轮执行授权**。本轮授权来自用户明确请求：“仅做报告，不要进行代码实现和部署”。

|归档文件|原始来源与作用|
|---|---|
|01_Research_Report.zh.md|E:/下载/Where_to_Refine_Research_Report.zh.md；固定源码复审、文献借鉴、证据边界|
|02_Agents.zh.md|E:/下载/Where_to_Refine_Agents.zh.md；A0–A7 待实施责任与验收|
|03_Figures_1_6.zh.md|E:/下载/Where_to_Refine_Figures_1_6.zh.md；六图证据设计，非实测结果|
|04_Review_and_Initial_Plan.txt|附件 2e54b598-f469-4384-a772-775ddc6b69ca/pasted-text.txt；复审结论、W00–W05、拟议代码包|
|05_Unified_Model_and_Minimal_Plan.txt|附件 f9bf9046-954d-4aa5-8e62-a01978e576af/pasted-text.txt；统一 WTR 主线、缩减矩阵|
|06_Original_Axis_Recovery.txt|附件 d5c1f9bd-4764-4a36-b90a-6e77c5fbae2e/pasted-text.txt；明确保留原轴恢复及 multidepth|
|07_State_Update_Study.txt|附件 2f2d5b94-0272-4f51-bea2-dfd6a90aca2e/pasted-text.txt；新增状态更新实验块并调整顺序|

按材料提供的逻辑顺序理解修订：初版六课程仍保留为价值学习研究；后续文本将三轴训练压缩为 T / T+D / T+S / T+D+S，明确 decoder 必须保留，最后把状态更新机制提前。A0–A7 是责任划分，不是八个同时启动的 GPU 作业。

## 2. 已吸收的研究主线

最终候选链条是：完整时间轴 cheap Scout → 窗口预算与任务条件价值预测 → T/S/D 固定容量重计算 → 带真实时间与来源的 sparse anchors → 原轴恢复 → TAD head。

T 继续表示 individual-frame heavy RGB evidence acquisition；先逐帧选择，再进行 VideoMAE 的 16-observation 打包。D 表示内部 attention/FFN 状态的重更新分配。S 表示 admitted token 内的 heavy/light FFN 分配，当前不是高分辨率 ROI 采集。不恢复已经撤销的原始连续 16 帧 clip selector。

局部价值目标为 `V(a|s) = L_TAD(s) - L_TAD(s⊕a)`，保留 cls/reg 分量及有符号值。固定预算交换不除以接近零的成本差；正成本追加才讨论单位新增成本收益。动作价值依赖当前状态、已观察证据和其他动作，不能默认可相加。

现有固定容量、真实 gather/scatter、selected-Q/full-KV、light residual 与 global TIA 应继续使用。需要改变的是局部排序及其监督：incoming attention 测量“被读取多少”，尚未直接学习“重算能改善多少”。现有全局 actual_delta 训练 FrameRouter/BudgetRouter，不等于已经建立每层 D/S slot 的价值监督。现有实现没有用 attention score 缩放输出残差，不能编造该掉点原因。

三个监督对象必须分清：冻结 full-D/S 参考的同支持轨迹；student 当前输入状态上的 heavy–cheap 算子残差；原始时间 query 的恢复误差。它们不能被统称为一个 KD 效应。`repair_delta` 的 teacher 特征替换属于诊断，不能充当真实可执行计算动作的标签。

状态动作的完整目标是 hold、light、heavy，并分别研究 light 相对 hold、heavy 相对 light 的收益。首版可以先学习 fixed-light 背景上的 heavy slot 选择；这不等于已经实现双价值、三档联合 allocator。

原轴恢复默认保留 physical-time interpolation + Cross residual + multidepth / provenance / Scout context。完整 query 栅格和物理坐标可以维持；未观察信息与 dense heavy feature 的语义一致性必须测量，不能承诺无损恢复。Graph 只作为证据访问或关系残差候选；Value-aware Recovery 留作后续增量。

任务加权残差先做 FFN-only：在同一当前 state 上测 `r=heavy(h)-cheap(h)`，在修正注入点取任务梯度，校准 `-<g,r>` 与实际重执行收益的符号、排序及选择 regret。这是 Taylor proxy，不是 oracle，也没有继承数值分析误差界。

未来教师只作用于价值学习。no-KD、frozen post（β=1）、EMA、future（β>1）必须匹配比较；同一输入、候选和支持上的三个 checkpoint 实际收益，用来验证是否能预测后续真实价值。教师推理时删除。若外推不超过 post/EMA，不保留 RISE 主张。若图不超过 Cross/静态物理时间控制，不保留 Graph 主张。

学习使用少量真实反事实进行校准，必要时在当前策略实际遇到的状态上补采样。旧记录未保存完整查询概率，不能追认无偏策略评估。先证明价值可用，再研究便宜代理与未来目标，不启动模块全笛卡尔积。

## 3. 应在哪一版代码继续

**建议从提交 `1955057508af5a5dfd59a98bddf49302bee5972c` 建立独立 WTR 工作树，沿用该版本的 `h65/paper` 与 Full-V2 执行骨架。** 拟议分支名可沿用材料中的 `codex/wtr-value-v1`；本轮没有创建它。第一版关闭 Graph 联动，保留 Cross 恢复。

本机对应仓库：`C:/Users/skywalker/Documents/ChatGPT/H65/h65_clean_adatad/graph_tad_20260914`。

|核对对象|实际状态与含义|
|---|---|
|当前分支|codex/graph-tad-20260914|
|当前 HEAD|1111e53da91e591e5ad702e1c4e499f905337841|
|1955057 与 HEAD|1955057 是 HEAD 的祖先；后续仅 0625cac 的 ANet 准备/恢复工具，以及 1111e53 的状态记录更新|
|模型执行代码|1955057 之后未改变本轮相关的 h65/paper 模型执行代码；固定到附件审阅版本能避免基线含混|
|工作区状态|存在未提交的 characterization 原型、协议及其他报告文件，不能把整个 dirty 工作区直接打包称为固定 WTR 版本|
|其他本地分支|paper、support_review、publication 的 HEAD 均不是1955057的线性后续，不应仅凭旧目录名切回这些版本|

“代码基线”与“初始化权重”不同。固定 checkpoint（如 V2-S40 EMA）用于诊断；机制训练应从共同的 H65/BMCR、Cross/R03 和任务资产起点开始，不让某一候选独享 V2 已训练40轮后再续40轮。各旧 checkpoint 保留自身 source_revision，不回填成1955057或当前文档提交。

本机限定库存中未找到 `wtr_core.py`、`wtr.py`、`wtr_probe.py`、`wtr_plan.py`，也未找到材料提到的精确名称 `Where_to_Refine_H65_agents_1955057.zip`。因此材料声称的26项 CPU/mock测试只记为材料自述，不能视为本机已接入、已验收。后续实施需先取得并审阅该包，或依明确规格完成最小实现；本轮报告不执行这些动作。

第一版待实施内容：固定支持局部反事实探针、重算前的轻量 value head、沿用现有容量核的局部选点、actual-value loss、执行成本和状态诊断、完整 checkpoint/EMA/optimizer 续跑覆盖。post/EMA/future 可随后接入。暂不联动 Graph、不重写 ViT 核、不同时训练整个三轴联合预算策略。

## 4. 首轮首先做测量，新增正式训练为零

最先回答两个问题：**有多少计算的任务边际价值确实不同？这种差异是否能转化为有限预算下更好的分配，而不仅是漂亮的相关性图？**

固定参考至少区分官方 dense AdaTAD 和 V2-S40 EMA 的诊断身份。已有 V2/Uniform 的40/60/80 checkpoint可用于后续漂移测量，但尚未产生的终点不能预先使用。

第一批局部采样沿用材料的小规模安排：32个训练视频，每视频最多4对 D exchange、4对 S exchange，最多256对；每对至少需要两次完整 student/head 重执行。这是样本内机制诊断，不是泛化成绩。合法候选不足时报告实际数量。

需要固定 RGB、augmentation、selected indices、其他层掩码、参数和随机态。D exchange 在同 packed-clip 竞争域，S exchange 在同 native-time 配额域；D交换保持相应FFN角色及预算。相同动作跨 checkpoint 比较时继续使用相同候选/支持。

**预算要说清楚：** 当前菜单 `force-plan 4` 是 K384/D50/S48，可用于同时产生 D/S 交换的压力诊断；它不是首轮训练的 D75/S100。首轮还应在 D75/S100 上核对 D 的结论。S100 没有未获 heavy 的合法 S slot，不能强行生成 S exchange；S诊断需单列 S<100 的预算。

首轮产出包括：

- signed ΔL_cls/ΔL_reg、近零比例、value分布；no-op给出数值噪声参照，联合2/4动作检验单点低价值能否共同省略。
- attention、uniform/static、简单norm/uncertainty与实际收益的排序、符号和选择regret；有限 GT 辅助搜索对均匀分配的改善空间。
- 同状态 hold/light/heavy 的实际任务收益；attention、pre-TIA、post-TIA 的NMSE、cosine及reference RMS；FFN残差代理与真值的校准。
- 完整推理算量分项：评分QK、heavy/light、TIA、恢复器、head；记录额外诊断/teacher查询与时间，不能把评分开销漏掉。
- 按真实 state age、边界/内部/背景、动作时长与支持缺口分组。边界不预设必然最重要，特征误差上升也不预设一定影响TAD。

局部 loss 不代替数据集 mAP。主性能仍使用全数据集 Avg-mAP、mAP@0.7及完整成本；视频聚类重采样时重算AP，不平均per-video AP。训练集诊断不称未见验证，已反复查看的 THUMOS test 也不改称新holdout。seed42的视频CI不等于训练seed不确定性。

有正向、可重复的分配空间后，才登记新增学习阶段。有限搜索未超过uniform不能证明数学上没有headroom；但也不应据此继续增加控制器复杂度。允许的精度差、最小有意义节省、目标标尺和选模规则应在正式比较前固定。

## 5. 两项必须纠正的实验定义

**旧 N 系列可以复用证据，但不能无条件复用为新的 matched control。**

N01=full-KV+hold；N02=N01+light；N03=N01+same-support；N04=light+same-support。它们确实覆盖状态更新的基础问题，不必重新跑一遍同配方。

然而这些课程的实际配置为 K384/D50/S48、fixed_plan=4、seed42、40轮/80轮LR前缀；N01/N02/N04沿用原交替路由层。N06/N07通过 `mod_start=4` 派生出 `[4,6,8,10]`，仍为D50/S48。新WTR基准是D75/S100，因此不能将旧N成绩与新M2/M3直接连成因果链，也不能只因名字相似宣布“只新增两条就完成全部严格对照”。

**原文的 age≥2 周期刷新，在当前交替路由配方上会退化。**

`engine.py` 每层默认 admitted=all；只有稀疏路由层才缩小集合；每个 admitted token 的 age 当层归零。稀疏层4、6、8、10之间的dense层5、7、9会对有效token进行heavy更新并清零age。因此在下一稀疏决策前，age通常重新为0；不能把“跨两个routed block没入选”称为“跨两层没有heavy更新”。`last_heavy_depth`还经过空间amax归约，不能直接替代逐token刷新历史。

本报告建议首轮M2采用**固定容量的周期配额轮换**：保持D75及原路由层，在路由决策之间轮换获得heavy的token；其计数明确叫“路由机会历史”，与真实state age分开。它是周期分配强控制，不作为长期陈旧状态假说的证据。真正研究连续跳层的误差累积，应另用明确连续路由条件先做固定checkpoint诊断，不能悄悄改变主课程层表。

## 6. 建议准备的首批训练任务

以下均为拟议，尚未实现、生成配置或登记。先完成上述测量和真实执行技术验证，再启动。

共同配方：VideoMAE-S、THUMOS14、seed42、K384、D75/S100、0-based路由层4/6/8/10、full-KV、原physical-time Cross+multidepth恢复、相同外部/shared监督与same-support设置；固定并冻结H65逐帧选择路径，包括Scout和frame-refiner行为，关闭动态预算。共同初始化，训练40轮，采用80轮LR前缀。D75只描述这些路由层的容量，不代表全模型省25%。

|拟议课程|唯一研究变量|主要比较|
|---|---|---|
|W00|incoming-attention分配，未选D状态light|现有排序的基准|
|W01 / M1|固定uniform分配，未选D状态light|便宜的简单选择是否已经足够|
|W02 / M3（首版）|实际反事实训练的局部value排序，未选D状态light，无新增KD|学习heavy相对light的价值能否胜过W00/W01|
|M0-matched|与W01相同的uniform重算位置，未选D状态改hold|严格隔离light correction的收益|
|M2|固定D75配额的周期轮换，未选D状态light|value能否胜过不学习的刷新/覆盖规则|

**首批建议5条 S×40轮。** 它同时回答排序和状态处理问题。W02与首版M3共用一条课程；W01与M1共用一条课程。M0采用与W01相同的uniform位置，避免“换hold/light时attention分数也随之变化”混淆状态更新归因。

W01保留同形小头可作机制成本控制；另报告完全不运行无用头的最低成本uniform系统。相同容量不等于相同完整FLOPs：attention评分、light与hold、value head都有成本差；先报告各自完整性能—成本点，需要严格同成本结论时再作明确预算匹配，不能靠空计算填账。

首版M3仅测 heavy相对light 的选点。完整 `V(light|hold)` 与 `V(heavy|light)` 双价值、两级配额分配留为后续有证据的增量，不能给首版换名后宣称已验证三档决策。

下一阶段保留原六课程中的教师部分：W03=frozen post（β=1）、W04=EMA、W05=future（β>1），各S×40轮，与W02采用共同起点/训练协议，而不是只让教师候选额外续训。三者配合相同候选上的三时间点actual-value预测；若声称训练效率，还需要额外优化步/teacher查询成本匹配。

这样，早期规划为**首批5条＋教师3条，共8条独立课程**，不是“六条W再机械加四条M”。这是本报告基于最新补充和实际代码提出的去重与匹配修订，区别于原材料未经核对的课程估数。诊断和技术预检不计正式训练课程，注册前决定阶段，注册后完成其原定40/80轮，不凭早期成绩停掉已登记课程。

## 7. 后续阶段与必须保留的旧证据

原有V2/Uniform S/B的80轮终点、Native AdaTAD K384直接评测及适配、既有N系列、Graph及其他已登记课程继续按原定任务处理。本轮没有取消、改排或重启任何任务。现有Uniform Full包含Cross及本框架训练，不能冒充原版AdaTAD直接降采样。

本机最新归集快照时间为2026-09-14 21:21:40 +0800。该归集记录尚无V2/Uniform80终点、Native K384或Graph的新完整成绩，N状态课程为待运行；这是本地快照结论，不是当前服务器实时检查。生成器中的旧 `PROPOSED_NOT_REGISTERED` 文字不能覆盖已更新的持久阶段登记。

较晚训练阶段依次为：

1. 胜出价值学习方式下的 T / T+D / T+S / T+D+S 四个S机制条件，匹配预算与训练；D-only/S-only可先用固定checkpoint诊断，不再训练完整七格乘全部模块。
2. 原轴恢复先测 interpolation、Cross、Cross+multidepth；新增continuous-time/Graph采用分阶段筛选，胜出者再进完整比较。Graph须面对static physical-time、dynamic no-referral及dynamic referral控制，固定selected frames、heavy features、head和训练预算。若用同容量对比，须同时公布完整成本差。
3. 最终胜出WTR-S/B各80轮；再做最强控制与最终模型的额外训练seed、ActivityNet、另一backbone/head。旧PaperModel的泛化结果不能重标成WTR。

本机还有未提交的 `research/characterization_20260914/PROTOCOL.zh.md` 及测量原型。该独立协议部分T动作是固定全RGB下的内部时间计算，并非WTR的逐帧heavy观察获取。可复用其记录、计费或统计设施，须逐项保持动作语义；不能把原型、CPU日志或原六图计划称为本轮WTR已实现/已部署。

## 8. 资源和后续部署对象

本轮收到另一任务的资源交接，现另存为 `RESOURCE_HANDOFF.zh.md`。交接报告：`connect.nmb1.seetacloud.com:44909` 上有两张RTX4080 SUPER；THUMOS14的200训练/211测试视频及annotations已同步；OpenTAD位于 `/root/autodl-tmp/OpenTAD`，提交 `346d09d19e2091372cec48172dbe40f7b28bdee6`，环境位于 `/root/autodl-tmp/envs/opentad`，尚未启动训练。

**建议这套新资源首先承接固定checkpoint probe和上述S机制课程。** 两卡可安排成对对照，并在真实单窗口前向/反向与显存核验后确定每卡作业数量；不预先承诺B或所有双教师课程均可直接跑。

OpenTAD环境可用不等于H65/WTR资源完整。交接没有确认H65/BMCR encoder/Scout初始化、Cross/R03、官方任务teacher/head、V2诊断checkpoint及resources映射已经在新服务器就位。模型构建实际依赖这些资产；后续部署前需对齐，不能用另一套初始化代替而继续叫matched复现。

原材料的唯一owner/Slurm、exit75与切片协议属于原实验集群。新AutoDL服务器不能直接套用旧Slurm作业号、PID、绝对资产路径或 `graph_deploy.py`。原集群的已登记课程留在原owner管理；新资源上的启动方式应按实际运行环境准备，保留断点、来源、参数及成本回执。本轮仅记录建议，没有建立任何控制器。

## 9. 图表、工单和结论边界的完整保留

Figure1区分observations/access/updates；Figure2测signed utility、null及coalition；Figure3诊断attention与value错配、实际D/S使用、评分成本及状态误差；Figure4给matched成本下dense/native/uniform/static/attention/value与GT辅助搜索参考；Figure5测原轴恢复和定位；Figure6测未来actual-value预测与最终预算收益。最新state-update补充应加入Figure3/5的真实age—误差—任务收益数据，不虚构age长尾。

A0版本/技术审计，A1反事实，A2局部价值，A3当前状态残差，A4时间价值与恢复，A5未来教师，A6统计与复现，A7部署，均已记录为后续责任。材料要求的legacy回归、同mask compact/dense-mask前向梯度、exact容量、真实参数更新、teacher冻结、fresh reload、EMA/optimizer/RNG恢复和计费核验，属于实施后的技术验收，不是本轮已经执行的测试。

有限GT辅助搜索不是数学上界；局部Δloss不是mAP；表示更接近teacher不自动表示TAD更好；官方test不是新holdout；单seed的video CI不是多seed稳定性；理论访问比例不是整机加速。仅实测数据进入正式图表，负结果和未完成项保留。

第一轮的成功标志是把“有价值差异”“能够学出更好的位置”“能在完整成本下改善TAD”连接起来，并说明原轴与状态如何被维护。若uniform或周期轮换已经足够，应收缩learned routing的必要性；若只有T有效，也不为了三轴标题强行保留D/S。Graph、未来教师和残差监督各自由对应证据决定去留。

## 10. 本报告关键代码依据

统一根路径：`C:/Users/skywalker/Documents/ChatGPT/H65/h65_clean_adatad/graph_tad_20260914`。

- `h65/paper/routing.py:26`：有限菜单；plan4=K384/D50/S48，plan9=K384/D75/S100。
- `h65/paper/model.py:97`：plan形成；配置容量可覆盖菜单；`:120`从mod_start推导实际路由层。
- `configs/paper_review/review5485_n04_s_seed42.json:56`、`review5485_n06_s_seed42.json:56`：N配方、容量与late设置。
- `h65/paper/encoder.py:117`：实际EnginePolicy及默认交替路由层。
- `h65/paper/engine.py:128`、`:135`、`:151`：admitted和age；`:207`、`:233`为depth-light；`:236`为当前state FFN诊断；`:254`为最终provenance归约。
- `h65/paper/interventions.py:100`、`:115`：真实actual_delta及BudgetRouter监督。
- `h65/paper/support_targets.py:55`：同selection的冻结full-D/S参考。
- `h65/paper/decoder.py:28`、`h65/paper/model.py:209`：原轴恢复、multidepth及Graph后接接口。
- `h65/paper/model.py:15`、`h65/paper/runtime.py:43`：初始化资产及数据资源依赖。
- `research/project_status_20260914/STATUS.zh.md`与`research/paper/graph/monitor_20260914_2114/after_recovery_snapshot.json`：带日期的运行证据，非本轮远端实查。
