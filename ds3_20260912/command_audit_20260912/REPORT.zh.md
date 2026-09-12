**命令文件总清单、实验实现/部署状态与待验证假设**

本报告按2026-09-13 00:06的远端快照和本地文件核对，23:00的初始快照也另行保留。范围包括本项目的工具/启动脚本、各工作树版本、外部包中的命令与agent任务书、启动manifest/模板、验证脚本及本任务的自动跟进配置。原始模型库模块与上游/ASFormer工具单列；没有保存成文件的一次性shell历史不伪装成命令文件。工作树副本不是额外实验，同名脚本也不一定同版本。

本次只生成索引和报告，没有执行外部包命令、重新启动控制器或修改运行中的训练配方。原始附件中的审批、hash、旧路径和预算是研究资料，不覆盖用户已经明确的200训练、211测试、单种子及80轮等要求。

完整逐文件索引见[FILES.zh.md](FILES.zh.md)，静态参数/来源见[command_inventory.json](command_inventory.json)，当前调度器全部73项入口见[REGISTERED_STAGES.zh.md](REGISTERED_STAGES.zh.md)，远端实际命令文件见[remote_command_files.json](remote_command_files.json)。

**目前最需要区分的是：有文件、可调用、已注册、已运行、完整评测和有效性得到支持，是不同状态。**

| 类别 | 当前事实 |
|---|---|
| 已完成历史实验 | 初期机制诊断；旧H65/BMCR四条60轮训练和六组完整评测；修正H65两条60轮训练和16次中间全测试 |
| 已实现并部署，仍未完成 | DS3 S/B D1 80轮及其参考、深度、空间、因子矩阵评测；已登记73阶段 |
| 用户已授权但部署还没落实 | 修正BMCR-T补实验；当前只有准备记录与可复用共享模块 |
| 暂停的部分实现草稿 | `mainline_20260911`里的recipe/time/fusion；没有连通的三阶段CLI和部署流程 |
| 后续研究/原包提案，未完整实现 | C2/CAL、J3/JOINT、STM merge/unmerge、动态总预算、输出KD、BMCR→DS3知识桥接、保基线初始化、128全时域等 |
| 明确跳过，不属于欠账 | 重训官方基线、独立U384完整训练、强制160/40划分、初轮全矩阵多种子 |

截至快照，当前DS3控制器有**73阶段：14完成、2运行、57 WAITING**。运行的是S训练1287797及B官方密集参考测试1287932。S已保存36/80轮、3600/8000更新；B正式D1尚未开始，等待自身参考评测和pilot依赖。S的5/10/15/20/25/30轮T24A完整测试mAP为50.6458%/48.9030%/48.1927%/49.1955%/49.4202%/49.9364%。后段有回升，但这些已测点仍低于零新增训练Z24的56.1065%；当前数据没有证明D1维持了精度，这也不是最终80轮结论。

**命令文件首先按下面几组使用和理解。**

| 组别 | 主要文件 | 实际作用与状态 |
|---|---|---|
| 初期诊断 | 主目录`tools/inspect_runtime.py`、`check_tia_connectivity.py`、`one_swap_audit.py`、`gpu_probe.py`、`run_gpu_probe.sh`、`verify_saved_probe.py`、`final_report.py` | 检查来源、TIA连接、交换效用、两步真实更新与重载；历史已完成，不是完整训练入口 |
| 旧完整实验 | 主目录`tools/full_dispatch.py`、`preflight_full.py`、`full_train.py`、`full_audit.py`、`full_eval.py`、`full_compare.py`、`summarize_audit.py`、`run_full.sh` | 原Phase2调度/训练/审计/测试，旧控制器16阶段已完成；此处是旧配方，不能和当前同名文件混用 |
| 当前共享H65/BMCR能力 | 活动树`tools/full_train.py`、`full_audit.py`、`full_eval.py` | 共享源码已修LR与裁剪边界；训练支持`--variant bmcr`，但不等于修正BMCR新实验已部署 |
| 已完成修正H65 | `tools/fidelity_dispatch.py`、`fidelity_select.py`、`fidelity_progress.py`、`fidelity_compare.py`、`fidelity_finalize.py`；原fidelity实验的`site/run_job.sh`、`setup_resources.py` | H65专用24阶段调度、25..60选模与结果汇总；没有BMCR阶段注册，旧实验已完成 |
| 当前DS3 | `tools/ds3_site.py`、`ds3_preflight.py`、`ds3_train.py`、`ds3_eval.py`、`ds3_select.py`、`ds3_dispatch.py`及`ds3_20260912/site/run_job.sh` | 已上传并运行的独立D1/Z0路线；当前唯一活动实验控制器 |
| 材料与报告工具 | `prepare_publication.py`（旧主目录）、`verify_results.py`、`retrospective_20260912.py`、`build_research_context_20260912.py`、`build_command_catalog_20260912.py` | 整理/核对已有产物，不构成新训练或新mAP实验 |
| 研究指令 | `docs/EXTERNAL_REVIEW_PROMPT.md`、`docs/BMCR_THREE_DIMENSION_RESEARCH_PROMPT.zh.md`及外部包prompts/agents | 给研究模型的任务文本，不是已经执行的GPU命令 |
| 资源与跟进 | `resources.example.sh`、本地`handoff_4090.md`、`h65-ds3-80/automation.toml` | 资源模板/使用说明及每30分钟主动跟进；跟进已ACTIVE，不自动把所有研究建议变成部署实验 |

活动源码根是`h65_clean_adatad/ds3_20260912`。旧主目录、`publication`、`fidelity`、`mainline`中有同名文件，但训练配方、参数、默认输出目录并不完全相同。主目录旧`full_train.py`不能当成当前修正版本；`mainline`也不是活动的已验证实现。

**下面这些是当前真实CLI，不是虚构的一键全实验命令。** GPU命令由站点环境和自有Slurm分配承载；表中列出接口并不表示本次再次执行。

| 入口 | 主要参数/前提 |
|---|---|
| `python tools/ds3_site.py` | 创建本目录到现有资源的链接；当前DS3配置已完成 |
| `python tools/ds3_preflight.py --backbone both` | S/B各两步真实GPU验证；当前已完成，不能当80轮成绩 |
| `python tools/ds3_train.py --backbone s --pilot` | 同一80轮轨迹的首100更新/第1轮 |
| `python tools/ds3_train.py --backbone s` | 读取本实验latest继续到80轮；B同理；不接受虚构的`--teacher`或`--epochs`参数 |
| `python tools/ds3_eval.py --backbone s --policy T24A --epoch 15 --metrics-only` | 需要对应已保存EMA；完整211/792精度，暂不profile |
| `python tools/ds3_eval.py --backbone s --policy Z24` | 零新增训练参考；不加载D1 aux checkpoint |
| `python tools/ds3_eval.py --backbone s --policy T24A --epoch 15 --profile-only` | 同一policy/epoch的完整metrics已存在才可补profile；不重跑mAP |
| `python tools/ds3_select.py` | 从已完成T24A候选选峰值并写选择JSON；不执行模型 |
| `python tools/ds3_dispatch.py [--once]` | 调度器；`--once`是一次调度tick，可能写状态并提交作业，不是只读查询；现有控制器无需另起 |
| `python tools/full_audit.py --backbone s` | 新BMCR必须指向自己的RUNS和对应修正warm；不能借用旧scales冒充新校准 |
| `python tools/full_train.py --backbone s --phase joint --variant bmcr` | 支持修正BMCR，但需新RUNS、修正warm20、资源和新audit scales；当前整套部署未接通 |
| `python tools/full_eval.py --backbone s --variant bmcr --milestone 25 --metrics-only` | 当前代码可评修正BMCR候选，但对应训练/检查点目前不存在 |

当前CPU DS3检查入口是`python -m unittest discover -s tests -p test_ds3.py -v`，此前7项已通过。测试文件的存在和其他脚本的语法检查，不等于对所有方案完成了真实GPU、完整mAP或性能验证。

实际DS3站点包装器是`ds3_20260912/site/run_job.sh`，使用`SLURM_SUBMIT_DIR`并设置DS3_RUNS_DIR。活动树中的通用`tools/run_full.sh`仍按`dirname "$0"`找源码根，直接作为sbatch脚本时可能落到Slurm spool目录；这是此前真实遇到过的目录问题，不能把它当当前已验证的站点部署器。旧主目录的`run_full.sh`则是硬编码旧站点版本，两者不要混淆。

**修正BMCR是当前明确的未完成项，不是在等旧附件再次批准。**

| 已有 | 仍缺 |
|---|---|
| 共享LR/真实裁剪边界修复、`--variant bmcr`训练能力、已完成修正warm20 EMA、warm状态结构核对 | 新实验RUNS/站点包装器与资源绑定、从修正warm重新生成并记录来历的utility scales、基于该warm/scales的真实BMCR预检、40轮joint、中间全测试、BMCR选模/匹配汇总和与DS3共用资源上限的调度注册 |

`bmcr_fidelity_20260912/`现在只有README和只读准备快照，没有训练/测试作业注册。DS3资源准备脚本只链接官方TAD与MobileNet；通用joint构造仍会先访问K400识别预训练，部署BMCR时也需处理对应资源/初始化路径。不能仅把Python命令放进现有DS3包装器就认为输出目录和资源已正确。

它的设计是：复用正确warm20 → 新训练集效用审计 → BMCR joint40 → 总25..60完整测试与峰值/终点profile，S/B各做一次。要检验的是**在相同正确配方下，BMCR的条件有符号效用是否比修正H65贡献监督带来净收益**。学习率修复不直接省去推理算子，也不能预先保证mAP上升。

**原来的原时间检测/粗细融合主线，属于部分代码草稿，缺口不仅是排队。**

本地`mainline_20260911`有7个已跟踪修改文件，另有未跟踪`h65/full/fusion.py`和`phase3_20260911/PLAN.md`。模型类接受`stage='recipe'/'time'/'fusion'`，几何恢复、融合、优化器和EMA的stage传递已有草稿；但是`tools/full_train.py:19-32`和`full_eval.py:175-195`没有`--stage`也没有传入stage，默认始终recipe。训练输出目录/元数据还没有按三阶段隔离；runtime配置没有引入后来修复的gt_boundary_validity。没有phase3专属验证、站点调度、训练/测试产物。不能把继承的旧train/eval文件叫作已经接通的三阶段命令。

| 方案 | 设计 | 原本要检验的假设 | 当前缺口 |
|---|---|---|---|
| recipe | K384、rank检测、global-TIA192、BMCR效用，先纠正配方 | 后续几何/融合比较是否有正确且一致的基础 | 老草稿只补LR，需合并裁剪边界及当前协议；修正BMCR正式实验尚缺 |
| time / G | native192按真实两帧支持中心恢复到原768；原GT/mask进入检测，不再rank GT和双重回映 | 原时间检测这一组网格/插值/监督几何变化是否改善结果 | stage CLI、配置、输出隔离、真实测试及完整20+40路线未接通 |
| fusion / F | 在time路线加96→D零初始化无bias普通粗残差；无gate；encoder梯度0.25，CNN仍由辅助动作任务训练 | 已有廉价全时域信息能否补足重分支缺失证据 | 新fusion模块只是未提交草稿；无完整训练/部署/效果证据 |

原计划每种结构各自20+40，保持相同BMCR设置，避免把新几何、融合、padding、KD同时改变。后来优先完成H65保真修正和DS3；此旧计划仍暂停，恢复时需更新用户后续要求的选模及配方。time不是“全部时间建模问题已经解决”，fusion也不能预先保证恢复缺失视觉语义。

**当前DS3模型/命令基本齐全，但训练、全量对照和最终解释尚未完成。** 原包24项计划在本项目的对应关系如下。原包文件全部写NOT_RUN是它交付时的状态，不能用它覆盖今天的真实运行记录。

| 原计划ID/当前名称 | 实际设计 | 要检验什么 | 当前实现/部署/验证 |
|---|---|---|---|
| D768 → D768G/D768L | 原global384与同权重local8的48clip密集参考 | 分离图转换损失与后续稀疏损失 | 已实现/登记；S完整测试69.0126%与67.5344%；B官方参考运行中，局部参考等待 |
| Z16/Z24/Z36 | 无aux训练，16/24/36完整clip＋原中心插值 | 零训练稀疏适配的代价、预算—覆盖曲线 | 已实现/登记；S43.2835/56.1065/62.5804%；B等待 |
| ZR24 | 同预算、有覆盖锚点的随机24clip | 均匀覆盖与随机集合是否同样有效 | S53.0022%，已完成；B等待 |
| AUX / D1 | 所有48clips/12层密集teacher前向，训练H0/H8/MLP近似与评分 | 密集辅助学习能否支持部署时少算 | 已部署；S36/80，B未开始；80轮有效性未验证 |
| PONLY | 只用廉价H0预测native全网格 | H0本身是否有可用于TAD的语义 | 已实现/登记，完整测试等待选中EMA |
| T24U | 均匀24clip深层＋其余H0 | 学到的补全是否优于Z24线性插值 | 已实现/登记，完整测试等待 |
| T24A | 预测效用选择24clip深层＋其余H0 | 学习决策是否优于同checkpoint T24U | 已实现/部署；S5..30每5轮已全测，尚未保持精度；其余等待 |
| D8 | 所有48clip真实执行前8层，H8输出 | 保留时间观察、减少深度是否更合适 | 已实现，代表性GPU预检通过，完整测试等待 |
| DAD | 24clip进8层，其中12clip进12层，其余H0 | 条件深度分配能否改善精度—成本关系 | 已实现/登记，完整测试等待 |
| S75/S50 | 仅9–12层heavy MLP处理75%/50%空间token；其他用近似器，attention/TIA保持栅格 | 部分昂贵空间更新是否可替代、真实省算是否值得误差 | 已实现/登记，代表性路径已预检；完整精度/时延未验证 |
| F000/F100/F010/F001/F110/F101/F011/F111 | T/S/D八组组合，共用选中aux EMA | 三因素是否互补，是否发生交互误差放大 | policy定义与调度对应已存在；全矩阵未测试完成 |
| STM | ToMe式局部merge/unmerge、attention token减少并恢复位置 | 比仅稀疏MLP更大幅的空间省算是否可保持TAD | 未实现对应模型/CLI，未登记作业 |
| CAL / C2 | 骨干/检测器冻结，辅助模块见到真实或虚拟稀疏混合图并做任务校准 | dense训练与部署混合特征分布失配是否是主要可修复因素 | 当前无C2训练入口/损失循环/冻结预算/任务注册 |
| JOINT / J3 | 稀疏路径中联合适配TIA/检测器等 | 稀疏适配后的能力上限与仅辅助学习的差距 | 未实现、未部署；不得归为D1 |

21个policy名称中，F100=T24A、F010=S75、F101=DAD复用同一策略评测；调度实际登记18个策略名称。73是作业阶段数量，不是73次独立训练。F000是全选框架控制，也不应被说成新的学习算法。

因子具体含义：T=1将prefix由48降为24；D=1让其中一半prefix继续最后4层；S=1让这些最后4层heavy MLP保留75%。F001因此是48个浅层clip中24个继续深层，F111是24个浅层中12个继续深层并使用MLP75。不能把三个保留率简单相乘为全模型FLOPs。

当前最终PONLY/T24U/D8/DAD/S75/S50/因子测试都依赖同一条D1的80轮训练及全部T24A候选评测，再使用同一选中EMA。它们WAITING主要是依赖设计，不代表没有实现。这也意味着目前还没有用早期checkpoint完成PONLY/T24U/D8等分支诊断；CLI已支持这些policy，调度只是没有提前执行它们。

**外部命令包要按“文件真的存在”和“它调用的模型已经存在”分别看。**

| 来源 | 已保存的命令/任务文件 | 应如何解释 |
|---|---|---|
| H65_agents_pack | `scripts/bootstrap_sources.sh`、`check_sources.py`、`run_research_agents.sh`、`run_approved_implementation.sh`；`prompts/01..06`与`AFTER_APPROVAL_implementation.md`；`TRAINING_HANDOFF.md`、manifest/approval模板 | 真正存在的是研究/来源/未来实现框架；目标锚点04c35a3与官方1aa8，不是当前346d09d。包本身无模型训练代码或训练结果 |
| H65_DS3包 | `COMMANDS_zh.md`；`bin/bootstrap.sh`、`source_audit.py`、`run_agent.sh`、`run_experiment.py`、`verify_bundle.py`；`agents/00..07`；CLI/route合同、experiments、launch/sbatch模板 | 原`tools/bata/ds3_validate.py`、`ds3_profile.py`和`configs/.../ds3/*.py`明确是待创建路径；当前项目以h65/ds3与tools/ds3_*完成部分同等功能，不能照旧路径直接运行 |
| BMCR第一次意见包 | `independent_checks.py/json` | 手工转录常数的算术/谓词核对；不是重新训练或mAP复测 |
| BMCRT第二次意见包 | `checks.py`、`arithmetic_checks.json` | 同属算术参考，Markdown/HTML报告不是可直接部署模型 |
| 当前外部研究Prompt | `docs/BMCR_THREE_DIMENSION_RESEARCH_PROMPT.zh.md` | 固定Git提交的研究请求；没有执行其中候选实验 |

原H65包M0来源/几何、M2思想机制、M3训练集效用审计、M4有符号效用监督已有本项目对应成果；M1中的官方完整重训/独立均匀训练按用户范围没有执行。M5要求保持global-TIA的空间/深度条件执行，当前DS3-L是local8变体，不能说严格完成了那个原合同。M6视频自适应总预算桶未实现：Z16/24/36是固定预算控制，DAD是固定数量下分配深度，短窗口min(K,L)也不是学习动态总预算。

原DS3包预算AUX3000、CAL3000更新、多个seed及20/100计时只是原始提案；当前D1经用户要求改为8000更新/80轮、seed3407，计时采用项目实际5次预热/20次采样。原包的NOT_RUN、AGENT_MUST_CREATE、审批/hash字段不能被伪造为执行回执，也不成为本任务再次请求授权的理由。

**近几轮讨论还有这些没有命令化的设计，不能把建议文档算成实现。**

| 设计/实验 | 想验证的科学问题 | 目前还缺什么 |
|---|---|---|
| BMCR/H65→DS3知识桥接 | 已学任务表示、边界先验、条件效用或teacher预测能否防止新稀疏分支大幅掉分 | 迁移接口、真实时间teacher对齐、具体损失/预算、训练入口与对照；目前只有研究说明 |
| 训练内clip特征缓存反事实 | local clip独立时，能否少跑重骨干而得到可靠任务效用标签 | 缓存/真实compact一致性核对、新clip/depth动作定义与标签/scales、C2集成；旧逐帧utility不能直接代入 |
| 保基线初始化/渐进替代 | 是否可先保持已验证输出，再学习廉价分支增量，减少初期坍塌 | 当前H0覆盖替换并不保证Z24/H65/BMCR初始等价；缺残差/路由初始化和课程实验 |
| 在线权重与EMA诊断 | 当前低分是否部分来自EMA滞后，还是分支/路由/目标失配 | load_aux有内部选择能力，但ds3_eval尚无online-state CLI或已完成配对评测 |
| 同配方逐帧/tubelet/micro-clip/16clip比较 | 粒度本身、运动统计、覆盖与时间图各起什么作用 | 尚无匹配训练/坐标/TIA/预算的统一入口与实验；Z0对比已训练H65不是该控制 |
| 检测输出自蒸馏/KD | 真实时间分类/定位teacher能否保持TAD能力 | 无通用输出KD模块、可微学生输出/匹配与相同teacher预算对照；旧BMCR是效用监督，不是此KD |
| 全时间128/多分辨率 | 保留时间覆盖而减少空间分辨率，是否比大块丢时间更合适 | 目前仅算子估算；没有登记配置、训练适配或新精度/时延实验 |
| 保留global-TIA、条件执行贵算子 | 廉价全局时间状态能否缓解local切换损失，同时减少attention/FFN | 现ClipEngine拒绝global scope；完整中间栅格状态、稀疏更新/恢复及新训练图未实现 |
| 扩大空间稀疏层范围/动态总预算 | 更早层或视频难度驱动预算能否进一步省算且不放大误差 | 当前仅后4层MLP和固定预算；没有新执行器/训练与实际计数 |
| 完整流水线瓶颈及等价工程优化 | 解码、传输、路由、检测、NMS分别占多少；哪些优化真提速 | 目前有总E2E及GPU模型/骨干计时，缺完整分段归因与解码复用/向量化等新比较 |
| DS3全矩阵最终比较/报告 | 选中、60/80终点和所有策略在精度/算量/时延上的关系 | 当前有单项eval、T24A选模及材料汇总；没有已完成最终矩阵，也没有专门的全流程最终比较入口 |

这些研究候选并非全部已经被用户指定为下一批必须全训。真正明确的执行欠项应先与当前已部署D1分开列清：修正BMCR需要落地；旧G/F草稿需要补接口与协议才可恢复；C2/J3及桥接等需要落实具体新设计。报告中的“想验证”是可被实验支持或否定的假设，不是已取得收益。

**本次审查中，几个容易误报的点已经校正。**

- mainline的EMA已经传递stage，不能把这一项列成缺失；真正缺的是训练/评测/audit命令选择、输出隔离及后来的边界配方。
- mainline存在未跟踪的fusion.py，但没有完整三阶段执行链；不能把“存在模块”写成“已部署融合实验”。
- `ds3_dispatch --once`可能提交作业；读取deployment.json才是状态查询。
- 外部模板缺`tools/bata/ds3_*`不意味着当前D1全部未实现；当前有独立、已验证的替代入口。
- DS3工程预检通过不等于TAD性能保持，当前截至第30轮的六个完整候选测试仍未支持这一结论。
- 已完成的旧控制器、旧报告中早于测试生成的状态字段和仍保存的命令文件，不是再次启动历史实验的理由。
