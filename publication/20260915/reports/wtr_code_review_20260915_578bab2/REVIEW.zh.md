# WTR 当前实现交叉审阅

审阅日期：2026-09-15。审阅者：本任务主代理与六个只读审阅代理。与实现任务「实现 Raw-v1 并部署并行实验」进行了具体代码问题讨论。本文记录独立审阅，不修改实验源码、部署或队列。

**最新复审结论：Core `40552945ad5d56b7404f833cd86f998e8558e8b1` 的 D-U/D-V/S-U/S-V 均通过代码与协议审阅；Raw `27d557ee3bd38a52aec8caaf16860a95957b44b6` 的 bank/Value 修正通过代码审阅。** 初审四项问题及进一步确认的收益排序问题均已修正。此 PASS 仅准入既定 GPU 验收流程，不表示本版本已完成 GPU 验收、训练或科学 gate。

下面保留初审出处与问题，文末记录修订版复核及可消费的逐配置回执。

## 审阅版本与结论

- Core：`578bab2a97431a908764fa0ab30a53c47fdfa336`，分支 `codex/wtr-fasttrack`。
- Raw：`1259d25bd49291aaf6ef0aceb0bc07ecae0aa222`，分支 `codex/wtr-raw-v1`。
- 审阅依据：当前 Fast-Track v3 计划及其每实验交叉审核要求；Raw 固定 episode、共同 preview、有限扩域、真实 swap 与 holdout 合同。旧串行实施优先级不覆盖最新授权。
- 本目录 `core/`、`raw/` 保存对应 Git 提交的只读审阅快照；文中代码链接指向这些固定版本，避免实现工作树变化后出处失效。

**初审结论：主要执行链正确，存在四项需要修正的问题；当前代码是固定容量的分阶段实验原型，不能登记为最终模型全部完成。** 实现者已接受四项修正，并已归档旧 D 候选回执，准备新版本复核。旧结果不用于解锁下游科学课程。

## 确认的问题

### F1 — P1：Core Value 尚无独立的 router-label holdout

[paper_train.py:177](core/tools/paper_train.py#L177) 对全部 200 视频采样出的 microbatch 直接执行 CF 并加入 Value loss，没有 fit 视频筛选。模型训练可运行，但这些视频上的 Value 排序不能再作为 out-of-fit router 证据。

影响 D-V、S-V、DS-V、T-V、TDS-V 的科学准入。D-U/S-U 的算子执行没有因此变错。

已与实现者达成修正：Value CF 回归限制到相同的 160 fit 视频；20 calibration、20 holdout 只用于冻结 checkpoint 的 re-query 与评估。detector 的 200 视频训练合同保留；当前配置 Scout 本身冻结。应称为 **router-label holdout**，不能声称整个检测模型没有见过这些视频。旧 D-V 头已经接触保留集标签，因此不能从它恢复新科学课程。

### F2 — P1（TDS-V）：D/S 标签收集绕过了当前 Temporal policy

[operator_training.py:18](core/h65/paper/operator_training.py#L18) 固定 `apply_refiner=False`；而 TDS-V 普通前向会在 [model.py:202](core/h65/paper/model.py#L202) 执行 Temporal Value refinement。因此 TDS 的 D/S CF 实际来自 Uniform T support，不能直接解释为当前部署策略产生的 D/S decision state。

当前 D-V/S-V 使用 Uniform T，不受此项影响。实现者接受：TDS baseline 先运行当前 T policy，changed 分支再复用 baseline 的 selection/preview，并仅干预指定 D/S action。D 干预后的 S 及后续策略仍需重执行。

### F3 — P2：Raw bank 中断后可能过早获得 publication eligibility

[raw_run.py:141](raw/tools/raw_run.py#L141) 按 domain/state 逐组保存；[raw_train.py:86](raw/tools/raw_train.py#L86) 只验证视频集合是否完整。因此在最后一个视频刚写完 O/state0 后中断，是实际可达的半成品状态：视频集合已经完整，其余 domain/state 尚未写完，训练入口仍可接受，且满足 holdout 条件时可写出 `publication_eligible=True`。

这不是针对人为篡改的防御要求。修正应使用已有 shard 完成回执与预期 group 清单，拒绝未完成采集；允许合法候选不足导致 action 数少于上限，不机械要求凑足 6400。

### F4 — P2：课程计划中的评测点与真正配置不一致

[wtr_fast_plan.py:38](core/tools/wtr_fast_plan.py#L38) 登记 `10/20/30/40/50/60/80`，但 [runtime.py:25](core/h65/paper/runtime.py#L25) 和实际配置要求 `10/20/40/60/80`。计划会承诺不存在的 30/50 轮评测产物。

实际训练及内联评测读取配置，尚未发现调度因此等待卡死；不能把可能的卡死当成已发生的故障。实现者接受从 config 读取单一评测点来源。

### F5 — P2：Raw 回归缩放改变了任务收益排序

进一步讨论确认，旧 Raw `utility` 与 holdout target 都先除 fit RMS 再求和，实际优化的是重新加权的 cls/loc 取舍。它甚至可能改变 STOP：实际两项收益为 -2 与 +1，总收益为 -1；若 RMS 为 100 与 1，标准化后得到 +0.98。

修订版统一按实际 `gain_cls + gain_loc` 排序、判断 STOP 和计算 regret，RMS 仅用于回归训练的数值缩放。对应回归测试已独立通过。

## 已确认正确的实现

1. **D/S 计算语义及容量。** D 是 packed token admission；DS 的 F50 以全部合法 token 为分母，并在 D admitted 集内按 native-time 整数 quota 分配。S⊆D。D-U/D-V 选中 token 执行 heavy attention 与 heavy FFN，未选中执行两类 light；这是 policy-level D，不是 attention-only。
2. **反事实确实执行。** 两支运行当前 student 的完整后续计算；D 干预后重新计算 S；S 干预保持共同 D，并在相同 native-time quota 内交换。cls/loc target detach，loss normalizer 恢复，controller 有梯度、输入 state 无梯度。
3. **执行与恢复。** 先 gather selected RGB，再按 16 观察打包；路由层为 4/6/8/10，保留 full KV 与 global TIA。Raw 显式保留 384 frames → 192 anchors → 384 Cross positions → 768 detector positions。
4. **Raw 公共输入隔离。** descriptor 使用共同 cheap preview 和公开几何/support，不读取 GT 或候选 heavy 输出；没有 domain one-hot。每次 swap 重排完整 support，并记录 changed pairs/packs/span。
5. **成本。** encoder、routing、light、TIA、recovery、head 的矩阵/卷积账本有实际计数与校准对照，未确认主推理 FLOPs 漏算或重算。decode、NMS、数据等待与 resident-RGB latency 的口径有区分；CF forward ledger 不应被扩写为完整训练 FLOPs 测量。
6. **保存与恢复。** trainable state、优化器、scheduler、EMA、RNG 恢复路径存在；Fast-Track 的 fresh reload 确实新建实例，不只是原实例 reload。旧课程取消标志被 dispatch 尊重。

## 与最终模型的距离

固定单 plan 的 D/S Value 实验可以回答局部计算价值是否可学，不需要先实现全部最终架构。但是：

- D/S head 当前输入是当前深层状态、pack 聚合、合法性与几何；尚无显式共享 Scout coarse context、完整执行历史或容量 plan 输入。因此不能把它描述为最终的 Shared Coarse Evidence + Joint Capacity Planning 已完成。
- Temporal 共用 Raw descriptor 代码；初审发现容量字段写死 1/1，修订版已由 Core 两个调用处传入真实 plan。独立单 plan 训练仍不自动提供跨 plan Value 泛化，联合预算能力尚未完成。
- Graph G0/G1、RISE-A/B、FVD、动态预算与 Raw matched retraining 的正式科学 runner/证据尚未完成，不能登记 PASS。它们不阻塞当前单轴实验。
- Raw 的收益口径已按 F5 与 Core 统一；fit-only 归一化保留在回归训练中。

## 每实验初审状态

|实验|代码/协议结论|当前证据边界|
|---|---|---|
|D-U|算子实现 PASS；随 pair 版本更换复核元数据|旧候选两次真实更新、新实例 reload、no-GT 均通过|
|D-V|需修正 F1|算子与 controller 梯度正确，旧候选技术验收通过；不能用作新 holdout 科学结果|
|S-U|算子实现 PASS；完整 GPU 验收未完成|CPU 容量/嵌套合同通过|
|S-V|需修正 F1；完整 GPU 验收未完成|没有把 A100 排队当作已验收|
|DS-U/DS-V|等待真实单轴开发 gate；DS-V 受 F1 影响|不能仅凭配置注册称完成|
|T-V|需完成 F1 修正及自身完整验收|共享 descriptor 接口存在，未获得完整课程 PASS|
|TDS-V|需修正 F1/F2 与容量 descriptor；等待 gate|不是已完成的最终 Core|
|Raw reader/bridge|首个完整视频接口 PASS|1 个训练视频、3 个窗口，不能替代 6 视频及全数据科学验证|
|Raw bank/Value|需修正 F3；domain/learnability 未完成|禁止半成品 bank 进入 publication eligibility|
|C0|严格 reference parity 尚未提供完整证据|未登记整体 PASS|
|Graph/RISE/FVD/DB/Raw matched retrain|未完成|未实现或未有证据不是 FAIL，也不是 PASS|

## 独立验证及代理讨论

- 4 项 Raw CPU 合同通过：候选域与物理 uniform、共享 tail validity、完整 tubelet 重配对、GT 投影/no-op。
- 3 项 D/S CPU 合同通过：精确容量/嵌套、交换守恒、controller-only 梯度。输出见 [cpu_operator_review.txt](cpu_operator_review.txt)。本地 Torch DLL 无法加载，因此将固定快照的纯 CPU 测试以内存脚本交给现有 4090 Python 环境执行；禁用 GPU，没有写远程项目文件或启动作业。
- 独立读回旧 D-V/D-U GPU 回执：均真实更新 2 次、fresh-instance strict reload 与 no-GT probe 通过；对应作业 1290324/1290325。回执现存实现者的 `review_candidates/578bab2_20260915_033052/`，摘要见 [gpu_candidate_receipts.json](gpu_candidate_receipts.json)。这些属于旧候选技术证据。
- Raw 已有回执验证 1 个完整训练视频的 3 个窗口，legacy/bridge 官方 AP cache 相同；它证明接口一致性，不证明 Raw 科学增益。
- 六代理分别审阅 Core 执行、CF/梯度、成本、训练/恢复、Raw 输入、Raw bank。主代理亲自读设计、核验代码疑点并作最终判断。实现者接受 F1–F4，正在修正。
- 已剔除误报：Raw proposal 限制在固定 episode 内是正确设计；DecordInit 打开容器不等于提前 materialize 所有 heavy RGB；本 THUMOS 配置 offset 默认为 0。padding 参与继承的 dense/full-KV 路径不是本轮已证缺陷，WTR D routing 还明确关闭 incoming-attention scorer，不能据此声称新 D Value 由无效 attention 分数决定。

## 修订版复核与最终审阅回执

独立检查 Core `40552945ad5d56b7404f833cd86f998e8558e8b1` 相对初审版本的修改；固定副本位于 `revised_4055294/`。Raw 新提交 `27d557ee3bd38a52aec8caaf16860a95957b44b6` 的相关 Raw 源码、训练入口与回归测试，与该副本对应文件没有差异。

- F1 已修正：分区验证要求 160/20/20、互斥且覆盖全部训练视频；在线 CF 条件与技术 preflight 的 full-window probe 均限制 fit。普通 detector 训练仍覆盖 200 视频；配置与 metadata 记录 router-label holdout 的范围。
- F2 已修正：有 Temporal Value 时，D/S baseline 执行当前 T policy；changed 分支复用它的 selection/preview，继续对目标 D/S 交换完整重执行。
- F3 已修正：读取所有 manifest 对应的 completed receipt，验证全部 shard 齐备及 group inventory 相等；中断和缺 group 不能再仅凭视频齐全获得训练资格。没有强凑 action 数。
- F4 已修正：课程 plan 的评测点直接来自实际 config，均为 10/20/40/60/80。
- F5 已修正：Raw routing、STOP、holdout regret 统一使用实际 cls+loc；另外 Temporal descriptor 的真实 D/S 容量参数已接通。
- 本次只重跑受影响的 3 项回归：收益符号、真实 plan 字段、最后视频半成品 bank 拒绝，全部通过。见 [cpu_revision_4055294.txt](cpu_revision_4055294.txt)。未重复未改变的 Raw 同帧 GPU 验收。

|配置/范围|本次代码与协议结论|尚未由该结论证明的事项|
|---|---|---|
|D-U|PASS|新 SHA 的 GPU preflight、正式 80 轮、最终 mAP|
|D-V|PASS|新 SHA 的 GPU preflight、独立 holdout 收益、正式 80 轮|
|S-U|PASS|新 SHA 的真实 GPU preflight 与完整训练|
|S-V|PASS|新 SHA 的真实 GPU preflight、独立 holdout 收益与完整训练|
|Raw bank/Value|PASS|6 视频 gate、完整 mini-bank、信号评议、完整 bank 与 learnability|
|T/DS/TDS 完整课程|未单独授予准入 PASS|各自完整验收及真实开发解锁条件|
|最终模型全部组件|未完成|共享粗上下文/联合预算及可选增强仍按证据推进|

已生成四个逐配置 JSON 与一个 Raw 审阅记录，位于 `revision_4055294_receipts/`。`scope=code_and_protocol_review`，绑定精确 Git SHA、配置 ID、本审阅任务和证据路径。实现者负责复制到自身准入目录；审阅者没有直接修改运行队列。新 `admission` 分别检查代码审阅和科学依赖，`paper_course` 继续负责本版本两更新、新实例 reload 与 no-GT 验收。

总计独立复跑 10 项 CPU 测试通过：初审 Raw 4 项、D/S 3 项、修订回归 3 项。旧 D pair 的 GPU 技术回执另行核验。没有由这些技术通过推断论文收益或将旧候选结果混入新科学版本。

## 2026-09-15 04:04 执行交接补记

实现任务确认逐配置审阅回执已原样复制并被准入流程消费。以下为实现任务回传、并与其 [EXECUTION_STATE.zh.md](../wtr_fasttrack_20260915/EXECUTION_STATE.zh.md) 对照的历史快照，不作为实时状态或科学效果结论：

- Core `4055294`：新 D-V/D-U 作业 1290328/1290329 均通过本版本两次真实更新、fresh-instance reload、no-GT 验收后进入正式课程。04:04:14 +0800 分别为 227/213 updates，均在 epoch 3；D-V 的 23 条 Value 标签中 `labels_outside_fit=0`。
- A100 的新 S-V/S-U 作业 248205/248206 仍 PENDING(Priority)，不登记 GPU 验收通过。
- Raw `27d557e`：独立 revision 目录的作业 1290331 执行六视频验证→mini-bank→诊断；没有据此解锁完整 bank 或 publication。原 A100 排队 Raw 作业已由实现者取消以避免重复。
- 旧 `578bab2` 候选没有作为新版本 resume；其结果保留归档。科学 gate、完整 mAP、各增强 runner 的未完成状态不变。

本次审阅与实现任务交接已完成。后续实质代码修改仍须对应具体配置与精确提交重新审阅；现有 PASS 不扩大到尚未审核的实验。

## 2026-09-15 11:10 内联评测修复追加审阅

两条 D 课程已完成 epoch 10，但首次评测因执行 plan 内的 `wtr_geometry` Tensor 无法被 JSON 序列化而失败，尚无全量 mAP。此为初审未发现的日志集成错误。补丁 `6e2fc7f78424e2485e8d33b079fac926eb0cf4a6` 仅改用既有公开 trace plan 并记录 evaluator 版本，已完成独立定点复核。

允许保留模型/训练 SHA 4055294，从各自 latest 恢复并先补 epoch 10 EMA 评测，不重复前 10 轮或两更新 preflight；评测运行另记 SHA 6e2fc7f。实际全量评测成功仍待原课程完成验证，科学收益未授予 PASS。详见 [定点复核报告](EVALUATION_FIX_6e2fc7f.zh.md) 与 [机器回执](evaluation_fix_6e2fc7f.json)。

## 2026-09-15 11:33 交接与 Raw 证据边界

已完整阅读实现任务的 [11:10–11:33 进度报告](../wtr_fasttrack_20260915/progress_20260915_1100/REPORT.zh.md)，并向实现者反馈以下判断：

- D-V/D-U 的补评续训作业 1290651/1290652 仍 Priority 排队，不能写成已完成修复后的全量评测。两条课程各保留 epoch 10/1000 更新；目前尚无新 Core mAP。
- S pair 的 A100 待执行作业经确认后由实现者取消并迁至 4090，1290654/1290655 仍 Priority 排队，尚无训练成绩。本审阅没有操作资源或队列。
- Raw 六视频/19窗口接口验收与24视频mini-bank采集完成：90个domain/state组、346个实际swaps，replay最大误差0，符合先300–500条的试采规模。这些是接口与标签可重放证据。
- 候选域确实扩大，但R+的180个实际动作仅24个插入off-grid帧；38/45个配对状态的O/R+动作集相同，holdout更是8/8相同。因此当前holdout没有提供区分两域的动作对照，无法检验扩域的增量，也不能据此否定Raw。
- 同意先在training/development内补足小规模off-grid动作覆盖，维持共同support和清楚的查询预算，再判断扩bank与Value拟合；不直接复制当前动作生成方式到6400条。当前尚无Raw learnability或完整检测增益PASS。
- Atlas S的有限时间分配headroom为正；B的8/12组CI跨零。现报告对GT辅助、额外查询开销、逐预算视频CI和未验证廉价Value的限制表述正确，不把它们改写为新Core训练成功。

后续最先决定科学准入的仍是：D pair实际完整评测、稳定checkpoint的router-label holdout ranking/regret，以及具有实际新增观察覆盖的Raw小bank。代码审阅PASS不因此扩大。
