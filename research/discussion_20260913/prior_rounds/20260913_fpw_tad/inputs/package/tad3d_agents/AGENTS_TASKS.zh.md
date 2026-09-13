# 可直接交给 agents 的科研实施任务书

## 0. 总控任务（整段可复制）

你是本项目的科研实现总控agent。基于`yuzbo/BMCR-T-AdaTAD@239d098cd899936c35989fae6243c70259a85adb`，实现、验证并分析FPW-TAD：单candidate frame选择、原时间轴latent解码回写，以及保持完整中间状态的深度/空间条件贵算子。先读同包REPORT.zh.md、SOURCES.md、SOURCE_MAP.md、plans/experiments.json、FIGURES.md。当前任务不保证任何mAP/速度收益；必须允许证伪，记录负结果。禁止启用T24/whole-clip时间选择路线；但H65选帧后的16-observation计算打包允许保留。

在独立git worktree实施，不改旧实验权重、结果或上游源码的语义，不覆盖`h65/full`、`h65/ds3`旧实验。新增模块放`h65/frame`，工具放`tools/frame_*`。对旧类只添加默认关闭的非破坏性native提取接口或read-only hook。所有新行为均由新recipe控制；运行路径、source commit、checkpoint identity、teacher来源、数据split、成功更新、原始结果和测量scope写入manifest。

源码与checkpoint可访问性不足时，先交付可运行CPU单测、配置、blocked reason；不要伪造checkpoint加载、Slurm成功、训练完成或211测试。不要默认分支替代固定snapshot。旧weights保留真实旧配方；修正BMCR用修正warm20和重新审计尺度再joint40。新恢复分支可先用已完成最强frame模型做readout实验，不能把待完成BMCR成绩填入表。

逐项实现以下角色任务，按照依赖次序合并。首先发布API契约和dry-run CLI，再运行技术测试，之后在获得明确GPU执行授权、资源核实和OWN job检查后才提交作业。本包不授权启动或取消任何服务器作业。禁止依据历史快照中的job ID直接scancel，也禁止无限重试。算法身份测试是正确性要求，不是自造1pp等研究审批门槛。

每次完成一个阶段，请输出：修改文件/符号；单测和实测receipt路径；新增与旧行为差异；真实状态；当前可得结论和不能得出的结论；下一项最小判别实验。不得为了论文结论把没改善的模块隐藏，不把只省MAC称为加速，不把oracle feature replacement称为真实frame计算收益，不把外部官方teacher KD称为已有self-KD。

## 1. Agent A：固定证据与fidelity负责人

**修改权限**：`tools/frame_audit.py`、`tests/frame/test_legacy_identity.py`、`research/frame/manifests`。其他科学模块只读。

1. 确认git基点及资源清单。阅读context/latest_snapshot/retrospective、P2/P4/P5/P6/P8/P9/P10；本包的SOURCE_MAP是导航不是替代原文。
2. 审计候选时间单位、native/检测长度、每个checkpoint配方、teacher来源。当前本地可能已有后续结果，必须另登记新commit/新证据，不能混进固定快照表。
3. 修正BMCR方案原样保留warm20→audit→joint40，不同时改变16cell/decoder/spatial/depth。独立工作目录复用合法warm，不覆盖旧checkpoint。
4. 给`FormalH65`新增native read-only捕获，比较旧`encode`、legacy预测、GT映射、NMS前回映。full/partial/short以及odd有效帧数量全测；FP32/AMP容差分别记录实际值与理由。
5. `FormalScout.condition`精确向量化可作为systems-only分支，先逐元素比较partner/feasible/utility/selection再计时。

**输出**：`baseline_manifest.json`、`legacy_identity.json`、`checkpoint_inventory.json`。正确读取资源前不能填`weights_loaded=true`。

## 2. Agent B：时间来源与decoder负责人

**修改权限**：`h65/frame/{geometry,decoder,readout}.py`、`tests/frame/test_{geometry,decoder}.py`。

1. AnchorBatch数据结构包含anchors、两个source candidate IDs、两个原frame时间、valid contributors、center、span、packed位置、last_heavy_depth、spatial_quality。
2. 输入anchors `[B,A,C]`，A=K/2；不是original native的直接子集。原查询 `[B,384]`来自frame_inds；存candidate/原frame/秒的明确单位与offset转换。
3. 实现physical interpolation作为单独baseline。当前`restore_tubelets`直接恢复candidate轴，不可重复插值或当native384。
4. 实现两层cross-query残差decoder，memory192→query384，context用已有scout。输出新head零初始化，必须等于**新插值基线**；不得声称等于旧rank预测。firststep上游梯度零可接受，第二步以后必须验证参数更新。
5. Query chunking要在没有query-self-attn、跨query norm/conv时才声明等价；变长memory要padding mask。invalid query在loss/head阶段正确mask。
6. 使用official原detector768读出；旧rank头独立保留。加入桥接norm/通道映射，不能直接把old pooledfeature当same teacherlatent。
7. 官方VideoMAE decoder复用是独立init实验：只在真实读取完整pretrain checkpoint后做key/shape/depth/norm审计。self-attn与cross-attn转换另有manifest，不能宽松load后叫严格兼容。random对照须同结构/宽度/参数数量。

**输出**：R01/R02/R03最小配置、shape contract、native/readout equality tests、new-head initial mAP待实测字段。

## 3. Agent C：任务损失、teacher与router负责人

**修改权限**：`h65/frame/{objectives,teachers,utility,router}.py`、`tools/frame_intervene.py`。

1. 外部officialglobalteacher严格冻结并eval，只teacher forward no_grad；frozen detector不能切断studentfeature梯度。
2. 默认GT+feature，两条独立可关的差分/输出KD消融。cls用独立sigmoid/Bernoulli语义，reg用物理端点/同轴distance。保留合法裁剪边界标记。
3. 三种teacher明确命名：`external_official`、`frozen_anchor`、`shared_full_student_stopgrad`；EMA_eval与EMA_utility_teacher独立。fullbranchself-KD增加GT与计算统计，不能偷偷double预算不记账。
4. 构造表示修复标签`repair_delta`与真实执行标签`action_delta`：frame固定K交换必须真实重新组装并编码；global图不允许只替换一个teacher位置冒充重算。depth/spatial也要从真实前缀状态重新执行后缀。
5. 干预使用相同样本/增强/teacher状态/后处理，记录cls、reg、漏检惩罚、actualMAC、actualroute变化。训练loss归一化状态需固定，不让两个counterfactual查询改变normalizer。
6. 学习有符号效用与误差校准；按训练video OOF估计尺度/超参，然后用全部200训练，不强制160/40。真实动作查询预算可降低频率，但记录调用数和GPU时长。
7. 先保留原局部伙伴，后比较跨cell/受coverage约束全局候选。hardtopK不可微；使用干预监督或明确STE/transport近似，不让GTloss自动背锅。S0→S1做有限改动，记录局部gain之和vs组合gain。
8. 推理禁止GT参与routing。边界prior来自scout，maxgap来自几何；uncertainty只可在实际task-error校准后解释。

**输出**：interventions记录、utility calibration、noGT-inference测试、teacher-student-gradient测试、各loss增量消融。

## 4. Agent D：真实深度/空间执行负责人

**修改权限**：`h65/frame/{engine,gates,cost}.py`、`tests/frame/test_engine*.py`。

1. 首版复用frame-selected packedK骨干和globalTIA(K/2)。状态 `[B,K/2,h,w,C]`必须保留到各层TIA后，不能用空间池化结果代替非线性Adapter输入。
2. 前8层全算，后4层可按native支持位置选heavyQuery和FFN；D是贵更新深度而不是所有位置完全停止通信。空间门先FFN，light残差可用低秩预测或identity对照。
3. 顺序严格：attn残差→FFN残差→scatterfullgrid→Adapter。Adapter本身已含identity。Fullgate回旧图，before/afterlayer tensors和最终输出都比。
4. 将attention qkv权重按Q/K/V切片：Q仅selected，K/V完整当前状态，output projection仅selected。先验证sameinput selectedquery数值等价；不要通过全qkv再gather骗projection节省。
5. 动态stage/空间token变长pack按当前实际batch统计，重复/invalid索引校验。以per-time空间块均匀配额为简单baseline，再学分配，不强制多个新router。
6. 实现dense-mask C2与compact J3两forward；参数复用、同state/gate/GPUfp模式时逐层验证。训练dense-cache必须来自相同学生输入；任何未满足状态一致性的teachercache只能是近似实验。
7. 和静态blockdrop/PBD式、分辨率128、结构化/非结构化FFN比较。只有实现真正省算且task误差合理，才进入联合三维策略。

**输出**：layerwise trace、FP32/AMP parity、真实hooks成本、静态vs动态的微基准。没有project GPU时只能输出synthetic CPU证据。

## 5. Agent E：实验编排与安全部署负责人

**修改权限**：`tools/frame_{train,eval,profile,plan,dispatch}.py`、`configs/frame`、`research/frame/plans`。

1. 所有工具支持`--help`；train/eval支持`--dry-run`打印完整配方/资源/成功更新上限且不创建CUDA上下文；dispatch默认只生成manifest与shell。
2. 新recipe与resume严格匹配数据、teacher、backbone、K、decoder/init、门策略、precision、optimizer、schedule。旧recipe权重作为初始化不复用错配optimizerstate。
3. B01优先；R阶段固定selector/backbone，decoder稳定后才开adapter/router，明确每组LR；seed3407主协议。技术2步和机制pilot不冒充fulltraining。
4. 列出每个实验额外teacher forward、缓存创建、学生训练步与GPU小时；共用warm/teacher缓存只计一次但披露分摊方法。
5. 正式eval验证211视频ID/792窗口，完整mAP五阈值；peak选择按既定用户规则披露并保留terminal。postNMS完整检测与preNMS诊断均保存。
6. 只在用户明确授权执行后，读取当前OWN队列和环境，资源按现场可用量核实。不得读取别人的allocation、取消别人/历史不明作业或无限重试。执行授权前所有command都是待执行计划。
7. profile用同GPU交错A/B、同输入与scope，full/partial/short；骨干、GPUmodel、decode/H2D、NMS/E2E分开。异常不删、非矩阵算子列出、方法和implementation-only优化分开。

**输出**：作业DAG、dry-run receipts、授权后ownjob_manifest、每experiment的results.json。失败写`failed`、未测null，不能0填缺失结果。

## 6. Agent F：统计、论文图表和反证负责人

**修改权限**：`tools/frame_analyze.py`、`paper/{figures,tables,analysis}`。

1. 按FIGURES.md和plans/claims.csv建立claim→experiment→rawdata→plot链。只读取complete/verified符合协议的记录作主结果；toy、算术预测、partial单独目录。
2. 五阈值AP/平均、short/boundary/repeat、coverageholes、confidence、DETAD分类全分析。定义durationbins时先固定，不挑对我们有利的阈值。
3. CI：每视频paired bootstrap重算AP，抽中重复video须改ID；seed方差与samplingCI区分。测试peakbias不能靠bootstrap消除。
4. 相关图utility必须对应actualinterventions，teacher repair分色/标记另图；不要用attention可视化或tSNE宣称因果。
5. 使用真实视频帧/annotations/trace；先规定案例选择规则，展示失败。模型推理不见GT，绘图离线joinGT合法。无原视频时拒绝生成伪时空案例。
6. 每panel单独矢量PDF/SVG+PNG，单栏3.28125in/双栏6.875in；建议8–9pt图内字，不修改官方template。Matplotlib无seaborn；defaultcolorcycle+distinctmarkers，真实数值轴/单位/CI。数据为null时拒绝画结果点。
7. 主表保存teacher/supervision/更新/scope脚注，别把论文@0.5值抄成ours平均mAP，别在不同硬件/node/cohort混算speedup。

**输出**：6主图组/12补充图组/4主表的source manifest、完整结果分析与反证段落。

## 7. Agent G：独立review与paper负责人

**修改权限**：`paper/main.tex`、`paper/supp.tex`、`research/frame/review`；实验与数据只读。

检查所有“首次/保持/更快/显著/因果/泛化”字样是否有证据。对比PBD、CrossMAE、ExpeditingViT、DyT、CoLT5等，不将已知子机制包装为首创。明确训练D1/C2/J3、external/selfKD、真实/估计cost。给出证据不足情况下可成立的最弱结论，提出一个最小反证实验而非无限堆模块。

最终论文不预写成功结论，不遗漏负结果。CVPR2026版式仅作模板，投稿届官方规则重新检查。必须加入单数据/单seed/测试选模局限；第二数据集与多seed有条件扩展，不伪造完成。

## 8. 合并与接口顺序

A(native/metadata合同) → B(固定selector恢复)；C(teacher/GT)可与B并行；D依赖A的layerstate合同；E先发布dry-run，再接B/C/D；F只读完成记录；G独立review所有主张。一个agent一次只改其文件范围，改公共dataclass由A发布版本后统一迁移。禁止在agent私有分支重复更改旧训练核心造成配方漂移。

验收按科学事实而非任意mAP门槛：legacy/fullgate恒等、noGTinference、梯度通路、state一致性、actualopcount、记录完整性是必须；性能是否值得扩大则由matchedcontrols与误差分析决定。
