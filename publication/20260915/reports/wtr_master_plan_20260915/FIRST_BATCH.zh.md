# 第一批模型实现、实验与部署推荐

> 最新执行规则见[双线交接](HANDOFF_DUAL_BATCH1_20260915.md)。用户已授权实施：Atlas/Core与Raw并行；Raw参数仅在training/dev确定；先300–500 swaps mini-bank再决定约6400；holdout ranking/regret通过才跑O-V/R+-V全211/792。下文1A/1B原指Raw内部阶段，现改用R阶段，不能与新的Core/Raw双线编号混淆。下文“未实施”描述的是上一轮报告时点，当前Raw进度以接手任务为准。

本文件是可审阅的实施建议。用户本轮要求推荐报告，故尚未创建新工作树、编写模型模块、生成实验配置或提交任务。

## 1. 首批目标与明确取舍

**建议首批交付“Raw-v1-S共同输入接口＋冻结检测器的候选域实验＋一个共享Temporal Value小头”。** 它同时为后续Core建立时间价值接口，并验证off-grid候选是否值得进一步投入。

首批不训练完整新detector，不同时修改D/S、Graph、DB、FVD、时间PE、micro-clip或THUMOS窗口。先得到可执行且有清楚归因的最小模型，不用模块数量代替科学结果。

任务分为：

- **1A：零新增优化更新。** 接通Raw输入、同一V2-S冻结参考，完成S0与O-U/R+-U/O-CF/R+-CF。
- **1B：仅拟合一个共享Temporal Value小头。** 从训练/development反事实标签学习，冻结其余模型，再补O-V/R+-V。

合计最多7个评测条件，其中2个是GT辅助参考，只有1条轻量router拟合实验；不是7个完整训练，也没有新增40/80轮detector课程。

## 2. 代码与资产基线

建议从Atlas已提交HEAD `50a49e328e2a9dc0080d5233286a4d512bcb68af` 创建独立分支 `codex/wtr-raw-v1`，拟议本机工作树为 `C:/Users/skywalker/Documents/ChatGPT/H65/h65_clean_adatad/wtr_raw_v1_20260915`。这些名称目前只是推荐，尚未创建。

模型来源仍为1955057执行框架和真实V2 checkpoint。新分支复用Atlas读取、重构、AP及profile能力；不改正在运行的Atlas工作树，不将旧Graph目录的未提交原型作为唯一实现源。

可复用能力：

- `h65/atlas/data.py`：官方episode/元信息及完整视频分组；需把episode生成与RGB materialization进一步分离。
- `h65/atlas/recovery.py::build_probe`：从官方teacher、V2 EMA导出与backbone/scout差异重构冻结V2，S已GPU验收。
- `h65/frame/geometry.py`、PaperDecoder、TaskReadout：原轴表示与读出；Raw需要显式适配，不能沿用全部长度比假设。
- Atlas官方AP、视频聚类统计、算子计费及按窗口恢复：优先复用已验收实现。

**不能误认的边界：** 当前Atlas temporal.py从 `data['inputs']` 索引，而该tensor由官方dataset pipeline生成。因此它仍受官方候选域约束，不具备任意原视频frame ID读取能力。Raw reader、坐标和descriptor需要新接入。

旧FrameRouter的294/300维特征也不是新Unseen Descriptor。可以参考其NLL、校准与checkpoint方法，但不能直接改标签名字就宣称支持Raw。

## 3. 推荐首批参数

以下是实施起点，须在开发/技术验收后一次性登记冻结，不能用正式test结果反复调节。

|项目|建议|
|---|---|
|数据/骨干|THUMOS14、VideoMAE-S、seed42|
|冻结参考|完整重构的Full-V2-S epoch40 EMA；固定heavy backbone、head、Cross、Scout参数及D/S策略|
|Heavy容量|K384，D100/S100；不以该标签声称总模型恰好省50%|
|Episode|重放相同物理窗口、GT/边界有效性和augmentation身份；正式211视频/792窗口|
|共同preview|192个物理时间均匀低分辨率观察，64×64；共用时间戳、像素、变换与Scout输出|
|Proposal|固定几何offset，起点每preview节点4个有界候选；非学习offset|
|域|Official O；Raw-Expand R+=O∪Poff；去重后的实际数量计账，R+上界为O＋192×4|
|Uniform|同一物理区间的中点等分，再按域映射最近合法帧；固定tie/重复/覆盖规则|
|Value最小形式|几何覆盖初始集合＋set-conditioned swap；首批不训练独立unary acquisition proposal|
|交换范围|建议最多4轮、每轮最多16个合法候选pair；几何候选规则与搜索预算预登记|
|Tubelet|排序后确定性两帧pair、16-observation pack；不加max-gap，不静默新增观察|
|恢复|保持现有Cross多层恢复与真实物理几何；不新增时间PE或Graph|
|精度/验证|沿用Atlas确定性FP32技术路径；其他精度作为后续独立系统测量|

共同192-preview会改变旧Scout的cadence，不能假定与原Standard表现相同。首批先按共同冻结表示评估；若需要适配，仅在训练/开发数据上完成后统一冻结，四格不得各用不同Scout却仍称完全共享证据。

## 4. 最小实现责任范围

下列为拟议模块责任，不表示这些文件已经存在。

|责任|必须完成的内容|
|---|---|
|Episode/坐标|EpisodePublic与GT监督对象分开；preview/heavy/query与native anchor映射；保留训练episode变化和正式测试清单|
|CheapPreviewReader|在heavy候选tensor构造前取得公共低分辨率流，返回真实frame IDs/time/validity|
|BoundedRawProposal|固定几何proposal、映射合法帧、去重；明确Expand/Replace身份和候选成本|
|Unseen Descriptor|preview插值上下文、时间/gap/coverage、全局摘要、拟议选集与plan；不读GT或未取得的heavy特征|
|RawSelection/TubeletBuilder|精确执行观察数、有效性、ordered contributors、pair/pack、span与重组影响|
|Raw冻结参考适配|复用NativeEncoder、V2 decoder/head，绕开旧candidate/detector整数ratio；不重写VideoMAE|
|真实CF收集|同episode、checkpoint、预处理和预算重执行；固定D/S；标签包含真实tubelet重排|
|Temporal Value头|一个共享轻量pair head预测cls/loc交换收益，两个域共用；首批无FVD与未校准风险惩罚|
|评测/回执|全窗口预测、官方AP、完整成本、训练/正式标签隔离、源版本与数据身份、可恢复输出|

关键恢复接口：K384→192个heavy anchors；THUMOS现有Cross在384个原配对query位置恢复；TaskReadout再插值到768检测位置。Raw须保留这条对应，并通过真实时间把preview、raw anchors和原query接起来。

## 5. 第一批评测条件

|ID|身份|训练状态|目的|
|---|---|---|---|
|S0|原V2-S输入/preview/selector参考，强制同K384/full-D/S配置|既有权重冻结|检查原路径与迁移差异；不是直接引用旧dynamic峰值|
|O-U|共同preview、Official域、physical Uniform|零更新|新共同输入条件的简单控制|
|R+-U|共同preview、Raw-Expand、physical Uniform|零更新|候选机制变化的控制|
|O-CF|Official域的有限真实交换搜索|零更新，GT仅诊断|被测动作空间参考|
|R+-CF|Raw-Expand的同预算有限搜索|零更新，GT仅诊断|检测off-grid的可利用机会|
|O-V|Official域、拟合后的共享Value交换策略|只训练Value小头，其余冻结|官方网格内的学习收益|
|R+-V|Raw-Expand、同一个Value小头|只训练Value小头，其余冻结|同一策略能否利用新增候选|

Dense-S官方68.9767%的结果可作为外部强参照，但不能拿它冒充S0的V2-K384结果。共同preview下O-U/O-V也不冒充原始官方AdaTAD。

CF使用有限候选与查询预算，完整记录起始支持、实际查询数和搜索成本，不称mAP上界。O/R搜索接近只表示当前实验未显示额外机会，不据此声称所有raw候选均无价值。

## 6. Value小头的训练数据与范围

建议从200训练视频按seed42固定划分160 fit、20 calibration、20 router-holdout，ID清单在实施时生成并保存。该holdout只对新router未参与拟合，既有detector可能训练过这些视频，不能称全模型未见数据。

初始bank可对每视频两套预登记episode/支持状态、每域至多8个合法swap取真实标签，最多约6400对；实际合法数量与完整前向次数另报。状态覆盖应包含uniform及确定的合法扰动，不能只收集最终test里表现好的动作。

新pair head只读取训练侧冻结描述子，以固定尺度的cls/loc真实收益进行监督。首个pilot可用2000次小头更新作为开发起点；完整detector更新为0，不加载旧optimizer冒充resume。若标签/状态覆盖不足，通过训练/开发材料调整并在正式评测前冻结。

**publication/test atlas、O-CF/R+-CF正式评测标签均不进入训练、归一化尺度或参数选择。** 冻结后一次性执行O-V/R+-V全评测，Uniform预测在输入和模型完全相同且回执有效时复用。

## 7. 验收顺序与阶段产物

|阶段|检查的具体问题|通过后的产物|
|---|---|---|
|R0 合同/manifest|是否真正同episode、同preview；GT是否与policy输入隔离|冻结协议、公开输入字段、split与版本记录|
|R1 CPU接口|时间nearest、域包含/上界、GT投影、确定性pair/pack是否正确|可复现输入与selection/tubelet回执|
|R2 S GPU技术版|同frame IDs读回RGB、原路径桥接、no-op、成本、坐标及AP复算是否一致|6训练视频技术回执，至少一个完整视频覆盖所有窗口含尾窗|
|R3 训练侧bank|未heavy读取描述子能否连接真实交换，尺度/状态身份是否一致|fit/calibration/holdout标签，查询成本|
|R4 正式冻结域测量|有限O/R候选是否产生完整TAD差异|S0、O-U/R+-U/O-CF/R+-CF全211/792结果|
|R5 小头拟合与冻结|共享Value是否改善训练侧未拟合视频的交换排序/regret|一个evaluation checkpoint、训练范围与状态回执|
|R6 完整Value评测|是否转化为全数据检测与成本收益|O-V/R+-V及四格交互、视频聚类CI|

R3与R4的数据和输出路径严格分开；R5可在训练bank就绪后开展，不读取R4标签。技术检查不能因为6视频通过就变成正文科学证据；也不重复执行已通过且条件未变的旧Atlas测试。

## 8. 部署建议

拟议远端目录：`/root/autodl-tmp/wtr_raw_v1_20260915`。数据只读复用 `/root/autodl-tmp/thumos14`；S权重/重构资产只读复用现有Atlas assets。Python沿用 `/root/autodl-tmp/envs/opentad/bin/python`，不为新实验擅自升级共享环境。

当前Atlas占用两张卡。可以先完成本机代码、CPU合同与小型manifest；GPU技术版由现有owner安排空出的合法时隙。原则为每卡最多一个本队列GPU任务，首个Raw GPU阶段先用S和一张卡，通过后再考虑并行独立评测。

如果现有owner不能接受新stage，先完成新入口和回执验证，再在Atlas阶段结束后进行明确资源交接；不临时修改正在测量的科学脚本，不另起竞争controller，不取消旧任务。

首批先允许算法级缓存/按episode读取，完整记录其成本边界；不预解码整个数据后宣称on-demand加速。真正异步按需decode、compressed-domain preview及系统加速是后续交付。

运行时长先由R2真实测得的decode/forward与查询数估计。本报告不虚报GPU小时数或完成日期，也没有提供尚不存在的CLI冒充已可运行命令。

## 9. 首批必须保存的分析

- 原始全视频预测、Avg-mAP、AP@0.3–0.7、完整neural GFLOPs与候选/控制器开销。
- 候选域identity、off-grid比例、实际distinct frames与执行观察数。
- ordered-pair span、N_changed_pairs/changed packs，区分frame替换与真实重组影响。
- Preview coverage：GT仅用于分析的无直接preview观察、gap/时长与失败关联；不称信息损失上界。
- 四格interaction按video-cluster联合重采样，每次重算四组AP。Frozen detector与后续独立重训结果分开。

不得以局部loss替代mAP，以相同K替代完整成本匹配，或从CF与Value差距直接断定唯一瓶颈。

## 10. 首批之后的决策

若发现稳定的Raw有限域机会，并被Value利用，再投入匹配重训的O/R四格及Raw-Replace；若Value未利用机会，先区分描述子、学习与preview覆盖，不盲目加Graph。若当前搜索未找到机会，保留负结果和有限域边界，避免按test反复调proposal。

完整Core的A/F价值与light适配、DB、FVD、Graph、B确认及跨数据集属于后续清楚登记的批次。当前首批结果不署名为完整WTR-Core或最终WTR-Full成功。

**本文件仅推荐实现与部署范围，所有新阶段仍未实施、未登记、未提交。**
