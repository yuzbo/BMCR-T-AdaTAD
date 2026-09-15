# 两版WTR方案比较：全数据证据框架与算子级方法设计

日期：2026-09-15。本次比较用户上一条“Graph-conditioned、Future-distilled统一T/S/D最终候选”与新建议及 `wtr_agents_1955057.zip`，并对照此前确定的全数据characterization计划。只审阅、解压到独立报告目录并保存材料；没有应用补丁、运行包内测试/CLI、训练、部署或修改现有研究计划。

**判断：具体模型研发路线更推荐新版的算子解耦、低成本替代与条件价值方案；论文方向论证继续采用此前的全数据characterization框架。** 上一版Graph统一三轴可以保留为后期组合候选。新版不是对已确定全数据证据要求的替代，也还不是已完成的最终WTR代码。

## 1. 核心差别

上一版的问题重心是：怎样借助共享Graph关系状态和未来价值教师，更好地决定T/S/D在哪里花计算。

新版进一步拆问：现有heavy算子究竟哪部分不能省；light是否是可用替代；替代路径改善以后，哪些位置仍然值得heavy；怎样训练这种条件价值。

两者最终目标都保留预算内TAD性能、真实逐帧选择和原时间轴恢复。区别在于：上一版优先设计统一决策架构，新版优先定义和改善被分配的计算动作。

|方面|上一版Graph统一三轴候选|新版算子方案|
|---|---|---|
|方法中心|共享关系上下文→T/S/D Value Heads→Allocator|attention/FFN解耦→训练可用light→算子条件价值分配|
|最先要排查的问题|现有局部排序是否不够好，Graph/future是否提供更好的决策信息|即使选点正确，当前cheap算子是否仍不足以维持任务状态|
|内部动作|D的heavy block与S的heavy FFN边界仍需明确|显式q_heavy_mask、ffn_heavy_mask，可分别启用，分开记录age|
|Graph角色|拟共同影响T/S/D，并参与恢复|先独立比较证据访问/恢复，之后才研究作为价值头输入|
|RISE实验起点|anchor/post价值预测函数输出外推，蒸馏三轴头|固定动作库测实际收益漂移，再做冻结detector的现有FrameRouter试验|
|新增监督重点|真实价值＋未来排序目标＋恢复/同支持约束|进一步突出同一个pre-op输入上的heavy/light输出对齐|
|允许的最终形态|完整共享Graph、三轴价值、训练期future作为全面候选|允许attention保留、FFN稀疏等非对称方案；Graph/future可不保留|
|失败结果的解释|需要辨别Graph/teacher/三轴交互中的来源|先辨别算子损害、替代质量、选点质量、监督质量，归因更直接|

这些并非全部是新发现。局部D/S缺乏直接价值监督、V2-S倾向D100/S100、保留Cross/TIA、Graph和future不能预设有效，之前材料已经提出。新版最实质的推进是**把Q/FFN从隐性耦合改成显式独立动作，并将轻重同输入对齐作为独立训练问题**。

## 2. 为什么更推荐新版的方法设计

### 2.1 它落实了上一轮指出的D/S语义问题

现有admitted同时限制attention query和heavy FFN，S再限制FFN集合。上一版虽然提出两个价值头，但两个“heavy相对light”标签仍可能覆盖同一次FFN更新。

新版合同明确允许：Q full/FFN full、Q sparse/FFN full、Q full/FFN light、Q sparse/FFN light，另保留旧coupled对照。这允许直接回答关系访问与非线性更新谁更敏感，避免把所有掉点归于一个depth mask。

T/S/D是执行位置的坐标，attention/FFN是算子种类。两者并不冲突：可以在不同时间、空间、层深位置分配不同算子。最终若继续声称三轴，需要明确坐标消融与算子消融的对应，不能仅把Q/F改名成D/S。

### 2.2 它补上了“可用动作”的前提

如果light对任务状态破坏很大，再好的排序也只能尽量回避它。这与“当前有可省计算但router没选对”是两种问题。

新版同输入对齐：在当前student pre-op状态u上比较 `F_light(u)` 与 `stopgrad(F_heavy(u))`。已有same-support参考则沿自己的full计算轨迹形成目标，二者不能互相替代。主目标仍需是TAD表现；局部NMSE改善不能直接视为方法成功。

这是一项有针对性的迁移实验。SSA原工作研究full/sparse attention的输出对齐；把原则用于H65的light FFN，是本项目的新假设，不继承原文的性能或理论结论。[SSA原论文](https://arxiv.org/abs/2511.20102)

### 2.3 它能检验简单方案是否已经足够

最有价值的新增控制应包括 **Uniform＋同输入算子对齐**。如果它已获得大部分收益，就应承认主要贡献来自cheap路径可用性；只有value routing在该强控制上继续改善性能—成本，才能证明“where”的增量价值。

低成本算子训练会改变实际价值分布，所以应在适配后重新测量关键反事实与分配空间，不能让后续router沿用旧cheap算子的标签。冻结原模型的CF参考没有gap，也不能排除训练出更好的替代算子后出现新机会；但这只能支持有限、明确的适配研究，不能成为无限增加模块的理由。

### 2.4 它保留更直接的最终目标

最终应区分不可省的计算、可被cheap替代的计算，以及仍需按位置分配的heavy计算。允许只压FFN、保持attention，或暂不保留Graph/future。模型是否在三轴都降低比例，不是单独的成功条件；任务收益、实际成本与强简单控制才是。

## 3. RISE部分发生了真正的实验对象变化

上一版核心公式外推的是共同状态/候选上的价值预测输出：

`predicted_future = predicted_anchor + β × (predicted_post - predicted_anchor)`。

新包的离线阶段则从anchor/current detector真实重执行记录读取 `actual_delta`，外推真实收益标签：

`target_future = actual_current + (β-1) × (actual_current - actual_anchor)`。

随后用current实际收益NLL和可选current/EMA/future分布JSD，仅训练既有FrameRouter。源码 `wtr/fit_router.py:62–83` 直接体现这一点。later实际收益不进入该训练程序，用于预测检验或另一个冻结detector上的迁移评测。

这让新pilot能较干净地测试“真实价值轨迹作为目标是否有帮助”，减少head自身预测误差的干扰。但正结果不能直接证明上一版的“预测函数外推”有效，更不能证明三轴在线递归学习已经完成或反事实成本已经降低。两者是前后可衔接的实验，不是等价实现。

新包的 `current` teacher是当前真实价值形成的软分布，与“冻结post预测器的snapshot KD”也不同。β=1在两版都回到各自外推的较晚端点；都不等于EMA。JSD与KL的选择本身不是新版优越性的主要理由。

固定loss normalizer、共同尺度、真实EMA重执行、later标签隔离和evaluation-only导出，都是有用的实验边界。不过future检验失败只影响future分支，不应阻断actual-value routing；教师质量不是算子解耦研究必须先通过的关卡。

## 4. 新版实验流程与此前全数据规划的差别

新版E0–E6更像方法开发路线：技术smoke→S算子损害→三checkpoint价值预测→冻结FrameRouter→算子路由→Graph恢复→B/完整任务/泛化。

此前确认的论文路线是：官方dense S/B全数据必要性→TAD条件结构→完整CF–Uniform改善空间→原轴恢复→训练后WTR利用机会。前者有利于定位实现与方法问题；后者更能让读者独立判断研究问题是否成立。

**不建议退回以32训练视频pilot承担正文方向论证。** 新文档也承认32视频不是最终全数据统计，但它把characterization扩展指向train-development覆盖，并未落实我们已经确定的“全211视频、792窗口、dense S/B总体测量”要求。新包最终调用完整官方mAP，只能证明相应策略的全测试表现，不能替代覆盖全数据的必要性分布、coalition和CF预算曲线。

另外，Graph先做恢复再测试价值上下文是合理的资源顺序，不是逻辑必要条件；恢复未获益不能自动推出图上下文不会改善选点。两个作用应有各自结论。

## 5. 六图论证的取舍

|图的职能|此前全数据版|新版论文草案|
|---|---|---|
|问题定义|Fig1受控干预与代表案例|Fig1观察/访问/更新案例，基本一致|
|必要性与交互|Fig2全数据分布，Fig3专门TAD结构|Fig2合并单项/组合和局部算子误差，TAD结构后移|
|分配改善空间|Fig4为独立的frozen全数据主图|Fig3质量—成本曲线，含frozen/adapted身份|
|原轴恢复|Fig5独立核心图|融入方法、恢复消融及Fig6解释|
|方法展示|Fig6承载真正的训练后WTR结果|Fig4方法/算子对齐，主结果也在表与Fig3|
|未来价值|Fig6子实验或附录|Fig5专门未来价值，篇幅权重提高|

建议保留此前六图主骨架，增强内容而不是全部换图号：Fig2/3加入Q/FFN解耦与交互；Fig4保留冻结/适配身份清楚的headroom；Fig5继续专门展示原轴恢复；Fig6展示新算子方法的真实结果与关键消融。future只有显示足够独立价值时才提高篇幅，避免再次让教师研究挤占WTR最终表现。

## 6. ZIP的实际交付程度

压缩包共23个文件，本次已静态检查文档与关键实现，没有运行其测试或程序。`VALIDATION.json`和日志记载26项CPU测试通过，但明确未覆盖OpenTAD集成、真实checkpoint、视频、CUDA、全mAP或Slurm；这是提供方的验收记录，不是本轮复验。

|内容|实际覆盖|
|---|---|
|动作库|`integration.py:50–55`取训练dataset；`probe.py:29–38`按视频去重且仅保留有效768帧窗口，每视频最多一个采样窗口|
|局部算子探针|固定其他masks，测2×2组的attention-hold、FFN-light及二者联合；不是全动作空间的独立Q/F训练|
|FrameRouter训练|`fit_router.py:74`只优化router.network；输出evaluation-only，不能resume旧训练优化器|
|算子方法组件|OperatorValueHead、同输入损失、真实价值损失已经有代码；未接入完整PaperModel/engine|
|四个配置|固定full-D/S、固定A-MoD、uniform D/S、mixed plan-aware，均是旧schema支持的控制，不是完整新WTR配置|
|未提供的正式套件|全211/792 population测量、完整observation replacement、通用coalition、全预算CF-reference评测及完整三轴在线递归训练|

因此，“把max-videos调大”不能得到211视频/792窗口正式characterization：数据split、每视频一窗与完整窗口过滤都不符合该协议，需要专门扩展。包中完整官方评测依赖原仓库 `paper_eval.py`，它不会自动补齐上述缺失实验。

新包确实比单纯设计文本更接近可实施的开发起点，但不能把已写好的局部组件和已连接的最终模型混为一谈。

## 7. 仍需保留的设计限定

同输入feature对齐是训练候选，不强迫每个cheap输出精确重建heavy；若只改善NMSE却损害TAD，应降低或删除该损失。attention-light若只有token-local输入，也不能预设能替代依赖其他token的关系访问。

STOP适用于可拒绝的追加动作、或保留当前合法配置的交换动作。固定K的原始选帧/slot分配不能简单插入STOP后少选，再继续称为相同容量实验。新包的exact_capacity本身只是等成本固定数量选择，也不解决任意联合成本优化。

Taylor残差代理应明确在修正注入点取梯度；不能用归一化输入上的梯度直接替代输出残差所需的敏感度。图访问、teacher查询、light训练和评分的成本继续完整记录。

已有N系列的D50/S48与拟议D75/S100不匹配，交替dense层刷新真实age等既有核对仍有效。把age拆为attention_age/ffn_age提高了定义清晰度，但不会自动产生不存在的长期陈旧状态。

## 8. 推荐采用的整体路线

1. 保留已确认的全数据S/B characterization、TAD结构、联合干预和CF–Uniform曲线，补入attention/FFN的清楚动作定义。
2. 对主要损害算子开展有限的同输入light适配，保留Uniform＋对齐强控制，并重新测量该动作空间的任务价值。
3. 训练便宜的算子条件价值头，确认其在适配后的简单控制之上提供完整性能—成本收益。
4. 独立运行新包的固定动作库forecast与冻结FrameRouter实验，把它作为future-target机制研究；再决定是否接入预测函数蒸馏和在线递归。
5. 原轴Cross保持，Graph/continuous-time独立验证，成功机制最后组合并进行最终S/B与跨设置确认。

**选择建议：采用新版作为方法研发基础，保留全数据版作为论文证据与展示框架；上一条共享Graph三轴完整版作为后续可比较的组合候选。** 这项建议没有自动改变现有课程或授权任何实施/部署。

参考核对：同输入对齐原则见[SSA](https://arxiv.org/abs/2511.20102)；attention独立动态深度见[Router-Tuning / MindSkip](https://arxiv.org/abs/2410.13184)；低成本嵌套视觉专家见[MoNE](https://papers.nips.cc/paper_files/paper/2024/hash/6b768359d0e8925164f61f381a748441-Abstract-Conference.html)。这些原领域结果均不构成TAD有效性证明。
