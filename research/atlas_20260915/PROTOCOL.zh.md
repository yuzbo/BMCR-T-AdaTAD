# Publication Action Space v1 - 正式测量协议

用户于2026-09-15授权在connect.nmb1.seetacloud.com:44909开展实验与论文级绘图，并明确确认：Fig.4采用16个覆盖候选域的合法动作组；逐点统计仍为每轴16个动作。以下规则在读取publication干预结果之前固定；不由test曲线修改阈值、动作粒度、边界定义或Static规则。

## 参考与两套数据

- 官方AdaTAD VideoMAE-S/B，官方state_dict_ema严格加载499键；FP32、TF32关闭、确定性cuDNN、math SDPA，所有参数冻结，无优化器更新。
- 官方dense复现目标分别为S69.0126%、B71.1204%，同211视频/792窗口，允许FP32/历史评测的0.10个百分点技术差异。超出则先定位协议/实现，不放行新的publication干预。
- Development：200训练视频中由seed42、不看GT/预测抽32视频，各取官方中间窗口；用于技术验证和Static全局排序。另测一个视频的全部窗口覆盖短/尾窗。开发数据不用于正文总体结论。
- Publication：固定全部211测试视频、792官方窗口，原模型处理、stride4、160中心裁剪。正式图用全覆盖测量，动作本身抽样。THUMOS test已被看过，不能声称untouched holdout。

## 官方heavy与light来源

官方dense模型没有H65 light模块。因此，full路径保持官方heavy权重、TIA和head；D/S干预的light替代读取已有Full-V2 epoch40 EMA中层4/6/8/10的depth-attention、depth-FFN和spatial-light算子，冻结并逐键加载。它是带明确来源的冻结算子比较，不是随机light，不是官方原模型固有的light，也不是新训练结果。

S/B的原始V2路径、source_revision及全部使用张量随提取资产保存。恢复实验进一步保存H65初始backbone相对官方权重的所有差异、Scout及完整V2 EMA，可精确重构既有V2；不把旧checkpoint换名成新代码版本。

## 逐点population动作

- O：每窗16个等距物理时间分层候选帧，不看GT。分别neighbor、interpolation、40→160低分辨率替代；仅改该帧RGB，shape/mask/time/GT不变。
- D：16个state-layer位置，均匀覆盖0-based层4/6/8/10和4个时间层，空间坐标按固定窗口/视频序号轮换。该位置attention及FFN从训练好的light升级到官方heavy，其他位置full，TIA始终保留。
- S：同样16个预定位置，D保持full admission，只把指定FFN从light升级到heavy。
- T：原逐帧K384均匀支持上，16个时间分层的合法remove→insert单帧交换；固定K，RGB实际重新gather与VideoMAE重新打包，恢复到原物理时间轴。若无合法未选帧，报告实际0动作，不虚构零价值。
- D/S升级值是L_light-L_heavy；O移除效应是L_intervened-L_dense；T是交换收益L_base-L_exchange。分别保存cls/reg、raw有符号值及成本，不能把不同动作的raw值当跨轴优越性。
- 每窗null完全重放一次；轻扰动控制在轮换的population位置执行160→158→160双线性变换。no-op容忍为1e-7原始cls/reg损失差，超出先检查数值实现。
- Population结果同时报告action-weighted与video-balanced；图中的near-zero曲线采用固定相对容忍网格1e-8至1，并突出0.5/1/2/5%。同时保留raw值、负部、正部和零部。

## Coalition与TAD条件样本

O、D、S在每个窗口联合执行1/2/4/8/16个动作，使用固定分层顺序及单点低绝对效应顺序；比较实测联合效应和单点效应之和。T使用不冲突的交换联合，另测高单点收益排序；不足时记实际数量。I=joint-sum，只解释超加/次加，不声称次模性或数学最优。

GT条件采样另存sampling=tad_conditional：每窗最多2个完整可见动作，取5/50/95%相对位置，并补最多2个背景位置；测O-interpolation及D/S升级。它不混入Fig.2总体分布。边界主带为动作时长10%，原始GT边界；多动作重叠单列，裁剪端点不是新边界。每条记录保存cls/reg和完整位置元数据。

## Fig.4有限分组参考

16组都由真实可执行primitive的并集组成；它们只界定有限搜索空间，不能称数学oracle或整个WTR动作空间的上界。

- D组：4个路由层 × 4个对齐VideoMAE打包边界的时间区域。组内仍是逐state attention/FFN heavy更新；未选state为light，完整TIA保留。
- S组：4个路由层 × 4个5×5空间区域，覆盖全部native time。D始终full，选中组heavy FFN，其余light。
- T组：4个时间区域 × 每区4种交错逐帧相位。每组是individual RGB帧的集合，不是原16帧连续clip selector。每组取同样数量的有效帧；余数及padding作为所有策略共同必选。为VideoMAE补至16的倍数，额外padding明确记录并实际计费。前4组确保每个时间区域都有观察支持。
- 预算点对应4/6/8/10/12/16组；横轴采用实际完整执行GFLOPs。D/S先以Uniform得到目标成本，然后在16组有限子集中寻找同组数、成本差不超过dense成本0.5%的可行集合；每个最终集合重新operator profile。T各策略同组数、同实际RGB执行长度。
- Random从满足预算的有限集合随机选；Uniform按固定均衡顺序；Static使用32个development视频的平均value/cost排序；Attention使用同冻结dense模型的诊断attention分数；Marginal CF最大化已测单组收益之和。T按单组收益排序加入覆盖组。
- 每个candidate group在共同cheap base上真实执行一次，保存V和DeltaC；不使用dense单点移除值替代cheap-base增量值。
- Sequential CF在预先固定的前20个publication视频中window_index能被4整除的窗口测前8步，每步重算剩余候选。该子集只检验单步近似，不代替主全数据mAP。
- 每个策略/预算必须重新执行全部792窗口、合并211视频并重算官方AP@0.3–0.7。不能用local loss或平均per-video AP代替。
- 图的主横轴是所选计划的完整执行成本。CF查询与dense-attention诊断评分属特权选择成本，单列账本；Attention在图注中标为dense-score diagnostic，不伪装成低成本可部署router。Random/Uniform/Static不使用当前test GT。

## Fig.5同支持恢复

固定既有Full-V2 epoch40 EMA、同一H65 K384选帧、D/S全量、同一组heavy anchors与多层特征及同一检测head。共享heavy计算只执行一次，比较packed linear、physical interpolation、Cross关闭multidepth、完整Cross+multidepth。关闭分支属于同checkpoint依赖消融，不冒充独立重训的架构对照。

特征目标是同一冻结V2 encoder使用全RGB/full-D/S后物理插值到原query的表示；其额外计算单列。保存每query support gap秒数、NMSE、cosine，及GT关联起止误差、未匹配比例、全数据AP。恢复器能维持完整物理query支持，不保证未观察语义无损。Graph/continuous-time本轮不新增。

## 统计、绘图与数据冻结

- 以video为cluster，正式bootstrap=10000，seed42；mAP每次按重采样视频权重重算完整AP，使用已验证的固定排序/匹配缓存加速。CF与Uniform使用同一组重采样，报告配对差值CI。
- Lorenz/Gini/top-p针对max(V,0)，同时报告负部；不对signed values直接算Gini。边界/时长/cls-reg统计不预设方向。
- 原始每窗口JSON写入后不覆盖。续跑跳过完成文件，参数与源版本写manifest；技术修改发生在publication前，若更改科学定义则新目录新版本，不覆盖旧结果。
- 分析和绘图只读原始记录，不执行模型。完整正式图必须核对211/792覆盖，缺数据时不生成正式Figure2/4/5。
- 先产Fig2及Fig4，Fig1按预登记集中度10/50/90分位病例，Fig3独立条件样本，Fig5并行。Fig6留待新WTR训练，本轮不生成虚假性能点。
- 英文图注，色盲友好配色，单/双栏尺寸下可读字号；输出SVG、PNG和PDF，逐页渲染检查。图注明确模型、split、动作语义、执行成本及特权查询成本。

本协议不启动WTR训练、Graph接入、RISE或三轴联合allocator训练。既有原集群任务不修改。

后续Value Head训练只能使用training/development记录；publication/test atlas用于冻结评估与展示，不能转作训练标签。
