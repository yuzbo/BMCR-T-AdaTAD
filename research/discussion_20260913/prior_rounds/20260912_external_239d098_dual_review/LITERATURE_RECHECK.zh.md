**16项文献机制复核：能支持什么，不能替本项目证明什么**

本表合并两份建议，重点核查原问题、保留状态、训练是否接触条件路径和省算位置。不是2026年SOTA或排他性新颖性检索；不复现作者权重或性能。部分条目只核查原文摘要/方法描述，没有全库实现审计。原文件版本与细节保存在附件中，不改写作者原报告。

|编号/来源|对原建议的判定|应吸收的机制与边界|
|---|---|---|
|L01 [DyT，v2](https://arxiv.org/html/2403.11808v2#S3.SS2)|原文式5/6支持训练全算后mask残差，后续学生已见缺失更新|它支持mixed-state训练，不支持当前纯dense fitting。MLP逐token可以同输入压紧；attention删K/V另有失配。第二份关于作者分割Block的限定正确；[固定代码](https://raw.githubusercontent.com/NUS-HPC-AI-Lab/Dynamic-Tuning/d1744f0b9366f79ad9b78f586e479af34e81807a/dense_tasks/Segmentation/backbone/segmentation_vision_transformer_IN21K.py)252–271行无compact eval分支，不外推全部入口|
|L02 [CoDA，v2](https://arxiv.org/html/2304.04947v2)|预训练重模型冻结，全部token轻Adapter、选中token条件重路径，训练router/adapter等|与冻结大模型的小数据适配更接近，但条件路径参加任务训练；不是拟合dense最终特征自然成功|
|L03 [CoLT5，v3](https://arxiv.org/html/2303.09752v3)|light/heavy成立，FFN、重Q、重KV分别路由，具有专门预训练/适配|需要更新的位置与提供上下文的位置分开；训练资源差异巨大，不能承诺200视频即可复制结果。①v2/②v3的核心结构判断不冲突|
|L04 [DToP，v2](https://arxiv.org/html/2308.01045v2)|早退输出与类别代表上下文成立；[固定README](https://raw.githubusercontent.com/zbwxp/Dynamic-Token-Pruning/2baefffc879622b6514504dff44cf04d7e0e7504/README.md)确认base后pruning训练|保留原位置任务状态；类别代表不能替代重复事件覆盖。①v2/②v1不应视为两份独立验证|
|L05 [TR-BERT，ACL正式PDF](https://aclanthology.org/2021.naacl-main.463.pdf)|§3.1保留退出层token状态，§3.2为任务微调、RL和联合训练；有启发式warmup及KD|可以少更新但仍留位置表示；原报告将其限定为任务/策略适配是正确的，RGB未观察与词token退出不同|
|L06 [AdaFocus V2](https://arxiv.org/abs/2112.14238v2)|视频识别、可微patch选择、全局低成本视觉与局部计算的机制支持建议|借鉴低成本观察和局部细化，不照搬整段早退到多事件TAD；本次未审全作者代码|
|L07 [SlowFast](https://arxiv.org/abs/1812.03982)|②新增参考：低帧率语义与高帧率低通道运动通路|对H0的顺序敏感轻状态有启发；不是直接移植双骨干。不能从分类/空间动作检测结果推定THUMOS时序定位收益|
|L08 [QueryDet](https://arxiv.org/abs/2103.09136)|已有FPN特征上以粗查询驱动高分辨率稀疏检测，方向正确|省的是后续head计算，不能证明先删RGB安全；卷积halo/坐标须适配。本次机制复核不复跑①所引qinfer实现|
|L09 [PointRend](https://arxiv.org/abs/1912.08193)|粗预测、已有细特征与选择性点细化支持原建议|说明不同训练/推理调度有时成立；不提供删除未观察视觉信息的保证。未复核全作者代码/精度表|
|L10 [RandLA-Net，v3](https://arxiv.org/abs/1911.11236v3)|随机采样与局部几何/特征聚合联合，机制正确|覆盖与聚合值得迁移；不能把无序点集邻域直接等同有方向的视频tubelet|
|L11 [3DSSD](https://arxiv.org/abs/2002.10187)|几何/语义融合采样及候选生成支持建议|纯语义与纯覆盖各有盲区，TAD应同时保留覆盖和任务证据；不搬用3D检测指标解释本项目mAP|
|L12 [Deep Feature Flow](https://arxiv.org/abs/1611.07715)|②新增参考：关键帧重特征通过运动场传播，训练传播路径|它提供“跳过的计算要有状态载体”的例子；flow成本、遮挡和新目标仍存在。当前VideoMAE跨窗深cache不是同一机制|
|L13 [Eventful Transformers](https://arxiv.org/abs/2308.13494)|保留token状态并按变化更新，支持从重算转向状态更新|当前窗口/clip相位/TIA上下文必须对应；不凭重复RGB宣称免费缓存，不引用其速度数字作本项目预测|
|L14 [Expediting ViT，v1](https://arxiv.org/abs/2210.01035v1)|浅特征聚类与重建可在论文密集预测任务中不经finetune|是“所有稀疏化都必须重训”的反例；其关系来自已获得的特征。TAD中每次TIA之前就需要合法栅格，不能只末端补齐|
|L15 [ETAD，v2](https://arxiv.org/html/2205.07134v2)|§3.2/3.3确认完整snippet前向、检测学习、部分梯度重放|两份把它限定为训练资源优化是正确的；没有证明少RGB推理保持精度|
|L16 [Temporal Corruptions，v1](https://arxiv.org/html/2403.20254v1)|原文摘要及结论确认中部腐坏往往造成更大下降|①保留动作内部证据的动机成立；其腐坏协议不是Z24，不能据此把本项目损失都归动作中部或边界|

最直接应吸收的是DyT/CoDA的条件状态训练与TR-BERT/DToP的退出状态保留；SlowFast对应轻状态的运动信息；QueryDet/PointRend等主要提供训练—推理调度边界。它们共同支持提出实验，不能替代本项目的matched checkpoint、实际compact和完整TAD测试。

对子代理初次检索中未打开的TR-BERT PDF、ETAD、Temporal Corruptions、DToP README和DyT固定Block，主代理已补读，见[主代理补核](C:/Users/skywalker/Documents/ChatGPT/H65/h65_clean_adatad/reviews/20260912_external_239d098_dual_review/evidence/PRIMARY_RECHECK.zh.md)。初步搜索记录继续保留，不能把其中的未核准状态当最终结论。
