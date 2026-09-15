> 2026-09-13 最新用户决定：当前阶段每个完整实验仅 seed 42 一次；取消所有重复种子课程。历史结果保留原始种子，不重新标记。

**论文完整模型与并行实验：2026-09-13最新用户执行授权**

用户要求直接落实Pro最终模型、论文完整实验/消融/图表，以及ActivityNet、InternVideo骨干和查询式TAD头；不再用逐步控制变量或其他路线的性能作为启动门槛。完整模型及其消融从同一实现派生并行执行。代码、52配置/217阶段和统一调度已部署；6条包含技术检查的完整训练、ANet-S技术检查及ANet数据准备已经进入Slurm，尚无新论文模型完整测试结果。实时状态以research/paper/deployment.json及DEPLOYMENT.zh.md为准。

活动树：paper_20260913，分支codex/fpw-paper-20260913，基于2abb417。旧FPW、BMCR80及已完成权重/结果保留，原16候选clip选择和DS3仍取消。主代理改代码与验证；探子只读探索。

完整模型：H65/BMCR候选帧先验；多层/来源感知的Cross全时间轴latent恢复；独立可训练student TAD readout；冻结external teacher或明确命名的shared-full student监督；真实frame/depth/space干预；带不确定性的有限预算菜单联合调度；A-MoD交替层、首末dense、attention排序；空间重轻FFN；与全分支GT及特征一致性联合训练。主要训练新decoder、student head、原Adapter、light、frame/action/budget router。另设完整骨干微调、冻结head、无多层恢复/效用/self-KD等并行消融。

预算分配先实现可执行的有限T/D/S菜单，按实际完整模型FLOPs约束选择；它不被宣称任意token全局最优。维内token排序保留用户指定attention规则。动作标签来自真实重执行，repair proxy与actual action分开；校准使用训练视频分组。固定K换帧不对接近零的增量cost机械相除。

数据：THUMOS保留200训练/211测试、完整792窗口及五阈值mAP。ANet使用全部可验证的官方训练/验证协议、class-agnostic定位加CUHK分类及0.50:0.05:0.95指标；原生encoder768候选帧与detector192须显式区分，重复物理帧时间必须被支持。数据准备未完整前不把7491个成品当完整ANet，teacher未存在时不能用随机/THUMOS teacher冒充ANet教师；可并行采用GT+shared-full学生学习并训练dense参考。

骨干泛化：先使用OpenTAD原生支持的InternVideo1-MQ ViT-L/24层/patch16/tubelet2，公开权重必须实际取得并核验；不以离线特征head实验冒充RGB三轴计算泛化，不将其称为InternVideo2。检测头泛化采用OpenTAD已有TadTR/DeformableDETR基础设施，与原point head共用同一恢复表示；输出KD须按head语义匹配，不能逐位置混合不同数量的query和point输出。

训练/评估：主模型与消融同数据/共同初始化/匹配课程并行，THUMOS主模型80轮、主要消融40轮且使用同一80轮学习率调度前缀；ANet15轮，InternVideo/TadTR40轮。初始THUMOS技术检查后在同一allocation直接完整训练，周期完整评测也在训练allocation中完成。记录实际更新、teacher/extra/full/action查询、训练分配时间、峰值与终点。具体课程和矩阵已经固定在configs/paper和research/paper/plan.json，不以早期排名否定路线。

论文判断：目前恢复器有正向证据，但尚有官方精度差距，缺完整跨数据/骨干/head证据，现有结果只能作为阶段性结果。目标是实际FLOPs—最佳mAP前沿改善与可靠机制证据，延迟/显存/E2E照常报告但不作路线淘汰门槛。所有未测结果为空。

交付：生产模型/训练/评估/统一调度；完整实验与消融矩阵；真实视频时轴案例、pair/gap/空间heavy/层间漂移/feature-task误差/校准风险图；三轴问题说明和真实算子成本分布；图表和论文主张可追溯到原始记录。原GPU资源配额和只操作自有作业规则继续适用，新的完整模型应优先获得后续空槽位，不等待旧探索矩阵全部完成。
