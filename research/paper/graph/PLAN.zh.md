# Graph TAD：完整实现和并行部署合同

用户于2026-09-14明确授权实现G-Repair-S、G-Context-S、G-Full-S/B及S机制对照。本轮在独立graph_tad_20260914工作树实施，原有79配置/374阶段继续；不修改正在训练的旧runtime。

新增7条完整课程：G-Repair-S、G-Context-S、G-Full-S/B为80轮；G-Full-S的fixed-local、no-referral、full-KV机制对照为40轮，采用同一80轮LR前缀。全部seed42，主课程保存10/20/40/60/80，机制保存10/20/40；不以中期成绩早停或作为其他课程启动门槛。

所有课程沿用Full-V2相同encoder/Cross资产起点、GT/外部feature/shared-full/同support目标、优化器与数据划分。从第1轮开始新的课程，不把已多训40轮的Full-V2 checkpoint伪装成同起点比较。新增图残差输出零初始化，保持同RGB/support下的旧恢复函数；图KV替换采用前5轮dense/sparse混合过渡，正式部署只执行稀疏路径。其过渡新增计算计入训练账本。

G-Context图KV在现有8×H×W packed attention域内工作：16个实际访问槽位，8个局部/8个多尺度初始候选，两跳referral最多4×4条路径，两次referral更新。原4/6/8/10路由层使用图访问，首4层、交错dense层和末层保留。当前A-MoD前层attention打分仍存在并计费，不能预扣那部分重复QK。K/V先投影一次后按索引读取，不为每个query重复投影。该实现是GM启发的单列表视频适配，不冒充原Qwen多边通道架构的忠实复现。

G-Repair使用[B,384,128]时间图、最终heavy融合和3次低维图更新；保持encoder不变。G-Full在L6/9/12更新时间图并通过零初始化残差反馈到heavy，同时使用图KV及最终anchor纠错。该时间图不包含5×5空间网格；空间关系来自已存在的packed patch网格，不能把池化Scout扩成虚假的空间观测。

G-Full在重encoder之前计算cheap初始图，给rate logits及actual frame utility网络提供上下文；32个有效时间分区至少保留一个真实观察，后续交换不得破坏覆盖。选帧不读取未计算的heavy特征。actual frame/T/D/S干预仍真实执行完整student，其差值包含图传播与恢复效果。

heavy写回按两个真实contributor的原candidate-pair位置分摊；不使用packed native下标硬覆盖原轴。跨度、有效贡献比例、实际depth/spatial质量参与可靠性权重。新增anchor一致性约束比较原轴预测在真实贡献位置的聚合与stop-gradient anchor，不重复已有边界加权feature loss。

图状态和地址为每次前向的局部变量，不跨视频/调用泄漏，不进入EMA作为运行时状态。参数纳入原learned-state/EMA/optimizer。full-plan shared student和frozen same-support参考关闭新增图模块；官方dense成本参考同样关闭新增图路径。

计算报告以真实矩阵/卷积2MAC为主：包含图query/key/geometry投影、候选兼容性、实际QK/AV、FFN、carrier反馈、decoder、scout/head。coalesce/top-k/gather/scatter作为非矩阵操作与延迟另报。逐窗ledger必须与实际算子计数一致；禁止按访问比例缩放整个encoder成本。

CPU验证涵盖稀疏图与小型dense oracle的值/梯度、重复地址归并、无效边/尾窗口、跨视频隔离、referral可达性、贡献坐标、零输出初始化、coverage保留。GPU预检在本模型的最终图路径上完成两次真实更新，并核验图关系参数梯度/变化、teacher冻结、EMA严格重载、无GT推理和实际profile；预检更新丢弃。随后同allocation直接进入完整课程。

全部配置和自身checkpoint评测立即进入现有唯一paper_dispatch持久队列，max-live10/long8/账户配额16。GPU不足仅决定调度先后，不能取消他人作业或静默替换既有模型。所有实现、已登记、Slurm提交、真实更新、完整成绩分别记录。
