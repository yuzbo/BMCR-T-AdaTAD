**完整论文实验协议与当前判断（2026-09-13）**

可以进入完整论文实验阶段；当前已测结果还不能作为最终论文结论。新的完整联合模型、独立训练消融和跨数据/骨干/检测头实验直接并行提交，不以另一条路线的 mAP 决定是否启动。运行正确性、完整数据和本实验自己的检查点仍是必要依赖。

模型从真实 H65-S60 / BMCR-B60 权重出发。THUMOS 的 Cross 恢复器复用已经训练完成的 R03 epoch20 作为共同初始化，独立训练对照随后各自更新；这不是随机从零训练。S/B 主模型和均匀选帧各 3 个种子（3407/3408/3409）、80 epoch；稠密对照 80 epoch；主要消融 40 epoch，与主模型的 epoch40 比较。ANet 全训练集 15 epoch；InternVideo1-MQ 与 TadTR 40 epoch，分别有同骨干/同头的完整、均匀、稠密对照。实际更新数由完整训练集长度和 accumulate=2 计算，ANet 不沿用 THUMOS 的每轮 100 更新。

40 epoch 的 THUMOS point-head 消融采用与主模型相同的 **80 epoch 学习率调度前缀**，在第 40 轮停止；主模型的前 40 轮与之使用相同更新数和学习率轨迹。不能把已经衰减完的 40 轮 cosine 与 80 轮课程的中点当作同配方。TadTR、InternVideo 的完整/均匀/稠密三条均使用相同 40 轮调度；ANet 三条均使用相同 15 轮调度。

|组成|最终实现|证据与限制|
|---|---|---|
|时间|768 候选帧的 scout；H65/BMCR 初选；真实换帧收益路由|均匀和随机独立训练，加同 checkpoint 替换选点；16 帧仅是 VideoMAE 的内部计算组织|
|深度|交替 A-MoD；首末 dense；前一个 dense attention 的入注意力排名|真正压缩 Q/K/V 与 FFN；保留未计算状态并执行全局 TIA；24 层按同规则扩展|
|空间|被深度接纳的 token 内再选 heavy FFN，余者 light FFN|包含 token/tile、attention/uniform、去掉 light、full-KV 和分辨率对照|
|恢复|两层 Cross latent residual decoder；中间层与最终层 memory；来源与 scout context|另外实现 TCN、插值、官方 MAE decoder 初始化和同架构随机初始化|
|任务学习|独立可训练 student point/TadTR readout；Adapter/scout/light/decoder/router 联训|external teacher 冻结；另有冻结 head/encoder、全 backbone 微调以及损失消融|
|联合预算|15 个可执行 T/D/S 菜单；有符号分类/定位收益与不确定性|以实际算子校验的预算表筛选；是有限菜单分配，不宣称任意 token 全局最优|
|干预监督|真实 frame/T/D/S 与 full-reference→joint-menu 重执行整个 student、全局 TIA、decoder、检测头|joint比较确保包括混合预算在内每一个菜单项都收到真实标签；feature repair只作代理|
|校准|固定最终 student、训练视频四折 OOF 拟合路由器，全部训练标签重拟合后部署|representation 已见全训练集；OOF 只指 router 拟合。拟合后覆盖率是校准诊断，不是独立测试保证|
|完整时间轴|THUMOS：768 candidate / 384 native / 768 detector；ANet：768 / 384 / 192|支持 ANet 重复物理帧，保留原时间坐标，插值基底聚合同时间锚点|

训练同时覆盖混合稀疏状态和 shared-full 分支，部署使用实际稀疏执行。它不是只在稠密状态训练、上线后才突然删 token。THUMOS 技术检查与正式训练合并在同一 Slurm allocation：先做两次真实更新，验证通过后丢弃这两步，从固定初始化接续正式完整课程；训练分片以 checkpoint 与游标续跑。课程每个预登记点做完整测试，保存 EMA 峰值及终点 online，训练仍跑满。

ANet 只在 10024 训练/4728 验证视频的共享准备正式 READY 后开始完整训练评测，训练集按官方 GT 过滤规则报告实际参与视频。定位为 class-agnostic，CUHK top2 分类，mAP@0.50:0.05:0.95。官方 ANet-S TAD teacher 已实际读取并核验。B 与 InternVideo 的完整本地权重已经取得，远端传输完成后由 tensor 校验激活相应任务；不使用随机权重代替预训练权重，也不以子集测试冒充完整 ANet。

主要判断为完整模型实际矩阵/卷积 FLOPs（2 MAC，含 QK/AV、attention 路由、scout、轻支路、恢复与检测头）和完整 TAD 测试的最佳性能。动态策略记录整个测试集的执行预算分布、总 GFLOPs、全窗口均值和分位数。延迟、解码/NMS E2E、显存、训练分配时间与各类 teacher/action 查询另外报告；不作为路线淘汰条件。显存同时报告进程峰值和扣除已有驻留数据后的增量峰值，独立推理不加载训练教师。

模型、预算、消融和泛化均在 configs/paper 与 plan.json 中完整登记。主实验和各类消融共 60 个配置；加上技术检查、终点测试、校准、八格因子测试、真实案例图和 1000 次视频配对 bootstrap，共 233 阶段。调度器最多占用本程序与继承旧 FPW 合计 10 个活动/排队作业，其中至多 8 条长训练；同时遵守账户总 16 作业限制。BMCR80 的原课程及独立评测控制器继续，不复训已完成模型。

目前的新论文协议还没有完整测试结果。旧探索的 Cross-B EMA10 最佳 68.5792%，Cross-S EMA20 64.5743%；相应官方稠密模型为 71.1280% / 69.0126%。B 的恢复效果值得继续，但官方性能差距尚未闭合。另保留历史 H65-S 65.3857% 的真实高分记录；它的课程和修正配方不同，不能把 64.5743% 称为全部历史 S 模型的最高分。现有 Cross-S 对插值基线的视频配对 95% 区间仍跨 0，学习选点的充分必要性也尚未证明。

绝对计算量比较还揭示：Cross-B 68.5792% / 4094.12G 被官方 S 的 69.0126% / 2347.89G 同时超过。不能只强调 B 对自身稠密模型节省约一半计算，必须同时呈现跨骨干比较。相关固定参照来源为 fidelity_20260911/FINAL_COMPARISON.json，绘图保留真实来源路径。

论文需要回答四个可被否定的问题：在相同完整计算预算下是否优于 H65/BMCR 与均匀选点；恢复误差与实际任务损失之间是否存在稳定可利用的联系；三轴联合预算能否降低多轴同时稀疏的负交互；这些结论能否跨随机种子、ANet、InternVideo1-MQ 和查询头成立。若只有特征相似度变好、TAD 不提升，或者三轴组合持续被简单时间采样支配，就不能支撑目前的完整模型主张。

图表生成器：tools/paper_analyze.py（结构、计算—精度、曲线、消融和三轴八格）；tools/paper_diagnostics.py（固定抽样真实视频、完整时轴选点、heavy mask、层间漂移）；tools/paper_calibrate.py（真实 repair/action 对比和 OOF 不确定性）；tools/frame_errors.py（已核对官方 AP 的 paired-video bootstrap 与误检分布）。所有结果图读取实际记录；未产生的数据不补点。

七项远端 CPU 验证已通过，包括真实 24 层执行/算子计数、混合预算反向传播的 TIA 时间轴、可训练 head 与冻结参照、ANet 坐标与重复物理帧、预算约束。S/B point、S/B TadTR、S 全骨干微调、S 官方 MAE decoder、ANet-S 这七个可用组合通过实际权重构建。GPU 上的真实长课程与完整结果仍须看权威 deployment.json，登记配置与进入 Slurm 队列分别报告。
