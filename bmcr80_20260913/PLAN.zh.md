**修正BMCR-T：总80轮完整S/B实验**

用户2026-09-13明确要求完整部署、提交修正BMCR并将训练拉长到80epoch。按总轮次执行：复用已完成的修正warm20EMA，新增joint60；每骨干6000次新joint更新，S/B共12000次，预检与效用审计另计。共享warm不重复训练。

本实验只使用H65/BMCR候选帧选择合同：K384/global-TIA192/native192/selected-rank384检测，NMS前回映。DS3及原16候选clip选择路线已取消，不在本实验重新注册。

固定200训练视频、211测试视频/792窗口、seed3407、batch2、每轮100次更新。冻结VideoMAE非Adapter参数，训练原Adapter、检测器、scout/条件效用；保留已修正ASFormer LR和裁剪合法端点，既有BMCR真实交换目标、uniform companion和2000更新课程不改。

学习率日程预先设置为joint60：原3个joint epoch的线性warmup与余弦衰减到第6000更新，最低LR沿用上游1e-8。warm20EMA只提供模型状态，joint重建optimizer/scheduler，EMA从该状态起始；不是加载warm optimizer续跑，也不续旧BMCR终点。

初始化：每骨干从自己的修正warm20重新做训练集128窗口效用审计，诊断fit/holdout按训练视频隔离，尺度只由fit计算。scales记录相应warm路径、EMA来源、修正版本和分量顺序。真实GPU预检与正式joint共用同一warm/scales加载函数，2步分别覆盖无反馈和完整joint反馈，S/B两预检都通过后才允许任一正式训练。

每骨干在总25/30/35/40/45/50/55/60/65/70/75/80轮做完整测试，共24个候选。选择五阈值平均mAP最高的EMA，同分取较早者；这沿用用户指定的测试峰值选择，不能称未见测试估计。单独保存、报告第60/80轮和峰值的完整/部分/短窗profile；若峰值就是60或80，复用对应profile，不重复mAP或测量。

每步保存loss分量、梯度范数、各参数组LR、时间与实际更新编号。报告聚合每轮均值及batch loss p10/median/p90、五阈值测试曲线、峰值位置和60→80变化。原始预测及checkpoint全部保留；单seed的batch分布不是训练种子方差。

第60轮的80课程LR不同于旧60课程终点，不能将新旧差值称为纯延长20轮效果。旧BMCR仍是旧错误LR/crop配方；修正H65仍为原60课程。新实验应分别比较同轨迹60→80、原60范围峰值与全80范围峰值、和历史H65/BMCR参考，不混合不同选模范围。

独立worktree/branch：`bmcr80_20260913` / `codex/bmcr80-20260913`。独立远端源根：`/data/run01/sczc063/yuzibo/h65_clean_adatad_20260910/bmcr80_20260913`，权威回执是其中`bmcr80_20260913/deployment.json`。新控制器只登记36个阶段：2audit、2preflight、2train、24test、6profile（实际可去重复用）。最多2个自有audit/preflight/train加1个test/profile，均包括PENDING；账户16作业上限，固定已验证4090池。不取消其他项目作业，不自动重试失败。

本次不加入新KD、受限交换、向量化路由、原时间融合、空间或深度压缩。它们作为下一轮独立实验建议，避免让本轮配方比较同时改变模型结构。
