# 原版 AdaTAD 直接等间隔降采样对照

本实验响应用户的新要求：用原版 AdaTAD 的直接降采样，检验完整路线相对简单方案的收益。此前 `point_uniform` 是内部完整框架的 Uniform Full，含 Cross 原轴恢复、scout、混合 D/S 课程及蒸馏，不能把它称为本实验的原版降采样。

## 新增矩阵

|对照|S|B|训练与评测|
|---|---|---|---|
|Native dense，K768|官方 EMA|官方 EMA|零更新，完整 211 视频 / 792 窗口|
|Native uniform，K384 direct|官方 EMA|官方 EMA|零更新，完全等间隔下采样，完整测试|
|Native uniform，K384 adapted|seed42|seed42|各 80 个新适配 epoch；GT-only，100 updates/epoch|

共新增 2 条训练课程、4 个冻结权重评测、2 个共享技术预检、2 个终点 online 评测，共 10 个持久阶段。80 轮内还自动执行 10/20/40/60/80 EMA 全测。原有 75 条课程保留；不启用多种子，不恢复已取消的原 16-candidate-clip 选择路线。

## 模型与时间坐标

保持 upstream OpenTAD `ActionFormer` 和 `VisionTransformerAdapter` 实现不变。官方 S/B 配置都使用 768 候选、源视频间隔 4 帧。首先执行官方 crop/sliding-window，再执行 `frame_inds[::2]`，将输入变成 384 帧；原视频起点、裁剪窗口、测试窗口列表均保留。

模型路径：384 RGB → 24 个内部 16 帧计算包 → 192 个 tubelet 时间位置 → 12 层完整 attention/FFN 与全层 TIA → 空间平均 → 官方原有 linear interpolation 到 384 → 官方 projection/neck/point head → 按源帧 stride=8 映射回秒。这里的 24 个计算包不是学习选择原始 clip。

GT 坐标除以 2、mask 取 `[::2]`、输出 `snippet_stride` 乘 2。后处理仍调用官方 `(segment * snippet_stride + window_start_frame + offset_frames) / fps`。K768 为采样恒等对照，模型配置与官方完全相同；K384 的检测轴和 padding 长度都为 384，不能暗中恢复成 768 再称原版直接降采样。

官方自身的 192→384 tubelet 上采样保留。没有新增学习式 decoder、Cross、scout、frame router、depth router、spatial heavy/light、budget menu、external teacher 或 self-KD。官方完整 dense checkpoint 作为初始化资产使用，不执行教师前向。

## 训练与解释范围

K384 adapted 从同一份官方 dense EMA 重新建立 optimizer，进行 80 个新 epoch，seed42，真实 batch2，全部 200 训练视频。采用官方 AdamW 分组：检测头/projection lr=1e-4、adapter lr=2e-4、weight decay=.05，并保留官方 head no-decay 规则；其余 VideoMAE 权重冻结。5 epoch warmup 后 cosine 到 80，EMA=.999，梯度裁剪 1。BF16 与当前论文评测一致。课程/GT 裁剪沿用官方实现，技术预检的两次更新丢弃，正式模型重新加载官方权重。

该课程采用官方优化器配方；当前 Full-V1/V2 是另一套完整方法训练配方。因此它检验相对一个经过充分适配的原版简单方案的系统级收益，不单独归因于某一个模块。分别报告 direct 与 adapted，避免把零训练采样失配当作完整模型创新收益。只有结果实际优于它，才声称性能提高；否则保留结果并调整论文主张。

全部 80 轮完成，不以早期分数决定早停。保存 optimizer、scheduler、EMA、随机状态与采样游标，资源切片后接续同一次 seed42。最佳与 epoch80 终点分别报告；最佳采用预登记完整测试里程碑，属于 test-based peak，不能当独立验证集选择。

## 计算与报告

按真实执行运算统计完整模型 GFLOPs（矩阵/卷积 2MAC，含 attention QK/AV、所有 TIA、projection 和 head）。采样没有 RGB 全轴 scout 前向；数据解码成本不计入 FLOPs，但保留全评测耗时。报告 mAP@0.3–0.7、平均 mAP、完整窗口和全测试平均计算、GPU resident latency 分布、显存、训练更新与墙钟成本。延迟不作为路线淘汰条件。

结果汇总给 direct 评测独立 `evaluation_id`，不会与 adapted 最佳 checkpoint 混合。Pareto 图分别标记 Native dense、Native uniform/direct、Native uniform/adapted、Uniform Full、Full-V1、Full-V2、PBD-style、Static。H65/BMCR/Cross 仍是作者内部方法；本地复现的稀疏 AdaTAD 不是官方已发表的稀疏性能数字。

## 验证与部署边界

CPU 检查覆盖：真实官方 checkpoint 严格加载；K768 模型配置恒等；K384 完整/尾部/短视频 RGB 与原输入 `[::2]` 完全相等；GT 与输出秒坐标不变；200/211/792 数据一致。GPU 预检必须实际完成两次 GT 更新、adapter/head 更新、EMA/state 重载和不依赖 GT 的有限推理，才接续该模型训练。

所有阶段加入现有唯一 paper 控制器，维持 10 个活动/排队作业、最多 8 条长训练及账户配额。新 K384 direct S/B 优先使用评测槽位，GPU 预检与 direct 评测在同一 allocation 接续执行。排队不是运行，CPU 检查通过不是 GPU 训练成功。
