# 原版 AdaTAD 降采样部署状态

2026-09-14 16:20，SSH 与唯一控制器已核验。代码 `0313f136d076552e84a459b91869ad59169f4c3b`；远端 runtime 为 `support_review_20260914`，统一 owner 为 `paper_20260913`，控制器 PID `2408086`。

|实验|当前状态|作业 / 接续条件|
|---|---|---|
|原版 AdaTAD-S，K384 直接降采样|Slurm PENDING|1289969；同 allocation 先做两更新技术检查，再重新加载官方 EMA 全测|
|原版 AdaTAD-B，K384 直接降采样|Slurm PENDING|1289968；同上|
|原版 S/B K384，GT-only 80 epoch|已登记，未取得 Slurm ID|仅等待各自技术预检和训练槽位；不等待直接评测成绩|
|原版 S/B K768，官方 EMA 原生参考|已登记|评测槽位自动补位|
|S/B 适配模型 epoch80 online 评测|已登记|只依赖自己的 terminal checkpoint|

原有 75 条课程保留。新增 4 个配置、2 条训练课程、10 个持久阶段：当前合计 79 个配置、77 条训练课程、374 个阶段。所有完整训练只使用 seed42。

CPU 验证已通过 3 项测试；S/B 各 499 项官方 checkpoint 严格加载成功。实际解码的完整/尾/短窗口输入逐元素等于原输入的 `[::2]`，有效帧分别 768→384、503→252、253→127，GT/输出秒坐标不变，200 训练 / 211 测试 / 792 测试窗口保持一致。GPU 预检仍未执行，因此不能声称已完成真实训练验证，也没有新的 baseline mAP/GFLOPs。

只延期了两个尚未开跑的自有补充任务（Dense-S 1289886、ANet-B 技术预检 1289888），保留所有历史 attempts 和重排回执。运行训练、V1 checkpoint 续跑和 ANet CPU 准备步骤均继续。既有持久跟进已更新为覆盖本次新增任务。

结果会将“官方权重直接降采样”与“80轮训练适配后的原版降采样”分开，使用实际完整模型 FLOPs、最佳完整测试 mAP 和 epoch80 终点比较。当前 Uniform Full 仍保留为完整框架内的强对照，不用新基线替换或掩盖它。只有新结果确实支持，才报告完整路线的性能收益。

完整设计：[PROTOCOL.zh.md](PROTOCOL.zh.md)；验证：[cpu_validation.json](cpu_validation.json)；提交回执：[submission_receipt.json](submission_receipt.json)。
