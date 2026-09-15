# Graph TAD 部署状态

2026-09-14 19:22:28核验。代码841307e3af9ad6547535a61c9c0b23b9f308f383已部署至服务器graph_tad_20260914并推送GitHub分支codex/graph-tad-20260914。

|实验|轮次|状态|
|---|---:|---|
|G-Repair-S|80|已进入持久队列；自身预检随训练allocation执行|
|G-Context-S|80|已进入持久队列；自身预检随训练allocation执行|
|G-Full-S|80|GPU预检1290083 PENDING；通过后自动排正式课程|
|G-Full-B|80|GPU预检1290082 PENDING；通过后自动排正式课程|
|Fixed-local / S|40|已进入持久队列|
|No-referral / S|40|已进入持久队列|
|Full-KV / S|40|已进入持久队列|

新增7配置/36阶段；总计86配置、84训练课程、410阶段。所有完整实验仅seed42，主模型80轮、机制40轮匹配80轮LR前缀，保留全部里程碑和终点。每个模型只依赖自己的技术检查、checkpoint和GPU资源，不设置成绩门槛。

已通过10项CPU数学/梯度/坐标/真实ViT/算量检查，以及7个实际资产构建、优化器覆盖和严格状态重载。GPU预检尚未运行，因此当前没有图模型的GPU性能、训练更新或mAP成绩。

唯一owner仍为paper_20260913，控制器已由2408086更新为3506502，现场确认健康。只延期了尚未开跑的V1续跑请求1289884/1289885，其checkpoint和80轮课程仍保留；没有取消任何运行训练。原版AdaTAD降采样1289968/1289969和PBD-B1289883保持原队列申请。

结构、训练和计费细节见[IMPLEMENTATION.zh.md](IMPLEMENTATION.zh.md)，原始登记和作业回执见[registration.json](registration.json)、[submission_receipt.json](submission_receipt.json)。图拓扑、来源可靠性、固定selection和D/S masks下的full-KV诊断及完整Pareto/机制图入口已经实现，待实际检查点自动运行。持续跟进已更新为覆盖本轮全部任务。
