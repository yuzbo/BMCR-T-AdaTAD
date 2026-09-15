**完整论文实验：执行与证据状态**

已登记 60 个训练配置、233 个阶段。状态：{'WAITING_INLINE': 7, 'PENDING': 7, 'WAITING': 219}。

新论文协议完整测试 0 条；旧探索协议完整测试 47 条。

以下是各已测配置的预登记 EMA 里程碑最佳值；终点、逐阈值 AP、原始路径和预算分布保存在 full_test_records.json。尚未测量的配置没有填入结果。

|协议|配置|最佳 epoch|mAP %|GFLOPs|
|---|---|---:|---:|---:|
|pilot|R03_cross_b|10|68.579|4094.12|
|pilot|C01_vector_condition_b|0|67.272|4093.09|
|pilot|R01_interpolate_b|0|67.272|4093.09|
|pilot|S02_token48_b|5|65.745|3745.89|
|pilot|D02_amod50_b|5|63.449|3355.67|
|pilot|R01_anchor_head_b|0|62.618|4093.09|
|pilot|J01_joint_b|5|60.503|3122.73|
|pilot|R03_cross_s|20|64.574|1226.36|
|pilot|R02_tcn_s|10|64.532|1226.34|
|pilot|R04_feature_only_s|10|64.263|1226.36|
|pilot|R01_interpolate_s|0|63.825|1225.47|
|pilot|S02_token48_s|5|63.465|1170.02|
|pilot|D02_amod50_s|5|61.137|1027.00|
|pilot|J01_joint_s|5|59.442|969.36|
|pilot|R01_anchor_head_s|0|57.754|1225.47|
|pilot|D02_amod125_s|5|51.219|871.95|

判定条件：以实际完整模型计算量与 TAD 性能判断；延迟、显存、E2E 和训练开销另行呈现。
论文主张须由同课程独立训练对照、三个种子、视频配对 bootstrap、ANet、InternVideo1-MQ、TadTR 和真实干预数据共同支持。
激活相似度只用作冗余代理，不能单凭相似度宣称信息可删除或任务无损。
