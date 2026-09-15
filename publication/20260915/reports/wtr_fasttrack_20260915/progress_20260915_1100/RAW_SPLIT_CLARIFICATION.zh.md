# Raw mini-bank划分口径

核对依据：远端revision_27d557e/runtime/protocol.json、mini_bank全部90个group记录、Core research/paper/resources.local.json。分区代码为tools/raw_prepare.py:38–41。

**划分单位是video_id，三个分区视频互斥，不是仅state隔离。** 实际组文件的视频集合与manifest一致，三个两两交集均为空。

|分区|200训练视频中的完整划分|Raw mini-bank视频数|配对support state数|
|---|---:|---:|---:|
|fit|160|16|30|
|calibration|20|4|7|
|router-label holdout|20|4|8|

Raw mini的4个holdout视频为video_validation_0000057、0000058、0000059、0000163。它们都属于完整20-video holdout，是其子集。Raw与Core使用相同seed42完整160/20/20视频分区，已逐集合核对一致；但Raw本批仅覆盖4个holdout video，不能称完成Core的20-video holdout验证。8个state不是8个独立视频。

detector及历史V2初始化可能已训练过全部200视频，因此只称router-label holdout。当前Raw Value头尚未拟合，已有数字仍是mini-bank诊断，不是Value泛化结果。

图表区分候选帧池、合法交换空间及实际查询动作。当前合法性主要是固定预算、移除已选帧、插入未选合法帧与显式padding；排序pair/16-observation pack是消费规则，没有额外max-gap约束。16-pair几何提议只是合法交换空间的子集；不把它冒充穷举。38/45相同的是实际查询的swap集合，candidate pool并不相同。

13820仅为Decord返回的解码帧数量；codec/GOP内部实际解码量仍未测。
