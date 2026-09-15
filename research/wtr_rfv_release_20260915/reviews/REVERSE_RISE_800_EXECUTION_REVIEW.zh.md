# 800bcd1：反向交换实现、实际启动与 RISE 选择复核

2026-09-15。固定已发布 source：800bcd10a66c69d0d57142ac02389adbc45d2975。现有工作树的后续修订另列，不能将其正确性回写为旧版本已经成功运行。本任务只读代码、已完成 GPU 回执及历史导出，并做 CPU 算术复核；没有启动模型训练、GPU重放或调整队列。

**后续更新：统计复用修复以4aa1ca2发布，六个 head 已完成，STRUCTURE_SIGNAL_FAIL已独立复核。** 见[4aa完整实验审阅](C:/Users/skywalker/Documents/ChatGPT/H65/reports/wtr_rfv_review_20260915/REVERSE_4AA_EXPERIMENT_REVIEW.zh.md)。下文保留800批次启动失败和重放通过的历史事实。

**结论：反向表示与 P1/P1-bi/P2 对照主链合理；6 对真实重放的数值足以通过技术准入。800 的 mini 在训练前因 CPU/GPU 统计逐位比较而退出，尚未产生六个 head 或科学结果。RISE 选择导出复核正确：本次留出集 Future 与 Post 的选择完全相同，旧 Forecast/FVD 未通过状态保持。**

## 1. 结构代码与设计一致的部分

- [reverse.py:26](C:/Users/skywalker/Documents/ChatGPT/H65/h65_clean_adatad/rfv_sprint_20260915/h65/rfv/reverse.py:26) 对每个候选构造真实 S'，交换端点特征，用有效 K 更新 support mean，重算四个间距，反向 membership 保持 (1,0)，支持集往返及 padding 一致。没有从 cheap192 近似恢复新增 hidden。
- [reverse.py:60](C:/Users/skywalker/Documents/ChatGPT/H65/h65_clean_adatad/rfv_sprint_20260915/h65/rfv/reverse.py:60) 限定 Plain-M，同一个 head 分别计算正反向。P1 两项 Huber 平均，P2 对差分输出做 Huber；分量尺度和 valid reduction 与原方案一致。
- mini 的 fit8/held8 按原物理排序交替划分；正反向保留同一候选归属。P1-bi 仅以同一 P1 raw 权重改变读出，无新增拟合。P1/P2 同 seed 初始化、state 顺序及 2000 updates，cal 只作描述性读出；不新增 inner10/outer20 评价。
- 主门为 held8 的 P2−P1-bi raw regret、视频配对区间和各 seed 方向，并与 STOP/随机合法交换比较。原 R1 FAIL、位置族外推局限和长训限制均保留。

两个只读子代理分别核对 mini 与 RISE 实现；主代理核对反向特征、真实重放入口、发布版本差异及实际执行记录。

## 2. 已确认的启动问题：重算统计不应要求跨设备逐位相等

已发布 800 的 tools/rfv_reverse_mini.py:155–158 在 CPU 上重算 fit8 normalization，并与原 CUDA 拟合保存的 buffers 使用 torch.equal 比较。实际运行在 19:57:32 结束：

|阶段|退出码|耗时|
|---|---:|---:|
|RISE 历史导出|0|4.24 秒|
|反向真实合同|0|22.84 秒|
|反向 mini|1|2.42 秒|

错误为：ValueError: Original fit normalization changed。尚未进入训练，因此不能写成六个 head 已运行、科学 FAIL 或仅等待结果文件。

主代理在 CPU 上使用同一 25 个 fit states/200 个候选独立复算，得到：

|buffer|CPU 重算对原保存值的最大绝对差|
|---|---:|
|input_mean|8.9406967e−8|
|input_scale|5.9604645e−8|
|target_scale|5.8207661e−11|

均值误差除以原 input_scale 的最大值为 1.5464479e−5。这支持舍入差异的解释，不支持输入合同已经改变。

**本次修复应直接复用已绑定原 bank/fit8 的原 buffers，保持本来要求的原 normalization。** 工作树已采用这一办法，并核对原各 seed 保存统计一致；这是后续修订，未冒充为已发布 800 的行为。本次 CPU 复核没有模型前向或 optimizer update。

证据：[独立统计比较](C:/Users/skywalker/Documents/ChatGPT/H65/reports/wtr_rfv_review_20260915/REVERSE_800_NORMALIZATION_RECHECK.json)。实际日志：AutoDL /root/autodl-tmp/rfv_20260915/logs/reverse_800bcd1.log；实际阶段回执为 results/800bcd1/sprint_finished.json。

## 3. 重放验收门的盲区与本次实际数值分开

800 的 [rfv_reverse_contract.py:89](C:/Users/skywalker/Documents/ChatGPT/H65/h65_clean_adatad/rfv_sprint_20260915/tools/rfv_reverse_contract.py:89) 以同四次前向定义：

\[
r=\max(\|L_0-L_3\|_\infty,\|L_1-L_2\|_\infty),\quad
\epsilon=\max(10^{-8},3r).
\]

但正反残差恒有

\[
\|(L_0-L_1)+(L_2-L_3)\|_\infty\le2r.
\]

所以仅用该残差小于 epsilon 不能证明重放稳定；本次 replay 本身还没有独立上限，缓存标签误差门也可能随之被放宽。

实现负责人已接受修订：以后由原 bank 的已保存 no-op/replay 误差确定独立界，并显式限制本次 replay。主代理已核对工作树差异；后续发布版本需要单独标记。

**本次实测没有落入这个盲区。** 我只读回收了全部 6 对、24 次完整前向的记录，最大值为：

|量|最大误差|
|---|---:|
|同支持重放|0|
|正反 gain 之和|0|
|对旧 bank 标签|0|
|旧 407 重建|0|
|反向 407 对原廉价视图直接重建|5.9604645e−8|
|反向差除以原 input_scale|5.6639328e−7|

支持集往返、normalizer 固定和主体计算量匹配也通过。因此可以基于实际数值确认本次技术准入，原干净 GPU 重放无需为上述门槛修订重复运行。只要反向输入/检测函数保持不变，新拟合修复版本应明确引用 800 回执及其独立复核，保留两个 source 身份。

原始回收：[REVERSE_800_REMOTE_RESULTS.json](C:/Users/skywalker/Documents/ChatGPT/H65/reports/wtr_rfv_review_20260915/REVERSE_800_REMOTE_RESULTS.json)。

## 4. RISE：分数与候选排序改变，但留出选择没有改变

此次导出使用原 d62ea55 的 anchor/post/真实累计 EMA，在同 s40 和候选集合重算完整函数，沿用 β=1.1。STOP 分数为0，零分并列优先 STOP；top2 margin 包含 STOP。没有重新拟合、查询 CF 或重选 β。

主代理对全部 129 条 seed-state 的原始候选分数独立重算外推公式、选择 ID、runner-up、margin、实际收益、排序变化及汇总。没有不一致；FP32 外推公式和 margin 最大误差均为0。

|分区|视频数|seed-state 数|最终选择改变|候选排序改变|STOP 改变|
|---|---:|---:|---:|---:|---:|
|fit|25|75|5|75|0|
|calibration|8|24|1|23|0|
|原 B0 holdout|10|30|0|29|0|

留出集上 Future−Post 的实际选择收益差为0，明确对应**最终选择相同**，不是不同选择恰巧得到相同收益。代码并非普遍禁止换选择，因为 fit/cal 中确实发生了变化。

30 条留出记录中，17 条满足 \(2\|\Delta z\|_\infty < \mathrm{margin}_{Post}\) 的充分不变条件；另外13条不满足这个较保守的界，但实际选择仍未改变。不能把“界未满足”解释成应该发生选择改变。

导出回执的 3 seeds × 4 functions 均复现旧六项指标，最大误差0。主代理此次独立复算的是候选分数上的算术与决策，没有再次加载快照重跑模型。历史 β 使用 later-cal 的回顾式边界仍在，fit/cal 的变化不构成未来泛化证据，也不支持继续增大 β 追逐选择变化。

证据：[独立复算](C:/Users/skywalker/Documents/ChatGPT/H65/reports/wtr_rfv_review_20260915/REVERSE_RISE_800_RECHECK.json)、[逐候选分数](C:/Users/skywalker/Documents/ChatGPT/H65/reports/wtr_rfv_review_20260915/RISE_800_CANDIDATE_SCORES.jsonl)。

## 5. D/S 更新与下一步范围

据实施任务回收的同 epoch40 指标：

|轴|Value mAP / %|Uniform mAP / %|V−U / 百分点|
|---|---:|---:|---:|
|D|64.2601|64.3341|−0.0741|
|S|64.0616|64.2459|−0.1843|

当前仍未领先。训练进度已超过40不代表已有更晚评测；这仍是中途单 seed 的跨课程比较。

四课程的 preclip grad_norm>1 比例均约96%，Uniform 也如此。该事实说明裁剪常发生，不能单凭它特指 Value 导致额外裁剪或定位掉点根因。继续固定 checkpoint 的策略切换与分组梯度/clip 系数诊断合理，旧课程不改。

证据：[DS_LIVE_PATHS_AND_CLIP.json](C:/Users/skywalker/Documents/ChatGPT/H65/reports/rfv_sprint_20260915/DS_LIVE_PATHS_AND_CLIP.json)。

当前唯一待接续的新拟合仍为反向 mini 的6个 head，由实施任务发布统计复用修复后继续；未产生其科学结论。packing 覆盖/partner96 暂缓，原 R1/G1/Forecast FAIL 及长训限制保持。
