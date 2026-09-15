# 现有 descriptor 的一次固定可分辨性诊断

2026-09-15。按实现任务确认，本任务独占完成这一次CPU观察性分析。未训练、未进行模型前向、未生成CF标签、未使用GPU或outer20，也未建立新的近邻router/proxy。

**结论：没有发现完全相同full407输入却有不同收益的直接反例；也没有检出该固定输入距离下稳定的局部收益一致性。** 这与8/8动作留出失败相符，但不足以在“根本观测不足”和“当前表示/拟合方式不合适”之间作唯一因果判断。

## 固定方法

只使用原mini fit中的25个有动作state，每state仍按已经执行的8拟合/8留出划分。脚本逐项核对原action IDs，没有重新选择划分。

使用已保存seed42 Plain-M头的fit8 `input_mean/input_scale`，按模型实际float32方式标准化full407。只在同一state内部计算留出8动作到拟合8动作的欧氏RMS距离；exact distance tie等权处理，本次没有tie。support/global两个96维块在同state内均确认为常量，因此不会通过跨视频外观近似混淆比较。

主统计为：最近邻的`|gain_held−gain_fit| / 同state gain range`，减去同state均匀配对的对应值。辅统计为明确符号下的异号率差。负值表示近邻更一致。

随机参照固定为：在每state内置换8个fit标签与输入的对应关系10000次，held标签、邻居身份及所有输入不变。置信区间按25个视频重采样10000次；它不是200个独立视频或训练seed区间。没有搜索距离函数、阈值或子集。

## 结果

共25视频/25 state、200个held→fit近邻比较：

|检查|观测值|随机参照/区间|
|---|---:|---|
|full407完全相同输入对|0|无直接表示冲突反例|
|近邻归一化gain差|0.203867|随机0.210178|
|gain差的近邻−随机|−0.006311|视频95% CI [−0.029751,+0.020092]；置换单侧p=0.334|
|近邻异号率|51.0%|随机50.875%|
|异号率的近邻−随机|+0.125个百分点|视频95% CI [−4.25,+4.75]个百分点；置换单侧p=0.537|

这里的“最近”只是8个可用fit动作中的相对最近，并非预先定义的绝对相似。标准化RMS距离的中位数为0.330、范围0.0615–2.148；最近距离/该query平均配对距离的中位数为0.566。因此不能把这些对全部写成“几乎相同输入”。

## 可以与不可以得出的判断

1. 当前固定度量下，没有足够证据表明近邻的实际收益比同state随机配对更一致；这不是局部可迁移规律已经得到验证的结果。
2. full407包含remove/insert时间等动作坐标，通常会把有限样本逐一分开。因此没有精确重复不等于信息充分。收益也不需要随时间或单个特征单调，不能把非单调本身当实现错误。
3. “近邻异号”没有证明所有可能的函数都不可学；距离度量、有限的8个fit参照和目标本身的高频变化仍会影响结果。若发现完全相同声明输入、相同state/plan且收益差稳定超过噪声，才是当前表示不足的直接反例；本次未发现。
4. G0b的标签辅助选优可以利用cheap输入难以预测的实际差异，因此有oracle headroom与廉价Value可学是两件事。现有材料尚不能把这种不可预测成分与表示/拟合方式问题分离。

本次唯一诊断已完成，不据结果继续扫描距离、加模型、扩bank或调整阈值。按已生效stop rule，32→64的触发条件未满足；不重复8/8，不加步数/容量，FVD不解锁。本结果不是新router/proxy或科学PASS。

## 前序结果交叉核对

- 8/8诊断：fit rho=0.9403；held rho=0.04444，regret=0.00149207，高于STOP=0.00141354；calibration rho=−0.17672。因此不能仅归因于跨视频覆盖，32→64的原定条件不成立。
- 历史B0：β=1.1时Future与Current/Post regret均为0.00327770015，差值CI为[0,0]；EMA为0.00326354553。Future未改善实际决策收益，forecast gate失败，不能据此解锁FVD。该结果仍是历史离线B0，不是新T长训轨迹。

## 复现文件

- [脚本](C:/Users/skywalker/Documents/ChatGPT/H65/reports/wtr_rfv_review_20260915/descriptor_discriminability.py)
- [完整结果JSON及逐state/逐近邻记录](C:/Users/skywalker/Documents/ChatGPT/H65/reports/wtr_rfv_review_20260915/DESCRIPTOR_DISCRIMINABILITY.json)
- 输入：AutoDL `/root/autodl-tmp/rfv_20260915/results/2ca4d4b/mini_c40_shard{0,1}`。
- 固定scaler：`/root/autodl-tmp/rfv_20260915/results/d62ea55/ACTION_HOLDOUT/plain_m_action_holdout_s42.pth`。

Permutation seed=42，video-bootstrap seed=43。全部原始输入、teacher、模型权重与数据划分未改变。
