# RFV capture等价与8/8诊断的定点审阅

2026-09-15。只读审阅；未修改源码、运行新训练或重复GPU标签查询。

## capture等价：接受现有回执的限定范围

对比 `2ca4d4b5f2ad657410bfc5c2c4cebd3bc6a11a32` 与 `3a7e93f3f77350e9598011db1dd821cedaee8f0b` 的完整diff：新增G1消费者、分析和测试文件未被capture runner调用；实际前向依赖未改变。capture路径的差异仅为：

1. `h65/rfv/bank.py:191–194` 比较episode的`official_frame_ids/official_valid`前将两侧转tuple。只统一JSON往返产生的容器类型，值、顺序、support、action、候选与continuation仍比较。
2. `tools/rfv_run.py:208–209` 在读取已保存预测进行AP汇总时设置`sliding_window=True`。这是神经预测/CF标签查询之后的CPU后处理；bank/replay不执行该汇总分支。

主代理用真实mini-bank JSON做独立CPU检查：JSON列表与live tuple比较通过；改变一个物理frame或selected support均被拒绝。结果见[CAPTURE_EQUIVALENCE_CHECK.json](C:/Users/skywalker/Documents/ChatGPT/H65/reports/wtr_rfv_review_20260915/CAPTURE_EQUIVALENCE_CHECK.json)，GPU重查询次数为0。

因此现有`research/rfv_sprint_20260915/CAPTURE_EQUIVALENCE.json`足够支持这两个精确版本的`rfv_label_forward_equivalence`。新A/B入口检查精确capture revision集合、scope、passed和evidence，且之后仍验证逐state action identity，使用方式正确。

该回执不声称不同checkpoint的标签数值相同，不覆盖AP汇总输出等价，也不授予完整B0/forecast/FVD通过。实际teacher、前端或下游策略改变时，标签仍须按对应语义重查询。

## 8/8脚本：代码与诊断协议PASS

精确提交：`d62ea556c56ca6157289a866aaafc79c4fa427c6`；范围仅`tools/rfv_action_holdout.py`及它调用的已核验Plain训练/统计接口。

- 第47–54行只选择原fit有动作state，按`(insert,remove)`物理顺序交错分8/8，不看gain；同步记录两组action IDs。
- 第16–20行同时切分action metadata及descriptor/remove/insert/target数组，节点级公共cheap输入保留合理。
- 第55–60行虽然把fit与calibration一同交给既有接口，`normalization`与`train_one`内部只筛`partition=fit`，因此输入统计、target RMS及优化只使用fit8。
- 第62–64行的held指标使用held8自己的target集合计算oracle/STOP/regret。calibration保持原16动作，是并列描述；不能把两个不同候选集合的绝对regret直接当同难度对比。
- 原inner10未参与优化或评估，正式outer20未载入该mini；`scientific_gate=False`正确。
- 模型返回时已为eval模式，且调用centered_error前再次执行evaluate，当前推理模式正确。

没有发现此脚本需要修正的实现问题。本PASS仅允许这项既定CPU诊断，不等于诊断结果成功、B0全部实现通过或正式课程解锁。

## 唯一执行顺序已与实现任务确认

先执行现有25个有效fit视频内的8/8动作留出诊断；Plain-M、2000steps、3seeds、目标与proposal保持不变，暂不新增GPU标签。

只有同视频未见动作可学、跨视频仍差，才预登记fit32→fit64的固定覆盖增量。届时保留原fit32，从剩余outer-fit池固定选新增32并排除inner10；calibration/inner/outer角色不重划，报告注册数量和实际有动作数量。两项不同时启动。

## G0b状态更新

主代理已直接读回完整calibration20/46窗结果：Uniform 92.49616075%、LocalCF 93.02457398%；Δ+0.52841322pp，95% CI [+0.08244043,+0.71468687]，AP@0.7 Δ+0.82774438pp。它确认所测有限动作空间在这些开发视频上有AP headroom。

Plain mini泛化gate仍失败，正式T/Graph/FVD课程继续等待。`rfv_analyze.py`中仅由G0b计算的`task_course_eligible`不能单独充当总准入条件，仍须与Plain及recipe技术gate共同判断。
