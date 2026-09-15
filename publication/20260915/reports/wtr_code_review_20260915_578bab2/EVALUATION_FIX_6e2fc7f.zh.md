# 首次内联评测日志故障定点复核

日期：2026-09-15。补丁 `6e2fc7f78424e2485e8d33b079fac926eb0cf4a6`；模型/训练科学版本 `40552945ad5d56b7404f833cd86f998e8558e8b1`。实现者负责部署与恢复，本任务只审阅。

**结论：日志修复 PASS。可从各自 epoch 10/latest checkpoint 恢复，先补跑 epoch 10 EMA 全量评测，再继续训练；不需要重训前 10 轮或重复两更新 preflight。**

## 问题与修改范围

实现任务在 11:10 核查发现两条 D 课程均已完成 10 epoch/1000 更新，但首次内联评测在 JSON 写入时失败，尚未产生 mAP。该故障是初审未发现的评测日志集成错误，先前模型/GPU技术通过不能证明完整评测路径通过。

独立完整 diff 确认补丁只修改 `h65/paper/evaluation.py`：

- 第 65 行将 window record 的 plan 从 `detail['plan']` 改为 `detail['trace']['plan']`。
- 第 96–97 行为评测结果记录 `evaluation_source_revision`。

`model.py:235` 把实际路由需要的 `wtr_geometry` Tensor 加入执行 plan；第 258 行已经构造剔除 `route_masks/wtr_geometry` 的公开 trace plan。使用该公开 plan 仅改变日志内容，不修改执行 plan、模型 forward、预测、成本计算、标签或预算配置。其余 window record 字段在当前 D-U/D-V 路径中为已转出的标量、列表或普通字典。

## checkpoint 恢复判断

- `paper_train.py:278–280` 在进入评测前保存 latest 与 milestone checkpoint。
- 第 86–105 行恢复 learned weights、optimizer、scheduler、EMA、更新数/epoch/cursor 与 RNG。
- 第 124–132 行优先完成尚无 completed 回执的 milestone EMA 评测。
- 第 146 行之后再从已保存的 epoch_index 继续训练。

因此不存在为修复 JSON 日志而废弃已完成 10 轮的理由。评测成功后 EMA context 恢复训练权重，原优化器与课程状态继续使用。此前科学回执仍适用于未变化的模型和训练代码。

## 版本与验收边界

部署必须写 `EVALUATION_REVISION=6e2fc7f78424e2485e8d33b079fac926eb0cf4a6`。模型/训练版本记 4055294，评测代码单独记 6e2fc7f；不能声称运行中的全部源码仍是 4055294。

本次为定点 diff、调用链与恢复路径审阅，没有新增 GPU 作业或重复训练测试。实现者的 py_compile/diff 检查是额外技术证据。恢复原课程后越过原失败点，并完成 211 视频/792 窗口及 completed/metrics 产物，才证明实际评测恢复成功；本回执不授予 mAP 或科学增益 PASS。

实现任务同时回传 Raw 六视频 19 窗口、346 个实际 swaps、replay error 为 0，A100 S pair 仍排队；这些是状态交接，不是本日志补丁的验收对象。Raw mini-bank 的科学信号与后续准入仍需单独判断。
