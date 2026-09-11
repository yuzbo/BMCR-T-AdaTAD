# H65完整中间测试结果

已测4/16个预设候选。S/B的60轮训练均已完成；每个表列结果覆盖全部211视频、792窗口，使用对应轮次EMA。

最终模型按总epoch25/30/35/40/45/50/55/60的平均mAP峰值选择，同分取较早轮次；第60轮另行报告。尚未测完全部候选时，当前最高值仅为已测候选中的最高值。

| 骨干 | 总轮次 | 平均mAP | @0.3 | @0.4 | @0.5 | @0.6 | @0.7 | 记录 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| S | 25 | 53.6248% | 73.3906% | 66.4384% | 56.5452% | 43.5166% | 28.2332% | [指标](evaluations/s_h65_test_epoch_25/metrics.json) · [预测](evaluations/s_h65_test_epoch_25/result_detection.json.gz) |
| S | 30 | 56.6237% | 74.7939% | 69.2087% | 59.1574% | 47.3272% | 32.6311% | [指标](evaluations/s_h65_test_epoch_30/metrics.json) · [预测](evaluations/s_h65_test_epoch_30/result_detection.json.gz) |
| B | 25 | 58.8581% | 78.5039% | 71.7255% | 62.0225% | 49.1216% | 32.9168% | [指标](evaluations/b_h65_test_epoch_25/metrics.json) · [预测](evaluations/b_h65_test_epoch_25/result_detection.json.gz) |
| B | 30 | 61.4086% | 79.7640% | 74.0397% | 64.1274% | 52.7398% | 36.3720% | [指标](evaluations/b_h65_test_epoch_30/metrics.json) · [预测](evaluations/b_h65_test_epoch_30/result_detection.json.gz) |

本表是测试集选模过程，不能称为独立未见测试性能。中间点不能直接当作第60轮终点；计算量与时延在峰值和终点的独立测量完成后报告。
