# H65完整中间测试结果

已测16/16个预设候选。S/B的60轮训练均已完成；每个表列结果覆盖全部211视频、792窗口，使用对应轮次EMA。

最终模型按总epoch25/30/35/40/45/50/55/60的平均mAP峰值选择，同分取较早轮次；第60轮另行报告。尚未测完全部候选时，当前最高值仅为已测候选中的最高值。

| 骨干 | 总轮次 | 平均mAP | @0.3 | @0.4 | @0.5 | @0.6 | @0.7 | 记录 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| S | 25 | 53.6248% | 73.3906% | 66.4384% | 56.5452% | 43.5166% | 28.2332% | [指标](evaluations/s_h65_test_epoch_25/metrics.json) · [预测](evaluations/s_h65_test_epoch_25/result_detection.json.gz) |
| S | 30 | 56.6237% | 74.7939% | 69.2087% | 59.1574% | 47.3272% | 32.6311% | [指标](evaluations/s_h65_test_epoch_30/metrics.json) · [预测](evaluations/s_h65_test_epoch_30/result_detection.json.gz) |
| S | 35 | 59.0293% | 76.3124% | 70.6971% | 62.1240% | 50.7314% | 35.2817% | [指标](evaluations/s_h65_test_epoch_35/metrics.json) · [预测](evaluations/s_h65_test_epoch_35/result_detection.json.gz) |
| S | 40 | 60.6321% | 77.3363% | 71.4753% | 63.8199% | 52.7586% | 37.7706% | [指标](evaluations/s_h65_test_epoch_40/metrics.json) · [预测](evaluations/s_h65_test_epoch_40/result_detection.json.gz) |
| S | 45 | 61.8558% | 77.9441% | 72.8948% | 64.7242% | 54.1085% | 39.6074% | [指标](evaluations/s_h65_test_epoch_45/metrics.json) · [预测](evaluations/s_h65_test_epoch_45/result_detection.json.gz) |
| S | 50 | 62.5107% | 78.6391% | 72.8612% | 65.5623% | 54.8983% | 40.5927% | [指标](evaluations/s_h65_test_epoch_50/metrics.json) · [预测](evaluations/s_h65_test_epoch_50/result_detection.json.gz) |
| S | 55 | 63.0833% | 79.3377% | 73.4698% | 65.8459% | 55.4467% | 41.3164% | [指标](evaluations/s_h65_test_epoch_55/metrics.json) · [预测](evaluations/s_h65_test_epoch_55/result_detection.json.gz) |
| S | 60 | 63.4094% | 79.1262% | 74.0298% | 66.3798% | 55.8804% | 41.6307% | [指标](evaluations/s_h65_test_epoch_60/metrics.json) · [预测](evaluations/s_h65_test_epoch_60/result_detection.json.gz) |
| B | 25 | 58.8581% | 78.5039% | 71.7255% | 62.0225% | 49.1216% | 32.9168% | [指标](evaluations/b_h65_test_epoch_25/metrics.json) · [预测](evaluations/b_h65_test_epoch_25/result_detection.json.gz) |
| B | 30 | 61.4086% | 79.7640% | 74.0397% | 64.1274% | 52.7398% | 36.3720% | [指标](evaluations/b_h65_test_epoch_30/metrics.json) · [预测](evaluations/b_h65_test_epoch_30/result_detection.json.gz) |
| B | 35 | 64.0615% | 80.8500% | 76.0793% | 66.6294% | 55.5015% | 41.2470% | [指标](evaluations/b_h65_test_epoch_35/metrics.json) · [预测](evaluations/b_h65_test_epoch_35/result_detection.json.gz) |
| B | 40 | 65.3995% | 81.5243% | 76.3374% | 68.7863% | 57.2553% | 43.0942% | [指标](evaluations/b_h65_test_epoch_40/metrics.json) · [预测](evaluations/b_h65_test_epoch_40/result_detection.json.gz) |
| B | 45 | 66.3111% | 82.3546% | 77.5714% | 69.5104% | 58.0567% | 44.0623% | [指标](evaluations/b_h65_test_epoch_45/metrics.json) · [预测](evaluations/b_h65_test_epoch_45/result_detection.json.gz) |
| B | 50 | 66.4645% | 82.3616% | 77.9074% | 69.6106% | 58.3978% | 44.0451% | [指标](evaluations/b_h65_test_epoch_50/metrics.json) · [预测](evaluations/b_h65_test_epoch_50/result_detection.json.gz) |
| B | 55 | 67.1328% | 82.9573% | 78.1886% | 70.6501% | 58.7011% | 45.1668% | [指标](evaluations/b_h65_test_epoch_55/metrics.json) · [预测](evaluations/b_h65_test_epoch_55/result_detection.json.gz) |
| B | 60 | 66.9026% | 82.8301% | 78.0807% | 70.0593% | 58.7906% | 44.7523% | [指标](evaluations/b_h65_test_epoch_60/metrics.json) · [预测](evaluations/b_h65_test_epoch_60/result_detection.json.gz) |

本表是测试集选模过程，不能称为独立未见测试性能。中间点与第60轮终点分别保留。

完整峰值、终点及计算量/时延比较见[最终报告](FINAL_REPORT.md)。
