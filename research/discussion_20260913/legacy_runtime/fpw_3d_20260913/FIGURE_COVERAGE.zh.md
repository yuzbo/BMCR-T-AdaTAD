# FPW 图数据覆盖

`tools/frame_analyze.py` 只消费 manifest 中的真实记录。每个 record 至少可含 `id`、`backbone`、`status`、`metrics`、`gflops`、可选 `latency_ms`、`source`、`train_updates`、`teacher_queries` 和 artifact 路径；缺失字段不会用估计值补齐。

当前接口支持：S/B 按平均 mAP 与实际矩阵/卷积 FLOPs 的 Pareto 图；五个 IoU 阈值 mAP 图；独立 latency cohort；记录带 `training_curve`/`curve` 时的训练曲线。输出同时包含 PDF/SVG/PNG、`source_manifest.json` 与 `coverage.json`。

执行量按 manifest 的 `gflops` 原样报告（full_eval 的约定是 2× matrix/conv MAC）；latency 只单独呈现，不参与 Pareto 或路线淘汰。不同阶段可并行，工具不依据阶段顺序推断 mAP。

执行次数、utility、错误/失败图只有在真实记录提供相应字段后才可生成；未提供时 `coverage.json` 标记 missing。该接口不声称已有 6 张主图和 12 张补充图全部完成，最终覆盖取决于输入记录。
