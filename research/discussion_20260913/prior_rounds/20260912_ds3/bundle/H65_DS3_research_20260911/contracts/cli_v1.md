# 拟新增 CLI 合同

这些命令在 anchor 提交中不存在。Agent 05/06 必须实现、记录 --help，并通过测试后才能运行；不得把此文档当作已实现证据。

## ds3_validate.py

```bash
python tools/bata/ds3_validate.py source --manifest M.json --out OUT.json
python tools/bata/ds3_validate.py config --manifest M.json --out OUT.json
python tools/bata/ds3_validate.py checkpoint --manifest M.json --out OUT.json
python tools/bata/ds3_validate.py gates --manifest M.json --gate G0 --out OUT.json
python tools/bata/ds3_validate.py gates --manifest M.json --gate G1 --out OUT.json --allow-gpu
python tools/bata/ds3_validate.py gates --manifest M.json --gate G2 --out OUT.json --allow-gpu
```

source/config/checkpoint 默认不训练；读取可信 checkpoint 也必须经过明确的来源批准，不执行下载。G0汇总来源、resolved config、checkpoint；G1实际CUDA身份/clip等价/shape/执行计数；G2训练smoke最少2次成功非零LR更新、冻结/梯度/参数差异及D1全dense图。G3–G6同一接口，具体checks见REPORT；没有测试实现必须NOT_IMPLEMENTED而不是空PASS。

`--allow-gpu` 必须同时要求 `H65_DS3_ALLOW_GPU_RUN=1` 和 Slurm allocation 或显式 `H65_DS3_COMPUTE_ALLOCATION_CONFIRMED=1`。CUDA验证只允许 manifest.review_status 为 APPROVED_FOR_VALIDATION 或 APPROVED_FOR_THIS_RUN；CPU来源检查允许 UNREVIEWED。不允许从缺失 gate 自动推断PASS。

为了避免“必须已通过G1才能运行G1”的循环，验证CLI按各gate依赖单独检查，只允许短界定测试；正式train/eval由执行包run_experiment.py检查完整gate清单。G1前必须G0真实通过；G2前必须G0/G1真实通过。测试checkpoint与临时输出只写manifest指定外部目录，不覆盖正式输出。

G1测试检查 `[all_selected, full_spatial, depth12]` 身份。FP32起始数值容差可设atol=1e-5、rtol=1e-4，但agent必须根据同后端重复run基线误差校准并说明；AMP单列，不能无理由放宽。

## ds3_profile.py

```bash
python tools/bata/ds3_profile.py --manifest M.json \
  --scope backbone --warmup 20 --iterations 100 --out OUT.json --allow-gpu
python tools/bata/ds3_profile.py --manifest M.json \
  --scope model --warmup 20 --iterations 100 --out OUT.json --allow-gpu
python tools/bata/ds3_profile.py --manifest M.json \
  --scope end_to_end --warmup 20 --iterations 100 --out OUT.json --allow-gpu
```

scope必须明确，model包含preview/router/pack/exit/reconstructor/detector，end_to_end还包含decode/preprocess/NMS；用真实视频、完整部署路径。无数据时只可提供标为synthetic的算子测试，不得输出正式加速结论。报告计数、GPU同步、p50/p95、batch/precision、memory、缓存状态、有效视频数与代码/模型身份。

## 既有 train/test CLI

新config注册成功后复用 `tools/train.py CONFIG --seed ... --cfg-options ...` 和 `tools/test.py CONFIG --checkpoint ... --checkpoint-state-key ... --expected-checkpoint-epoch ... --metrics-json ...`。不得发明旧入口不支持的参数。训练初始化路径写新config或明确支持的DS3_TEACHER_*环境变量，并封存resolved配置。
