# 发布验证

2026-09-11 对独立发布副本执行：

```bash
OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES= PYTHONPATH="$PWD:$PWD/upstream" python -m unittest discover -s tests -v
python tools/verify_results.py
```

19项CPU测试通过，输出见[CPU记录](cpu_tests.txt)。公开证据检查通过，见[结果一致性记录](results_consistency.json)：六组各211视频/792窗口、422000预测，指标/profile一致；两次预热各2000更新、四个分支各4000更新，cost与梯度范数有限。

这些检查针对发布打包与局部代码合同。它们没有重新从标注计算mAP、没有进行新GPU实验、没有证明离散代理梯度无偏或学习目标达到最优。原正式GPU训练与评测证据位于 `phase2_20260910/runs/`。
