# 交付包说明

主审查提交：239d098cd899936c35989fae6243c70259a85adb。
实验事实截至2026-09-12 21:56；仅基于该固定仓库快照。

`REPORT.zh.md`：完整研究报告，按（a）现状与科学问题、（b）跨领域机制、（c）三条模型族、（d）最小判别实验、（e）成本与测量、（f）Codex交接组织，附固定源码索引、原论文/作者代码版本和证据边界。

`CODEX_HANDOFF.zh.md`：可以单独交给Codex的接口、文件定位、张量合同、最小测试与分阶段实现说明。没有实施patch，没有启动任务。

`mechanism_checks.py` / `mechanism_checks.json`：已在CPU运行的独立数学/小型机制验证。没有加载项目权重、原视频或仓库模块；结果不是TAD精度或GPU时延。运行需要Python和NumPy：

```bash
python mechanism_checks.py --output mechanism_checks.json
```

本交付不包含模型权重、视频、字体或训练数据。报告中所有新模型精度、真实GPU速度和未来课程结论均为待测，不含捏造的新成绩。主仓库中的训练/评估结果是记录核查，不是本次独立复现。
