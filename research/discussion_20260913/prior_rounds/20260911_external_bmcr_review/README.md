# 外部BMCR-T建议：完整归档与二次复审

主结论：总体诊断与研究顺序接受；已实证确认ASFormer学习率分组错误；新架构和阈值只作为有条件的研究方案，不能整包照单执行。

- [58项逐项评价与结论](REVIEW.zh.md)
- [20项文献及外部源码复核](LITERATURE_RECHECK.zh.md)
- [原始报告](inputs/BMCR_T_fixed_commit_review.zh.md)
- [完整粘贴回复](inputs/pasted-text.txt)
- [原始ZIP](inputs/BMCR_T_review_15280e5.zip)
- [包内独立检查脚本](inputs/package/independent_checks.py)与[原始检查结果](inputs/package/independent_checks.json)
- [原样归档记录](ARCHIVE.json)
- [实际scout与真实优化器分组核验](actual_scout.json)
- [记录及算术复核](records_and_math.json)
- [附件脚本复制重跑记录](external_script_rerun.json)

原件未改写，包内报告与单独提供版本逐字节相同。主代理的CPU核验脚本为`check_actual_scout.py`，实际构造scout但没有执行训练forward、加载checkpoint或更新参数；`check_records_and_math.py`读取原实验日志计算结果。附件脚本先阅读再在`external_script_rerun/`副本执行，未覆盖原件。

本轮只做复审和归档，没有更改已发布的模型源码、重启训练或公开上传新收到的附件。固定仓库仍保存旧配方的完整实验，后续比较必须明确纠正配方与历史结果的区别。
