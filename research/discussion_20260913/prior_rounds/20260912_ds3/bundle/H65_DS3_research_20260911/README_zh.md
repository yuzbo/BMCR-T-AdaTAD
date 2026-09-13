# H65-DS3 研究与 agents 执行包

本包回答：如何在 H65 上建立 dense-forward 训练、时序/空间/网络深度条件稀疏推理的 TAD 方法。

**阅读顺序**：`REPORT_zh.md` 是完整调研、代码核查、方法和实验计划；`COMMANDS_zh.md` 是分阶段命令；`agents/` 是8个完整角色任务书；`experiments.json` 是24个拟议实验；`contracts/` 固定训练定义、shape、坐标和新CLI；`sources.json` 包含29项文献/官方文档及11项固定源码来源。

## 本包实际提供了什么

提供了报告、任务书、可执行的隔离worktree/bootstrap、只读源码哈希核验、Codex调用封装、受控实验计划/启动器、Slurm模板，以及11个CPU数学参考测试。

**不包含已经完成的新H65模型实现或新ds3配置**。它们必须由agents按任务书创建。现有仓库入口和拟新增入口在命令文档明确分开。启动器发现新config缺失、来源未审查、证据不足会拒绝运行，不会静默改用其他模型。

## 证据范围

代码锚点：`04c35a3b76897e6c1569eeede41ed3aecaf7f854`。它不是自动确认的当前默认分支最新HEAD。远端GPU、checkpoint、训练日志和正式TAD评估未访问/运行。本包生成的验证结果在 `ARTIFACT_VALIDATION.json`，只覆盖文件格式/脚本语法/参考单测，不等于H65功能或性能验证。Codex CLI调用本身也未在此环境执行。

## 首个动作

```bash
python /absolute/path/H65_DS3_research_20260911/bin/verify_bundle.py
```

再按COMMANDS第1节在现有仓库旁创建隔离worktree。不能把本包整体复制进原仓库作为代码提交；源码中只加入服务新方法的模块、配置和focused tests，数据、运行记录、checkpoint和本报告保存在外部研究目录。

## 主方法原则

主线D1：所有训练clips/patches/layers完整前向，训练廉价最终特征预测器与浅层出口；推理选择cheap/8/12，始终恢复原native时间接口，再使用原dense检测器。C2允许缺失/稀疏辅助校准，但独立命名和计费。Z0是没有新增拟合的直接部署基线。三者不得混报。

本包中的门槛、预算和速度目标是研究预注册建议，不是已取得的mAP或加速结果。请先做完整clip身份/时间轴测试，再训练新模块。
