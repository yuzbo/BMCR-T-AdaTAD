# H65/BMCR 三维去冗余研究与 agents 实施包

## 从这里开始

1. `REPORT.zh.md`：论文研究方向、跨领域机制、严格区分事实与假设、完整实现/分析论证。
2. `AGENTS_TASKS.zh.md`：可直接复制给总控与7类agent的任务书、职责、依赖和验收。
3. `COMMANDS.zh.md`：本包已经可运行的命令，和agents仍须实现的production CLI合同（两者明确分开）。
4. `FIGURES.md`、`plans/figures.csv`：6主图组、12补充图组、4主表的原始数据与绘制规格。
5. `plans/experiments.json`、`plans/claims.csv`：27个有限、分层实验项与5项可证伪命题；不是27组都要立即完整训练。
6. `SOURCE_MAP.md`、`SOURCES.md`：固定源代码接口、一手文献、版本与读取边界。
7. `reference/`、`tests/`：独立PyTorch CPU原型；不是已集成OpenTAD、不是新GPU模型。
8. `TAD_experiments_figures_data.xlsx`：实验、图表、主张、来源与历史图表数据。历史平均mAP由公式计算。
9. `figures/historical/`：根据固定快照真实旧结果产生的S/B六张独立图（PDF/SVG/PNG），每张附来源JSON。
10. `validation/`：实际CPU单测、计划校验、工作簿检查与预览、PDF检查记录。

## 确定的范围

项目固定`239d098cd899936c35989fae6243c70259a85adb`；时间选样单位是单candidate frame，不推进T24整clip路由。旧选帧后16-observation打包仍允许，局部16slot反事实先验先保留再消融。原始200/211/792、seed3407作为主协议；测试峰值选模须披露。

本包没有运行Slurm/训练作业，没有访问项目权重与原视频，没有新mAP/新延迟。当前通过的是13项自写合成CPU机制测试；计划校验覆盖27个实验、6+12图组、5个假设。**这些不是生产模型验证。**

历史图仅含旧phase2的official/H65/BMCR，同一次记录口径但节点可能不同；不能当新的同卡benchmark。官方/新模型训练来源不同，也不是单因素选择器消融。修正H65表见报告及source P9；修正BMCR不能从旧权重推断，需先真正完成授权配方。

## 可运行命令

```bash
python scripts/run_reference_checks.py
python scripts/check_plan.py
python scripts/plot_results.py --input data/historical_results.csv --out figures/historical
```

原型依赖：Python、PyTorch；绘图依赖Matplotlib。生成工作簿使用artifact_tool，仅用于制表；本包不要求升级原服务器PyTorch/CUDA环境。`scripts/prepare_worktree.sh`只创建指定的固定基点worktree。训练、干预、评估、dispatch的工具名是agents需实现的CLI合同，固定仓库中尚不存在。

## 最重要的禁止混淆

- 零残差只等于新插值基线，不等于换head后的旧BMCR mAP。
- anchors有真实frame来源，但不是dense teacher特征的直接子集。
- teacher feature修复不是实际增加一个frame的计算反事实。
- 外部teacher蒸馏与共享模型self-KD分别报告。
- all-heavy旧权重不等于官方权重；满预算接口恒等测试必须指定同一模型。
- MAC减少不是端到端加速，训练freeze不是执行skip。
- 没有数据就不生成结果点/真实时空可视化；new metrics保持null。

所有脚本与示例用于科研复核。部署前必须由agents接入真实资源、完成同图/梯度/执行计数测试，再依据明确执行授权提交OWN作业。不能凭历史job ID取消作业。
