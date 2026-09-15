# 固定5485c3c论文复审与并行agents包

先读REPORT；交给总控agent时使用AGENTS_MASTER。主审查快照固定5485c3cb8dfbd823acbb177a0b51292dadc44360。所有完整新实验单seed42；完整模型计算量和最佳TAD性能主导，终点/选模/其他成本披露。没有执行服务器作业。

## 文件

- `REPORT.zh.md`：逐题覆盖16个问题，当前/建议网络、核心因果诊断、三条候选、结果边界与论文摘要。
- `AGENTS_MASTER.zh.md`：可直接复制的总控任务、7个agent边界、张量/梯度/状态测试、生产CLI实施规格。
- `EXPERIMENTS.json`：18项有限新增同框架规格；不是已经提交的18作业。与既有52配置重复时去重；配置继承不等于等待前一模型mAP。
- `EVIDENCE_MAPPING.csv`：22项原始建议—实现—状态—支持主张映射。
- `FIGURES.zh.md`：8主图组、12附录组、6表规格（按篇幅选择主文项）、字段/图注/禁止解释。
- `SOURCE_MAP.md` / `ACCESS_MANIFEST.json`：固定源码阅读与未读边界；live_snapshot大文件未完整解码。
- `REFERENCES.md`：一手论文、版本、官方代码与阅读深度。
- `diagrams/current.mmd`、`diagrams/proposed.mmd`：已实现/建议结构分开的可转绘草图。
- `data/`：只含旧完整测试记录摘录和独立算术复算。
- `scripts/audit_records.py`：本地算术/计划约束审计，不加载权重。
- `scripts/plot_archived.py`：从已有记录生成3张历史图，PDF/SVG/PNG与来源清单。
- `figures/`：已生成并渲染检查的三张历史图；不是新seed42结果。

## 可直接运行的本包命令

```bash
python scripts/audit_records.py
python scripts/plot_archived.py
```

第二个命令需要matplotlib；不访问GitHub、数据服务器或GPU。生产训练/部署工具见AGENTS_MASTER，新增CLI须由agents实现后再在作者授权环境运行。不要把建议实验的null结果填成0。

## 本次实际完成

只读源码/文献审查；旧八格交互和四个历史精度—计算点的算术审计；18配置均seed42单次、无mAP依赖、无虚构指标的本地验证；3张历史图并视觉核查PDF渲染。**没有完成新模型单测/GPU预检/权重导入/训练或完整测试。**

## 需要等待的证据

当前PaperModel正式完整结果；depth-light/support-matched KD实际收益；官方decoder初始化在本任务中的收益；18新规格的实现/运行；PBD正式同框架对照与匹配查询成本；ANet/InternVideo1-MQ/TadTR完整泛化。
