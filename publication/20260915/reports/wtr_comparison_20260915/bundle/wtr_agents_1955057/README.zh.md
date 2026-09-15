# Where to Refine：固定 1955057 的诊断与 agents 执行包

基线提交：`1955057508af5a5dfd59a98bddf49302bee5972c`。

## 交付边界

这不是已经验证有效的新 TAD 方法，也不是完成 GPU 集成验收的 PR。本包提供完整源代码的**诊断/未来价值最小实验**、已核对现有接口的控制配置生成器、待 agents 集成的算子价值模块，以及逐项实现合同。它不改写远端仓库、不提交 Slurm、不取消现有课程。

本地环境只有 CPU PyTorch，已通过 `tests/test_core.py` 的 **26 项测试**；另做了语法编译与各 CLI `--help` 检查。没有 THUMOS 数据、官方/训练 checkpoint、OpenTAD CUDA 扩展或可用 GPU，因此 `PaperModel` 集成路径、真实训练、完整 mAP、Slurm 回执均**未运行**。接收 agent 必须先做本文列出的 GPU 契约检查，不能把 CPU 成功当成模型有效。

## 文件

| 文件 | 提供的内容 | 状态 |
|---|---|---|
| `wtr/core.py` | 原始单位价值外推、共同支持/尺度 JSD、STOP、精确等成本配额、Taylor 残差代理、算子价值头 | CPU 单测通过 |
| `wtr/operators.py` | 同输入算子残差蒸馏与真实价值监督损失 | CPU 单测通过；未接入原 engine |
| `wtr/probe.py` | 不改源码建立训练视频动作库、跨 checkpoint 重执行逐帧交换、固定掩码 attention/FFN 干预 | 接口静态核对；GPU 待验 |
| `wtr/forecast.py` | 当前/EMA/预声明外推对更晚真实价值的预测检验，按视频 bootstrap | CPU primitives 已测 |
| `wtr/fit_router.py` | 在冻结 detector 上训练现有 FrameRouter：GT-NLL + current/EMA/future JSD，输出 evaluation-only checkpoint | 接口静态核对；实际 checkpoint 待验 |
| `wtr/transfer_router.py` | 将已拟合router转移到更晚冻结detector的独立诊断 | 实际checkpoint/GPU待验 |
| `wtr/snapshot_metrics.py` | 导出固定快照的同epoch差值和实际D/S路由比例 | 源数据接口已核对 |
| `wtr/make_configs.py` | 只使用原仓库已支持字段生成 4 个控制，不静默添加未实现的新方法 flag | 语法/CLI 检查 |
| `docs/AUDIT_AND_METHOD.zh.md` | 实际代码发现、论文迁移、完整方法与证伪条件 | 研究方案，不是新结果 |
| `docs/AGENTS.zh.md` | 每个 agent 的读码点、实现合同、验收项、禁止事项 | 待执行 |
| `docs/COMMANDS.zh.md` | 固定工作树、动作库、预测检验、训练/评测、单控制器调度安排 | 命令草案，不是已提交回执 |
| `docs/PAPER.zh.md` | 论文结构、图表与证据要求 | 实测后填结果 |
| `docs/LITERATURE.zh.md` | 按瓶颈排列的跨领域论文阅读与迁移方案 | 非TAD实测结论 |
| `references.json` | 一手论文标识和可定位的审查文件 | 参考索引 |

## 最小顺序

先读 `docs/AUDIT_AND_METHOD.zh.md`，再按 `docs/COMMANDS.zh.md` 执行。

1. 不改变现有 84 条课程。另建固定工作树与独立输出目录。
2. 在 THUMOS **训练视频**中先做 6 视频技术试验，再做 32 视频诊断。用 `--plans 1` 固定 K384 / D100 / S100；不要从 Graph 开始。
3. 用同一 bank 在较早、当前、较晚 checkpoint 重执行。输入/裁剪/时间/选帧哈希不一致立即停止。
4. 测试预先声明的 β 网格是否改善更晚动作价值预测；不能用最终官方 test 选 β。
5. current / EMA / future 三种教师，保持相同训练动作标签、CPU 更新次数、初始化、温度、固定分量尺度。
6. 冻结比较方案后才执行完整官方评测。价值相关性不是 mAP，离线 frame pilot 不是完整递归 RISE。
7. 算子路由、在线递归、恢复模块等新方法由 agents 按合同接入；代码中的 `OperatorValueHead` 不会自动改变原模型。

## 防止误用

- **逐帧采样 ≠ 连续16帧选块**；16只是选后执行打包。局部交换 `//16` 是搜索范围限制。
- `actual_delta` 是 cls/reg GT 差，`repair_delta` 是另外的 teacher 替换代理，绝不能混成标签。
- K不变的 frame swap 不能计算 `V/ΔC`，因为 ΔC≈0。不同成本 action 也不能假定全局可加。
- β=1 是 current teacher，不是 EMA teacher。外推前必须恢复原始单位或使用共同固定尺度。
- 第三份 `future` 记录只用于诊断评估，`fit_router` 根本不接受这个参数。
- 第一版 frame bank 使用 plan1：无 D/S 稀疏，避免次级掩码漂移混淆。扩展 `--plans 1,3,4` 时，frame 干预的下游 D/S 掩码按各 checkpoint 重算，应明确叫闭环策略价值；另做冻结掩码控制。
- 算子 probe 的 `attention_hold`、`ffn_light` 是解耦干预，不等于直接复制历史 A-MoD 双算子 bypass。
- 算子 probe 记录 `branch_parity_max_abs`：all-true 路由与未路由分支不等价时不继续科学解释。先按 FP32/BF16 设容差。
- `--measured` 的成对计量可能复用 preview，并包含 task-loss 算子；不能拿其绝对值冒充独立推理全成本。完整评测继续用原 `paper_eval` 的全窗口账本。
- 本包不包含/下载数据或模型，不提供训练服务器凭证，不读取 secrets。

- 跨checkpoint重执行使用动作库固定的point-head loss normalizer，并在每次loss后恢复原对象；不将归一化分母漂移误判为价值漂移。
