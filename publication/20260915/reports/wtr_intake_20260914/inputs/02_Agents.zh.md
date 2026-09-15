# 可执行 agents 工单

每个 agent 先读根 AGENTS.md。工单编号不是8个并行GPU作业；代码开发可并行，真实运行通过唯一owner排队。每项需返回commit、diff、测试回执、资源与成本、未完成项。

## A0｜基线审计与集成守门

输入：fixed SHA、现有resources、最新真实运行回执、本包。

逐一验证7个blob；在新worktree dry-run/apply。阅读engine的selected-Q/full-KV、light、TIA、age、physical support、cost校验。确认没有output score multiplier。复核40/60/80结果文件和对应source_revision，统计实际D/S使用率。不要把训练计划注释的PROPOSED覆盖更新后的真实stage状态。

产物：`research/wtr/audit/{source_manifest.json,behavior_matrix.json,AUDIT.zh.md}`。

技术验收：WTR关闭时forward/gradient/EMA/optimizer groups不改变；K768全量接口与旧full路径一致；头部新参数初始化不改变原参数RNG；base不匹配必须拒绝。不能在这里宣称算法有效。

## A1｜真实反事实与冗余测量

先执行已有 `tools/wtr_probe.py` 的legacy checkpoint诊断。D pair在同packed clip竞争域、S pair在同native-time竞争域；固定其他掩码和selected RGB，比较actual_delta、符号和rank。空合法集合明确计数，不隐式改变budget。

扩展实现（本包未实现完整）：
- temporal frame exchange使用现有 `swap_selection`，保存原始frame/time/contributor及D/S计划。换帧会改变tubelet组成和后续状态，是完整temporal action，不能声称像D/S一样内部状态完全一致。
- 输入信息干预为shape-preserving neighbor replacement/interpolation；真正省算干预另列。0强度no-op检验数值噪声；改1/2/4候选帧作敏感度诊断，不回退连续16clip选择。
- 合法coalitions多budget、多history采样，报告共同删除与各自删除的interaction。GT仅用于诊断target，不输入部署policy。
- 对不同checkpoint重新执行完全相同action/support manifest。不要用已变化的selection索引比较“同一动作”。

产物：`records.jsonl`、`manifest.json`、视频集合hash、augmentation/checkpoint/route hashes、原始prediction文件、查询/耗时/成本账本。完整propensity只有在整个抽样过程均记录后才能宣称存在。

验收：相同action重复重执行可复现；所有非干预mask不变；实际成本可追溯；作用损失含cls/reg且无teacher replacement。Null与联合干预都必须实测。

## A2｜局部价值路由（本包核心）

接入本包 `wtr_core.ValueHead` 与 `wtr`，不重写ViT计算核。使用当前state+预算+age/quality预测heavy update效用；full-KV和light保持。WTR启用时不能还算一遍旧incoming-attention QK。trace新增MAC必须与matrix_counter校验一致。

验证 six variants W00–W05。W01 uniform保留同形小头用于机制对照；最低成本uniform系统另报。value监督独立于hard top-k梯度；不得写成全路径可微。头预测差值有gauge freedom，按实际容量竞争域训练/蒸馏。

GPU gate：legacy回归；compact与dense-mask同mask前向/梯度；无效token；exact capacity；FP32/BF16稳定性；损失mean参数真更新；teacher不变；两步后新实例strict reload；完整MAC账本；SIGUSR1切片恢复和不中断参考等价。测试优先用真实完整窗口，再覆盖短窗口。

## A3｜Goal-oriented residual / state repair

这是本包仅提供数学原语、尚待接入的扩展。

在 `engine.py` 的attention/FFN更新入口分别捕获当前state。比较同一 `h` 上 `heavy(h)` 与 `cheap(h)`，与frozen same-support参考自身轨迹目标分开。先做FFN-only，teacher heavy执行no_grad，student cheap保留梯度。重算不得更换RGB/数据增强。

记录 `-g*r` 的有符号task Taylor proxy，分别cls/reg。做real-intervention校准与gradient-norm、residual-norm、attention基线。不能用Taylor作为oracle标签或声称PDE界。训练候选只加入一项任务敏感的residual supervision，并与已有same-support loss分开匹配权重/teacher查询。

产物：`operator_value.jsonl`、before/after-TIA统计、standalone CUDA等价测试、matched新增config。若只改善NMSE不改善TAD，不保留最终方法。

## A4｜Temporal value与原轴恢复；Graph为独立分支

保持individual-frame selection接口及所有physical-time/contributor metadata。先为现有FrameRouter补共同候选集合、fixed target scale与online query records，再比较current/post/EMA/future；不得默认替换整套H65采样器。

原轴recovery对照保持相同selected frames、heavy features、head、训练预算：现有Cross；同架构fresh Cross；continuous-time reference（本包原语）；static physical-time graph；dynamic no-referral；dynamic referral。把相同容量比较和相同完整GFLOPs比较分开。

Graph新鲜节点/可靠anchor允许来自真实已算heavy evidence；重编码前选帧不可读取本次未算heavy。禁止GT boundary构图作为可部署结果。GM完整multi-edge机制与当前degree16适配明确区别。粒度/方向：TAD离线可双向，物理邻居不以不规则packed rank替代。先做repair/access，最后才单独评估GraphKV。

产物：recovery adapter、geometry/invalid/duplicate/coalesce测试、cost ledger、Graph是否必要的matched结果。不在WTR v0默认解除Graph拒绝保护。

## A5｜RISE-inspired未来价值研究

先定义两个未来：额外inference action的回报；未来训练checkpoint的action价值。论文只在二者明确时使用“future”。

离线三时间点probe：同输入、同候选ID、同预算/支持，真实重执行得到Va、Vp、Vf。fit/calibrate视频与score视频disjoint，β可在预登记小集合{1,1.05,1.1,1.2}中用开发数据选择；最终测试不选β。训练value目标标尺固定或映回raw loss后比较，禁止外推scales/calibration/indices。

强控制：noKD、frozen-post(beta1)、EMA、future(beta>1)、相同额外优化/teacher查询成本。当前包实现前四个teacher条件，不包含完整compute-matched extra-step generator；由本agent登记明确预算控制。

假设检验：forecast rank/sign是否改善、finite-budget chosen-action regret是否下降、最终mAP/算量是否改善。只提高教师自信不算通过。early→late状态重排不一定具有可外推线性，报告失败区域。实际condition变化/curriculum改变时reset或重新验证anchor。

保存anchor/post/EMA、interval、β、训练步、固定尺度和loss权重，resume和freshload一致。可预计算value-head函数降低teacher成本，但不得把模型teacher原forward成本写零。

## A6｜统计、论文和复现

重新读取全部结果，不照抄聊天中的峰值作为终点。按视频聚类bootstrap，每次重算全数据集AP。seed42结果的video CI不等于训练seed CI。公开方法reported、内部复测、特征输入、RGB全链路成本分开。

绘图只能用实测数据。保留负utility，局部loss/全局AP分开；样例按预登记median规则并加失败例。标注label-assisted search不是上界；预训练dense特征诊断不当成省算部署。图表定义详见FIGURES。

产物：Figure1–6数据源JSON、CSV、SVG/PNG、英文caption、未测占位清单；最终claim-evidence.csv逐项对应。加入原AdaTAD K384、Static、UniformFull、同预算同endpoint和跨数据集最终config。

## A7｜唯一 owner 部署

先执行RUNBOOK dry-run，核验现有dispatcher脚本及source_revision。不得复用陈旧PID或硬编码绝对asset路径。资源/代码/数据未满足时stages WAITING_ASSET，不伪造complete。

本包 `wtr_register.py` 默认不写；真实登记必须owner已授权维护且现有锁free，旧running/pending jobs不改变。owner自行按原命令恢复唯一控制器。新course可执行脚本均绝对路径；继承账户配额与max-live/max-train，不批量新增独立sbatch。

首次只是S六个固定D课程；B及S轴/T轴/Graph/泛化为后续明确登记，不生成全部笛卡尔积。已注册课程按原40/80日程完成，切片exit75继续，技术预检更新丢弃。

验收：dry-run文件无写入；重复stage拒绝；登记备份；source/args/resources可追溯；complete要求实际epoch/update/全量评估；提交必须有Slurm回执才写已部署。
