# Agents 实现合同

## 总规则

所有 agent 从 `1955057508af5a5dfd59a98bddf49302bee5972c` 创建独立开发分支。最终由一个 owner 集成；不要多个 agent 同改 engine.py。原课程继续跑，旧 checkpoint/配置/回执不覆盖。新代码版本、新配置 ID、新 init 来源、新训练成本各自记录。seed42 是现有正式课程约束；多 seed 验证需要另登记，不擅自复制全部旧课程。

新方法建议 recipe `wtr_value_v1`，但它目前不在原 `read_config()` 允许列表中。只有 Agent C/D 完成实际功能和状态契约后才注册。不要把未实现字段写进 JSON 当作功能落地。

## Agent A：证据与实验合同

读取 `h65/transport.py`、`h65/paper/{encoder,engine,routing,interventions,profile,training}.py`，以及 `tools/{paper_train,paper_eval,paper_dispatch}.py`。

实现/交付：
- `audit.json`：固定 SHA、源文件 Git blob hash、参数组 LR、冻结/可训练列表、plan 菜单、head 初始化、teacher 来源。
- 从 `current_results.json` 导出 matched-epoch 表、逐方案分布、真实 D/S 启用率、每层 heavy query/FFN 数。
- 为现有 action JSONL 补充 window/crop/selection/budget/checkpoint/scales/operator IDs。旧日志不能凭 video_name 强行跨时刻配对。
- 区分 pure GT `actual_delta`、feature substitution `repair_delta`、local residual NMSE 三个字段。
- GPU baseline parity、checkpoint 严格重载、资源合同与计费检查。

验收：FP32 全保留开关与原执行结果接近；batch tail/padding/no-valid query tests；probe 前后参数、buffer、训练态不变。所有后续 agents 依赖该验收。

## Agent B：测量价值，而不是重新训练所有模型

直接使用本包 `wtr.probe`，先 6 视频，再 32；最终 characterization 扩大到按类别/动作长度覆盖的训练-development 视频集。其余视频独立校准和机制检验。

实现扩展：
- 支持 frame bank 的固定 D/S masks 与闭环 D/S masks 两种语义；保持 action ID 不混淆。
- 保存当前层 action 前状态和所有可见预算/年龄元信息，不把 action 后 heavy target 拼入 predictor 特征。
- observation 替换与 compute 干预分开；增加一次性移除1/2/4/8个观测及 sampled coalition。
- 每个随机组固定 seed，并且记录搜索了多少候选、执行了多少 forward。
- 在小候选集合允许穷举最优；在大集合只叫 GT-assisted greedy/beam search，不叫数学 oracle 上界。

输出：`action_bank.pt`、`paired_values.jsonl`、`interaction.json`、signed CDF、视频聚类 CI、样本时间轴。不得捏造 dataset mAP 或把局部 loss 差转写成 AP 提升。

## Agent C：算子解耦与状态可恢复性（主实现）

主要修改 `h65/paper/engine.py`；接口与状态在 `encoder.py/model.py/runtime.py` 登记。

### C1 解耦 attention 与 FFN

当前 `admitted` 同时控制 attention query 和 FFN heavy，空间筛选又限制在 admitted 内。新 recipe 中显式维护：
```
q_heavy_mask[layer], ffn_heavy_mask[layer], valid_mask
attention_age[layer/state], ffn_age[layer/state]
```
旧 recipe 完全保持旧逻辑。新模式允许 Q-heavy 与 FFN-heavy 不相同，不能用同一个 `admitted` 隐性绑死。

每个新 mode 都必须独立可配置：
- Q full + FFN full（parity）
- Q sparse/full-KV + FFN full
- Q full + FFN heavy/light
- Q sparse/full-KV + FFN heavy/light
- 原 A-MoD coupled（历史实现，不改名新算法）

未选 Q 可 hold 或用已有 attention-light；未选 FFN 用已有 light。两种替代形式分开消融。完整 global TIA 的调用和数据形状保持。默认继续保护头四层和最后一层。

### C2 学习算子价值

复用 `wtr.core.OperatorValueHead`，先用 native tubelet 内2×2空间组；这一分组不改变时间选择仍是逐帧。
输入包括 pre-op state 的低维摘要、原始时间跨度、operator/layer ID、q_age/ffn_age、已用预算、cheap action/boundary cues。不得读取 GT、teacher logits 或 action 后 features。

输出 `[group,operator,{cls_gain,reg_gain,logvar_cls,logvar_reg}]`。显式 Gaussian NLL/排序监督训练 scorer；不能指望 hard top-k 自动获得所有 task gradients。精确配额通过 existing capacity mask 或 `exact_capacity`，先维持当前 per-native-time 最低覆盖；不独立 sigmoid 后再随意修预算。

### C3 同输入局部恢复训练

使用 `wtr.operators.same_input_residual_loss`：heavy 和 light 输入必须都是当前 student 的同一 pre-op state。heavy 结果 stop-grad，记录 heavy target 的 checkpoint。

- FFN：先做局部同输入 heavy FFN → light FFN，原 support loss 可保留为对照。
- Attention：若 light 只有 token-local 输入，不能假设其能恢复全局交互；单独比较 local-only 与 cheap-context-conditioned light，或保留 attention dense。
- 每个 batch 只抽1个指定层/部分组形成 target，避免第二整条 backbone 重算；记录实际开销。
- SSA 式双向对齐是独立后续选项。当前大部分 QKV/FFN frozen，不能声称它们已在双向训练中学习稀疏性；先明确哪些 adapter/input/weight 接收梯度。

验收：未启用新功能时 bitwise/容差 parity；all-true masks parity；all-light 安全；teacher 无梯度；scorer/adapter/head 有实际更新；两种 ages 正确；D/S 不可行组合被拒绝；计费覆盖 light/scorer/attention额外评分。

## Agent D：RISE-inspired 价值外推（不依赖 Graph）

先运行本包 `forecast` 和冻结 detector 的 `fit_router`。这只是第一阶段，不得写成在线递归算法已完成。

在线集成修改 `interventions.py/objectives.py/training.py/tools/paper_train.py`：
- 在每个 grounded block 更新前保存 anchor 的 scorer state、scales 和候选 ID。
- GT task + 实际反事实标签产生 post-update scorer；两者在同一原始输入、同一 plan、同一 action集合上评估。
- `raw_anchor = normalized_anchor * anchor_scale`；`raw_post = normalized_post * post_scale`；统一尺度后构造 teacher。
- `p_future = softmax((v_post + (β-1)(v_post-v_anchor))/temperature)`，含 STOP；teacher全程detach。
- 继续保留 current GT value NLL，以 `JSD(student,p_future)` 为辅助项，而不是全靠自我反馈训练。
- β预声明1.025/1.05/1.1/1.2，默认保守；可衰减至1。EMA 与 β=1 分别实现。
- 模型结构、预算课程、算子替换发生离散变化时重置anchor；不能跨变更外推。
- 外推不可信时回退current，由独立 calibration 定义可信度；不得偷看最终test或未来真实checkpoint标签。
- 新checkpoint保存anchor、scales、教师版本、candidate bank ID、优化器和RNG；重载后同一步next update可复现。

对照：no additional JSD / current / genuine EMA / future / 等额外GT优化 / 直接采用外推值策略。主比较固定总GT probe数量和训练更新数；另报训练forward/backward与wall time。外部 teacher loss 去掉不等于消除原初始化资产依赖。

## Agent E：Graph/时间恢复（独立可选）

保留当前 Cross 作为主线。固定 selected frames、各层掩码、初始化、训练预算和 head，仅改恢复机制：
1. 物理时间插值；2. 当前 Cross；3. 连续时间 attention；4. 固定稀疏物理邻接；5. 同degree动态referral。

现在已有 `edge_ops.py` 与 `graph_recovery.py`，不要另写同名Graph模块假称新增。Graph KV替换、Graph recovery、frame context、L6/9/12反馈必须解耦开关。backbone graph不等于全原始时间轴graph；索引支持从packed域投射回physical域必须明确。

度数扩展前先修改 `EdgeRouter`/`initial_edges` 固定16合同以及状态重载测试；不能只在JSON改 degree=8。

验收：local/referral一致候选预算，false/invalid edges不参与softmax，coalesce重复地址守恒，只有保留权重有梯度，不构造NxN中间矩阵，报告fullKV投影与referral/sort成本。仅在同cost机制结果支持后纳入主论文。

## Agent F：实验与统计，独立于模型作者

- 主结果必须 matched80epoch终点 + 预声明选择规则；旧最佳峰值仅作为历史progress。
- 至少给T-only、D/S-only、联合TDS的固定预算方法；每条实际profile后匹配成本，不能只匹配K。
- 完整211视频/792窗口官方mAP复现；95% CI按video cluster bootstrap重算全数据集AP，不平均per-video AP。
- 记录@0.3…@0.7、短/长动作、边界错位、false positive/negative，并包含失败案例。
- seed42探索先行；确认新增多seed授权后只复验decisive pair，不能自动铺全部实验。
- P0完成不能因新方法故事改写；已提交课程没有性能淘汰门槛。新方案是否扩大实验由独立诊断决定。

## Agent G：唯一部署 owner

读取原 `paper_dispatch.py` 和当前实际 deployment receipt。原快照 PID 只是历史信息，先核实时刻存活owner。不要第二个 `--submit` 控制器，不把旧deployment复制到新工作树继续调度。

新工作树可以是执行源；调度元数据统一归原owner。stage `args` 必须指向冻结的新源码、绝对配置路径；done/requires依赖绝对真实存在或owner约定的相对根。先dry-run打印，不提交。

只新增明确审批的阶段；保留账户配额与现有max_live/max_train约束，首批建议同时最多2个新GPU diagnostics，CPU forecast 不额外占GPU。新增 kind 必须实现完整receipt validator，不借用 `analysis` 冒充official_AP_reproduced。

训练 --preflight 输出目录和正式训练分离。exit75为planned slice，不标失败或结束；续跑必须同config/fullstate。新架构warm-start不是resume，不加载旧optimizer，更不能多训40轮当公平80轮结果。
