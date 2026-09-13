# Codex实施交接：先保护锚点，再校准混合状态

## 任务边界

基线仓库为 `yuzbo/BMCR-T-AdaTAD`，唯一代码起点为 `239d098cd899936c35989fae6243c70259a85adb`。本任务只编写代码、配置、测试和实验执行说明，不提交Slurm/服务器训练/全测试作业，不覆盖旧checkpoint，不改写历史成绩。主报告为同目录 `RESEARCH_REPORT.zh.md`；它的来源索引给出固定源码和论文阅读范围。

先核对固定ref与用户目录中真实可用的依赖/资源。不要从默认分支继续；不要将历史04c35a3或15280e5旧配方当当前实现；保留上游OpenTAD来源346d09d19e2091372cec48172dbe40f7b28bdee6的来源记录。资源路径从现有配置和用户环境解析，不编造绝对路径、不创建假checkpoint。

## 第一提交：只增强诊断与合同描述，不改变历史行为

### 1. 正式joint初始化的检查覆盖

修改 `tools/full_train.py::main`：现有joint且非preflight分支才加载corrected-warm及utility-scales，导致preflight不能检查同一路径。抽取共同初始化函数，允许“加载真实warm与尺度、打印合同、执行明确的检查后退出”，不修改正常joint训练的LR/采样/损失。

保持修正warm20→新训练集效用尺度审计→joint40；joint入口不得静默用旧尺度。输出父checkpoint元数据、来源epoch/update、参数角色、LR组、GT裁剪合法性；这些是复现信息，不是新增审批制度。未开始的修正BMCR不得写任何新成绩。

### 2. DS3推理因素显式解耦

修改 `h65/ds3/routes.py` 和 `tools/ds3_eval.py`：把 route、fill、checkpoint state拆为独立参数，同时保留全部旧policy别名的原行为。新增参数是本次拟议接口，当前源码尚不包含：

```text
--state {ema,online}
--route {legacy_policy,uniform,adaptive,random}
--fill {legacy_policy,interpolate,preview,residual_interpolate}
```

`h65/ds3/runtime.py`中的加载能力已有online/EMA选择，评测入口需向其透传；不要修改正在进行的D1课程或历史EMA文件。先实现uniform/adaptive × interpolate/preview的同24clip对照，深度全部12、空间比率1、local-TIA8、原detector768。

嵌套0/8/12的uniform控制应显式在uniform的浅集合上确定深集合，不能默默随机。为保留历史可重现性，可另保留`legacy_nested_random`名称。零logits必须使用明确均匀fallback或显式uniform先验；stable top-k相同分数偏向前序，不等于均匀。

### 3. 元数据与成本台账

给 `DS3.native` 的trace以及 `FormalH65.route` 的诊断输出增加原样可序列化字段。保持训练数据流不变；不要为日志反复将大量CUDA张量转CPU导致时延污染。性能测量时关闭非必要日志，另测日志成本。

建议合同字段：

```text
source_commit, parent_checkpoint, teacher_checkpoint, teacher_graph,
native_axis, detector_axis, tia_scope, candidate_stride, tubelet_phase,
padding_policy, route_unit, route_policy, fill_policy, regime, state_key
```

逐样本计数字段：

```text
valid_candidates, physical_candidates, valid_tubelets, partial_clips,
attention_clips_by_layer, query_tokens_by_layer, key_value_tokens_by_layer,
heavy_mlp_tokens_by_layer, surrogate_tokens_by_layer, tia_state_tokens_by_layer,
heavy_input_clips, observed_mask, estimated_mask, physical_time, support
```

保持官方、local dense、Z0数值输出不变；不要用新的padding规范悄悄改旧基线。

## 第二提交：Grid-C2固定路由残差校准（优先新实现）

### 4. 新的数据接口

可在 `h65/ds3/` 新建 `state.py` 或将以下概念集中在 `model.py`，不要求分散成许多模块。类型是设计要求，不是声称现有类已存在。

```text
RoutePlan:
  depth[B,48] ∈ {0,8,12}
  valid_candidates[B,768], valid_native[B,384]
  physical_time[B,384], support[B,384,2], valid_count[B,384]
  optional spatial_ids, source_kind

NativeState:
  features[B,C,384]
  valid[B,384], observed[B,384], physical_time[B,384]
  support[B,384,2], optional confidence[B,384]

LocalCache:
  native12[B,C,384]
  exit8[B*48,C,8]
  teacher contract + exact augmentation/window/position/mask identity
```

C=384对应S，C=768对应B。输入始终[B,1,3,768,H,W]；clip RGB[B*48,3,16,H,W]；160输入每clip token[B*48,800,C]。native384→detector768是DS3原轴合同。BMCR的native192→rank384应保持另外明确的类型/元数据，不允许同形状近似加载替代语义检查。

### 5. 装配函数

新增显式 `assemble_native(...)`，只负责选中特征、物理插值、残差和mask，不暗中启动heavy backbone。

第一版仅固定uniform24、depth12、space1：

1. 取得完整local teacher native12缓存，但只在训练用于标签/虚拟学生装配；缓存不包含梯度。
2. 按RoutePlan提取选中clip的真实native12作为observed状态。
3. 用现有`interpolate_native`在物理中心插值，形成R_S。
4. 新残差网络末层权重/bias为零，其输入可以包含现有H0隐状态或预测、R_S、Δt、有效支持及缺失标记；只更新missing位置。
5. `mixed = R_S + missing * residual`，观测位置保持原heavy状态。

不要在残差末层为零的同时再用一个永远零的乘性gate截断所有梯度。第一版不添加可靠性gate、MoE、多ROI或新空间网络；它们不是C2有效性的前提。

推理版本调用现有 `ClipEngine.execute`，真实只运行选定24clip。不得调用完整48clip teacher forward后再mask。epoch0与Z24装配等价应在固定route下测试；不能要求它等于H65/BMCR。

### 6. C2损失与梯度

在 `h65/ds3/losses.py` 增加单独的C2入口，原D1入口原样保留。第一版比较D1与C2时保持相同teacher/初始化/更新/增强/route，唯一主要变化为真实混合图的GT检测监督。

现有 `DenseTeacher.loss(native, masks, metas, gt_segments, gt_labels)` 可对输入native反传；参数冻结不妨碍输入梯度。使用它时必须保持teacher eval以及已有FP32 detector路径，检查其参数/buffer不变。不要在学生检测器输入周围使用`no_grad`或`detach`。

损失组合建议按实验逐项启用，不默认全部堆叠：

```text
L = L_TAD(mixed, GT)
  + lambda_feat * valid/boundary/interior-aware feature constraint
  + lambda_cls * semantically matched prediction distillation
  + lambda_reg * matched physical-coordinate localization distillation
```

第一配对试验可以只增加L_TAD，避免把C2与新KD一并混淆。KD后续使用学生可微pre-NMS输出，匹配独立sigmoid分类语义；box回归必须物理坐标对应，保留GT。不得把NMS输出detach后的指标当可微监督。

重建器先在固定route下校准；router不因L_TAD存在就自动得到hard索引梯度。不要给“hard top-k可微”这一错误解释配上看似非零但实际只来自别的aux损失的梯度日志。

### 7. 单次local dense缓存的反事实标签

只有固定local teacher、eval、同窗口/增强/位置与mask时，clip深特征才可独立复用。为下一阶段实现训练集标签函数：给定当前mixed状态和RoutePlan，生成合法同预算clip交换，重新装配并主要重跑detector得到signed ΔL；不要重跑每个交换的完整VideoMAE。

标签必须注明当前填充器、预算、teacher版本及动作：0→12、0→8、8→12分别校准。不要把当前0→12 proxy默认赋给0→8。损失分量、miss penalty饱和、交换伙伴、同预算互补关系都写入审计。

比较S0单交换与S1整组重解码时，二者实际动作集合必须保留，不能拿局部交换label评价整个CDF变化而不承认差异。GT oracle仅用于训练诊断，不部署。

缓存不能推广到global-TIA，也不能将dense最后层缓存视为稀疏MLP递归状态的真实输出。不同compact batch形状的AMP浮点差异要单独报告。

## 第三提交：只在前两阶段证据支持后实现全状态global engine

新增独立 `GlobalStateEngine`，不让现有`ClipEngine`绕过local断言。表示为[B,384,P,C]，每层按原顺序：attention残差→MLP残差→完整栅格TIA。所有位置至少保存原状态；被选位置gather重算、scatter回写，未重算位置仍进入合法全栅格TIA。

优先复用原重算子权重。第一版selected-Q/full-KV应拆分现有QKV投影权重并验证all-Q等价，保留原位置/clip attention支持，不把attention改成全768候选全连接。这样省去的是部分query相关计算而不是全部KV。空间tile/深度gate要共享同一执行账本。

训练可以保留全量teacher，但改变后的学生global尾层必须真实运行；不能复用所有dense尾层缓存。全heavy初始与同合同官方等价，再逐步降低heavy预算。BMCR的global192必须在另一个rank合同包装器实现，不能偷偷改成global384。

此阶段不强制同时实现ROI、merge、MoE、8层出口或降分辨率；先用静态128、D8作为独立低复杂度对照，只有匹配成本和性能证据支持才增加动态复杂度。

## 必须新增/扩展的测试

测试文件名为建议，实施后方可执行。先检查仓库已有测试入口并扩展，不假设下面新文件已经存在。

| 测试组 | 断言与证据 |
|---|---|
| `test_ds3_contracts.py` | native384/original768与native192/rank384拒绝混用；all-heavy同scope等价；strict load之外校验语义合同 |
| `test_ds3_routes_and_fill.py` | 固定预算、索引唯一、零logit均匀fallback、nested深度；历史Z0可重放；残差epoch0=Z24 |
| `test_ds3_mixed_grad.py` | frozen teacher/detector参数和buffer相等；mixed输入/残差梯度非零；router梯度路径单独解释 |
| `test_ds3_cache_equivalence.py` | local cache/virtual/compact FP32与AMP误差；global明确禁止缓存交换冒充真实运行 |
| `test_h65_counterfactual_contract.py` | signed交换方向、固定/重指派分类目标、漏检饱和、尺度来源、S0/S1动作集合 |
| `test_partial_window_geometry.py` | full/partial/short、奇数有效候选、部分tubelet、padding与观察计数；有效端点语义 |
| `test_execution_ledger.py` | 禁用算子确实不执行；QK/AV及MLP token计数；未知矩阵算子不被静默忽略 |

数值容差根据FP32/AMP和同/不同batch布局分别记录；不把一个任意宽阈值当函数等价证明。零末层首步上游梯度为零可以正常，应检查末层首步可学、后续上游可学，而非要求所有参数每一步都有非零梯度。

同目录 `toy_audit.py` 为已可独立运行的标准库代数脚本：

```bash
python toy_audit.py --out toy_results.json
```

它不导入BMCR/DS3，不加载权重，不可替代上述项目一致性测试。

## 实验说明文件应按如下顺序产出

E0：修正BMCR同配方，修正warm20＋新训练集scale审计＋joint40；尚未开始，不填成绩。

E1：已有D1 checkpoint的route/fill/online/EMA诊断；保持local、24clip、depth12、space1；明示fill成本不同。

E2：固定route的D1继续训练与C2配对校准；相同更新、teacher与增强，额外detector反向成本单列。

E3：固定重建后再做条件route/clip交换校准；oracle只作训练诊断。

E4：需要时做tubelet/micro-clip计算粒度控制，再做真实RGB粒度控制；区分观察数匹配与FLOPs匹配。

E5：静态128/D8与条件重更新比较；证据支持后才global全状态和三维组合。

不要形成全部选项完整笛卡尔积。不引入未经接受的160/40、独立U384完整训练或1pp门槛。按现有用户规则用测试峰值选EMA，另报终点并说明选择偏差；固定checkpoint的按视频bootstrap不消除训练seed和选模偏差。

## 交付内容

提交代码diff、修改文件/符号说明、已执行的本地测试及未执行项、可重现配置、同scope等价结果、真实执行计数、训练/teacher/缓存成本预算与未来服务器命令草案。没有服务器实测的部分明确标记“未执行”。

支持扩大实验的依据是因果控制清晰、混合任务收益与最终多阈值/困难事件表现一致、真实执行节省可测；不依赖未接受的任意精度门槛。若只能降低MAC但不改善同口径GPU/端到端时延，应优先查路由同步、装配和数据链路，不给出“已加速”的结论。
