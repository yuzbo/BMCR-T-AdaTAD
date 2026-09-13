# Codex 实施交接：BMCR 强锚点、DS3 诊断与 C2 原型

基准：`yuzbo/BMCR-T-AdaTAD@239d098cd899936c35989fae6243c70259a85adb`。本文件是设计交接，不是已提交的代码改动或启动作业指令。完整论证、来源及证据边界见 `REPORT.zh.md`。只按用户已授权范围操作；本文不新增服务器部署许可或精度审批门槛。

## 1. 首批改动只做两件事

先完成用户已授权的同配方修正 BMCR：复用修正 warm20 EMA，训练集效用尺度重审，joint40，同规则完整测试和 EMA 选模。不要同时改路由、检测轴、TIA 或 loss。

紧接着增加**只读 checkpoint 诊断**：同一个 D1 checkpoint 的 I(U24)、H0(U24)、I(A24)、H0(A24)，以及固定选点的 online/EMA 对照。先不增加 H8/MLP/depth 的组合。当前 T24A 不使用它们，不能用其结果判断这些模块。

这些诊断确认后才实现一个新 recipe 的“U24 + 零末层残差补全 + mixed GT loss”C2 原型。先固定 U24，后研究新路由；不要第一版同时换 global teacher、clip粒度和骨干训练方式。

## 2. 不可覆盖的资产与语义

保留旧 `15280e5` 训练出的 H65/BMCR 权重，明确旧配方；保留修正 H65 S60、B55 peak和B60 terminal、各自修正 warm20；保留 train-only utility scales 及来源；保留官方 S/B `state_dict_ema` 权重；保留全部 D1 aux/aux_ema、optimizer、LR、更新次数、80轮课程，不将中途导出的C2写回原D1目录。

当前 checkpoint 合同至少区分：

| contract | native | detector | TIA | 备注 |
|---|---|---|---|---|
| `rank384_global192` | `[B,C,192]` | selected-rank384 |192| K384重排成24clips；NMS前回映 |
| `original768_local8` | `[B,C,384]` |原768|8| DS3原clip独立前提 |
| `original768_global384` | `[B,C,384]` |原768|384| 官方/新全局状态图 |

C为S384、B768。strict state_dict加载是必要条件，不是跨合同语义兼容证明。绝不把 `04c35a3` 项目当本轮工作树，不把历史S65.385724%的90轮global192成绩贴到local8。

## 3. 新接口建议

下面是接口说明，不要求无条件重构全部现有代码。保持旧默认行为；新增字段/策略使用明确版本和显式选项。

```python
@dataclass
class TimeGrid:
    # All arrays describe the same original window, including physical padding.
    frame_indices: Tensor     # [B, 768], original-video frame units
    candidate_valid: Tensor   # [B, 768], bool
    native_centers: Tensor    # [B, 384], same frame units
    native_valid: Tensor      # [B, 384], bool; handle partially valid pairs
    clip_ids: Tensor          # [B, 384], original 0..47 identity, not selected rank
    tubelet_phase: Tensor     # [B, 384], original 0..7

@dataclass
class NativeState:
    values: Tensor            # [B, C, 384]; original native only
    valid: Tensor             # [B, 384]
    source_kind: Tensor       # observed / H0 / H8 / approximate / interpolate / invalid
    exit_depth: Tensor        # [B, 384]
    # optional: age, nearest-heavy temporal gap, calibrated reliability

@dataclass
class RoutePlan:
    # Each layer has explicit original-grid IDs and physical/effective counts.
    active_clip_ids: list[Tensor]
    active_mlp_ids: list[Tensor]
    active_query_ids: list[Tensor] | None
    active_kv_ids: list[Tensor] | None
    # Budget accounting is based on actual packed execution, not mask.sum alone.
```

原始输入当前是 `[B,1,3,768,H,W]`；`DenseTeacher.prepare_clips` 负责原有归一化和转为clips。不要让新preview或cache重复/改变normalization。native恢复后继续使用现有原插值至detector768与metadata转原视频时间，避免重复乘stride/fps。GT候选坐标、frame坐标、秒必须带单位或在一个统一函数内转换。

global内部维护 `[B,48,8,Hg,Wg,C]`，160下Hg=Wg=10；flatten顺序必须与原VideoMAE/TIA保持一致。VideoMAE此处没有分类CLS token，不复制NLP/DyT的“永远保留第一个CLS”的索引规则。

## 4. 最小文件改动

### `tools/ds3_eval.py` 与 `h65/ds3/runtime.py`

新增 `--aux-state online|ema`，复用已有 `load_aux(..., ema=False)`；输出metadata包含源checkpoint、成功updates、epoch、EMA decay、teacher strict来源、policy、route来源与completion来源。

增加固定RoutePlan输入/导出，并支持 `completion=interpolate|h0|residual`。I(A24)必须使用与H0(A24)完全一样的选点；比较EMA/online补全时固定选点，另做路由对照。不得因为策略名字不同重新随机生成route。I(A24)可能仍需preview，按真实执行计成本。

保留211视频/792窗口、相同NMS与峰值/终点输出。增加pre-NMS错误诊断与分阶段timing，但不要覆写已完成JSON，也不把部分集诊断写成完整测试。

### `h65/full/scout.py::FormalScout.condition`

可把窗口分为 `[B,48,16]` 后张量化相反membership伙伴的局部距离矩阵，再进行masked argmin及集合特征聚合。必须保持原16候选cell、伙伴tie、有效mask、候选/伙伴/selected mean定义。不要用“近似更快”的nearest neighbor替代后称恒等。精确测试后再profile Python/GPU同步减少，不预填加速比。

### `h65/full/model.py::route` / `h65/full/utility.py`

新增只读诊断 `S0_to_S1_changed_ids`、物理位移、伙伴变化、多交换交互；保留旧输出默认值。新的效用接口建议 `evaluate_action(state, action, targets, teacher)`，分别定义 swap、0→8、8→12、spatial refinement，不把frame score求和直接当clip utility。

先保存实际分类/定位分量和联合差值，再在train-only集校准尺度。保持旧标签版本和漏检dummy/cap语义，不在原recipe下静默改变常量。若增加限制S1变化的决策，单独命名；推理不能使用GT或teacher来决定接受交换。

### `h65/ds3/auxiliary.py`

新增独立 `ResidualCompletion`，不删除现有H0。输入至少含 `base_native`、preview特征、valid、heavy mask、物理gap/source。输出层权重和bias置零；上游正常初始化或复用已训练preview特征。

```text
base = interpolate_physical(selected_full_features, selected_centers, native_centers)
delta = residual(base, preview, time_grid, heavy_mask)
student = base + valid * (~heavy_mask) * delta
```

所选heavy位置保持exact observed。step0功能等于Z24，但若仍执行preview，成本未必等于Z24；两个结论分开。不要把旧绝对H0 output直接当delta。H8零残差只能初始等于H8输入，不能承诺等于H12。

### `h65/ds3/model.py::DS3.native` / `ClipEngine`

拆出“按给定route取重特征”和“重建/混合native”的纯接口，便于同route四格对照。local cache只缓存相同原clip的H8/H12及必要元信息；显式检查local8、eval、无随机层、同空间crop和tubelet phase。

禁止缓存模式用于global384/global192反事实；禁止把dense late-MLP输出用于已改变输入的递归近似层。先测真实compact与cache的FP32/BF16误差及预测一致性，保留原预检精度容差定义，不要求跨不同GEMM batch布局bitwise相同。

### `h65/ds3/losses.py`

新C2 loss必须有真实混合学生检测损失。冻结参数不等于关闭输入梯度：

```text
teacher_targets = teacher(...).detach()
student_native = mix_and_reconstruct(...)
# detector parameters frozen/eval, but this call MUST NOT be no_grad
loss_gt = frozen_detector_loss(student_native, original_gt)
```

具体复用DenseTeacher原native→detector插值和loss接口；固定teacher buffer与lossnormalizer状态。分类KD匹配20类独立sigmoid/Bernoulli；当前左右回归为非负距离，通过point stride还原端点，不是categorical distribution logits。跨rank teacher只在原时间预测端匹配，避免逐index KD。匹配teacher框时采用固定detached关联并保留GT，避免teacher漏检把GT监督删除。

路由监督仍需单独的条件效用loss或明确的ST/gate路径。hard top-k索引不会因增加TAD loss而自动可微。不要用 `abs(grad * residual)` 冒充有符号真实损失改善，保存符号/值/成本及状态条件。

### 新 `h65/ds3/global_engine.py`，仅在前述诊断后实现

不要通过给ClipEngine换temporal_size直接运行compact clips。原Block顺序是attention residual→MLP residual→Adapter，Adapter已经返回input+delta。

```text
x1 = x0 + scatter(attention(norm1(gather(x0, active_clips))))
x2 = x1 + scatter(mlp(norm2(gather(x1, active_mlp_tokens))))
x3 = original_global_adapter(x2_on_FULL_spatial_grid)
```

上式省略原drop_path和norm/shape细节，实际必须照原Block。不得再做 `x2 + adapter(x2)`，否则双重加identity。未active位置保留state并参与完整TIA。若为选query attention，全KV或选KV必须分别定义，C2按相同attention图训练。

原TIA在每个空间格点先down-projection/GELU再时间卷积，`mean(GELU(Wx))`一般不等于`GELU(W mean(x))`。因此只缓存池化native384不能重现原TIA；低分辨率/merging路径需在每次TIA前恢复公共空间格，而非只在最终detector前补齐。

## 5. 课程与成本元信息

保留Z0/D1/C2/J3明确命名：Z0无新增训练；D1完整teacher前向+辅助dense fitting；C2学生见过虚拟缺失但主干可仍完整算；J3真实compact路径联合适配。改变课程必须新recipe，而不是重标记已有80轮D1。

C2首版：固定U24→补全GT校准→条件utility路由→必要时一种出口/空间策略。teacher full forward、teacher input-gradient backward、学生forward/backward、swap detector查询、cache建立/读取/存储各自计数；相同optimizer steps不代表相同计算。

global C2多层输入改变后不能沿用固定dense teacher下一层cache，可能需要teacher+学生双流；4090上采用batch1+梯度累积、checkpointing等需明确是否改变有效batch/随机统计和课程。不要先假定训练也省算。

## 6. 单测与诊断的最小集合

| 建议测试名 | 必须验证 | 失败含义 |
|---|---|---|
| `test_full_budget_original_identity` | global原图与新全开门输出/框一致 | 实现不等价，不能进入性能比较 |
| `test_t24a_only_h0_and_full12` | 不调用H8/sparseMLP | 防止错误归因和policy漂移 |
| `test_residual_zero_equals_z24` | 同route/precision，step0预测回到Z24 | 初始化承诺不成立 |
| `test_zero_logits_is_not_uniform` | 明确旧tie route；新uniform显式生成 | 防止隐含路由先验错误 |
| `test_local_cache_vs_compact` | 原clip缓存与单独运行在容差内一致 | cache使用条件不成立 |
| `test_global_cache_rejected` | 跨clipTIA反例与拒绝错误cache模式 | 防止假反事实监督 |
| `test_pointwise_mask_vs_packed` | 同输入MLP一致，实际tensor变小 | 避免只mask不省算 |
| `test_attention_graph_contract` | query/KV策略与训练图一致 | 防止dense-mask/compact错配 |
| `test_valid_physical_counts` | full/partial/short、odd pair、空clip，effective≠padding | 预算/覆盖/时延解释错误 |
| `test_legal_crop_endpoints` | 不改upstream裁剪，裁剪伪端点不监督 | 既有纠错回归 |
| `test_frozen_detector_student_grad` | teacher/检测器参数buffer不变，学生梯度存在 | C2实际没有任务学习或teacher被污染 |
| `test_sigmoid_box_kd_semantics` | 独立类别和物理端点语义正确 | 蒸馏目标不匹配真实head |
| `test_condition_vectorized_exact` | 原伙伴、tie、set mean、结果与梯度一致 | 向量化改变算法 |
| `test_route_declared_vs_executed` | 每层clips/QK/AV/MLP计数一致 | 预算声明与真实执行不符 |
| `test_inverse_before_nms` | 原时间候选后再做NMS | 重复/边界评价失真 |

本目录CPU toy脚本不是这些仓库单测的替代品。上述测试尚未实现/运行；不得把列表转换成“全部通过”报告。

## 7. 扩大实验和改路线的依据

完成修正BMCR后才能讨论其新成绩；完成D1既定课程后才能讨论80轮终局。诊断可提前开展，但所有新指标留空直到有对应回执。现有中间全测试峰值选择需要继续披露；同时保留终点。

先报告同checkpoint的因果信息和实际计算趋势，再决定投入完整课程。若route排序无真实条件收益，先改监督/动作；若固定route的混合表示失配，先做C2；若输入本身丢失不可观测短事件，改覆盖/轻观察/粒度；若算量下降而延迟不降，优化执行和批量，而不是加更多模块。没有额外强制160/40或1pp门槛，也不把未知的哈希/审批脚本当科学标准。
