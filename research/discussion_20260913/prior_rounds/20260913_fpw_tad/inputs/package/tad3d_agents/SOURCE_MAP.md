# 固定源码改动定位

根提交：`239d098cd899936c35989fae6243c70259a85adb`。行号以本快照文本为准，agent编辑后必须同时输出新行号与diff；符号是主要稳定定位。

| 固定文件/符号 | 已有行为 | 最小接入/禁止改变 |
|---|---|---|
| `h65/full/model.py::FormalH65.__init__`14–43 | budget影响TIA、计算打包、rank detector长度 | 新wrapper持有已有模块，legacy默认行为不动 |
| `h65/full/model.py::route`45–57 | S0条件效用修正后重采样S1 | 先保持原选点；后加intervention记录，不把16cell自动删除 |
| `h65/full/model.py::encode`59–71 | gather→VideoMAE→rank384 | 在backbone后处理的最后Interpolate之前捕获native192；不能反向下采样 |
| `h65/full/model.py::train_batch/raw_route` | GT rank映射、EMA反事实、head hook | 原轴分支独立GT合同与head，原rawroute保持可复现 |
| `h65/full/scout.py::FormalScout.forward/condition` | 全时间低成本表示和16slot局部交换 | 复用上下文与action/boundary先验；保持旧梯度分流；新的全局候选单独recipe |
| `h65/full/utility.py::counterfactual_targets` | 真实旧route交换重编码 | 存actual action；feature repair另命名不能替代 |
| `h65/full/utility.py::classification_cost/localization_cost` | 固定分类指派、含miss定位惩罚 | 迁移原轴需重查目标与尺度，不丢漏检 |
| `h65/transport.py::sample_rates/gather_with_transport` | frame exact-K、prefix有效性、硬forward/代理gradient | 首版不重写选择器；新metadata不改变hardindices |
| `h65/transport.py::restore_tubelets/interpolate_irregular` | 物理中心插值到candidate T | 新native384查询显式实现，不误用当前函数输出长度 |
| `h65/full/geometry.py::TrueTimeMap` | rank↔true映射，NMS前回映 | legacy保留，original分支禁止二次inverse |
| `h65/full/data.py::LoadFramesWithBoundaryValidity` | 保留真实端点有效性 | GT辅助loss继续沿用，不更改crop随机过程 |
| `h65/full/runtime.py::optimizer_for/EMA` | 修正LR分组、旧EMA对象构造 | 新模块另optimizergroups与EMA构造，禁止默认只复制旧类漏掉decoder |
| `upstream/.../backbone_wrapper.py::unflatten_and_pool_features` | pool/rearrange/interpolate | native捕获/专用wrapper，不改upstream全局行为 |
| `upstream/.../vit_adapter.py::Adapter`19–75 | down/GELU/fullgrid时间conv/up+identity | scatter完整空间state后调用；不要再加一次identity |
| `upstream/.../vit_adapter.py::Block.forward` | attn→FFN→Adapter | 新engine复制算子顺序并同input parity，不只改temporal_size |
| `upstream/.../dense_heads/anchor_free_head.py` | sigmoid类别、左右距离、GT分配/normalizer | KD按真实语义；frozen head保留输入gradient |
| `h65/ds3/model.py::ClipEngine` | 仅localTIA可compact整clip | 参考gather/scatter算子，不复用local独立性做global标签 |
| `tools/full_eval.py::ArithmeticCounter/profile` | 实际矩阵卷积2MAC含QKAV | 增加新算子scope、未知矩阵告警、decoder/router/light counts |
| `tools/full_train.py/tools/full_eval.py` | 修正warm/joint/完整测试 | 保持旧baseline入口；新工具frame前缀隔离 |

## 新公共dataclass

`FrameSelection`: indices[B,K], valid[B,K], membership[B,T], continuous_proxy[B,K], input_frame_times[B,T]。

`AnchorBatch`: features[B,A,C], contributor_indices[B,A,2], contributor_times[B,A,2], contributor_valid[B,A,2], centers[B,A], span[B,A], clip_position[B,A], last_heavy_depth[B,A], spatial_quality[B,A], feature_space_id。

`NativeBatch`: features[B,C,T/2], native_valid[B,T/2], candidate_mask[B,T], query_frame_times[B,T/2], source_descriptor，禁止把任意position标为teacher-exact。

`ExecutionTrace`: 每层实际q_rows/kv_rows/mlp_rows/light_rows/tia_rows、patch实际输入shape、unique/physical candidate数、decoderQ/K数量、scope、precision；必须与forward hooks核对。

`InterventionRecord`: video/window/augmentation_id、checkpoint/recipe、state_id、action_type、remove/insert或depth/space坐标、repair_delta、actual_delta_cls/reg、实际costdelta、misscounts、S0/S1changes、teacher_kind；不能只存总标量label。
