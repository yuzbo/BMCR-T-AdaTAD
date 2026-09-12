**FPW并行实现公共接口（主代理发布）**

基于add94ec（其训练源码90ce8bb，模型主体继承239d098）；保留BMCR80运行，原clip选择禁止。用户最新明确：全面并行实现/部署，不串行等待上一阶段mAP胜出；主要判断量为完整模型矩阵/卷积FLOPs和最佳五阈值mAP，延迟报告但不作淘汰条件。

公共dataclass由主代理维护于h65/frame/contracts.py。各实现只改自己的文件，不能覆盖别人的改动。

Reader接口：`FrozenAnchor(model_cfg,checkpoint,variant,budget=384)`；`.model`持有冻结eval FormalH65，`.vit`暴露已有ViT。`select(inputs,masks,metas,mode='anchor') -> (selection,scout_output,provisional)`；支持anchor/uniform/random候选帧选择。`selected_rgb(inputs,selection)`保持旧硬gather和无效RGB置零。`prepare_clips(selected_rgb)`按原归一化/原计算打包返回[B*K/16,3,16,H,W]。`encode_native(inputs,selection) -> (native_BAC,trace)`在原postprocessing最后一次插值前捕获，不能从rank384下采样。`pool_tokens(tokens,h,w) -> native_BAC`采用原norm/空间池化顺序。`configure_budget(K)`允许16倍数K，仅更改明确的运行时间组织，K变体不宣称旧rank恒等。Reader不得修改旧h65/full/upstream。

Geometry接口：`make_anchors(features,selection,masks,metas,trace=None) -> AnchorBatch`；`make_queries(masks,metas,selection) -> QueryBatch`；`interpolate_anchors(anchors,queries) -> B,Q,C`；`decoder_metadata(anchors,queries) -> (anchor_meta[B,A,10],query_meta[B,Q,8])`；`scout_context(output,masks) -> B,Q,96`。时间单位为原视频frame，GT进入original detector仍为原candidate坐标。Query不是teacher-exact；中心只取有效contributors。

Decoder接口：`build_decoder(kind,channels,context_dim=96,width=192,layers=2,heads=3,use_provenance=True,use_scout=True)`，kind为interpolate/tcn/cross。`forward(anchors,queries,context) -> B,C,Q`。零末层等于指定物理插值底座（含query_valid mask），不是旧rank mAP。元数据/scout消融保留相同模块/参数规模，只将对应输入置零。

Engine接口：`PackedStateEngine(channels,light_width=32)`；`forward(vit,clips,policy,native_valid=None) -> (tokens,h,w,trace)`。最新用户指定深度主方案为A-MoD：12层中的第2/4/6/8/10层为MoD、首层末层dense；路由分数来自前一dense层attention矩阵对query行和heads求平均的incoming attention（arXiv2412.20875v1式4），不乘分数缩放输出，不新增可训练深度router。主A-MoD压紧所选token的Q/K/V和FFN，未选token保持identity；scatter完整空间状态后仍执行原globalTIA(K/2)，这是TAD适配处。selectedQ/fullKV作为另命名上下文保持消融。静态block-drop对照同样保留首末层，不直接砍掉最后层。空间FFN token/2x2 tile门可独立执行；AMoD首末层不压空间。dense_mask训练须使用与compact同样被选K/V，可用attention mask模拟，不能用fullKV冒充等价。若SDPA不能直接返回分数，可用已投影的detached Q/K分块计算incoming scores，必须计入额外QK，不宣称零额外FLOPs。trace包含逐层实际q/kv/heavy_mlp/light/tia、routing-score QK和depth/spatial masks；spatial_quality可记录实际FFN更新比例，last_heavy_depth保持真实值（首末dense时可能都12）。不得选原clip作为时间路由单位。All-heavy应走原block或严格等价展开。

Teachers/loss接口：`OriginalTeacher(model_cfg,checkpoint)`提供`.dense_native(inputs)`、`.loss(native_BCQ,data)`、`.predictions(native_BCQ,masks,metas)`、`.post_processing(...)`；参数/buffer冻结eval，student feature梯度保留。可复用已有DenseTeacher global封装作为库，不使用DS3路由。`training_objectives(student_native,teacher_native,data,teacher,feature_weight=1.,gt_weight=1.,difference_weight=0.,output_kd_weight=0.) -> dict(cost,cls_loss,reg_loss,feature_loss,...)`。GT+feature默认；各辅助项独立，Bernoulli分类KD及距离回归必须符合真实head。

主代理负责`h65/frame/model.py/runtime.py/utility.py/router.py`以及train/eval/intervene/plan/dispatch，连接上述接口；辅助代理A reader/condition，B geometry/decoder，C engine/gates/cost，D teachers/objectives，E analyze图表，F official decoder/PBD原文与权重证据。核心配置从已完成S修正H65、B旧BMCR各自强锚点独立初始化，并行恢复/监督/深度/空间/联合训练；技术验收后同时启动，不等待前一路线性能排名。B01复用当前BMCR80，不另起joint40或重训warm。
