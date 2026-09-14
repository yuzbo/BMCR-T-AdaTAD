# Figure caption and source notes

**English caption.** Overview of the implemented Full-V2 for temporal action detection. (a) A low-cost Scout supplies evidence for budget allocation and H65/BMCR individual-frame selection. Selected observations are packed and embedded for VideoMAE. The first four and final blocks are dense; alternating later blocks support depth admission and spatial heavy/light updates. Multi-layer selected-support features are decoded onto the original temporal axis before a trainable TAD readout. (b) Routed attention retains the full key/value context within each packed attention domain, while depth-bypassed tokens receive light updates. The FFN uses disjoint heavy, spatial-light, and depth-light masks, followed by TIA across the selected temporal support. Explicit residual paths preserve the input state. (c) The Cross decoder predicts a residual over interpolated anchors, conditioned on Scout context and physical-time metadata. Training additionally uses a frozen same-selected-support full-depth/spatial reference, an external dense teacher, and the shared-full student branch. Dimensions and lengths shown correspond to THUMOS with VideoMAE-S/B.

**中文图注。** 当前已实现Full-V2的结构。(a) Scout提供预算分配和H65/BMCR单帧选择依据；选中观测打包进入VideoMAE，前四层和最后一层保持dense，后段交替层支持深度路由及空间heavy/light更新；多层特征经Cross恢复原时间轴，再进入可训练TAD读出。(b) 路由注意力在打包注意力域内保留完整KV，对深度绕过位置使用轻量更新；FFN按互斥掩码分配heavy、spatial-light与depth-light计算，之后在选中时间支持上进行TIA交互。(c) Cross预测插值底座上的残差，利用Scout上下文与物理时间元数据恢复原轴。训练监督单独列示，不构成部署时的carrier分支。

## Exact interpretation

- Original block IDs 4/6/8/10 (zero-based) are drawn as blocks 5/7/9/11. The other blocks, including the first four and last, are dense. This is a 12-block model, not a 12-fold repetition of the entire dense-plus-routed sequence.
- D/S masks are active only at configured routed layers. `M_H` is contained in the depth admission mask `M_D`. Complements are restricted to valid tokens. Layer norms and stochastic-depth operators are omitted for readability.
- Selected input frames are held fixed during a single encoder pass. Per-layer D/S masks can vary. Packing 16 selected observations for the backbone is not selection of 16 original clips.
- TIA uses the selected `K/2` support; `full KV` does not mean full-video or complete-original-grid attention.
- The main path is the existing Cross decoder with R03 initialization. Official pretrained VideoMAE decoder experiments are separate, and are not depicted as the current model.
- Decoder query seeds contain the interpolated anchor base, Scout context and query metadata. Anchor memory contains physical-time and computation provenance, with gated features from earlier layers.
- The figure depicts implemented mechanisms. It does not claim a measured depth saving, a validated three-axis advantage, or a tested carrier architecture. The experiment thread reported that both epoch-10 Full-V2 evaluations selected D100 on all windows; that operating point cannot establish a benefit from dynamic depth skipping.

## Source mapping

- `h65/paper/model.py:122`: budget, selection/refinement, encoder, anchors/queries and decoder execution order.
- `h65/paper/model.py:47`: original block IDs and trainable light operators.
- `h65/paper/encoder.py:95`: selected-observation packing; `:113`: selected TIA length and feature capture.
- `h65/paper/engine.py:168`: attention and depth-light residual; `:175`: disjoint FFN masks; `:207`: residual and TIA.
- `h65/paper/decoder.py:28`: interpolation, query seeds, multi-depth memory and residual output.
- Scientific structure was reconfirmed by experiment thread `01a09654-63fa-7831-9eef-af21a5a73ac1` on 2026-09-14; its reported running model revision is `db9c749`, unchanged structurally from `da26af3`.

## Deliverables and layout

- `model_full_v2_cvpr.pdf`: vector PDF, 7.1 x 4.7 inches, suitable for two-column figure placement.
- `model_full_v2_cvpr.svg`: editable vector text and objects.
- `model_full_v2_cvpr.png`: 3550 x 2350 pixels.
- `tools/draw_full_v2_cvpr.py`: reproducible source. It checks canvas clipping and emits the three formats.
- The PDF was rendered with Poppler and visually inspected. Inference and control edges use distinct styles; every directed edge has one terminal arrowhead; residuals use local channels; panel references replace crossing zoom leaders.
