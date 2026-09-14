# 全部实现导航

本索引对应`codex/graph-tad-20260914`，运行事实以[STATUS](STATUS.zh.md)为准。下列“已实现”指代码已存在并接入配置/执行路径，不自动表示该配置已通过GPU预检、完成训练或有效。

## 模型与训练

|责任|主要源码|当前设计|
|---|---|---|
|完整模型入口|[model.py](../../h65/paper/model.py)|`PaperModel`集成encoder、scout/预算/选帧、原轴decoder与独立student readout；按配置开启Graph|
|时间几何和原轴回写|[geometry.py](../../h65/paper/geometry.py)、[frame geometry](../../h65/frame/geometry.py)|真实物理时间、有效性、contributor来源与anchor/query接口；避免用打包rank冒充原时间|
|时间/预算路由|[routing.py](../../h65/paper/routing.py)、[interventions.py](../../h65/paper/interventions.py)|H65/BMCR初始化；多预算训练；实际重执行student的选帧/D/S干预收益监督|
|深度、空间和TIA执行|[encoder.py](../../h65/paper/encoder.py)、[engine.py](../../h65/paper/engine.py)|真实selected-Q执行，selected/full/graph KV路径、heavy/light FFN、depth轻残差、global TIA|
|Cross/Interp/TCN/MAE恢复|[decoder.py](../../h65/paper/decoder.py)、[mae_init.py](../../h65/frame/mae_init.py)|完整时间轴latent输出；多层Cross；fresh Cross不加载旧R03；官方MAE或同架构随机及输入统计适配|
|Point/TadTR readout|[readout.py](../../h65/paper/readout.py)、[runtime.py](../../h65/paper/runtime.py)|student TAD head可训练；不同head接相同原轴接口。query head不等于已完成新的公开压缩基线|
|同支持状态参考|[support_targets.py](../../h65/paper/support_targets.py)|同一选帧的full-D/S冻结初始参考；attention/FFN/TIA状态监督；分块统计/梯度重算|
|任务、外部/自蒸馏|[objectives.py](../../h65/paper/objectives.py)、[training.py](../../h65/paper/training.py)|GT、原轴feature、full GT、shared-full self feature、实际干预；记录teacher与student额外前向|
|PBD/静态深度|[compression.py](../../h65/paper/compression.py)、[model.py](../../h65/paper/model.py)|训练集候选删层、逐步恢复、保留原层ID对齐；P00静态、P01渐进至9层，空间100%|
|原版AdaTAD下采样|[native_adatad.py](../../h65/paper/native_adatad.py)|官方窗口后[::2]及坐标/mask/stride同步；原encoder/TIA/head，无恢复/路由/KD改造|

Full-V1沿用交替A-MoD与当前Cross框架。Full-V2固定前4层dense，后段原0-based层4/6/8/10路由，末层dense；selected-Q/full-KV，独立depth-attention/FFN轻残差，空间完整grid heavy/light，增加同选帧支持监督。时间初始化、Cross与可训练head继续使用。两者是组合候选比较，不能将差值全部归于单一模块。

这些课程采用混合稀疏预算训练，并有训练期full辅助分支、外部teacher及部分同支持参考；并非只训练dense、到推理才首次变稀疏。encoder并非默认全参数训练：主课程训练Adapter/head/新增模块等；全量微调是单独已注册控制。

## 最新Graph实现

|模块|代码|具体落实|
|---|---|---|
|稀疏边和图KV|[edge_ops.py](../../h65/paper/edge_ops.py)|degree16索引/连续边权、局部/多尺度种子、受限两跳referral、去重聚合、实际gather KV的分块SDPA；不先构造完整N×N注意力矩阵再mask|
|原轴图状态与可靠anchor|[graph_recovery.py](../../h65/paper/graph_recovery.py)|[B,384,128]低维时间状态、真实contributor落点、质量/跨度信息、边消息、受约束anchor修正|
|图辅助选帧|[graph_frames.py](../../h65/paper/graph_frames.py)|重编码前的廉价图frame context/rate残差，32分区覆盖，refinement时保留覆盖；不使用尚未计算的heavy特征决定本次选帧|
|编码器反馈|[engine.py](../../h65/paper/engine.py)、[model.py](../../h65/paper/model.py)|G-Full在L6/9/12将真实heavy状态回写时间图，零输出初始化的残差接口保护初始路径|
|图训练与GPU预检|[paper_train.py](../../tools/paper_train.py)、[graph tests](../../tests/graph/test_graph_contracts.py)|图分支预检、梯度合同、前5轮attention过渡、EMA/optimizer/checkpoint覆盖；技术更新丢弃后正式课程|
|图诊断与计费|[graph_diagnostics.py](../../tools/graph_diagnostics.py)、[profile.py](../../h65/paper/profile.py)|图度/来源/距离/可靠性、固定选择与D/S掩码的full-KV参照、实测算量、参数和路由分布|

G-Repair偏向原轴恢复，保留当前encoder；G-Context改变晚层KV访问、保留Cross；G-Full组合两者、图辅助帧决策与中间反馈。它们从与V2相同的既有H65/BMCR和R03资产开始，不是从V2最佳epoch40再额外训练80轮。Graph并不把全部空间patch改成统一5×5图载体。

晚层图KV在现有打包域内选择访问；全原轴长时间联系由128维时间图承担。普通full plan0保留为无Graph共享full参考，因此plan0干预是整个策略变化，不能当“只改一个轴”的纯消融。Graph的非零固定plan和机制对照已注册。

A-MoD原有额外attention评分QK仍存在并计费；图访问少并不保证K/V投影和FFN也同比减少。当前实现没有把图度16或理论比例直接写成实测节约率。

官方VideoMAE decoder复用的是预训练transformer/projection等可匹配参数，输出端改成latent特征残差并接TAD任务训练；不是直接将原RGB像素重建头当成已适配的TAD decoder。预训练是否优于随机还没有本轮完整结果。

## 实验、部署与分析工具

|责任|入口|
|---|---|
|完整论文原52课程|[paper_plan.py](../../tools/paper_plan.py)、[configs/paper](../../configs/paper/)|
|独立复审＋Full-V2/简单强控制|[paper_review_plan.py](../../tools/paper_review_plan.py)、[configs/paper_review](../../configs/paper_review/)|
|Native原版控制|[native_adatad_plan.py](../../tools/native_adatad_plan.py)、[native_adatad_run.py](../../tools/native_adatad_run.py)、[native_adatad_deploy.py](../../tools/native_adatad_deploy.py)|
|Graph完整课程与持久登记|[graph_plan.py](../../tools/graph_plan.py)、[graph_deploy.py](../../tools/graph_deploy.py)、[configs/graph](../../configs/graph/)|
|唯一dispatcher、课程/断点/内联评测|[paper_dispatch.py](../../tools/paper_dispatch.py)、[paper_course.py](../../tools/paper_course.py)、[paper_train.py](../../tools/paper_train.py)、[paper_eval.py](../../tools/paper_eval.py)|
|全部结果去重和归档|[paper_review_snapshot.py](../../tools/paper_review_snapshot.py)、[paper_review_analyze.py](../../tools/paper_review_analyze.py)|
|官方dense复测、配对统计|[paper_public_dense.py](../../tools/paper_public_dense.py)、[frame_errors.py](../../tools/frame_errors.py)|
|论文图/训练曲线/成本|[paper_review_plot.py](../../tools/paper_review_plot.py)、[paper_training_trajectory.py](../../tools/paper_training_trajectory.py)、[paper_epoch_comparison.py](../../tools/paper_epoch_comparison.py)、[paper_routing_cost.py](../../tools/paper_routing_cost.py)|
|本次全项目索引|[paper_project_status.py](../../tools/paper_project_status.py)|

## 仍未闭合的实现与证据

注册配置已经能够形成候选、控制和数据生成任务，但不等于最终论文已完全覆盖。最新Graph只登记THUMOS S/B与S机制，ActivityNet/InternVideo/TadTR现有配置仍是较早PaperModel；不能把它们将来的结果自动署名为Graph或V2。

七个独立axes配置之外以主full表示TDS，但动态预算设置不同；同checkpoint干预、独立训练比较和严格匹配八格因子表需分开。现有多套代码已具备大部分组件，最终候选的实验版本与图表口径仍需冻结。PBD-style未实现/验证为官方LoRA配方的忠实复现。完整训练总计算目前仍是分项前向代理＋查询次数＋GPU时间，不能伪称精确反向总FLOPs。

本次没有为“报告”重新训练或改变科学参数，也没有把上述待证部分隐去。详见[PAPER](PAPER.zh.md)。
