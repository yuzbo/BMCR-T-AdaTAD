# H65 的稠密训练与时空、深度稀疏推理研究方案

## 1. 研究结论与范围

可以把 H65 改造成“训练时充分利用全部时空观测和完整深度，部署时按任务需要选择昂贵计算”的 TAD 系统。最稳妥的主线不是一次性将 patch embedding、TIA、注意力和检测头全部改成非规则结构，而是：**保持原始完整 clip 的局部计算语义，蒸馏统一的最终层特征接口，推理时选择 cheap / shallow / full 三类计算，并始终向检测器提供原来的稠密时间网格。**

本文将新分支暂称为 **H65-DS3**，表示 dense-supervised、temporal/spatial/depth selective execution。这是研究方案名称，不是当前仓库已有模型。核心问题是“昂贵算子在哪里、执行到哪一层”，而不只是“输出 token 数变少”。

本次源码依据为 `yuzbo/OpenTAD_C3_CoarseClean_20260702` 的固定提交 `04c35a3b76897e6c1569eeede41ed3aecaf7f854`，提交时间为 2026-08-23，提交说明为 `test: require nonzero-lr H65 stage2 smoke`。文献检索截至 2026-09-11。这个提交是本次讨论的可复核锚点，**不代表已核实的 9 月 11 日默认分支 HEAD，也不代表服务器上的最新实验状态**。关键源码经过读取；服务器 checkpoint、训练日志、GPU profiler 与完整评估结果未读取或执行。文献结论依据原论文与官方页面；FastBERT、ToMe、LOUPE另查阅了PDF相关方法页，其余条目不声称完成逐公式或复现实验审计。

建议实施次序为：**代码与 dense teacher 审计 → 零训练完整 clip 抽取基线 → 全 dense 前向的廉价特征/浅层出口蒸馏 → 真正的分层推理 → 空间 MLP 稀疏更新 → 三维组合 → 可选稀疏校准。** 不承诺任何 mAP 或加速数值。文中门槛和预算是待预注册的研究设计。

## 2. 必须先固定“dense 训练”的定义

拥有完整原视频、不等于训练时没有稀疏计算；冻结 backbone、不等于没有新增训练；恢复成长度 768 的张量、不等于恢复了全部信息。这三个区别决定论文主张是否成立。

| 代号 | 训练期操作 | 推理期操作 | 可以使用的准确表述 |
|---|---|---|---|
| Z0 | 原 dense TAD checkpoint；没有新参数拟合或阈值监督拟合 | 直接抽取完整 clips、确定性插值、训练自由 token 压缩 | dense-trained / zero-shot sparse deployment |
| D1 | 每个训练样本的所有 clips、空间 patches 和主干层完整前向；在完整特征上训练 cheap predictor、出口、轻量算子替代器 | 条件执行 clip、深度和空间更新 | dense-forward auxiliary training / sparse execution at deployment |
| C2 | B、D 冻结，但补全器或策略训练时看过 mask、缺失特征或实际稀疏路径 | 使用已校准的稀疏策略 | dense backbone + sparsity-aware auxiliary calibration |
| J3 | backbone/TIA/detector 在稀疏路线下联合微调 | 相应稀疏路线 | sparse-aware joint training |

**最贴合目标的主研究对象是 D1，必须同时报告 Z0 和 C2。** D1 允许训练时增加辅助模块，但禁止将 masked-feature reconstruction 混入而仍宣称从未见过稀疏输入。低分辨率 preview 在自己的网格上可以是密集的；“dense-forward”指部署选择发生前，训练主干没有跳过任何候选 clip、patch 或层。

D1 也不是“完全不用为部署增加训练”。完全禁止任何新增拟合时，只剩 Z0，性能上限和稳定性都应作为实验问题，而不是先验保证。LayerDrop、DynamicViT、A-ViT、LOUPE 等可以借鉴，但它们各自的稀疏训练或采样训练不能被重新命名为 Z0/D1。[R05,R08,R11,R12]

## 3. 当前 H65 的真实实现与缺口

### 3.1 已经存在的结构

固定提交的 dense base config 使用 768 个候选帧、16 帧一组的 VideoMAE、小模型 C=384、12 个 block，160×160 输入对应 10×10 空间 patch。`VisionTransformerAdapter` 中 TIA 的 `temporal_size` 明确设置为 `num_frames // tubelet_size`；在该配置下是 8，而不是全窗口 384。标准 patch embedding 是时间 kernel/stride=2 的 Conv3d。主干处理完整 clip，随后空间池化、串联各 clip 的 tubelet 特征，再插值给检测器。[C04,C05,C08]

由这些代码得到的 shape 推导为：

```text
输入候选窗口                 [B, 1, 3, 768, 160, 160]
原始 clip 分组                [B*48, 3, 16, 160, 160]
单 clip patch tokens          [B*48, 8*10*10, 384]
单 clip 最终特征图             [B*48, 384, 8, 10, 10]
空间池化、保留原 clip 顺序      [B, 48, 8, 384]
串联 native 时间轴            [B, 384, 384]  # channel=384, time=384
原 post-processing 插值       [B, 384, 768]
ActionFormer                  原来的 dense 坐标、标注、先验与后处理
```

这里的 768 是数据管线产生的候选观察网格，不应未经检查就解释为视频所有原始解码帧。真实秒数、视频 stride、窗口起点仍应由 loader metadata 得到。[C05,C08,C09]

### 3.2 当前训练进度的可验证范围

cycle4 的 stage1 是 **uniform K=384、30 epoch / 3000 successful updates、epoch29 EMA**；它不是 dense768 teacher。stage2 的 `duca_h65_first_singleclock_cycle4.py` 继承 joint384 配置，明确关闭 packed route，设置 `paper_claim_allowed=False`。其中 `total_frames=768` 不是“backbone 实际吃到 768 帧”的充分证据，必须沿 selector 和 clip grouping 看真实输入。[C01,C02,C03]

当前提交主要加强了 smoke：至少两次迭代，而不是第一步可能仍处于零学习率的单步检查。两次迭代本身仍不能替代“至少一次非零 LR、成功 optimizer update、参数实际变化”的日志证据。历史 20+40 配方的 epoch19 与 cycle4 的 epoch29 不应混用。[C01,C03]

### 3.3 packed 代码不等于已经具备三维稀疏

已有 packed block 实现会 gather 选中 token，运行 attention/MLP，再 scatter 回 dense 张量后执行 Adapter。它提供了实现参照，但与新方法不是同一功能：原 route 是 temporal tubelet group，不是任意空间 patch 路线。packed 与非零 physical-time 残差同时出现会 fail closed。现有功能不能直接解释为空间剪枝、任意早退或已验证加速。[C04]

`ActionFormer` 构造器当前没有本报告的新 `ds3` 参数；现有 `token_compressor.target_len` 必须与 `projection.max_seq_len` 一致。新增路线要通过明确的新 wrapper / 子类 / opt-in 参数接入，不能只改一个 `max_seq_len` 后绕过旧约束。[C07]

### 3.4 需要单独测试的坐标残差问题

当前 physical-time mask 使用 `delta=actual-canonical`，构造 `delta_i-delta_j`，第一 block 将其乘标量后加到 attention logits。[C04,C06]

对固定 query i，数学上：

\[
\operatorname{softmax}_j(A_{ij}+a(\delta_i-\delta_j))
=\operatorname{softmax}_j(A_{ij}-a\delta_j).
\]

因此它相当于一种 key 侧偏置，不能直接解释为一般的成对时间距离建模。这个结论是对公式的代数分析，不是声称已观察到性能错误。若所有训练样本都满足 actual=canonical，这一项为零，无法提供“非规则偏移下应该如何处理”的训练信号。新完整原 clip 路线应保持 dense 模型的局部位置编码，不要无意复用另一条 global-rank 稀疏重排语义。

### 3.5 新增模块最容易出现的训练故障

`_freeze_layers()` 在 forward 内执行，旧 optimizer 对 backbone 也有特殊零 LR / Adapter 白名单。新出口、替代器或 selector 若放进错误的子树，会被静默冻结或落入零 LR 参数组。必须检查 `requires_grad`、optimizer membership、实际 LR、梯度范数、参数哈希变化，不能只看 loss 下降。[C04,C08,C10]

## 4. 跨领域证据：哪些真正值得借，哪些只是名称相似

### 4.1 视频识别与视频密集预测

**Deep Feature Flow** 是 sparse heavy computation + dense feature propagation 的直接先例：稀疏关键帧做重网络，其余帧通过运动传播获得表示。但原论文包含端到端训练；它支持架构合理性，不证明任意 dense checkpoint 经删帧即可保持精度。[R01]

**Listen to Look** 对本方案的启发更具体：便宜的图像/音频信息可以蒸馏 expensive clip-level 表示，随后选择昂贵处理的时间位置。H65 首版只用已有低分辨率 RGB preview，不必新引入音频。核心借鉴是“每个时间位置都有廉价表示，昂贵计算用于纠正”，而不是在大片空白上盲目插值。[R02]

**Clockwork ConvNets** 提示不同语义层的更新频率不同。DToP 则在语义分割的密集预测中让容易 token 早退，同时保留必要上下文。这两者说明密集输出和非均匀深度可以共存，但需要处理特征语义与上下文依赖。[R07,R15]

2026 年的 **Temporal Cluster Assignment** 与 **Video Patch Pruning** 为视频分割中的时间一致性、早期剪枝提供较新参照。它们是 CVPR workshop 工作，应与主会议和成熟部署证据区分；它们也不能替代 H65 的 clip、TIA 和 TAD 边界实验。[R22,R23]

### 4.2 NLP 与自蒸馏：最重要的深度借鉴

**FastBERT** 先训练 backbone，再利用深层输出自蒸馏中间分类器，推理依据不确定性决定是否继续。这与“训练完整计算，推理条件早退”非常接近。对 H65 的迁移不是直接把中间分类 logits 当作 TAD 输出，而是让每个出口预测检测器熟悉的最终 native feature。[R09]

**Be Your Own Teacher** 提供深层向浅层蒸馏的另一条路线。它支持出口对齐设计；不意味着不同层未经映射就可拼接。**LayerDrop** 是有价值的对照，但训练时已经随机删除层，应归入稀疏鲁棒训练一类，不可作为 D1 的证据。[R10,R11]

视频分类中的 entropy early exit 对 TAD 尤其需要警惕：背景类别很确定，不代表附近动作起止点可以安全省略。出口决策应由特征误差、任务敏感性、边界证据和覆盖约束共同决定。

### 4.3 ViT token 压缩

**ToMe** 与 **ATS** 支持对已有 ViT 进行无需重新训练的 token reduction；这是 Z0 空间路线的现实依据。它们改变的是内部 token 计算，不能自动节省已发生的解码、patch embedding 与早期层。ToMe 的合并质量、token mass 和解除合并映射需针对 VideoMAE/TIA 检查；ATS 的评分机制也不能假设 H65 有可直接使用的 CLS token。[R03,R04]

**DynamicViT、A-ViT、DiffRate** 分别提供动态剪枝、token halting 和压缩率优化的参考。它们涉及额外学习或校准，不宜混称为“原 dense 模型零训练推理”。D1 应优先利用不改变 selected-token 上游输入的操作，例如条件执行逐 token MLP；更复杂的 attention 稀疏放后面。[R05,R06,R08]

### 4.4 MRI、深度补全和非规则时间序列

**LOUPE** 的真正价值是把采样与恢复作为联合设计，而不是固定采用动作概率 Top-K。但其原始方法是概率采样、回顾性欠采样与重建训练；不能照搬成“逐样本内容自适应且硬 exact-K”的结论。严格预算需要额外的排序、离散约束或可行集合设计。[R12]

**Sparse-to-Dense** 深度补全提示必须显式传递 observation mask，并尽量保留真实测量而非让网络覆盖它们。对 H65，可学习补全属于 C2；D1 首版用廉价全时间表示做基底，使未重算位置不再是零值。RGB 引导深度与 RGB 引导时间特征不是同一物理逆问题，因此不应将可恢复性直接类比。[R13]

**mTAN** 提供连续时间 embedding 与参考网格聚合。H65 无需第一版就引入 ODE：显式真实时间、左右间隔、来源状态，加局部插值或注意力已足够形成可检验基线。[R14]

### 4.5 扩散模型、多模态大模型与 TAD 自身

**DeepCache** 的训练自由复用发生在扩散去噪步骤之间，而不是视频相邻帧之间。它启发缓存与误差监控，但动作边界没有扩散轨迹的相同结构。**LLaVA-PruMerge** 和近期 **Spatio-Temporal Token Scoring** 说明视觉 token 数和模型不同阶段的开销必须共同核算；节省 LLM token 不等价于跳过视觉 backbone。[R16,R17,R24]

**TALLFormer**、**TURN TAP**、**SP-TAD** 需要放在相关工作中，但分别涉及训练长期特征记忆、temporal unit 复用、稀疏 proposals，与部署时少跑 backbone 的含义不同。**Finding Action Tubes with a Sparse-to-Dense Framework** 处理时空动作管，不是 H65 的 temporal-only TAD；它足以说明不能把“sparse-to-dense 用于动作检测”本身当作新颖性。[R18–R21]

AdaTAD 的 TIA 主要缓解训练适配与显存问题，冻结大 backbone 仍需要前向计算。VideoMAE 的高比例 masked pretraining、MVD 的 masked feature distillation 都支持研究表示冗余，却不证明 TAD 可以在推理丢掉同样比例的观察而保持边界。[R25–R27]

**文献归纳：最匹配 D1 的组合是 Listen to Look + FastBERT/BYOT + dense prediction 接口；最匹配 C2 的组合是 DFF + LOUPE + sparse-to-dense completion；ToMe/ATS 是独立的 Z0 空间对照。**

## 5. 推荐总体结构：稠密廉价基底与稀疏昂贵修正

### 5.1 统一最终特征接口

令原始第 j 个 16-frame clip 为 C_j。完整 dense 模型产生：

\[
F^D_j=B_{1:12}(C_j)\in\mathbb R^{8\times384}.
\]

低成本 preview 提供：

\[
F^0_j=H_0(P(C_j^{low})),
\]

其中 H0 必须预测 **8 个有区别的 tubelet 特征**，不是预测一个 clip 向量后复制八次。第 l 层出口为：

\[
F^l_j=H_l(\operatorname{pool}_{xy}(X^l_j)),\quad l\in\{4,8\}.
\]

在推理时决定深度 d_j，输出：

\[
\widehat F_j=\begin{cases}
F^0_j,&d_j=0,\\
F^4_j,&d_j=4,\\
F^8_j,&d_j=8,\\
F^D_j,&d_j=12.
\end{cases}
\]

串联所有 clip 的八个位置得到 native384，再使用原 post-processing 到 detector768：

\[
\widehat Y=D\big(U_{\rm original}(\operatorname{concat}_j\widehat F_j)\big).
\]

这里 `D` 包括原 projection / neck / head，保持原坐标、GT 表达和 NMS；`U_original` 不是随便增加的插值层，而是经过 full-grid identity 测试的原转换。**原检测器结构不变是工程策略，不是精度不变的定理。**

### 5.2 训练和推理的图不同，但辅助预测器的输入可保持一致

D1 训练时所有 clip 经过完整 backbone；从同一 dense 前向取 X4、X8、F12，同时训练所有 H0/H4/H8。没有 clip mask、稀疏 attention 或动态跳层。低成本 MLP surrogate 也在完整 dense token 输入上学习对应 block 的完整 MLP 输出。

部署时，完整原 clip 的 dense prefix 在相同 eval 模式和变换下，与训练用于该出口的 prefix 接口相同；只是该 clip 不再执行后续层。这比“训练从没见过非规则 tubelet，推理突然任意丢帧”更可控。不同出口的 feature 混合后经过跨时间 detector，仍可能产生分布偏移，应以 D1 对 C2 的实验量化，而不是隐瞒。

### 5.3 训练阶段

阶段 A：取得真正 dense768 checkpoint，或按预注册预算训练 dense teacher。无此 checkpoint 时，不得把 H65 uniform384 warmup 直接换名使用。

阶段 B：冻结 B 与 D，所有训练 clip 完整前向获得 teacher native features 和出口输入；训练 preview feature head、H8、必要时 H4，以及空间 MLP surrogate。训练完整 teacher 的额外成本需要统计。缓存只允许训练集，记录窗口、空间 crop、增强、teacher checkpoint 哈希；不能使用未对齐 crop 的 teacher 特征。

阶段 C：部署 D1 推理策略，在训练内部开发集上固定预算与规则。阈值拟合、性能选型属于模型选择，需披露；不在官方 test 上调参。D1 不使用实际混合缺失特征训练其参数。

阶段 D（可选 C2）：允许冻结 B/D 下的 route-mix calibration、masked reconstruction 或少量实际稀疏前向校准。它是单独实验臂，有独立训练成本和命名。

### 5.4 冻结但允许梯度传播

Teacher full forward 可以在 `no_grad()` 下产生目标。学生 `D(F_hat)` 的损失需要穿过冻结 detector 传回 H0/H8：应冻结 D 的参数，而不是把学生 D forward 放进 `no_grad()`。同时固定 BN/Dropout 等 eval 行为，确认 B、D 哈希不变，新模块哈希确实变化。调用 `model.train()` 后还需要验证内部 `_freeze_layers()` 不会覆盖新参数策略。[C04,C10]

## 6. 时序稀疏：先保完整 clip，再讨论非规则帧

### 6.1 首版采样单元

优先选择原窗口分组的完整 16-frame clip。dense 48 clips 到推理 24 clips，不重新把跨时段任意帧凑成新的 clip。不改变 tubelet 内邻接关系、局部位置编码和 TIA 栅格。这是选中 clip 特征等价命题成立的必要条件之一。

在 eval、同一输入变换、相同局部坐标、无跨 clip 操作下：

\[
\operatorname{Gather}_{J}B_{clip}(C_{1:48})
\simeq B_{clip}(\operatorname{Gather}_{J}C_{1:48}).
\]

容许 GPU 浮点次序导致的小量差异。**这个关系不能延伸为空间剪枝后的 attention 或任意跳层后的输出等价。**

### 6.2 零训练 Z0 与 D1 的区别

Z0 只对选中的完整 clips 做重 backbone，在 native 时间位置插值缺失 tubelet 特征。必须保留每个 selected clip 的全部 8 个 tubelet，不应先池化成一个时间点。短窗 padding、端点外推、重叠滑窗、局部 clip 边界与掩码要有测试。

D1 将所有 clip 的 F0 作为 dense cheap base，再替换 selected clip 的真实高质量特征。比较“插值补全”和“cheap base + heavy correction”能判断性能提升来自邻域可预测性，还是廉价视觉输入仍提供了关键信息。

### 6.3 路由器的目标

单纯 actionness Top-K 会偏向动作主体，忽略边界和短动作。建议 score 由预测的表示误差、任务敏感性、边界概率、时间覆盖共同组成。训练中可计算：

\[
u_j=\sum_{u,c}|g^D_{juc}(F^D_{juc}-F^0_{juc})|,
\quad g^D=\partial L_{TAD}(D(F^D),Y)/\partial F^D.
\]

只让低成本 preview 去预测 u_j。它是 **gradient-weighted residual proxy**，不是真实“添加该 clip 后的边际 mAP 增益”。真正 counterfactual value 需要额外比较路径，放 C2 或离线诊断，并计入训练成本。

先用固定数量的覆盖 anchors，再将剩余 budget 分配给高 score clips。必须显式检查最大未观察间隔；正的 density floor 并不自动给出 deterministic max-gap 保证。首版采用定额分桶预算更利于 GPU 压紧执行，动态 K 留作后续。

### 6.4 采集成本的边界

若 preview 已经解码全窗口，方法的准确命名是“稀疏昂贵 backbone 计算”，不是“只采集 24/48 的全部视频数据”。真正减少解码需要另做关键帧、低码率代理流、分辨率分层或解码器访问方案，并测量 GOP 随机访问成本。首版不把这类尚未实现收益写入结论。

## 7. 深度稀疏：可控的前缀出口，而不是任意删层

首版只用 0/8/12，验证后再加 4。出口接收空间池化后的 8 个 tubelet 特征，用低秩线性或两层小 MLP 预测最终层表示，保留逐 tubelet 时间结构。训练完整 dense branch 的 H8 和推理只跑到8层的 H8 输入最容易保持一致。

推理使用嵌套 active clip 集合：

\[
I_{12}\subseteq I_8\subseteq I_4.
\]

先对 I4 执行1–4层，再把需要继续的 clips 压紧运行5–8层，再执行9–12层。退出 clip 保存经桥接的最终接口。不能先算完所有12层，再选择某个中间输出，这没有深度节省。

示例只是计数说明：若 |I4|=24、|I8|=18、|I12|=12，则重 backbone 的 clip-block 数为 4×(24+18+12)=216，对比 dense 的48×12=576，比例为0.375。它不是端到端时间比例，也没有包含 preview、出口、调度和 detector。

简单地用 identity 跳过任意 block 会改变后续层输入分布，也可能跳过 TIA。应先采用 prefix exit，不在第一轮同时引入非连续 block skipping。后者更接近 LayerDrop/A-ViT 的训练问题，需要独立实验臂。[R08,R11]

## 8. 空间稀疏：先稀疏逐 token MLP，后稀疏注意力

### 8.1 低侵入版本

Transformer MLP 的作用是逐 token 的。在同一个 dense attention 输出 x 上，选中 token 的重 MLP 可 gather 后独立计算；未选中 token 用密集训练出的便宜 surrogate 预测残差。随后 scatter 回完整 `[tubelet,h,w]`，再让 TIA 按原栅格运行：

\[
r_i=\begin{cases}FFN(x_i),&m_i=1\\h_\phi(x_i),&m_i=0.\end{cases}
\quad x'_i=x_i+r_i.
\]

D1 训练时 FFN 与 h_phi 都在全部 dense token 上执行并蒸馏，不执行采样。推理时重 FFN 只处理 m=1 的压紧 token。未选中 token 是否仍算轻量 h、gate、scatter 都计入成本。

这个版本可以准确称为 **spatially sparse heavy-MLP updates**；attention、QKV 和 TIA 仍 dense。不可因为 token score/mask 存在，就宣称整层 attention 也节省。

### 8.2 更激进版本

局部 ToMe merge/unmerge 或 packed MHSA+MLP 作为第二步。在每个 block 内保存原 raster x，只对合并/选中 token 计算重残差，再将残差映射回原 token 位置供 TIA 使用。跨 tubelet 合并、空间碎片剪裁和 arbitrary token 顺序首版均禁用。

必须保存 merge membership、token mass、位置映射和未更新来源。ToMe 的 size-aware aggregation / attention 校正在本模型上做消融，不把其它模型的开关直接当作真理。由于 key/value 集合改变，选中 token 的 attention 输出也会改变；这时从 dense teacher feature 中直接 gather 不能精确模拟 sparse forward。[R03]

### 8.3 误差累积

逐层替代误差会向后传播。形式上，当各层存在可用 Lipschitz 上界时，最终误差受各层局部误差乘后续层增益的和控制；但 dense 数据上的平均蒸馏误差并不能给出部署稀疏轨迹上的统一上界。因此只在后半层、温和比例开始，并与“仅末层 MLP 替代”做对照。C2 的收益可能主要来自修正这种 rollout distribution shift，而不是模型容量增加。

## 9. Deformable embedding / TIA 应放在哪里

可变形采样的 offsets 改变读取位置，不自动改变输出 token 数，也不自动省掉 backbone 的密集乘加。要减少计算必须让昂贵算子的输入尺寸或执行次数真正减少。

对本路线，优先级应是：先完成完整 clip 抽取和统一最终特征接口；再把 deformable temporal aggregation 作为 native-grid reconstructor 的可选模块；最后才考虑替换 TIA 或 raw token embedding。若在非规则坐标上操作，offset 应以真实时间单位解释，并显式处理邻近点插值、搜索范围和 padding。

D1 中若只见过规则时间，不要期待 zero-init offsets 或 physical residual 自动学会不规则时间的行为。C2 可通过实际观察稀疏间隔训练这种模块，并准确标注其训练设置。不要同一轮同时改采样器、embedding、TIA、位置编码、detector 和蒸馏目标，失效后几乎无法归因。

## 10. 损失设计与信息边界

### 10.1 完整网格的辅助训练

对出口 e，使用：

\[
L^{e}_{feat}=\sum_t w_t\{\operatorname{SmoothL1}(F^e_t,\operatorname{sg}F^D_t)
+\lambda_c[1-\cos(F^e_t,F^D_t)]\}.
\]

再加入沿真实 native 时间轴的一阶差分一致性，权重来自训练集的合法动作边界；截断窗口形成的假边界要由 `boundary_validity` 排除。权重和损失要按有效位置归一化，不能让大背景数量淹没短动作。

D1 可对整个 F0 序列、整个 F8 序列分别运行冻结 D 并施加任务监督/蒸馏；这不是在训练异质稀疏混合图。先用 feature+temporal derivative；task loss 是后续明确的消融，而不是一开始堆多个固定系数。

分类 KD 应匹配实际 head：若是独立 sigmoid 多类分数，就使用逐类 Bernoulli KL / soft-label BCE，而不是未经核对使用 categorical softmax KL。回归 KD 必须在同一 dense 坐标、同一尺度和对齐的正样本/高置信位置比较；不要对所有背景回归输出做无意义的 L1，也不要直接对数量不同的 NMS 后 proposals 做逐项 KL。[C07,C09]

### 10.2 C2 的补全器

C2 可以采用 physical-time interpolation + 小残差 TCN，输入包含 cheap dense base、真实 observation mask、左右距离和出口来源。保持真实重观测位置不被随意覆盖；是否允许小的 task-aware correction 单列实验。训练 mask 分布必须与真实 clip/block policy 一致，而不是只用独立随机 frame dropout。

如果训练时使用 teacher native 特征模拟完整 clip 缺失，应先通过选中 clip 等价测试。空间剪枝和深度替代引入的 feature distribution 不能用同一 gather trick 假装精确；至少做 matched actual-forward 校准对照。

### 10.3 不可恢复的信息

存在两个视频在所有实际被使用的观测上相同，而在未被识别的短片段中动作不同，则任何后端都不能从这些相同观测必然恢复正确事件。cheap preview 缓解这个问题，但低分辨率、极短事件与遮挡仍可能使它不可辨认。

因此目标是经过实证约束的 rate–distortion / risk–cost 优化，而不是“保持 dense 信息但零成本”。覆盖、高 tIoU、短动作召回与背景误报必须是主指标，不能只看平均 feature MSE 或视频分类精度。

## 11. 成本模型、可实现加速与 profiler

对单 clip，token 数 N=8×10×10=800，通道 C=384，MLP expansion r=4。忽略 bias、norm、激活等项的 MAC 近似为：

\[
C_{block}\approx(4+2r)NC^2+2N^2C=12NC^2+2N^2C.
\]

其中 FFN 占 8NC²，QKV/output projections 占4NC²，attention matmul 占2N²C。空间只稀疏 FFN 时，不能把整个公式乘空间保留率。时间上完整 clip 数减半主要减少独立 clip 调用数，而不是把每个 clip 内 attention 的 N 减半。

全链路必须记录：

\[
C_{all}=C_{decode}+C_{preview}+C_{router}+C_{patch}
+\sum_{j,l}C_{jl}+C_{pack/scatter}+C_{exit/recon}+C_D+C_{NMS}.
\]

不能将时序、空间、深度三个保留率简单相乘当作实际加速。至少分别测 CPU/IO 完整链路、GPU model-only、backbone-only 三种口径；同步 CUDA、warmup、同 batch/精度/软件环境，报告 p50/p95、吞吐、峰值显存以及 clip-block/token 的实际执行计数。

若开销大头在 dense detector、preview 或解码，即使 backbone 节省明显，端到端仍可能不达标。动态 GPU kernel 还可能因为 gather/scatter、bucket 小 batch、SDPA 后端变化和内存搬运变慢。应从固定预算、少量深度档、整组 clips 开始，先得到可靠的 wall-clock 收益，再增加策略复杂度。

训练成本单列 teacher 完整前向、出口蒸馏、route 校准、counterfactual、缓存构建和搜索成本。可比较“相同训练开销”与“相同部署开销”，不能只比较一个更长训练的新方法与历史短训练基线。

## 12. 实验矩阵与阶段准入

`experiments.json` 给出 24 个具名实验计划；所有新配置路径均标记为待 agents 创建，不是当前提交已经存在的文件。

| 阶段 | 主实验 | 因果问题 | 继续条件 |
|---|---|---|---|
| P0 | D768 + 历史 U384/cycle4 对照 | 真正 dense teacher 是否存在、dense 精度能否复现 | G0 checkpoint/配置/数据来源完整 |
| P1 | Z36、Z24、Z16、ZR24 | 仅完整 clip 冗余能省多少；均匀和随机差多少 | G1 native identity、等价与坐标测试通过 |
| P2 | AUX、PONLY、T24U、T24A | cheap base 是否有价值；学习路由是否胜过均匀 | G2 梯度/冻结/无稀疏训练审计，G3 时序运行通过 |
| P3 | D8、DAD | 固定浅层与条件深度分别贡献什么 | 实际后层 clip 数减少，边界误差可接受 |
| P4 | S75、S50、STM | MLP 稀疏与 attention 压缩的风险/收益 | G4 full raster/TIA 合同和 actual execution 通过 |
| P5 | T/S/D 的 2³ 因子矩阵 | 三维收益是否可叠加、是否相互抵消 | G5 同预算同 checkpoint 的完整质量/速度评估 |
| P6 | CAL | C2 稀疏校准相对 D1 改善多少 | 单列训练成本，不重命名成 D1 |
| P7 | JOINT | sparse-aware joint 上限 | 仅在前面有效时投入 |

P1 失败不代表方向失败。如果 uniform clip interpolation 降很多但 cheap base 方法恢复，结论是“仅平滑不足”；若 cheap-only 几乎相同，则需要证明昂贵修正的必要性。若学得 selector 不胜均匀，则保留简洁均匀策略，不用额外模块掩饰。

首版 pilot 用 seed3407；最终 Pareto 前沿模型用3407/3408/3409。新增 dense teacher 可预注册6000成功更新作为起点，须核对实际 batch 数和学习率曲线；不要直接继承 base 中 max_epoch100 / end_epoch60 的组合并声称是60epoch完整 cosine。辅助训练先100成功更新验证，再对所有候选统一3000更新；C2 与对应 D1 额外训练匹配对照。数字是启动方案，不是已证实最优超参数。

THUMOS14 为首个目标；按现有官方 evaluator 记录完整 tIoU 列与均值，重点高 tIoU 和短动作。训练/开发/test 的视频 ID 集合必须显式保存，不能仅按目录名“validation”猜用途。路由、早退阈值和插值选择在训练内部开发集固定，最后才报告 official evaluation。其他数据集如 ActivityNet/FineAction 作为最终泛化验证，需要先核实该数据集的完整原生配置和数据协议。

建议开发集预注册门槛：平均 mAP 下降不超过1.0个百分点、高 tIoU 不超过1.5个百分点、短动作召回不超过2.0个百分点，并要求端到端实测加速至少1.3倍。可根据任务需求收紧，但必须在正式评估前固定。不要为满足门槛挑选特定 checkpoint 或视频子集。

## 13. 必须实现的测试

**身份路径**：all clips / full spatial / depth12 下，native features、detector input、raw head outputs 与 dense reference 一致；FP32 容差先固定，再对 AMP 单独建立容差。容差不能放宽到掩盖坐标错位。

**执行证明**：用 hooks / profiler 验证跳过 clip 从未进入重 backbone、退出 clip 从未进入后层、未选中的空间 token 从未进入重 FFN。禁止“dense算完再乘mask”的伪加速。

**泄漏测试**：推理 API 不接收 labels/teacher features。将未选中高分辨率数据替换为毒值，保持合法 preview 不变，结果应不变；若改变 preview 则允许路由变化。推理禁止读取训练特征 cache 与旧 raw predictions。别把端到端需要使用的 preview 也毒掉后错误判失败。

**形状与来源**：短窗、padding、1个clip、batch异长、端点、全选、最小预算、最大间隔、不连续原始时间戳、重叠滑窗均测试。valid_mask 表示有效时间，observed_mask 表示真正重观测，两者不得复用。出口输出与补全特征要标注来源，不能都标成 fresh dense feature。

**训练证明**：至少2次成功更新且非零LR；AMP有限；新参数变化；teacher B/D 参数与buffer未意外更新；loss对新模块有梯度；D1记录所有训练clips/tokens/layers计数恒为dense；增加学生task loss时仍能传梯度穿过冻结D。

**兼容性**：旧 cycle4 配置、原 dense base、旧 focused tests 不受默认行为改变；packed + relative-time 的原 fail-closed 约束不被删除。状态字典加载仅允许预期新增模块缺失，旧核心权重不允许静默漏载。

**数学参考测试**：执行包附带轻量参考实现，测试 irregular interpolation、complete-clip gather/forward 交换、softmax residual cancellation 和预算计数。它们是设计单元测试，不是 H65 GPU 集成测试。

## 14. agents 的分工与合并顺序

完整任务书在 `agents/`，可逐个交给 Codex/其他 coding agent；`COMMANDS_zh.md` 给出真实入口和受控的新路线命令。分工如下：

| Agent | 所有权 | 交付 | 禁止事项 |
|---|---|---|---|
| 00 coordinator | 协议、接口、阶段依赖 | 决策记录、合并顺序、状态矩阵 | 不直接宣布训练成功 |
| 01 audit | 只读代码、配置/ckpt来源 | source/provenance audit、G0状态 | 不修改历史代码或下载/启动训练 |
| 02 time | 新clip前端、native网格 | Z0、dense identity、真实compact | 不改GT到selected axis |
| 03 dense aux/depth | preview、出口、depth runtime | D1全dense训练和真实早退 | 不偷加mask训练而仍标D1 |
| 04 spatial | MLP替代、可选merge | full raster合同与真实token计数 | 不直接复用旧temporal route为spatial |
| 05 integration | 注册、训练入口、配置、manifest | 新研究配置、冻结/优化器审计 | 不放松旧训练guard |
| 06 eval/profile | 指标、计时、预算 | 可追溯质量/速度/训练成本报告 | 不把MAC或模型内耗时当端到端 |
| 07 independent review | 独立否证测试 | PASS/HOLD/BLOCK逐项结论 | 不在无证据时生成PASS |

共享文件（backbone wrapper、detector 注册、训练入口）只由 integration agent 写；其他 agents 先提交独立新模块及明确 patch 建议。默认命令按顺序运行，避免多 agent 在同一 worktree 同时写。需要并行时用独立 worktree，人工审核后 cherry-pick；不自动处理冲突，不 reset 用户工作树。

## 15. 工程接口建议

新文件名为提案，尚不存在于 anchor commit：

```text
opentad/models/ds3/contracts.py
opentad/models/ds3/clip_frontend.py
opentad/models/ds3/native_grid.py
opentad/models/ds3/preview.py
opentad/models/ds3/exit_heads.py
opentad/models/ds3/depth_runtime.py
opentad/models/ds3/spatial_runtime.py
opentad/models/ds3/losses.py
opentad/models/ds3/detector.py
configs/adatad/thumos/ds3/*.py
tools/bata/ds3_validate.py
tools/bata/ds3_profile.py
tests/test_ds3_*.py
```

`DenseNativeBatch` 至少包括 features、native centers、原始 frame indices、valid_mask、full_heavy_observed_mask、estimated/source_kind、exit_depth。clip router 的输出是原始 clip id 和预算，而不是已重排成伪规则时间的原始帧序列。

由 integration agent 选择 wrapper/子类接法，但必须维持旧 `ActionFormer` 的默认构造参数和原 state_dict 兼容。新 checkpoint 显式包含 frozen teacher identity、aux版本和部署策略；纯Z0继续读dense checkpoint，不允许随意 `strict=False` 漏掉旧核心层。

执行包只提供研究、任务书和受控启动工具，**没有假装实现这些新 H65 模型文件**。研究路线的配置文件、profiler 和 validators 都必须由 agents 创建并通过测试，运行工具会在文件缺失或 gate 未通过时拒绝执行。

## 16. 何时应停止加复杂度

若 Z0 已达到质量/速度目标，应先形成强 baseline，而不是为论文结构强行增加学习模块。若 preview-only 模型已足够，应与它进行同训练成本比较；若空间模块 wall-clock 变慢，先保留时间+深度两维，不把“理论MAC更低”当成成功。

若 D1 无法达到高tIoU约束而 C2 可以，应诚实将最终系统定义为“dense backbone + sparse-aware auxiliary calibration”，同时保留 D1 的负结果。若所有路线在短动作漏检上不可接受，应提高最小覆盖和重观测预算，必要时放弃某档压缩。

新颖性应围绕 **TAD所需的边界安全、完整clip契约、最终feature出口对齐、三维实际预算控制和dense训练图审计** 建立，而不是宣称“首次sparse-to-dense”或“无需训练、无损三维剪枝”。本次代表性文献检索不能支持世界范围不存在同构方法的排他性主张。

## 17. 最终建议

第一里程碑应是 `真正dense768 checkpoint → 原始24/48完整clips → native384恢复 → 原768检测器`，得到没有新增训练的可信结果。

第二里程碑才是主方法：**所有训练clips完整前向，训练H0/H8最终特征接口；推理选择cheap/8/12，并用dense廉价基底填满原时间网格。**

第三里程碑增加后半层空间heavy-MLP稀疏更新；证明真实GPU收益后，再讨论更激进attention压缩、deformable reconstruction和C2校准。这样最终失败或成功都能被清楚解释，也最能复用当前H65的selector、TIA、native feature与验证基础设施。

## 参考来源

以下来源标记对应文中的研究事实或源码结论。实现建议、门槛与实验预算是本文提出的设计，并非原论文已验证结果。原论文的模型、数据集、硬件与本H65配置不同，未将其加速或精度直接迁移。


**[R01]** Deep Feature Flow for Video Recognition. CVPR 2017.

https://openaccess.thecvf.com/content_cvpr_2017/html/Zhu_Deep_Feature_Flow_CVPR_2017_paper.html

用途/边界：稀疏关键帧重网络与稠密特征传播；原方法包含端到端训练，不是纯 dense-only 证据。

**[R02]** Listen to Look: Action Recognition by Previewing Audio. CVPR 2020.

https://openaccess.thecvf.com/content_CVPR_2020/html/Gao_Listen_to_Look_Action_Recognition_by_Previewing_Audio_CVPR_2020_paper.html

用途/边界：廉价模态蒸馏昂贵 clip 表示；视频分类，不直接保证 TAD 边界。

**[R03]** Token Merging: Your ViT But Faster. ICLR 2023.

https://arxiv.org/abs/2210.09461

用途/边界：可直接压缩已有 ViT；训练自由版本和训练版本须分开；合并改变注意力。

**[R04]** Adaptive Token Sampling for Efficient Vision Transformers. ECCV 2022.

https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/4901_ECCV_2022_paper.php

用途/边界：参数自由 token sampling；评分接口与无 CLS 的 VideoMAE 不可盲接。

**[R05]** DynamicViT: Efficient Vision Transformers with Dynamic Token Sparsification. NeurIPS 2021.

https://arxiv.org/abs/2106.02034

用途/边界：学习稀疏策略、蒸馏及训练时模拟稀疏；不属于严格 dense-forward-only。

**[R06]** DiffRate: Differentiable Compression Rate for Efficient Vision Transformers. ICCV 2023.

https://arxiv.org/abs/2305.17997

用途/边界：可优化压缩率；冻结 backbone 不等于无需率校准。

**[R07]** Dynamic Token Pruning in Plain Vision Transformers for Semantic Segmentation. ICCV 2023.

https://openaccess.thecvf.com/content/ICCV2023/html/Tang_Dynamic_Token_Pruning_in_Plain_Vision_Transformers_for_Semantic_Segmentation_ICCV_2023_paper.html

用途/边界：密集预测中的逐 token 早退，保留上下文；存在辅助训练。

**[R08]** A-ViT: Adaptive Tokens for Efficient Vision Transformer. CVPR 2022.

https://openaccess.thecvf.com/content/CVPR2022/html/Yin_A-ViT_Adaptive_Tokens_for_Efficient_Vision_Transformer_CVPR_2022_paper.html

用途/边界：训练 token halting；不能据此证明任意预训练 backbone 可直接跳层。

**[R09]** FastBERT: a Self-distilling BERT with Adaptive Inference Time. ACL 2020.

https://aclanthology.org/2020.acl-main.537/

用途/边界：先训练主干，再自蒸馏浅层分类器，推理条件早退；D1 深度路线最直接类比。

**[R10]** Be Your Own Teacher: Improve the Performance of Convolutional Neural Networks via Self Distillation. ICCV 2019.

https://openaccess.thecvf.com/content_ICCV_2019/html/Zhang_Be_Your_Own_Teacher_Improve_the_Performance_of_Convolutional_Neural_ICCV_2019_paper.html

用途/边界：深层到浅层自蒸馏，借鉴出口表征对齐，而非直接浅层特征等价。

**[R11]** Reducing Transformer Depth on Demand with Structured Dropout. ICLR 2020.

https://arxiv.org/abs/1909.11556

用途/边界：LayerDrop 训练中已有层随机删除，不是从未稀疏训练的例子。

**[R12]** Learning-based Optimization of the Under-sampling Pattern in MRI. IPMI 2019.

https://arxiv.org/abs/1901.01960

用途/边界：LOUPE 学习概率采样与重建；原始掩码不等价内容相关逐样本 exact-K。

**[R13]** Sparse-to-Dense: Depth Prediction from Sparse Depth Samples and a Single Image. ICRA 2018.

https://arxiv.org/abs/1709.07492

用途/边界：RGB+稀疏深度重建稠密深度；训练中使用稀疏观测，借鉴 mask 和观测一致性。

**[R14]** Multi-Time Attention Networks for Irregularly Sampled Time Series. ICLR 2021.

https://arxiv.org/abs/2101.10318

用途/边界：mTAN 将非规则观测映射至参考时间网格；借鉴真实时间条件化。

**[R15]** Clockwork Convnets for Video Semantic Segmentation. ECCV 2016.

https://arxiv.org/abs/1608.03609

用途/边界：不同语义层不同更新频率；时间复用不代表跨网络深度的特征相同。

**[R16]** DeepCache: Accelerating Diffusion Models for Free. CVPR 2024.

https://openaccess.thecvf.com/content/CVPR2024/html/Ma_DeepCache_Accelerating_Diffusion_Models_for_Free_CVPR_2024_paper.html

用途/边界：训练自由跨去噪步缓存；去噪步与视频动作时间不可等同。

**[R17]** LLaVA-PruMerge: Adaptive Token Reduction for Efficient Large Multimodal Models. ICCV 2025 / 作者项目页.

https://llava-prumerge.github.io/

用途/边界：多模态视觉 token 压缩；下游 LLM 节省不同于省掉视觉编码器输入。

**[R18]** TALLFormer: Temporal Action Localization with Long-memory Transformer. ECCV 2022.

https://arxiv.org/abs/2204.01680

用途/边界：TAD 长期特征记忆与训练效率；不能直接算作部署时跳过 backbone 的证据。

**[R19]** TURN TAP: Temporal Unit Regression Network for Temporal Action Proposals. ICCV 2017.

https://openaccess.thecvf.com/content_iccv_2017/html/Gao_TURN_TAP_Temporal_ICCV_2017_paper.html

用途/边界：复用 temporal unit 特征，提示 clip 单元计算与 proposal 计算的区别。

**[R20]** SP-TAD: Sparse Proposals for End-to-end Temporal Action Detection. 2021 预印本.

https://arxiv.org/abs/2109.08847

用途/边界：稀疏 proposal 不等价稀疏帧采集或稀疏 backbone。

**[R21]** Finding Action Tubes with a Sparse-to-Dense Framework. AAAI 2020.

https://ojs.aaai.org/index.php/AAAI/article/view/6811

用途/边界：时空动作管定位；不是 temporal-only H65，但限制宽泛 sparse-to-dense 新颖性声明。

**[R22]** Temporal Cluster Assignment for Efficient Real-Time Video Segmentation. CVPR 2026 Workshop ECV.

https://openaccess.thecvf.com/content/CVPR2026W/ECV/html/Yung_Temporal_Cluster_Assignment_for_Efficient_Real-Time_Video_Segmentation_CVPRW_2026_paper.html

用途/边界：近期无需微调的时间聚类与窗口一致性方向；workshop 证据。

**[R23]** Video Patch Pruning: Efficient Video Instance Segmentation via Early Token Pruning. CVPR 2026 Workshop ECV.

https://openaccess.thecvf.com/content/CVPR2026W/ECV/html/Glandorf_Video_Patch_Pruning_Efficient_Video_Instance_Segmentation_via_Early_Token_CVPRW_2026_paper.html

用途/边界：近期早期视觉 patch 剪枝；训练策略与 token 保真须单独核实。

**[R24]** Spatio-Temporal Token Scoring. 2026 预印本.

https://arxiv.org/abs/2603.18004

用途/边界：视频语言模型 token 评分/压缩；不同于同缩写的早期 token selection。

**[R25]** End-to-End Temporal Action Detection with 1B Parameters Across 1000 Frames. CVPR 2024.

https://openaccess.thecvf.com/content/CVPR2024/html/Liu_End-to-End_Temporal_Action_Detection_with_1B_Parameters_Across_1000_Frames_CVPR_2024_paper.html

用途/边界：AdaTAD/TIA 训练显存效率；冻结 backbone 不等于推理免算 backbone。

**[R26]** VideoMAE: Masked Autoencoders are Data-Efficient Learners for Self-Supervised Video Pre-Training. NeurIPS 2022.

https://arxiv.org/abs/2203.12602

用途/边界：高 masking 预训练不证明 TAD 高稀疏推理的定位精度。

**[R27]** Masked Video Distillation: Rethinking Masked Feature Modeling for Self-supervised Video Representation Learning. CVPR 2023.

https://arxiv.org/abs/2212.04500

用途/边界：特征蒸馏及空间/时间教师；属于 masked feature training，适合 C2 对照。

**[R28]** Codex CLI reference. OpenAI 官方在线文档，检索 2026-09-11.

https://developers.openai.com/codex/cli/reference/

用途/边界：exec、stdin prompt、sandbox 与 output-last-message；本机仍以 --help 为准。

**[R29]** torchrun / torch.distributed.run. PyTorch 官方文档.

https://docs.pytorch.org/docs/main/elastic/run.html

用途/边界：分布式启动参考；运行包采用仓库已验证的单进程环境变量，不升级现有环境。

**[C01]** scripts/run_duca_h65_matched_cycle4_n16r4.sbatch. 固定源码 04c35a3b76897e6c1569eeede41ed3aecaf7f854.

https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/04c35a3b76897e6c1569eeede41ed3aecaf7f854/scripts/run_duca_h65_matched_cycle4_n16r4.sbatch

用途/边界：真实 PRECHECK/PRE_RUN 与 stage1/2 启动入口

**[C02]** configs/adatad/thumos/duca_h65_first_singleclock_cycle4.py. 固定源码 04c35a3b76897e6c1569eeede41ed3aecaf7f854.

https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/04c35a3b76897e6c1569eeede41ed3aecaf7f854/configs/adatad/thumos/duca_h65_first_singleclock_cycle4.py

用途/边界：K384、single-clock、packed 禁用、paper_claim_allowed=False

**[C03]** configs/adatad/thumos/duca_sampling_rate_curriculum_stage1_uniform384.py. 固定源码 04c35a3b76897e6c1569eeede41ed3aecaf7f854.

https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/04c35a3b76897e6c1569eeede41ed3aecaf7f854/configs/adatad/thumos/duca_sampling_rate_curriculum_stage1_uniform384.py

用途/边界：30epoch/3000updates uniform384；epoch29 EMA

**[C04]** opentad/models/backbones/vit_adapter.py. 固定源码 04c35a3b76897e6c1569eeede41ed3aecaf7f854.

https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/04c35a3b76897e6c1569eeede41ed3aecaf7f854/opentad/models/backbones/vit_adapter.py

用途/边界：perclip TIA、patch embedding、packed scatter、冻结语义

**[C05]** opentad/models/backbones/backbone_wrapper.py. 固定源码 04c35a3b76897e6c1569eeede41ed3aecaf7f854.

https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/04c35a3b76897e6c1569eeede41ed3aecaf7f854/opentad/models/backbones/backbone_wrapper.py

用途/边界：clip grouping、坐标转发、native pooling/postprocessing

**[C06]** opentad/models/utils/temporal_grid.py. 固定源码 04c35a3b76897e6c1569eeede41ed3aecaf7f854.

https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/04c35a3b76897e6c1569eeede41ed3aecaf7f854/opentad/models/utils/temporal_grid.py

用途/边界：真实/规范时间坐标与残差

**[C07]** opentad/models/detectors/actionformer.py. 固定源码 04c35a3b76897e6c1569eeede41ed3aecaf7f854.

https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/04c35a3b76897e6c1569eeede41ed3aecaf7f854/opentad/models/detectors/actionformer.py

用途/边界：selector 前置、mask/GT、detector 长度及 compressor guard

**[C08]** configs/adatad/thumos/e2e_thumos_videomae_s_768x1_160_adapter.py. 固定源码 04c35a3b76897e6c1569eeede41ed3aecaf7f854.

https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/04c35a3b76897e6c1569eeede41ed3aecaf7f854/configs/adatad/thumos/e2e_thumos_videomae_s_768x1_160_adapter.py

用途/边界：dense768 配置；不是已验证 checkpoint

**[C09]** tools/test.py. 固定源码 04c35a3b76897e6c1569eeede41ed3aecaf7f854.

https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/04c35a3b76897e6c1569eeede41ed3aecaf7f854/tools/test.py

用途/边界：CLI、数据读取 cfg.dataset、评估与提交状态检查

**[C10]** tools/train.py. 固定源码 04c35a3b76897e6c1569eeede41ed3aecaf7f854.

https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/04c35a3b76897e6c1569eeede41ed3aecaf7f854/tools/train.py

用途/边界：训练守卫与 checkpoint 初始化

**[C11]** AGENTS.md. 固定源码 04c35a3b76897e6c1569eeede41ed3aecaf7f854.

https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/04c35a3b76897e6c1569eeede41ed3aecaf7f854/AGENTS.md

用途/边界：代码范围、远端写入范围和运行规则
