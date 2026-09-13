# BMCR-T：固定提交独立审查与改进研究

审查对象：`yuzbo/BMCR-T-AdaTAD`。代码、配置、分析、结果固定于 **15280e5dc29e3df18d085aaba21067e04107ce84**。任务说明单独固定于 **8a8529093f8a96b228e138c272de0f9df999bda8** 的 `docs/EXTERNAL_REVIEW_PROMPT.md`。本文中的“当前”只指这一代码快照。

本轮仅进行源码、原始结果记录与文献审查，以及独立算术/谓词检查；**没有启动训练，没有加载并运行模型权重，没有重跑完整评测，也没有重训官方 AdaTAD**。普通网页/源码归档下载失败后，通过 GitHub 连接器读到了关键固定版本材料。源码证据与作者记录核对不等于独立数值复现；未逐条重算全部预测，也未逐行审查仓库所有无关文件。

## 一、三个核心问题的裁决

**实现：主体调用链基本自洽，但不能签署“完全正确”。** 真实 RGB 固定预算执行、约束 sigmoid 的隐式梯度、ASFormer 动作监督、均匀 companion、EMA 的隔离、交换收益的符号、非互反伙伴的处理、推理前坐标回映和完整视频聚合基本正确。发现一项实际生效的学习率分组错误：内部 ASFormer 注意力输出投影被误分入动作分类头学习率组。另有三项更可能限制方法上限、但不能伪称确定性 bug 的设计问题：检测损失在 selected-rank 而非原时间中定义；同一换帧效用与最终全局集合重解码不一致；已经计算的全时间粗特征没有进入检测器。[C1][C2][C3][C4][C5][C6]

**效果：已经减少矩阵/卷积 MAC，但没有保持官方精度，也没有实现已测口径下的推理加速。** S/B 相对官方平均 mAP 分别下降 5.86/3.79 个百分点，@0.7 分别下降 7.24/4.27 个百分点。MAC 下降约 48.45%/49.55%，指定完整窗口推理中位时延却变成官方的 2.376/1.301 倍。推理 allocated 显存下降约 7.92%/25.50%；没有同配方官方训练成本，不能声称训练更便宜。[R1][R2]

**下一步：优先做原时间轴上的粗细融合，而不是继续扩大效用 MLP 或堆叠蒸馏损失。** 先补齐同日程 K384 均匀采样，再分离测试“检测网格恢复”“粗信息进入检测器”“学习选帧本身”的贡献。主方案是原生 tubelet 特征按真实时间恢复到完整检测网格、加入已计算粗特征的小型残差；两个独立备选分别是互补视图输出自蒸馏，以及保持预训练 clip 内部结构的采样。后文给出可执行的设计合同，而非未经实验的性能承诺。

## 二、论文身份与实验边界

BMCR-T 是仓库研究提案的实现名称，展开为 Boundary-aware Conditional Marginal Compute Routing 的时序版本，不是已经给定的一篇发表论文的复现。仓库说明它来自两份研究报告；本文没有取得这两份原始报告的完整内容，不能替它们认证公式或预期结论。[C0]

能够核实的 AdaTAD 原论文是 Liu、Zhang、Zhao、Ghanem 的 **End-to-End Temporal Action Detection with 1B Parameters Across 1000 Frames**，CVPR 2024，arXiv:2311.17241v2，2024-04-20 camera-ready 版本。论文核心是通过 TIA 与参数高效适配降低端到端训练显存、扩大骨干和输入规模；不是承诺把高成本 RGB 观察减半后无损且更快。[P1]

固定仓库 vendored 官方表中，VideoMAE-S/B、768 个候选位置、160×160 输入的平均 mAP 为 69.03/71.14；本仓库官方 checkpoint 重评为 69.01/71.13。应以本仓库同一评测器重评值作为当前外部锚点，不能替换为其他大骨干、1536 帧或最新 README 的最佳分数。[C10][R1]

当前新方法从 K400 recognition pretrain 开始，不是用官方 TAD 训练权重初始化。200 个训练视频，20 epoch 均匀预热/2000 次成功更新，加 40 epoch 联合训练/4000 次成功更新，batch=2；每个骨干的 H65-C/BMCR-T 共享其预热末轮 EMA。只报告一个种子。官方 checkpoint 与新方法并非同训练配方、同更新数的预算消融；checkpoint 的 epoch 字段本身也不足以重建官方实际选模过程。[R2][C2][C3]

数据命名必须区分：raw-video 目录名、annotation 的 subset 名、代码 `val` 并不等义。当前 200 个训练视频使用 annotation `training`；最终 211 个视频使用 annotation `validation`。`cfg.dataset.val` 对应测试来源，不能拿它调参。候选轴步长由 `snippet_stride=4` 给出，**768 个候选位置不是 768 张连续原视频帧**；实际秒坐标由 stride、window offset、fps 共同决定。[C11][C12]

## 三、沿真实调用链的审查

### 3.1 实际图，而不是方法名称

训练图为：

```text
all-candidate RGB X [B,1,3,768,160,160]
  ├─ resize64 → CNN → ASFormer encoder/decoder
  │              ├─ actionness / transition auxiliary losses
  │              └─ policy descriptors → calibrated rates → S0
  │                   └─ BMCR: condition on S0 → 2-d utility → re-decode S1
  └─ mix: half rows S1, half rows uniform U384
       → actual RGB gather [B,1,3,384,160,160]
       → 24×16-frame clips → frozen VideoMAE + trainable global TIA
       → native [B,Ce,192] → rank interpolation [B,Ce,384]
       → ActionFormer on selected-rank grid
       → GT mapped from original candidate time to rank

BMCR supervision events:
  same-model EMA, eval/no_grad
  → actual S0 RGB forward + up to two actual swapped-set RGB forwards
  → signed class/localization deltas → detached utility targets
```

推理没有 teacher；全部候选 RGB 仍在解码和粗网络路径中，高成本骨干才只运行选中的物理槽位。原生两帧 tubelet 输出被上采样到 384 个 rank 特征，不能把这 384 个输出当成 384 次独立的重模型观察。[C1][C4][C7][C9][C11]

### 3.2 正确合同及其边界

| 环节 | 已核实内容 | 不能推出的结论 |
|---|---|---|
| 固定 K | prefix-valid 窗口中选 min(K,L) 个有序唯一位置，物理张量补到 K；有效长度不足 K 时其余槽位无效 | 短窗口重计算自动降到 min(K,L)；当前仍固定 24 clips |
| 约束率梯度 | sigmoid 阈值满足总质量 K；自定义 backward 减去加权基线，符合隐式约束求导 | 后续硬采样和 RGB slope 的整条梯度无偏或方向正确 |
| 硬前向 | 重骨干看到精确 gather 的 RGB；transport bridge 的前向值为零 | 已跳过解码、CNN、ASFormer、全输入存储 |
| ASFormer | 动作 BCE 更新 CNN、encoder、decoder；encoder replay 复用随机状态，使策略梯度不回到 RGB stem | `detach` 意味整个 ASFormer 没有训练；事实并非如此 |
| 均匀 companion | batch2 随机一行均匀、一行学习采样，各自监督各自视频 | 这是同视频 teacher/student 蒸馏；它实际不是 |
| EMA | 同模型、0.999、成功 optimizer step 后更新；teacher eval/no_grad；目标 detach | 已有检测 logits/特征自蒸馏，或 EMA 必然更准确 |
| 效用符号 | 已选点的保留价值是替换后损失减原损失；未选点插入价值取相反方向；仅互为伙伴时补互反标签 | 任意两候选的标签可直接互换，或该值是 Shapley 值 |
| 集合条件 | 自身/伙伴 hidden、集合均值、转变表示、成员身份、覆盖间距等 358 维输入确实被使用 | 模型只做单点 MLP；或已有充分的高阶集合建模 |
| 时间几何 | GT→rank，预测→真实候选时间，随后→秒；回映早于 NMS | 骨干、TIA、loss 已经在真实间隔上建模 |
| 评测 | 检查测试 ID 集合 211；全部窗口后按视频聚合、类内 SoftNMS、统一 evaluator | 本轮已经独立重评 checkpoint，或逐条认证全部预测 |

证据：[C1]–[C9]、[C12]–[C14]。

约束率可写为

\[
r_i=\sigma((s_i-\lambda)/\tau),\quad \sum_i r_i=K,
\qquad a_i=r_i(1-r_i)/\tau.
\]

代码实现的向量–雅可比积为

\[
\frac{\partial L}{\partial s_i}
=a_i\left(g_i-\frac{\sum_j a_jg_j}{\sum_j a_j}\right).
\]

这验证的是 sigmoid 总量约束，而不是离散集合损失。RGB bridge 用相邻像素差传递局部代理梯度，可能与一次真正换帧、重新配对 tubelet 后的任务损失变化不一致。应直接比较局部代理方向与真实交换增益，不能用非零梯度测试代替。[C4]

### 3.3 确定性错误：ASFormer 学习率组越界

**位置**：固定提交 `h65/full/runtime.py:89–108`，其中第 **101 行**用

```python
lr = action_lr if '.conv_out.' in name else trunk_lr
```

注释第 99–100 行明确希望只有 encoder/decoder 的动作分类器使用 action_lr。但 `references/ASFormer/model.py:44–52` 中 `AttLayer` 也有 `conv_out`（定义在第 52 行），于是以下内部投影全部被误分组：

```text
temporal.encoder.layers.{0,1}.att_layer.conv_out.{weight,bias}
temporal.decoders.0.layers.{0,1}.att_layer.conv_out.{weight,bias}
```

**触发条件**：实际 S/B、H65-C/BMCR-T 的 warm 和 joint 配置均满足，无需特殊输入。warm 时这些内部参数使用 1e-4 而不是 5e-5；joint 使用 2e-5 而不是 1e-5。[C2][C8]

**规模与运行记录交叉验证**：四个 48→96 投影共 **18,816 个参数**误分组；两个 96→2 真正分类头仅 **388 个参数**。实际 `s_bmcr/config.json` 的 action_lr 分组记录是 18,816 个有衰减权重、388 个无衰减 bias，合计 19,204，与错误谓词完全一致。这里“18,816”的两种口径巧合相同：前者含内部投影的权重和 bias，后者是错误大组中的全部 weight。[R3]

**影响**：改变被宣称的 trunk/action 优化配方，足以影响收敛与路由特征。它不直接证明已有 mAP 是假的，也不能从静态分析断言修复后会上升多少，或把 S/B 差异全部归咎于此。共同 bug 不会让“BMCR 只比 H65 改一处”的比较完全失效，但使“严格预定优化配方”的说法不成立。

**最小修复**：按模块对象/参数身份精确分组，比子串更可靠。

```python
action_parameter_ids = {
    id(p) for p in model.scout.temporal.encoder.conv_out.parameters()
}
for decoder in model.scout.temporal.decoders:
    action_parameter_ids.update(id(p) for p in decoder.conv_out.parameters())

# 保持原有分支和 weight_decay 规则；只替换内部 LR 判断。
for name, parameter in model.scout.named_parameters():
    if name.startswith('spatial_stem') or name.startswith('temporal.'):
        lr = action_lr if id(parameter) in action_parameter_ids else trunk_lr
    else:
        lr = scorer_lr
```

最小回归测试应断言分类头参数集合恰好 388、全部四个内部 attention 投影在 trunk 组、所有训练参数恰好出现一次；分别覆盖 warm/joint 和 S/B。旧结果保留为“旧配方”；新比较必须共享**修复后的 warm**，不能只修一个方法后与旧基线宣称提升。

### 3.4 三项核心设计疑点，不能写成已证实 bug

**D1：正确的逆映射不保证正确的定位目标。** `h65/full/model.py:85–88` 将 GT 映到 rank 后交给 unchanged ActionFormer；`anchor_free_head.py:169–216` 的损失与后续 target assignment 使用 rank 间隔，DIoU、中心采样半径、regression range、最短实例选择均依赖该坐标。[C1][C13]

数学反例：rank 轴 `[0,1,2,3]` 映到真实时间 `[0,1,2,12]`。GT rank `[0,3]`；预测 A `[0,2]`、B `[1,3]` 的 rank IoU 都为 2/3。但真实 IoU 分别为 1/6 与 11/12。映射严格单调、逆映射也正确，目标仍严重不同。这是数学反例，**不是对真实数据错误比例的估计**。

最小机制试验可只把 decoded prediction 和 GT 映回真实时间计算 localization loss，并固定分类 target assignment；这仍没修正 rank 上的中心采样和 FPN range。更完整的主方案是把 detection grid 恢复到原候选轴。特别注意当前 `TrueTimeMap` 在插值中 detach 输入，适用于 GT/推理，直接拿去算 student 的 physical-time regression loss 会切断梯度；需另加保留 query 梯度的函数，并做有限差分测试。这是**未来修改的陷阱**，不是指当前推理有 detach bug。[C5]

**D2：S0 的局部交换标签指导 S1 的全局重采样。** `model.py:45–57` 的条件头在 S0 测量/预测局部收益，最后把所有候选的校准收益加到 rate logits，重新解码整个集合。它不等价于执行被监督的某一个 swap，可能改变更多位置和 tubelet 配对；条件特征也没有在 S1 上重算。没有证据证明一定有害，但它是当前效用可迁移性的核心假设。[C1][C6][C7]

最小验证：在训练来源 holdout 上，记录 S0→S1 的对称差、覆盖变化、真实任务损失增益，并比较“预测总增益”与“实测整体增益”。修正候选是最多一至两次、每次重建条件的显式交换，允许“不交换”，而不是更大的 MLP。要把新增 teacher 查询算入预算。

**D3：cheap representation 没有作为检测证据。** `FormalScout.forward` 算过全部时间位置的 CNN/ASFormer，但 `train_batch` 只把 selected heavy features 喂给 detector。未选时间的粗特征没有提供可被检测损失直接使用的兜底信息。该设计不是错误；但它使“全视频都被便宜地观察过”没有转化为“全时间都可用粗证据定位”。[C1][C7]

另外，VideoMAE 的两帧 tubelet 与 16-frame attention 把打包相邻当作规则相邻，TIA 也沿压缩后的 192 个 tubelet 位置做卷积。这一预训练结构偏移发生在输出回映之前，不能靠插值特征或回映边界完全修复。[C9]

### 3.5 确定存在的计算浪费与待定位的系统瓶颈

`h65/transport.py:69–95` 在 `alpha==0` 时仍先做平滑和 48 次 sigmoid 阈值二分，再覆盖成均匀率/端点 linspace。可在保留原 indices、valid、density/rates 数值与零梯度语义的前提下提前走均匀 fast path；这是无意义计算，但尚未计时证明是主要瓶颈。[C4]

`h65/full/scout.py:64–88` 的条件伙伴计算有 batch×48 cell×两种成员的 Python/GPU 动态索引，此外 `int(mask.sum())`、逐行处理等可能导致同步和小 kernel 密集启动。BMCR 条件头本身仅约 0.0177 GMAC，绝不能以此推断整个条件决策开销近乎为零。先对原实现做 CUDA/CPU 时间线，再向量化固定 16-cell 的两两距离与 mask；不要先更换算法造成性能归因混乱。[C4][C7][R1]

## 四、当前结果究竟证明了什么

### 4.1 精度

| 骨干/方法 | @0.3 | @0.4 | @0.5 | @0.6 | @0.7 | Avg-mAP |
|---|---:|---:|---:|---:|---:|---:|
| S 官方 checkpoint | 83.83 | 79.09 | 72.33 | 61.57 | 48.24 | 69.01 |
| S H65-C | 78.38 | 73.08 | 66.33 | 55.20 | 41.31 | 62.86 |
| S BMCR-T | 78.77 | 73.89 | 65.91 | 56.20 | 41.00 | 63.16 |
| B 官方 checkpoint | 85.98 | 81.84 | 74.93 | 63.26 | 49.62 | 71.13 |
| B H65-C | 82.37 | 77.57 | 69.93 | 58.89 | 44.31 | 66.61 |
| B BMCR-T | 83.02 | 78.36 | 70.67 | 59.30 | 45.35 | 67.34 |

原始分数来自固定 `comparison.json`；本文仅做百分比换算/差值计算。[R1]

BMCR 对 H65-C 的平均增益为 S +0.297、B +0.728 pp；S 的 @0.5 **−0.419 pp**，@0.7 **−0.302 pp**，B @0.7 +1.039 pp。因而“BMCR boundary-aware，所以严格定位得到保持/普遍改善”不成立。单种子、无 paired interval，不能把小均值变化称为显著提升。没有同日程纯 K384 均匀控制，尚无法证明学习选帧优于同预算均匀观察。[R1]

### 4.2 MAC、时延与显存必须拆开

| 主窗口口径 | S 官方 | S BMCR | B 官方 | B BMCR |
|---|---:|---:|---:|---:|
| 矩阵/卷积 GMAC | 1173.95 | 605.17 | 4041.08 | 2038.74 |
| 相对 MAC 降幅 | — | 48.45% | — | 49.55% |
| 时延均值 ms | 51.21 | 121.91 | 118.31 | 153.75 |
| 时延中位数 ms | 51.17 | 121.59 | 118.17 | 153.74 |
| peak allocated GiB | 0.935 | 0.861 | 1.508 | 1.123 |

实际 heavy clips 48→24，MAC 下降是真执行变化，不只是对已完成的 backbone 输出乘 mask。报告的 MAC 是指定矩阵、卷积与注意力乘法统计，不等于完整程序每条算术操作或能耗。[R1][C14]

计时是同一指定完整窗口、batch1、去掉计数钩子、warmup 后同步计时；不包括完整视频解码、视频级 CPU 聚合/SoftNMS 等。因此已有记录足以否定该口径的加速，但尚不能定量说明全视频端到端会慢多少。显存是该 profile 的 allocated 峰值，不能替代 reserved、整个进程物理占用或训练峰值。缩短 token 轴不缩减冻结骨干参数的存储，也不消除完整候选 RGB 和粗网络。[C14]

### 4.3 训练成本

S/B BMCR 完整 warm+joint 记录约 **6.05/6.52 GPU 小时**，训练 allocated 峰值约 **7.72/13.59 GiB**。两种骨干、两种方法、共享 warm 计一次的六阶段总记录约 **21.22 GPU 小时**，不含预检、效用审计、调参和测试。[R2]

按当前成功更新与 feedback 日程，teacher 查询最多 416 个事件，每事件最多 3 个单行高成本前向，即最多 1248 次单行 K384 前向；joint student 是 4000×2=8000 行前向。这只给出最多 15.6% 的**前向行数**附加量，不是训练时间或训练 FLOP 增幅：student 还有反向/checkpoint recompute，teacher 只有前向，H65-C 还执行分类/回归两次额外 input-gradient VJP。teacher eval 不保留激活，并不等于没有训练成本。[C3][C6]

### 4.4 为什么 S 损失更多：观测与因果分开

观测是：不只 BMCR，H65-C 对官方的差距也已经呈 S 更大；BMCR 改动不是 S/B 差异的唯一来源。至少应区分：预算减半本身；压缩 rank 导致的物理 receptive field/FPN assignment 变化；打包运动间隔与预训练 mismatch；代理梯度/teacher 信号质量；全时粗信息丢弃；优化配方与成功更新数。[R1][C1][C9][C13]

“较小模型对观察稀疏更脆弱”可以是一个假设，不是现有证据给出的唯一因果解释。S adapter LR 与 B 不同、实际采样集合/梯度统计也可能不同。先在训练来源开发集按动作长度、最大采样空洞、边界最近观察距离、类别分组；再做固定采样轨迹/同预算 U384 控制。只把 S 的 sampler 换给 B 做一次推理，仍有训练–推理分布变化，不能单独当成干净因果实验。

预热审计本身有负证据：S 的旧贡献代理与 holdout 真交换收益的 Spearman 接近零（分类 −0.030、定位 −0.028）；B 定位为负（−0.203）。但这些是**warm attribution/ridge 诊断，不是 joint 终点 BMCR utility head 的评估**。不能混淆这两个结论。[R4][R5]

## 五、四方向的跨领域机制比较

本节按“机制能带来什么”和“不能借用什么结论”组织。论文中的他任务收益不作为本项目预期 mAP。以下链接是原论文/作者代码；标注“源码已读”的条目实际查看了关键实现，其余只确认论文和作者入口，不宣称已逐行复现所有外部项目。

### A. 粗细融合：最应先补的能力

**Uni-AdaFocus：保留廉价全局信息，稀疏运行昂贵局部编码器。** 证据是视频识别，不是 TAD；ActivityNet 出现也不意味着已经验证动作起止检测。对当前代码最直接的借鉴是把 coarse hidden 用于检测，而不仅用于 policy。原文还明确讨论了像素插值梯度过于局部、策略梯度干扰全局编码器，以及用粗深层特征提供更语义化的策略监督。可迁移前提是粗特征在原坐标有可用的任务信息；失败方式是粗分支二分类表示不足、策略/检测梯度冲突。落点为 `FormalScout.forward` 暴露 raw coarse hidden、`FormalH65.encode/train_batch` 融合。其“cheap all + heavy selected”已是先例，不能作为新颖性本身。[P2]

**PointRend：粗预测与细粒度特征在原图坐标上的不确定点融合。** 已验证实例/语义分割，适合借鉴边界查询与 coarse-to-fine refinement。已读官方 `point_features.py`：采样使用原坐标归一化，且必须先插值 logits 再算 uncertainty，二者不交换。迁移到 TAD 可用 start/end 邻域的真正时间位置查稀疏细证据，保留廉价全轴。失败在于过度集中模糊边界、漏掉语义判别所需动作内部与负上下文。PointRend 原方法节约的是精细预测头，不是已经计算过的图像 backbone；要在 BMCR 实现 RGB 节约，必须沿用真正的先 gather 后 heavy encode。[P3][X1]

**Deformable DETR：在原坐标参考点附近跨尺度取少量证据。** 已读官方 `ms_deform_attn.py`：query 产生 offsets/weights，从各层原空间 reference points 取值，而输入 value projection 仍覆盖全部输入特征。迁移到本项目可用 coarse 的 `[B,768,96]` 作 query、native fine 的 `[B,192,96]` 作证据，先按真实时间限定每个 query 的四个邻近锚，再添加相对时间偏置；不同层的时间步长必须分别归一化。它适合替换不能处理大间隔/跨边界混合的简单插值，但缺失的视觉证据不会因 attention 出现而生成。新增投影、索引和 kernel 必须计时；初轮不加此模块，只有 plain fusion 的失败明确来自证据对齐才升级。参考点和跨尺度采样本身已非新贡献。[P20][X5]

**Deep Feature Flow：稀疏关键帧重编码、其余帧通过流传播。** 对视频检测/分割提供了比视频分类更接近的稀疏观察先例。借鉴的是对未重算位置显式补偿，并随不可靠性刷新；不能直接把空间 flow 变成时间轴线性插值。快速动作、出现/遮挡与新动作边界会破坏缓存有效性。更重要的是当前 TIA feature 依赖整个窗口/所选集合，跨窗口缓存不能只按“原帧 ID 相同”复用。首轮只做本窗口 coarse buffer；永久 heavy cache 需上下文、权重和几何一致性证明。[P4][X6]

**CoLT5：每个 token 保留 light path，heavy attention/MLP 只给选中 token。** 原证据是长文本理解/生成，Q 与 KV 可分别稀疏，坐标和便宜状态不消失。对本项目的启发不是把其数值直接搬来，而是把“保留检测坐标”与“持续执行高成本算子”解耦。它在 Base 规模也不是全面无损，论文表 3 的平均指标比 LongT5-Base 低；总体速度还受新预训练与 multi-query decoder 等因素影响。不能据其 XL 成功断言 VideoMAE-S 必然成功，也不能把仅 mask attention 当成跳过 MLP。本文读了原论文，未认证其完整官方实现入口。[P5]

**Sparse DETR / Focus-DETR：任务驱动的 encoder token 更新。** 在目标检测上更直接：前者用 decoder 相关性选择更新 token，后者更强调前景语义与定位监督，说明一般 attention 热度不是充分的重要性定义。可借鉴“固定原坐标+只更新部分表示”，或给 sampler 提供任务相关的 foreground/coverage 监督；失败方式是把分类性强的位置误当边界性强的位置。原 dense CNN backbone 成本不能因 sparse encoder 名字而被扣掉。代码落点是 coarse predictor 或 sparse feature update，不是简单替换整个 TAD detector。[P6][P7]

**最近先例/边界。** 2026 的 MSDR-Mamba TAD 也出现多率与双路径融合，但公开描述采用冻结预提取特征、双分支都执行，其 per-level routing 不等于跳过 RGB 计算；作者的 routing 变体还有增时延的结果。这里只将它作为“融合/路由不自动省算”的近邻，不把它作为已公开可复现的 RGB 稀疏方法；源码未取得。[P18]

### B. 自蒸馏：先定义 teacher 多提供了什么

**Soft Teacher：同架构 EMA，跨增强坐标变换，分类和框回归分开处理可靠性。** 官方代码已读，teacher 冻结/no_grad，并明确对齐样本和变换坐标。证据是半监督目标检测，不是有标签 TAD 必然受益。可迁移成同一模型的另一 K384 观察视图，提供当前 student 缺失的证据；teacher 高置信但边界不准时必须减少回归蒸馏，不得用 teacher 缺失覆盖掉真实 GT。落点新增输出级 KD，而非把当前 utility teacher 改个名字就称自蒸馏。[P8][X2]

**深浅层自蒸馏与 data2vec 2.0：层级差异或观察差异创造教学信息。** BYOT 用深层教浅层，主要分类证据；data2vec 2.0 的 EMA contextual teacher 与多 masked student 提供了摊薄 teacher 代价的思路，主要是表示学习。TAD 的迁移前提是 shallow/coarse 对相同物理时间有定义良好的语义/边界表示。当前 coarse 只有 actionness 二分类输出，不能直接与 20 类 sigmoid 检测 logits 做 KL；需在共享 detector feature/head 层蒸馏，或新增训练用 20 类 coarse 头，并计入成本。均值 hidden L2 容易被大量背景支配；先做一个输出信号，而不是 feature、relation、boundary、class 四项同时开。[P9][P10]

**CrossKD 与蒸馏负证据。** CrossKD 在目标检测中专门处理 teacher 目标与真实标签对 student head 的冲突；可借鉴梯度隔离，不必整体复制跨模型教师头。Cho 与 Hariharan 的研究也说明更大/更好的 teacher 不保证更好 student。这些是反对“增加 KD 项总会提高上限”的证据，不是本项目 S 失败的归因结论。官方 B checkpoint 教 S 是跨模型/跨规模蒸馏；同一 S 的 EMA 教 S 才是本轮所说的同模型自蒸馏。[P11][P12]

不同预算 teacher（例如同一参数的 K768 EMA）理论上可提供额外证据，但当前 wrapper 的 TIA temporal_size、pre/post reshape、检测 grid 与 K 绑定，必须完整改成 per-call 合法预算，并实付两倍观察前向；不能只传一个更大的索引向量就声称支持。首轮建议 K384 互补视图，避免隐藏的 teacher 预算升级。[C1][C9]

### C. 非线性采样：问题不是“是否再加非线性层”

当前同时具备非线性内容分数、非均匀离散索引、单调累计率解码、集合条件 MLP 和代理梯度。需要区分以下六件事：位置不均匀不等于决策有集合交互；MLP 非线性不等于效用语义正确；连续参数不等于连续 RGB 观察；可求导 surrogate 不等于离散无偏梯度；正采样率不等于所有边界都有足够观察；逆坐标正确不等于特征邻接与定位损失度量正确。[C1][C4][C7]

**SampleNet：软投影训练、硬近邻匹配推理。** 已验证分类/重建/配准，不是其原论文已经解决 3D 检测。已读 registration sampler：训练把学习点软投影到输入邻域，推理通过 KNN 与补齐恢复实际输入点；还存在 CPU NumPy 往返。借鉴价值是显式处理“连续位置训练→离散实物输入”的差距，而不是照搬到 GPU 热路径。视频必须另加单调、唯一 K、tubelet 合法性，不能把混合 RGB 当真实帧。落点为 transport 和独立梯度质量审计。[P13][X3]

**SAMBLE（CVPR 2025）：边缘细节与全局覆盖不是同一目标。** 原文批评过分聚焦 sharp edges 的采样偏置，采用 shape-specific 分配平衡局部细节与全局均匀；证据主要是点云分类/分割。TAD 对应的是边界附近也要有上下文，而不是把所有 K 都贴近起止；可以使用候选 cell 配额、最大 gap 限制和负背景保底，但这仍不能保证高 tIoU。先与既有 coverage_floor 比较，别把已有覆盖改名为创新。[P14]

更一致的集合修正可定义：

\[
S^{r+1}=S^r-\{i_r\}+\{j_r\},\quad
(i_r,j_r)=\arg\max_{(i,j)\in\mathcal A(S^r)}\widehat\Delta(i,j\mid S^r),
\]

只接受通过 coverage 约束且校准收益为正的交换，并允许 stop。每轮只更新廉价 set descriptors，最终才跑 student heavy。teacher 标签必须在相同 S^r 上测量；多轮不默认可加，最多两轮用于验证局部监督是否能对齐实际决策。该改动属于纠正学习目标与执行动作的一致性，不是因为用了迭代就具有论文新颖性。

对连续时间参数，可令正密度积分得到单调 warp，但原视频离散读取仍需 round/gather。使用 RGB 局部代理、特征代理、随机策略梯度或有限交换回归，是四种不同估计器；比较应包括与真交换增益的方向相关性、方差、teacher 调用和执行开销，不能仅比较训练 loss 是否下降。

### D. 跨领域定位：保留的是坐标/身份，不一定是深层计算

| 任务 | 已验证机制 | 删除/停止更新后如何定位 | 真正少做了什么 | 迁移边界 |
|---|---|---|---|---|
| 视频目标检测/分割 | Deep Feature Flow | 将关键帧特征按 flow 映回目标帧像素 | 非关键帧 heavy CNN | 缓存与动作边界变化必须受控 |
| 图像检测 | Sparse/Focus-DETR | token 的原空间坐标和未更新状态保留，box head 回归原图坐标 | 部分 transformer encoder 更新 | 不等于免除 dense CNN |
| 实例分割 | PointRend | 在原 mask/图像坐标上查询并回填 | 高分辨率 point head 的密集执行 | 不确定性可能吸引噪声点 |
| 抽取式 span QA | Length-Adaptive Transformer | drop-and-restore 按原 token index 回填最近有效 hidden | 被丢 token 的后续 encoder 层 | 恢复坐标不恢复丢失的新上下文 |
| NER/POS/CWS | TOKEE | halt-and-copy，退出 token 仍可被其他 token 读取 | 退出 token 的 query/FFN 更新 | 独立单点置信度不够，要考虑邻域依赖 |
| 3D 目标检测 | IA-SSD | 被选点保留原 XYZ，并通过中心相关采样/投票定位实例 | 后续 point feature abstraction | 前景采样不等于边界完整 |
| 3D 检测稀疏执行 | Ada3D | voxel/BEV 原几何索引与 sparse tensor | 被删位置的后续 sparse conv | 算子、归一化、真实稀疏率共同决定加速 |
| 点云配准 | SampleNet | 软点训练、推理匹配实际点，再做几何变换估计 | 下游较小点集处理 | task utility 与可匹配几何仍须一致 |

来源：[P3]–[P7]、[P13]、[P15]–[P17]、[P19]；LAT 官方 restore 实现已读。[X4]

LAT 的关键负面边界是：仅永久删词的分类方法不能直接服务每个 token 的 span 输出，需恢复身份；TOKEE 在 NER 等序列标注中还发现平均不确定性可能被容易 token 淹没，所以采用邻域判据，并额外训练缓解 early-exit 分布偏移。映射到 TAD：不能因为片段中间语义很确定就让整个片段的两端都失去可定位表示。文本事件触发词/论元 span 是合理后续迁移目标，但上述 NER/QA 论文并不自动证明已经在事件抽取上验证。[P15][P16]

**共同设计原则不是“删得越多越好”，而是：输出坐标系仍完整、每个位置保留最低成本状态、高成本更新只作用于真正执行的子集、实例与边界不因索引压缩而被改写。** 原坐标、插值、EMA、固定 K、cheap–heavy 双支路都已有强先例；可争取的新颖性是证明 TAD 中“稀疏观察—定位度量—高成本执行”的具体失配，并给出最小、可测、真实更快的修复，而不是命名模块组合。

## 六、主方案：原时间轴粗细残差融合

### 6.1 单一假设

在 K384 不增加高成本观察的条件下，已计算的全时间粗特征能补充稀疏重特征空洞，而把检测损失恢复到原候选轴可以避免 rank 度量失真。这是要检验的假设，不是已成立结果。首轮不加新的 cross-attention、大型记忆、空间 pruning、动态深度，也不同时加入 KD。

### 6.2 张量与时间合同

粗分支输出 `C: [B,96,768]`，取 ASFormer 原始 encoder hidden，显式 transpose；不要直接拿已按 policy adapt_scale 缩放的 `output['hidden']`。在 `FormalScout.forward` 的原 encoder 之后、replay 替换之前保存 `coarse_hidden`，不增加 encoder 前向次数。[C7]

高成本输出采用原生 tubelet 特征

\[
H\in\mathbb R^{B\times C_e\times192},\quad C_e=384\;(S),\;768\;(B).
\]

对完整窗口，位置锚为

\[
\tau_j=(s_{2j}+s_{2j+1})/2.
\]

这是 tubelet 的几何锚，不表示其上下文感受野只有这两张帧；attention/TIA 已扩大上下文。短窗口按 valid frame 的加权均值定义锚，保留 pair 支持数；无效 native tubelet 不进入插值，单有效帧 pair 标注低支持。保持当前物理 K 槽位合同，不通过伪造末尾独立特征填空。

去掉 wrapper 最后从 192→384 的 rank 插值，改成基于真实锚的插值：

\[
\widetilde H_t=(1-a_t)H_{l(t)}+a_tH_{r(t)},\quad
 a_t=\frac{t-\tau_{l(t)}}{\tau_{r(t)}-\tau_{l(t)}}.
\]

在锚范围外采用端点保持，并把外推距离/支持质量传给 gate；不假装长空洞被线性插值恢复了真实视觉信息。索引/邻接几何 stop-gradient，特征值保留梯度。完整 detector 输入长度为 768；GT 不再压到 rank，预测不再做第二遍 rank→time，仍调用原 metadata→seconds 与原视频聚合。[C5][C12]

### 6.3 小型门控粗残差

\[
R_t=\operatorname{LN}(W_c C_t),\quad R_t\in\mathbb R^{96},
\]
\[
g_t=\sigma\{\operatorname{MLP}_{200\to32\to1}
[R_t,W_f\widetilde H_t,\gamma_t]\},
\qquad Z_t=\widetilde H_t+W_o(g_tR_t).
\]

其中 `Wc:96→96`、`Wf:Ce→96`、`Wo:96→Ce`，`Wo` 零初始化；`gamma_t` 是固定定义的 8 个标量：归一化时间、左右锚距离、包围锚间距、锚外位置指示、粗 actionness、粗 transition probability、局部有效观察覆盖率。短窗口 mask 与支持信息参与上述距离/覆盖量，不另设伪有效锚。

第一项消融固定 `g=1`，判断普通粗残差是否已足够；只有门控在相同计算合同下带来可重复优势才保留。CNN 与 Transformer 的特征无需强行同分布：投影/LayerNorm/小幅残差负责适配；不能未经实验断言它们无法融合或必然互补。

### 6.4 梯度与目标

建议初始完整设计的 coarse detection 梯度缩放为 0.25，另外用 SG(C) 的单因素消融判断增益来自现成信息还是重新学习。原始 `coarse_hidden` 路径可将检测梯度传到 CNN 与 ASFormer encoder；decoder 仍主要由 actionness 监督。原 policy replay 隔离保持不变。不同路径的共享参数梯度会相加，不把“stop-gradient policy stem”误称“所有任务都不能训练 stem”。

\[
L_{\rm main}=L_{\rm focal}(D(Z),Y_t)
 +L_{\rm DIoU}^{\rm original\ time}(D(Z),G_t)
 +L_{\rm existing\ auxiliary}.
\]

初始使用 U384，以排除政策学习的混杂；验证增益后，在同一融合结构、同一数据/更新/teacher 预算下比较 BMCR 采样。若它不优于均匀，最终主方法直接保留均匀采样。方法价值不能依赖保留 BMCR 名称。

### 6.5 训练与推理伪代码

```python
# Research pseudocode: no job is launched by this document.
# Backbone weights frozen; adapters, detector and fusion trainable.
for batch in training_source_loader:
    X, valid, gt, meta = batch
    scout = model.scout(X, valid, curriculum)
    C = scout.coarse_hidden.transpose(1, 2)  # raw encoder state, not policy-scaled

    # First attribution run: fixed uniform policy / identical replayed indices.
    S = policy.select_exact_k(scout, valid, K=384)
    rgb = hard_gather_with_existing_proxy(X, S)
    H_native = backbone_without_final_rank_upsampling(rgb, S.valid)
    tau, native_valid, support = native_tubelet_geometry(S)
    H_full = irregular_value_interpolation(H_native, tau, native_valid,
                                          query=original_candidate_grid(768))
    Z = fusion(H_full, scaled_gradient(C, coarse_grad_scale), geometry(S), valid)
    prediction = detector(Z, valid)  # fixed original 768 grid
    loss = original_time_detection_loss(prediction, gt)
    loss += unchanged_auxiliary_losses(scout, S, gt)
    assert_finite(loss)
    loss.backward()
    clip_grad_norm_(trainable_parameters, 1.0)
    optimizer.step(); scheduler.step(); successful_updates += 1
    update_ema_after_success()       # no output KD in the main experiment

@no_grad()
def infer_full_video(windows):
    all_predictions = []
    for X, valid, meta in windows:
        C, S = coarse_and_select(X, valid, fixed_final_policy)
        H, tau, native_valid = encode_actual_24_clips_and_get_native_anchors(X, S)
        Z = fusion(interpolate_to_original_grid(H, tau, native_valid), C, geometry(S), valid)
        boxes, scores = detector(Z, valid)
        all_predictions.append(convert_candidate_coordinates_to_seconds(boxes, scores, meta))
    return original_video_aggregation_and_class_soft_nms(all_predictions)
```

做架构归因时固定/重放选帧轨迹，否则 fusion 改变 policy gradients 会改变输入集合，无法把收益纯粹归于融合。最后允许一次明确标注的联合优化实验。

### 6.6 真正新增成本

原 profile 中，head/projection 从 384 恢复 768 的矩阵/卷积增加量分别约 **S 7.5804 GMAC、B 7.8069 GMAC**。上述 gate/投影增加矩阵 MAC 约 **S 0.0686、B 0.1253 GMAC**。若保留原 BMCR router、其余算子维度和 profile 口径不变，总估计约 **612.82/2046.67 GMAC**，相对官方仍约减少 **47.80%/49.35%**。[R1]

这些是由实测部件计数加新增矩阵维度得到的**设计估算**，不是已测模型结果，未计 LN、插值、索引、点算子、kernel 启动等时延。新训练时间、峰值显存、完整视频 latency 全部未知；768 长度 head 的激活增长也需实测。新增 coarse hidden 不增加 RGB 或 backbone 前向；允许梯度回 CNN 会改变反向成本。没有在本轮运行这个模型。

## 七、两个有实质差别的备选

### 备选一：同模型互补采样视图的输出自蒸馏

这是**训练目标改动**，推理不保留 teacher，不必更改重模型的 K384 或增加推理分支。

teacher 为同架构同参数族 EMA，S→S/B→B，0.999，每次成功 student 更新后更新；teacher eval/no_grad，另一视图使用相同视频、相同空间 crop、相同候选时间网格。首轮保持 K384，不使用官方 B→S。

先比较两个单损失版本，而非相加：

\[
L_{\rm clsKD}=\frac{1}{\sum_t w_t}\sum_{t,c}w_t\,
D_{KL}\big(\operatorname{Bern}(\bar p_{tc})\,\|\,\operatorname{Bern}(p_{tc})\big),
\]

或更贴近当前短板的

\[
L_{\rm locKD}=\frac{1}{\sum_m w_m}\sum_m w_m\,
\operatorname{SmoothL1}\!\left(
\frac{\hat b_m-\operatorname{sg}(\bar b_m)}{\max(|G_m|,\epsilon)}\right).
\]

ActionFormer 是 20 类独立 sigmoid，不能无依据改用跨类别 softmax KL。分类对齐先把同层概率/特征定位到同一个原时间；跨 FPN 层不要把相同 index 误当同一物理位置。定位更稳妥地按训练 GT 实例匹配 teacher/student，匹配结果 stop-gradient，而 student boxes 的物理映射保留导数。[C13][P8][P11]

`w` 同时考虑类别置信度与边界可靠性，例如复用已有 teacher views 的边界离散度和 teacher–GT 一致性；不能只看 actionness 自信。没有合格 teacher match 的实例保留原 GT 损失，不蒸馏该项。真实 GT 始终为主，不把同模型错误稳定化当“知识”。

**预算公平的起点**：先复用 BMCR 原有同一 S0+swap teacher packet，对比“只学 utility”与“只学输出”。第二个单因素实验再把 teacher views 换成 uniform/互补采样，严格匹配事件、每事件实际前向次数和 K；不是把“最多三次”当成每个方法都恰好三次。可把一次查询版作为更低训练预算的独立部署方案，但不能在论文表中装作等 teacher 成本。

```python
student_pred = student(current_view)
loss = supervised_detection_loss(student_pred, GT)
if common_teacher_event(successful_updates):
    with no_grad():
        packet = ema_teacher(common_budgeted_views_of_same_video)
    aligned, weights = align_in_original_time_and_filter(packet, GT)
    loss += kd_weight * ONE_selected_kd_signal(student_pred, aligned.detach(), weights.detach())
# normal optimizer/EMA steps; inference retains student only
```

若同预算局部互补 teacher 都不能改善验证集定位，先停输出 KD，而不是添加关系矩阵、NIG、更多 teacher 或更大预算来稀释归因。

### 备选二：非均匀 clip 中心、规则 clip 内部观察

这是**采样/骨干输入几何改动**，不是融合或蒸馏。

把 768 候选分成 48 个互不重叠的 native 16-candidate clips，选择 24 个；每个 clip 内保持原候选 stride=4 的规则顺序和两帧 tubelet，clip 中心可不均匀。完整窗口仍精确 384 个真实 RGB 输入，重骨干仍 24 clips。不要把“native16”误写为 16 张原视频连续帧。[C9][C11]

基线必须同时包含“均匀 K384 单帧采样”和“均匀 24 native clips”；后者才是该路线的采样政策控制。可以用覆盖约束、邻域不确定性和显式 clip swap，不能默认连续块选择不会伤短动作。

预训练 clip 内部运动/PE 更兼容，但 **global TIA 仍跨不规则 clip 间隔**；只换 clip 中心未解决全部非均匀性。必要时在 TIA 瓶颈将所选 native states 散射/插值到完整原生 tubelet 时间网格，执行廉价时间卷积，再 gather 回去；或加明确的 Δt-conditioned 局部核。两者是后续单因素，不与 clip 政策同时首次上线。它会增加所有 TIA 层的瓶颈计算，不能只报主 ViT 的 24 clips。[C9]

```python
cheap = scout(all_candidates)
clip_score = aggregate_per_native_clip(cheap)
selected_clips = fixed_budget_select_24_of_48(clip_score, coverage_contract)
S = expand_to_original_16_candidate_positions(selected_clips)  # no overlap
H = actual_video_backbone(X[S])
prediction = same_detection_interface(H, original_time_metadata(S))
```

最主要失败模式是短动作/边界恰在未选块或块交界处；如果“规则 clip 的表征收益”不足以抵消观察覆盖损失，停止此备选，不以更细碎多 ROI/更多 clip 修补到无法归因。

## 八、修改文件与测试合同

| 文件 | 修改范围 | 最小验证 |
|---|---|---|
| `h65/full/runtime.py` | 修复 action LR 参数身份；注册 fusion 参数、EMA state、dev split | 精确参数名/数量、无重复遗漏、warm/joint 对照 |
| `h65/full/scout.py` | 暴露 raw coarse hidden；policy 隔离不变；可选向量化伙伴计算 | coarse/action/policy 三条梯度路径；RNG replay 一致 |
| `h65/full/model.py` | native feature 输出；original-grid detector；fusion 接线 | native192/原轴768、mask、K384、无二次时间回映 |
| `h65/transport.py` | 均匀 fast path；支持 native anchors 的 value interpolation | K、唯一性、短 prefix、有效支持、hard forward 不变 |
| `h65/full/geometry.py` | differentiable query mapping 与推理 mapping 分离 | round-trip；不规则点/端点/短窗口；student 梯度 finite difference |
| 新 `h65/full/fusion.py` | 96 维粗残差及可消融 gate | `Wo=0` 等价于 fine-only 原轴模型；无效位置为零 |
| 新 `h65/full/distillation.py`（备选一） | 同模型输出 KD、原时间对齐与过滤 | teacher 梯度为零；multi-label KL；box 梯度非零；GT 不被覆盖 |
| `h65/full/utility.py`（仅政策试验） | 显式 swap 与最终集合增益评估 | 互反条件、符号、目标集合和实际执行集合一致 |
| `tools/full_train.py` | 记录成功更新、实际 teacher 次数/预算；模式开关 | 同数据顺序、同更新、无意外额外 teacher/dense 路径 |
| `tools/full_eval.py` | 新网络入口与真实成本分解，不动评测协议 | 211 IDs、全部窗口、同 NMS、真实 clip hook、去计数器计时 |

主方案无需改 vendored `upstream/` 源码，只在 wrapper 的 config 副本调整最后插值和 detector max_seq_len。只有备选的 true-time TIA 可能需要明确的 adapter 派生类，不能偷偷改官方基线骨干。

## 九、最小、可解释的消融与停止条件

### 9.1 数据和优化先锁定

在原 200 训练来源视频中登记 fit/dev，例如 160/40，按视频而非窗口切分，并检查类与时长覆盖。当前 warm 已见过全部 200，不能直接拿其中 40 当干净的模型开发验证集；新的调参比较必须从 fit-only 初始化轨迹训练。为保持成功更新合同，在 160 视频上比较固定 2000+4000 更新，不强称仍等于 20+40 epoch。

超参数与最终模型选择冻结后，研究模型使用全部 200 训练来源、同预训练、同成功更新；最终对固定 211 视频完整测试。官方 checkpoint 仅作外部锚点，本轮与计划都不要求重新完整训练官方 AdaTAD。不得用测试 211 的子集调参，也不得用最优测试 epoch 替代末轮 EMA。[C3][C11][C14]

修 LR 后同结构比较共享纠正后的 warm；跨结构比较需明示初始化迁移，逐项核对权重与 shape-dependent buffers，不能用宽松加载静默跳过不兼容参数。固定 seed 仍不足以保证相同 augmentation，因为不同分支消耗不同 RNG，需按 `(seed,video,crop/update)` 确定输入增强，或存轻量采样轨迹。teacher 预算用实际 forward 数×每视图 MAC/帧数计，而不是只写“每8步”。主结果至少三种子；视频 paired bootstrap 与种子方差分别报告，二者不能互相替代。

### 9.2 分阶段矩阵，而不是所有模块一起上

| 编号 | 唯一变化 | 回答的问题 |
|---|---|---|
| U | 修复配方；K384 真均匀；同6000成功更新 | 是否连均匀同预算都没超越？ |
| B | 修复后的 BMCR；原 rank head；原 teacher 合同 | 整体方法在记录额外训练预算后有什么增益？ |
| B→U（仅推理干预） | 同一个 B checkpoint 强制 uniform；参数、训练历史和 teacher 完全相同 | 已训练模型上，学习的推理集合是否真的优于均匀？ |
| G | 固定与 U 相同的 RGB；native192→原轴768；无 coarse fusion | 恢复检测网格/目标能否改善？ |
| F | G + 粗残差，先不加 gate | 已计算粗信息是否真的有用？ |
| Fg | F + 小 gate；其余不变 | 是否需要条件融合，还是普通残差足够？ |
| Kd | 固定同一结构、数据、teacher packet，只把训练信号换成一种输出 KD | 自蒸馏是否优于相同额外预算的 utility？ |
| C24 | 规则 native clip 输入；与均匀24 clip直接比较 | 预训练结构兼容能否胜过细帧观察覆盖？ |

**teacher 预算的比较边界**：U/G/F/Fg 首轮全部 teacher=0；B 与独立训练 U 的差异不能单独归因为采样，因为 B 还有额外 teacher。B→U 与 B 的推理干预使用同一 checkpoint，teacher 预算严格相同且无需追加无效 teacher 前向，隔离的是已训练模型上的推理政策，不代替纯均匀训练基线。Kd 与对应 utility 版本使用同一实际 teacher 查询合同；若进一步声称端到端学习政策的纯收益，应把相同 teacher 输出监督施加于均匀和学习采样两组，再做成对比较。不能把不同 teacher 预算的行标为同成本。

F 的 coarse SG/live-gradient 对比仅在 F 有信号后做；clip 的 true-time TIA 修复仅在 C24 表征问题诊断后做。最后只组合已证实互补的胜出项，不把全排列训练作为“研究充分”。U 的归因版本保留同样的 scout 辅助训练/开销；最终 uniform 部署可合法删去不再使用的 policy-only 部件，单独报告真实 latency，不能把这两个 U 混在一行。

### 9.3 必要读数与可视化

报告五个 tIoU 与平均值、短动作≤3秒 recall/@0.7、按长度/空洞分组的 start/end 误差、每个 GT 边界至最近重观察的距离；同时画全视频时间轴上的粗预测、选帧、真实时间间距、最终检测、GT。热图必须来自预先登记的完整视频，不只挑成功片段。

utility 诊断用 heldout 视频的真交换标注：正负号准确性、gain 排序、收益 regret、S0→S1 真整体收益，并给视频级区间；相比相关系数，更重要的是“按它决策是否比随机/覆盖受限交换减少真实损失”。若 F 的增益值得进一步归因，再加同参数时间打乱 coarse 控制：若打乱也一样好，应怀疑容量/优化效应，而非时间互补。

计时要同卡、同 batch/精度、交错次序、warmup、CUDA 同步；分解 decode、resize/H2D、CNN、ASFormer、固定 K 解码、条件伙伴、gather、heavy、detector、NMS。报告完整/短/尾窗与按真实视频组成加权的时间；补充完整端到端视频吞吐与 p50/p95。allocated/reserved/训练含 EMA 峰值分开。不能以 GMAC 比率替代任何一项。

### 9.4 停止与改道

**代码门槛**：LR 分组、mask、native anchor、学生物理坐标梯度与评测 ID 任一失败，先修复，不进入性能结论。

**政策门槛**：同日程 U384 不劣于 BMCR，且终点 holdout 决策增益不能稳定胜过无信息/覆盖控制，则删除条件效用头，停止扩 MLP 和继续调 gain。现有预热代理低相关只提示风险，不代替终点门槛。

**融合门槛**：G 有效但 F 无效，就保留原轴定位，放弃“粗细互补”论文主张；F 有效但 gate 无效，就用普通残差。不要为了宏大故事保留没贡献的机制。

**蒸馏门槛**：等 teacher 预算下边界未改善、分类冲突增加或只在更大 teacher 预算下有效，则不把 KD 当主贡献。允许报告它是额外训练预算换性能，但不能混入零成本上限保持。

**clip 门槛**：规则 clip 改善表征却让短动作/边界覆盖更差且总体不占优，停止该路线；不预设块采样比帧采样更优。

**效率与保持门槛**：先定义业务/论文的非劣界，例如提出 Avg 和 @0.7 都不劣于官方超过 1 pp，作为待登记的“保持”定义；这不是预期分数。实际加速必须在主部署口径下快于同硬件官方，且差值超过重复测量波动。若 MAC 下降而端到端仍慢，就只能报告 arithmetic-efficient，不能写 accelerated。为补精度不断增加训练/推理模块仍无 Pareto 改善时，应改道到更简单的均匀稀疏观察+原轴检测，而不是继续包装 BMCR。

## 十、最终科学立场

当前最值得保留的是“真实减少高成本观察”的执行事实，而不是边界条件效用已经有效的主张。最值得补的是被计算后丢弃的全时粗证据和定位度量的一致性。先修真实 bug、补均匀对照，用简单融合检验上限，再决定是否需要自蒸馏或结构兼容采样。高水平贡献需要的是可证伪的问题、因果可归因的改进和真实系统收益；本文不预测任何 mAP。

## 附：原始来源与固定源码链接

以下 C/R 均固定用户指定提交；X 为已读取的外部作者实现，外部仓库不属于本次固定审查快照。外部代码入口没有被声明为与用户源码同样完整的审计对象。部分 CVF 页面再次读取返回 403，已通过 arXiv/作者代码核对可取得的信息；MSDR-Mamba 的后续页面请求被限流，未取得其源码；不把未访问内容列为已读。

[C0]: https://github.com/yuzbo/BMCR-T-AdaTAD/blob/15280e5dc29e3df18d085aaba21067e04107ce84/docs/METHOD.md
[C1]: https://github.com/yuzbo/BMCR-T-AdaTAD/blob/15280e5dc29e3df18d085aaba21067e04107ce84/h65/full/model.py
[C2]: https://github.com/yuzbo/BMCR-T-AdaTAD/blob/15280e5dc29e3df18d085aaba21067e04107ce84/h65/full/runtime.py#L89-L108
[C3]: https://github.com/yuzbo/BMCR-T-AdaTAD/blob/15280e5dc29e3df18d085aaba21067e04107ce84/tools/full_train.py
[C4]: https://github.com/yuzbo/BMCR-T-AdaTAD/blob/15280e5dc29e3df18d085aaba21067e04107ce84/h65/transport.py#L21-L115
[C5]: https://github.com/yuzbo/BMCR-T-AdaTAD/blob/15280e5dc29e3df18d085aaba21067e04107ce84/h65/full/geometry.py
[C6]: https://github.com/yuzbo/BMCR-T-AdaTAD/blob/15280e5dc29e3df18d085aaba21067e04107ce84/h65/full/utility.py
[C7]: https://github.com/yuzbo/BMCR-T-AdaTAD/blob/15280e5dc29e3df18d085aaba21067e04107ce84/h65/full/scout.py
[C8]: https://github.com/yuzbo/BMCR-T-AdaTAD/blob/15280e5dc29e3df18d085aaba21067e04107ce84/references/ASFormer/model.py
[C9]: https://github.com/yuzbo/BMCR-T-AdaTAD/blob/15280e5dc29e3df18d085aaba21067e04107ce84/upstream/opentad/models/backbones/vit_adapter.py
[C10]: https://github.com/yuzbo/BMCR-T-AdaTAD/blob/15280e5dc29e3df18d085aaba21067e04107ce84/upstream/configs/adatad/README.md
[C11]: https://github.com/yuzbo/BMCR-T-AdaTAD/blob/15280e5dc29e3df18d085aaba21067e04107ce84/upstream/configs/_base_/datasets/thumos-14/e2e_train_trunc_test_sw_256x224x224.py
[C12]: https://github.com/yuzbo/BMCR-T-AdaTAD/blob/15280e5dc29e3df18d085aaba21067e04107ce84/upstream/opentad/models/utils/post_processing/utils.py
[C13]: https://github.com/yuzbo/BMCR-T-AdaTAD/blob/15280e5dc29e3df18d085aaba21067e04107ce84/upstream/opentad/models/dense_heads/anchor_free_head.py
[C14]: https://github.com/yuzbo/BMCR-T-AdaTAD/blob/15280e5dc29e3df18d085aaba21067e04107ce84/tools/full_eval.py
[R1]: https://github.com/yuzbo/BMCR-T-AdaTAD/blob/15280e5dc29e3df18d085aaba21067e04107ce84/phase2_20260910/comparison.json
[R2]: https://github.com/yuzbo/BMCR-T-AdaTAD/blob/15280e5dc29e3df18d085aaba21067e04107ce84/phase2_20260910/TRAINING_SUMMARY.json
[R3]: https://github.com/yuzbo/BMCR-T-AdaTAD/blob/15280e5dc29e3df18d085aaba21067e04107ce84/phase2_20260910/runs/s_bmcr/config.json
[R4]: https://github.com/yuzbo/BMCR-T-AdaTAD/blob/15280e5dc29e3df18d085aaba21067e04107ce84/phase2_20260910/runs/s_audit/review.json
[R5]: https://github.com/yuzbo/BMCR-T-AdaTAD/blob/15280e5dc29e3df18d085aaba21067e04107ce84/phase2_20260910/runs/b_audit/review.json
[P1]: https://arxiv.org/abs/2311.17241v2
[P2]: https://arxiv.org/html/2412.11228v1
[P3]: https://arxiv.org/abs/1912.08193
[P4]: https://arxiv.org/abs/1611.07715
[P5]: https://aclanthology.org/2023.emnlp-main.309/
[P6]: https://arxiv.org/abs/2111.14330
[P7]: https://arxiv.org/abs/2307.12612
[P8]: https://openaccess.thecvf.com/content/ICCV2021/html/Xu_End-to-End_Semi-Supervised_Object_Detection_With_Soft_Teacher_ICCV_2021_paper.html
[P9]: https://openaccess.thecvf.com/content_ICCV_2019/html/Zhang_Be_Your_Own_Teacher_Improve_the_Performance_of_Convolutional_Neural_ICCV_2019_paper.html
[P10]: https://proceedings.mlr.press/v202/baevski23a.html
[P11]: https://openaccess.thecvf.com/content/CVPR2024/html/Wang_CrossKD_Cross-Head_Knowledge_Distillation_for_Object_Detection_CVPR_2024_paper.html
[P12]: https://openaccess.thecvf.com/content_ICCV_2019/html/Cho_On_the_Efficacy_of_Knowledge_Distillation_ICCV_2019_paper.html
[P13]: https://arxiv.org/abs/1912.03663
[P14]: https://arxiv.org/abs/2504.19581
[P15]: https://aclanthology.org/2021.acl-long.508/
[P16]: https://aclanthology.org/2021.acl-long.16/
[P17]: https://arxiv.org/abs/2203.11139
[P18]: https://www.mdpi.com/2079-9292/15/17/3797
[P19]: https://github.com/A-suozhang/ada3d
[P20]: https://arxiv.org/abs/2010.04159
[X1]: https://github.com/facebookresearch/detectron2/blob/main/projects/PointRend/point_rend/point_features.py
[X2]: https://github.com/microsoft/SoftTeacher/blob/main/ssod/models/soft_teacher.py
[X3]: https://github.com/itailang/SampleNet/blob/master/registration/src/samplenet.py
[X4]: https://github.com/clovaai/length-adaptive-transformer/blob/master/length_adaptive_transformer/modeling_bert.py
[X5]: https://github.com/fundamentalvision/Deformable-DETR/blob/main/models/ops/modules/ms_deform_attn.py
[X6]: https://github.com/msracver/Deep-Feature-Flow

其他作者入口：[Uni-AdaFocus](https://github.com/LeapLabTHU/Uni-AdaFocus)、[Sparse-DETR](https://github.com/kakaobrain/sparse-detr)、[Focus-DETR](https://github.com/huawei-noah/noah-research/tree/master/Focus-DETR)、[IA-SSD](https://github.com/yifanzhang713/IA-SSD)、[Ada3D](https://github.com/A-suozhang/ada3d)、[SAMBLE](https://github.com/stevenczwu/SAMBLE)、[TOKEE](https://github.com/LeeSureman/Sequence-Labeling-Early-Exit)。

独立检查随附 `independent_checks.py` 与 `independent_checks.json`：只复核分组谓词/参数计数、单调映射 IoU 反例、结果差值与设计 MAC 估算，不是模型单测或性能复现。
