# RFV-T 当前执行计划：反向一致性 Mini

2026-09-15。本文是 Fast Sprint v3 的当前增量执行命令。RFV-T 优先，D/S 保留旧课程，Atlas 收尾，Raw 暂缓扩张；独立分支为 `codex/wtr-rfv-20260915`。

## 证据与时间顺序

三份新审阅文件的代码审阅基点是 6d1f924。随后完成的 b644d87 Value-R1 结果必须保留，不能把本轮写成发生在 R1 之前，也不重跑 R1。审阅中的建议属于待验证方案，不是新的实验结果。

- 当前部署 ≤4 轮、每轮 ≤16 候选的 T-local-CF，在 cal20/46 windows 的 mAP 增量为 +0.528413 pp，视频配对 95% CI [+0.082440,+0.714687]。这是 GT 辅助有限搜索的机会证据，不等于廉价输入能预测这份机会。
- b644d87 的 R1 共 18 个 head 已完成。full inner10 的 R1−R0 regret CI 跨 0；原 within 8/8 的 R1 regret 为 0.001598842，R0 为 0.001519199，STOP 为 0.001413537；R1 held Spearman 0.00540。`VALUE_R1_GATE=LEARNABILITY_FAIL` 不改写。
- R1 fit Spearman 约 0.944，说明可拟合已见数据。原 fit200 全部处在 tubelet 局部位置5，held200 为位置1/2；这是执行位置族外推，不能将其混称为普通同位置插值，也不能从现有结果唯一归因到某个表示缺陷。
- 旧 G1 和历史 RISE-B0 未过科学 gate。实际 value drift 已测到，不意味着 future 选择已改善。所有 T/Graph/FVD 长训继续锁定。

## 唯一新增的 Value 结构实验

固定原 detector、checkpoint、增强、D/S/Cross、loss normalizer 与一次交换后的完整执行函数：

`S′ = S − i + j; g+ = L(S) − L(S′); g− = −g+`。

正向输入为 `phi(C,S,i→j)`；反向输入必须为 `phi(C,S′,j→i)`。在 S′ 中 j 是 remove、i 是 insert，因此两项 membership 仍为 (1,0)。交换端点 hidden/时刻/来源/actionness/transition，signed distance 取负，四个 gap 按 S′ 重算，support mean 以有效 K 更新，global mean 不变。复用完整407及公开物理元数据，不用 cheap192 近似重建新 hidden。

先做 CPU 合同与真实 GPU 合同：少量原 fit 样本，包含完整与短窗；最多6对，每对独立执行 S、S′、S′、S 四次。检查支持集往返、候选合法性、原407复现、反向407对完整廉价视图的误差、归一化后误差、两个 loss 分量的正反残差、原标签复现和 normalizer 固定。no-op 是独立同状态重放；不把 i=j 加入合法交换域。

技术门通过后只拟合两组共享 Plain-M，407→128→64→2：

|读出|训练|推理|
|---|---|---|
|P0|复用 b644d87 的原 within Plain-R0|f(x+)|
|P1|正反两个 component-normalized Huber 的平均|f(x+)|
|P1-bi|直接复用 P1 同一个 raw snapshot，零额外更新|(f(x+)−f(x−))/2|
|P2|直接对差分输出作原 Huber|(f(x+)−f(x−))/2|

P1、P2 均 fresh zero-init，同42/43/44 seeds、AdamW、2000 updates、state batch 顺序、原 fit8 normalization、原200个独立正向标签。反向标签是派生监督，不扩大独立样本数。P1/P2 都训练两次 head forward；P1-bi/P2 推理也增加反向构造与第二次 head forward，分别计账。P2 的共享最终 bias 相消，不声称其梯度等同 P1。

不同时加入 JS、新 descriptor、Graph、RISE teacher、额外步数、温度或 STOP 搜索。原正向归一化保留；反向 signed_distance 实际可能为 −65 至 −75σ，记录数值幅度、有限梯度、fit 表现。非有限/明确优化故障报技术失败，不包装成科学负结论，也不自动扩大归一化网格。

## 评价与约束

沿用原25 state 的物理排序交替8fit/8held。主评价 held8；fit8 与原 calibration 只做描述性读出，不做选择。此次不再读取 inner10，正式 outer20 继续封存。

Primary：P2−P1-bi 的 raw cls+loc regret，视频配对10k bootstrap和逐 seed 方向；同时报告 P1-bi−P1、P2−P0、P2 对 STOP/随机合法交换、chosen gain、负收益执行率、NDCG、Spearman。STOP=0、零分 tie 选 STOP。结构信号 PASS 要求 P2−P1-bi regret CI 上界<0、三 seed 同向且优于 STOP/随机控制；它只支持本 mini，不直接解锁任务训练。

原8/8执行位置差异保持并显式报告；不更改旧 split 和 R1 FAIL。此前共同 held4 的覆盖拟合、partner96 表示修改暂缓，不能与本轮同时启动。

## 两条并行诊断

**D/S 固定 checkpoint。** 先回收原 train.jsonl 已记录的 clip 前 global grad_norm 与 milestone。定位原405课程的准确 checkpoint 后，复制到独立诊断，不修改课程；同一 Value checkpoint、同一数据/RNG/容量/recovery，仅切 Value/Uniform 路由。分别记录实际任务差、重/轻路由差、per-loss/group 梯度与 global clip 系数。训练入口实际为 bf16 autocast，没有 GradScaler；诊断匹配该事实，不臆造 unscale 步骤。不执行 optimizer.step，不把换 checkpoint 的差异称为纯路由效应。

**RISE 既有 B0 决策导出。** 复用 d62ea55 的真实 anchor/post/累计EMA snapshots，同 s40/candidates，β=1.1 冻结。导出原始分数、STOP、top2 margin、外推位移、argmax/排序/实际选择收益变化，逐视频复现旧六项指标。零新拟合、零 CF、零 β 重选。这是解释历史 B0 的诊断，不是新 T-RISE/FVD PASS。

## 资源与输出

- 4090：四条原405 D-V/D-U/S-V/S-U 继续，不抢占，不增变体；CPU 记录和新代码合同并行。
- AutoDL：使用已交还的 GPU0 做反向真实合同，成功后6个短 head；同队列运行旧 RISE 导出。GPU1 的 Atlas interaction_v1_s 已于19:10完成，后续资源以 owner 最新回执和实际占用为准。
- A100：部署同一不可变版本并执行独立技术核验；需要新 CF/固定 D/S GPU 诊断时以单独 Slurm 作业承载，不等待旧课程，也不重复 AutoDL 正在做的同一实验。
- Atlas：原S已完成GPU测量；另行授权的 interaction_v1_s 792窗已完成，CPU AP/bootstrap收尾，与旧 Atlas science 分开记录。B暂停，不新增科学分支。
- Raw：仅 official/off-grid coverage开发诊断，不扩6400、不长训。

本轮必须产生 `REVERSE_CONTRACT.json`、`REVERSE_MINI.json`、`RISE_CHOICES.json`、D/S诊断结果及真实 launch/finish receipts。先保存结果与日志，再做独立交叉审核，更新 ledger 和 GitHub。已完成实验不因状态文件过时重启。

## 最终路线仍待证明

完整训练路径可实现与科学有效性是两件事。当前证据支持 T 有有限机会，不支持廉价 Value 已稳定泛化。Graph 是否增益、RISE 是否改善未来选择及最终任务 mAP/成本都未成立。只有 Plain learnability、完整视频技术准入和相应机制门通过，才恢复 matched80轮；epoch80永久primary，official test不用于模型选择。
