# 跨领域文献：按H65的真实瓶颈排序

检索截止2026-09-14；论文标识/一手来源在 `../references.json`。以下“可借鉴”是本项目研究建议，不是原论文已经在TAD中验证的结论。

## A. 当前最高优先级：算子伤害与替代路径

### A1. SSA: Sparse Sparse Attention by Aligning Full and Sparse Attention Outputs in Feature Space（arXiv:2511.20102）
**原问题。** LLM稀疏注意力微调中的full/sparse输出偏移与梯度更新问题。方法在每层同一输入上计算不同注意力模式的输出，并进行双向对齐；非主路径不向下一层递归。
**H65价值。** 现有同支持teacher仍然沿另一条完整轨迹演化，并非在当前学生同一个pre-operator输入上评价heavy/light替代。对后者做局部函数监督能把“累积状态偏移”和“当前算子近似误差”分开。
**最小改动。** 先针对FFN-light：`target=heavy(current_normed_input)`，只采部分token计算teacher；light分支接受残差匹配，GT任务损失始终保留。再对attention full/sparse做相同实验。
**限制。** 当前QKV与heavy FFN大多冻结。不能把双向loss写成“更新所有冻结权重”，也不能照搬SSA对梯度缺失的解释到H65，因为H65 full-KV、TIA仍能传梯度。该论文是预印本；不假设其TAD有效。

### A2. Router-Tuning / MindSkip（arXiv:2410.13184）
**原问题。** LLM动态深度的训练成本和关键层跳过风险；区分注意力模块的动态执行。
**H65价值。** 当前admitted同时改变attention query与heavy FFN。应先分离这两个决定，再讨论“深度冗余”，而不是推断所有block可同等跳过。
**最小实验。** fixed support下query-only、FFN-only、coupled三种干预。保留global TIA。不预设哪个更安全。

### A3. MoNE: Mixture of Nested Experts（NeurIPS2024，arXiv:2407.19985）
**原问题。** 图片/视频token分配到不同计算量的嵌套专家，使便宜计算仍产生有效表示。
**H65价值。** 从“跳过”转向“可用的低成本替代”。但V2已有light；新的研究点只能是参数共享/嵌套结构、局部函数监督、替代误差对真实任务的影响，而不是重复添加light。
**最小实验。** 当前light vs同输入蒸馏light vs共享/嵌套FFN低容量实现，固定路由mask与总预算。

### A4. LayerSkip（ACL2024，arXiv:2404.16710）与Once-for-All（ICLR2020，arXiv:1908.09791）
**原问题。** 训练中建立早退出/多容量子网络的有效性，而非部署时任意截断。
**H65价值。** 比较现有15plan轮换与预注册渐进容量课程，避免突然同时改变T/D/S分布。现有全分支GT、混合plan、support KD已存在，新增贡献应是明确的课程因子，而非泛泛“加sandwich”。
**限制。** 训练步数、LR horizon、辅助分支曝光、总查询量都必须对齐；LLM自推测解码不是TAD的直接实现。

### A5. A-MoD（arXiv:2412.20875）与原MoD（arXiv:2404.02258）
这是理解现状的必要对照，不是本轮新增模块。A-MoD用前层attention排序且不引入新可学习router；原MoD强调固定capacity与学习式token分配。H65当前局部D/S评分接近前者，真正可研究的差异是task-value身份分配。

## B. 定义和低成本测量真实计算价值

### B1. EDDI（ICML2019，arXiv:1809.11142）
从医疗检测等有成本的信息获取问题借鉴“基于当前已知信息的下一次获取收益”。H65动作价值必须条件于已有frame、空间质量、深度状态与预算。不需要照搬Partial VAE；原论文的期望信息增益也不能直接等同于TAD loss差。

### B2. SAGE（NeurIPS2020，arXiv:2004.00668）
以预测能力度量特征贡献并考虑交互。比用Data Shapley直接套训练样本价值更贴近这里的输入/执行动作。建议在小型动作集合上做随机coalition和pair interaction：单个删去无损并不说明多个一起删也无损。

### B3. ROAR（NeurIPS2019，arXiv:1806.10758）
删除后重新训练的解释性评价，提示冻结模型删除敏感性混合了分布偏移。H65要分别测冻结执行损失、固定支持的短适配、完整配方匹配训练，不用其中一个替代另一个。

### B4. Taylor Importance Estimation for Neural Network Pruning（CVPR2019，arXiv:1906.10771）
用任务loss的梯度近似删除/替代效应。H65中可用cheap状态上的`-grad dot (heavy-light)`预测增加heavy计算的收益，但必须用heldout真实重执行值检查相关性与regret。梯度在dense/cheap基点不同，符号和解释不同。这个代理不能被称为oracle。

## C. 非规则时间与关系访问

### C1. mTAN（ICLR2021，arXiv:2101.10318）
临床非规则时间序列的连续时间嵌入与reference-time注意力，是Cross/Graph的强简洁对照。原轴query按真实时间向selected anchors读取，保留coverage与validity；不按packed rank插值。

### C2. TadTR（TIP2022，arXiv:2106.10271）
TAD已经存在少量时间采样点的deformable attention。对H65，reference-relative temporal sampling可作为Graph referral的竞争解释：需要多跳地址传播，还是直接学习少量时间偏移已足够？不能宣称首次稀疏时间访问。

### C3. Graph Machine（arXiv:2609.02881v1，2026-09）
保留线性数量的节点状态，动态访问少量节点，边索引负责地址、边权承载可微质量。优先借鉴状态/访问分离和受限referral，而非整体替换H65。当前仓库已是GM-inspired简化实现，full-KV投影仍存在、degree16写入合同。
**最小实验。** 同selection/masks/head：Cross、continuous-time、fixed local、多尺度static、dynamic无referral、dynamic有referral。尚无独立收益时不绑定RISE。

### C4. NSA（ACL2025，arXiv:2502.11089）
硬件对齐的层级稀疏attention。H65只有在访问机制有效后，才研究固定块、批量gather、复用KV和融合算子。完整矩阵FLOPs减少与排序/索引/GPU延迟必须分账。

## D. 教师来源与轨迹优化

### D1. RISE（arXiv:2609.05295v1，2026-09）
RLVR提供有依据的方向，历史/当前位移构造外推教师，蒸馏而非直接采纳外推模型。迁移到H65时，缺失的监督是计算行为的收益，不是动作GT标签。最先做真实固定动作库上的future-value forecast，再做冻结detector的router训练，最后才考虑在线递归。
**必须控制。** current、真实EMA、lagged snapshot、future、等额外优化；β=1是current；future效果不由PCA保证；所有checkpoint收益用共同物理支持和共同单位。

### D2. Meta Pseudo Labels（CVPR2021，arXiv:2003.10580）
teacher可通过student在有标签数据上的表现反馈改进。借鉴的是“教师是否实际帮助学生”的判据，而非直接为H65添加昂贵二阶元学习。先独立development检验未来教师是否减少真实routing regret、是否改善mAP；不通过就回退current/EMA。禁止拿官方test拟合teacher。

### D3. Mean Teacher / Snapshot / SWA
作为RISE的稳定性和额外优化控制，而不是三个并列新贡献。当前H65已有EMA与full self-feature；新增teacher试验必须明确到底改变teacher参数、输入支持、目标对象还是数据分布，不能全改后归因于“未来”。

## E. 组织方式与优先级

Uni-AdaFocus（TPAMI，arXiv:2412.11228）提供廉价预览—昂贵局部计算的结构借鉴与跨维度实验组织。其三个维度是空间、时间、样本级，并不是H65的时间、空间、网络深度，且当前H65已经有Scout/Cross。借鉴它的实验组织，不重复声称新增原有架构。

第一批：A-MoD审计 + MindSkip式算子拆分 + SSA式局部对齐 + EDDI条件价值；并行做不依赖Graph的RISE最小检验。
第二批：MoNE/LayerSkip/OFA建立有效低成本路径；SAGE/ROAR检查集合与适配因素。
第三批：mTAN/TadTR/GM定位恢复机制；需要真实系统加速时再做NSA式实现优化。

任何外部论文均不直接证明H65三轴压缩能保持TAD精度。以上优先级是基于当前失败模式的研究判断，不是性能预测。
