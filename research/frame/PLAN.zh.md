**FPW：H65/BMCR单帧选择后的原时间回写与A-MoD三维计算，全面并行推进**

用户最新明确：以总计算量和最佳五阈值mAP为主要判断；延迟、显存、端到端成本照常报告，不用延迟否决路线。各阶段全面实现、独立初始化并行部署，不等待前一路线mAP胜出。唯一启动依赖是生产接口/真实GPU梯度/实际压紧正确性，不设任意精度门槛。

保留当前BMCR80训练；B01由该已部署实验承接，不重训warm、不重新提交joint40。FPW源起点add94ec（模型主体继承239d098，加入已验证BMCR80和DS3取消状态）。以已完成修正H65-S60和旧BMCR-B60作为固定强锚点，明确旧/新配方，不等待尚在训练的BMCR80才启动新路线。

所有时间选择仍为单candidate frame。K384由选择帧重新配对为native192，不是原native384的子集；新的384物理时间query读取这些带来源的anchors。R01为独立的插值+official original768检测头零训练基线，不能声称等于旧rank头。R02瓶颈TCN与R03独立cross-query decoder从零残差出发；R04去掉检测GT loss；R05输出Bernoulli/距离KD和差分分别独立；R07来源/scout分别消融。

深度主方案按用户指定采用A-MoD（Attention Is All You Need For Mixture-of-Depths Routing，arXiv2412.20875v1）：第2/4/6/8/10层为MoD，首层、末层dense；前一dense层incoming attention对heads及有效query取均值，按capacity选token，不乘分数缩放输出，无新可训练深度router。主方案打包所选token的QKV与FFN，未选token保留identity贵路径状态，再在完整空间grid执行globalTIA(K/2)。保留TIA是本项目TAD适配。另设uniform路由、selectedQ/fullKV和静态保首末层drop等控制。50%和12.5%capacity同时部署。

SDPA不能返回attention maps时，复用已经投影的Q/K分块计算scores。额外QK真实计入FLOPs，不能宣称免费。空间FFN token/2x2 tile、规则128、独立selectedQ/fullKV均作为独立支线；联合支线从同锚点启动并训练混合K/门预算，不等单维实验获胜。实际dense-mask与compact使用相同K/V与固定mask进行数值核对。

首批32个配置：2个零训练R01评测，30条独立训练支线；包括恢复/监督/来源、A-MoD及其dense-mask训练控制、空间、效用、联合、K320/K256和主恢复器的另外两个种子。每支固定20epoch、全部200训练视频，batch1并累积2次形成100 optimizer updates/epoch，以保持包括K768在内的同一有效batch与内存口径。所有支线同样按microbatch loss规范训练，不把它称为与旧batch2损失归一化逐位等价。

新增模块基础LR1e-4，Adapter LR1e-5，100步warmup后余弦衰减，EMA0.99用于这些短程新模块实验；既有BMCR80仍EMA0.999不变。每5epoch全211视频/792窗口测试EMA，并在terminal评online；全测试峰值选模和选择范围均披露。初始20轮是明确的新模块完整数据课程，不是宣称相当于BMCR80训练成本或已充分收敛。

新调度器最多8个自有FPW活动/排队作业，最多6个长训练，给评测保留流动槽位；总账户16作业及站点GPU限额仍由Slurm约束。与BMCR80控制器并存。若实际排队/缺GPU，已获用户条件授权联系指定PVR任务；只有其未达原目标且未超本任务可比性能时，核对当前job身份后协调让位。PVR的两个旧GPU收尾已自行完成，无须取消；其较低FLOPs结果不被误写为严格科学支配。

R06官方decoder初始化需要实际可读、匹配key/shape的完整预训练权重；当前已有K400权重的decoder keys待核验，不以encoder-only冒充decoder。G02第二数据集需实际视频/标注/对应teacher资源；这些外部条件不会阻塞任何可运行的主路线。PBD-style进步删层、真实干预与完整分析接口继续接入，不能用占位脚本伪称已运行结果。

论文/结果判断依据是实际完整模型2MAC矩阵/卷积FLOPs与最佳mAP的前沿；单seed、测试选模和不同训练/teacher成本明确报告。模型延迟和E2E单列，不作路线淘汰门槛。原包未测结果保持null，所有图取自真实记录。
