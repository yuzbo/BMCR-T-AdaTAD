# 实施 owner：诊断与正式 WTR 的边界

2026-09-15。已完整阅读8bef34df用户附件，并亲自点验transport、FormalScout、FormalH65和NativeEncoder原代码；只读探子核验S/D执行范围。此回复是设计讨论，不修改在跑配置或启动新的正式课程。

**我同意双线分离。之前将局部learnability门扩展到所有T模型，范围过宽，应纠正。** 当前407D/Uniform邻域/最多4交换的负结果只约束这组局部预测机制；不能否定有不同选择空间、表示与task梯度的全轴模型。反过来，全局任务学习有效也不能自动证明Value监督有贡献。

## 四层分类

|对象|层级|当前结论|
|---|---|---|
|Atlas、T-local、Raw mini/coverage、407D R1/Reverse、离线G1、历史RISE、D/S固定诊断|诊断与测量|结论限于具体输入、状态和选择域|
|D-V/D-U/S-V/S-U 80轮|完整训练的组件消融|是真实完整TAD课程；固定T/配额下的组件对照，尚未完成80 endpoint|
|历史BMCR完整训练|历史完整方法与对照|具有全轴选择能力；不能替代当前Paper框架matched基线|
|T-Global、Global-TaskValue、全轴Graph/FVD、逐步S/D与联合|正式论文候选|需独立登记、实现和训练；目前未证明|
|现有局部T-V-G/F/GF|旧选择域下的待验证候选|不得与新的Global版本混为同一结果|
|已证明最终模型|当前为空|没有新RFV组合取得完整matched endpoint和成本证据|

## T：恢复全轴机制，保留当前定位执行

FormalH65.route在 h65/full/model.py:45–57 确实执行rate→S0→condition→全轴sample_rates。h65/transport.py:52–102是校准、平滑、coverage floor后的systematic exact-K，仍有采样归纳偏置，不能称为穷举任意子集。一次重采样可改变大量帧，16-cell限制的是conditional partner描述，不是最终支持只能局部变化。

当前Paper也已有关键接线：h65/paper/encoder.py:73–96在selector=anchor、encoder.variant=bmcr时做两次全轴采样，prepare调用gather_with_transport。variant目前由资源source.get('variant','h65')提供；需明确到新recipe，避免改共享resources影响旧模型。旧D/S课程selector=uniform、train_scout=false；这条学习路径没有启用。

首版T：Standard全768合法候选，K384先作预算点，D100/S100与当前physical-time Cross/readout固定。启用BMCR条件分配和train_scout，关闭额外局部frame_router.refine。保持真实selected RGB→16观察pack→tubelet contributors→物理anchors/queries→Cross→原轴TAD head。

transport.py:105–125保持hard RGB前向，通过continuous coordinate与detached邻居RGB slope传代理梯度；encoder.py:96训练且train_scout时bridge_weight=.25。它不是温度，也不是整数索引的精确梯度。旧FormalH65在full/model.py:85–88把GT映射到压缩rank；不能把整套旧detector覆盖当前原轴恢复接口。

新T技术准入要验证：全轴分数可改变超过4帧；有效K/去重/短窗正确；未选RGB不进heavy；task梯度真正到达rate/conditional层，关闭bridge后该路径消失；保存重载、完整视频所有窗→JSON→指标通过。然后可以做自己的matched80轮，不等旧8/8通过。

## S/D：部署是全域打分，但当前预算约束仍局部

我同意“全栅格轻状态，逐层分配昂贵attention/FFN，D允许重入”的正式Core取舍。现有D/S部署本来就对合法token整体打分，单次CF只是监督采样，不代表部署仅改变一个token。

不过 model.py:143–146 当前固定层4/6/8/10、fullKV；operator_value.py:26–50的D为每pack整数quota，S依据admitted比例largest-remainder分给8个native-time再各自选择。这些是当前executor/实验约束，不能全部提升为WTR定义。

D不是单调early-exit：engine.py:153–174是重/轻token路径，后层可重入。有效计算深度可按实际heavy路径次数表述；fullKV投影仍保留，不冒称整层省掉。S是patchify后heavy/light FFN token分配；mask可形成不规则区域，但并未实现原RGB任意ROI裁剪或高分辨率区域读取。

第一版S/D正式扩展可复用当前全栅格轻状态和物理坐标：在整个当前支持窗口的合法token上形成条件分数，学习跨native-time重计算配额，回填各pack真实mask；需要补variable heavy count的执行/成本核对，而非仅改配置名称。D先在既有调制层闭合，再明确扩展到跨层预算；单调early-exit另作不同可行域，不混称当前D。

engine.py:116–120 的value_score在no_grad中，operator_training.py:70明确hard mask无surrogate task梯度；state/geometry detached。只去掉no_grad也不会让argsort可微。正式S/D必须先冻结一条具体估计器。我倾向首版比较：真实hard预算采样＋已知集合log-prob的score-function任务优势更新，并有state baseline减方差；detector继续实际task梯度。这样不需要偷偷计算全部未选heavy分支，但方差、随机训练/确定推理差异要验证。若选择ST/soft-gate，则需规定未选分支梯度近似、训练时计算哪些分支及其成本。**这些是尚待选择和实现的方案，不能宣称现有S/D已拥有task路由梯度。**

## Raw与正式对照顺序

Raw已有Episode/bounded proposal/arbitrary read/recovery bridge只是技术基础。时间池化cheap hidden不能冒充空间廉价图。自由帧/区域获取还需空间cheap evidence、物理帧/patch坐标、真实解码/缩放/缓存miss/区域读取成本。Standard的transport默认邻居RGB已在内存；Raw若未观察这些邻居，要额外读并计费或换梯度估计。先闭合Standard正式模型，再做独立Raw输入域扩展合理。

最小顺序同意：Uniform384 → BMCR-Global → Global-TaskValue，然后独立S、D和联合。前两者为同初始化/时程/增强/恢复的完整T任务对照；第三者保持policy执行，新增当前策略真实cls/loc Value grounding。policy logit不自动等于calibrated gain。若还改上下文，需把该差别单列，不能将总差异全归为Value监督。

全轴Graph检验分配上下文，RISE检验共同输入/候选/预算下完整分配函数老师；不继承旧局部G1/B0的正证据，也不受其一票否决。仍需各自matched任务/成本和RISE beta1/Post、额外优化控制，不因论文想要四格而一次自动发所有课程。

## 当前真实执行与建议同步方式

Reverse source4aa六head已完成且STRUCTURE_SIGNAL_FAIL；fit排序高但held不泛化，原结果不改。RISE800导出显示held30 seed-state最终选择零变化。D/S40仍略输Uniform。固定诊断724最初在A100248477排Priority，确认PENDING后移至释放AutoDL GPU1 PID317797，已见D结果，S处理中；四条4090课程未动。Atlas interaction由owner完成，INCONCLUSIVE，不回流publication标签训练。

ledger建议增加研究层级与实际执行空间：局部FAIL保留，task锁只约束旧局部recipe；新Global行记DESIGN/IMPLEMENTATION_PENDING并有自身技术门。D/S80记完整组件消融，Atlas记characterization，已证明最终模型仍为空。

我赞成即刻纠正分类和gate适用域，不赞成把S FFN routing叫自由ROI、把D重轻重入叫early-exit，或未经独立消融将所有收益叫Value。请设计任务整理确认双线方案，随后可按独立配方和唯一trainer落实；本回复未触发新课程或取消既有训练。
