# H65/BMCR 三维去冗余的论文研究与实现方案

## 1. 研究判断与证据边界

建议把论文主线收敛为 **Framewise Predictive Write-back for Budgeted Temporal Action Detection（FPW-TAD，工作代号）**：保留H65/BMCR已有单candidate选择和稀疏VideoMAE，再用带来源与物理时间信息的解码器恢复检测时间轴；在同一骨干状态上对深度和空间的昂贵残差进行条件执行。目标不是叠加MAE、router、KD三个名字，而是回答：**在已有真实与预测状态的条件下，哪个额外计算动作最能保护检测质量？**

唯一项目依据为`239d098cd899936c35989fae6243c70259a85adb`；快照截至2026-09-12 21:56。科研规划不把旧模型已完成成绩当成新方法成果。当前只有源码/记录核查、算术推导及本包独立CPU合成测试；没有项目权重、原视频、真实4090复测或新mAP。[P1–P10]

T24整clip路由从新的实验计划退出；历史结果只归档，不据此宣称整clip在科学上被排除。H65按单candidate选K384后仍可打包为24个16-observation计算块，这不是T24路由，保留该实现有助于复用训练好的权重。BMCR每16-slot内选择交换伙伴只是局部监督先验，不应因退出T24就自动删除；应作为后续消融。[P2–P5]

现有新模型中旧BMCR-B 67.3404%最高；S上修正H65 63.4094%更高；官方S/B 69.0126%/71.1280%仍领先。约一半矩阵/卷积FLOPs没有转化成这些旧训练模型的指定窗口加速。修正BMCR截至固定快照尚未开始；同配方warm20、效用尺度重审、joint40应先完成，但新恢复器的接口/CPU测试可并行准备。[P1,P8,P9]

## 2. 先纠正六个会破坏论文因果解释的概念

**零残差不等于保持旧mAP。** `F_hat=I(Z)+R`且R末层零，只保证初始等于指定插值路径。若同时从rank384 detector换成original768 detector，就换了表示和检测合同，不能说初始仍等于67.34%的BMCR。必须单测旧rank头、插值+新头、训练恢复器+新头。保留旧头作为不可覆盖的真实锚点，不靠未校准的双头融合包装“保精度”。

**192 anchors不是384原tubelets的直接子集。** selected帧两两组合成latent，还经过clip attention与global-TIA192。一个latent可能由不相邻两帧及更广上下文构成，不能简单贴回某个原tubelet，也不能把192 anchor加192 mask视为完整原网格。用384个原时间query读取192个带来源的anchors，允许所有输出位置进行表示转换。[P2,P5,P6]

**teacher替换值不是实际动作值。** 把teacher某位置feature写回学生，再跑head，测的是表示修复敏感度。真实增删帧会改变pair、clip上下文和global-TIA；真实深度/空间升级也会改变后续状态。便宜替换可作诊断/预标签，但需与实际同预算干预对照。不能用global teacher缓存“免费”产生所有真实动作标签。[P4,P6]

**蒸馏要区分来源。** 当前BMCR有EMA反事实监督，不等于已有检测输出KD；旧H65的贡献代理也不是KD。官方TAD教师属于外部强监督；同一checkpoint全K分支与稀疏分支之间才是共享模型self-distillation。EMA评价器、EMA效用teacher、固定official teacher必须分别标记。[P1–P4]

**单帧选择不代表深层单帧独立。** VideoMAE的两个candidate在patch embedding后混合，后续深度门作用于带frame provenance的tubelet/空间token，不应伪称每个原frame独立退出。全原时间浅patch通路是另一个后续变体，不得偷偷替代“先选帧减少RGB重编码”。[P6]

**满预算不会自动等于官方。** 只有权重、归一化、padding、图、时间接口都与官方一致的独立官方模式，才应做官方恒等测试。H65训练权重即使K=768，仍不是官方权重。任何fallback的成本与行为需实际测试，不能承诺无条件mAP保持。

## 3. 跨领域机制：哪些是直接证据，哪些只是类比

### 3.1 直接TAD压缩：PBD是必须比较的近期工作

CVPR2025 Progressive Block Drop用渐进式去block与跨深度对齐压缩TAD，在THUMOS14和ActivityNet实验。它已覆盖“删除深层冗余再对齐”的论文叙事，不能把早退出或层间蒸馏本身说成新颖贡献。我们的差异应是单帧不规则观察、原轴恢复、条件动作价值与三维分配协同，并与静态block drop/同KD预算对照。其论文@0.5和MAC统计与本项目口径不完全相同，原表不得直接并列数值排名。[1]

ETAD主要解决梯度与proposal采样的训练开销；AdaTAD++研究时空adaptation与高分辨率训练。二者是相关背景，但不是同等的RGB稀疏推理保精度证据。[15,19]

### 3.2 Masked latent prediction并不自动变成推理恢复

VideoMAE原decoder预测RGB patch；MVD预测image/video teacher latent；data2vec把完整上下文latent预测扩展到语音、视觉和语言。它们支持选latent而非像素作为学习目标，但预训练时能预测并不能证明下游固定K可以恢复真实短动作。[6,8,21]

MultiMAE以任务query从稀疏多模态latent恢复结构化输出；CrossMAE去掉query间self-attention，用每个输出位置读取可见latent，并利用多层encoder表示。对FPW-TAD的可迁移点是坐标query、浅解码器、多层skip信息。原方法主要提高预训练效率，不能拿下游encoder成绩充当“推理写回模块”收益。[2,5]

V-JEPA2.1（2026，v3）强调密集预测监督与多深度自监督；值得检验真实和预测状态是否需要共同约束。其消融也体现不同任务目标的权衡，因此不能把“多加context/depth loss”写成统一提高mAP的保证，更不能用其大规模预训练结果推定200视频可重复相同规律。[7]

MGD进一步提醒：mask学生feature、生成teacher target可能只用作蒸馏损失分支；生成器出现在训练图，并不等于下游推理head使用生成结果。FPW-TAD必须画出真正连接detector的恢复feature与梯度路径。[20]

### 3.3 真正的下游回写：先有状态，再减少昂贵精炼

Expediting ViT在密集检测/分割/深度等下游中，先获得完整浅特征，用token clustering减少后续计算，再重构高分辨率feature。它是实际dense-interface write-back的直接证据；但保留了浅层状态，不是彻底未观察RGB。因此需要控制“廉价完整观察”是否才是收益来源，而非把一切归功于decoder。[3]

Eventful Transformers维护状态并按变化重算，支持视频中的增量计算；但不同窗口、空间crop、tubelet相位和global上下文可能不一致，缓存不能默认免费，遮挡和新事件也是失败场景。[13]

### 3.4 长文本与密集分割：保存输出位置和上下文角色

CoLT5的轻重路径与独立Q/KV预算，TR-BERT的token深度分配，DToP的密集位置早退出与代表性上下文，都提示应区分“需要被重更新的位置”和“需要提供上下文的位置”。CoLT5在条件计算图上预训练，DToP/TR-BERT也非冻结teacher后只训练一个final decoder；TAD迁移必须做真实稀疏学生适配。[9–11]

DyT最接近本项目的训练设计：训练完整算block，再按gate选择进入后续学生状态的残差，并可用完整分支做self-distillation。它已占据full/light-state与动态PEFT组合的先例；本项目应证明task-grounded action calibration、真实时间恢复和更合理预算分配的额外收益。tokenwise MLP的dense-mask/packed等价不能不加条件推广到丢K/V的attention。[4]

### 3.5 点云与数值计算：几何与目标误差优先

Point-M2AE的几何一致多尺度mask与skip式解码启发我们保存原坐标和浅/深层信息，而不只线性插值。3D缺失几何和短动作缺失都存在不可恢复性；几何插值是归纳偏置，不是语义保证。[14]

Dual Weighted Residual等目标导向数值方法按最终观测量的误差分配网格/精度，而非按局部残差大小。可借鉴“预测误差×下游敏感度”的原则，但不能把PDE误差定理移植到不连续的AP/NMS。[16]

**发散后的收敛：** 论文创新不应是“MAE + KD + router”。最值得检验的是：来源/坐标感知恢复是否减少任务分布差；廉价表示修复代理是否可校准为真实计算动作效用；保持上下文的可执行3D预算是否优于同算量静态分配。

## 4. 可证伪科学假设与论文主张

H1：在相同frame selector、K、teacher、训练更新下，带来源/物理坐标的恢复器优于等参数的普通插值残差器。反证：feature误差虽降但高tIoU/短动作没有改善，或只换teacher/head就解释收益。

H2：表示修复代理与真实frame/depth/spatial干预存在系统偏差；经少量真实干预校准的有符号utility优于abs-gradient、feature error、actionness和未校准oracle标签。反证：实际gain相关性/route后GT损失未提高，或同预算统一覆盖即可达到效果。

H3：同一参数模型下，真实混合学生GT训练优于只在dense输入上拟合；同预算self-KD有额外价值。反证：没有KD同样好、收益仅来自更多更新/查询teacher，或C2与J3在严格同输入测试不一致。

H4：深度/空间贵残差选择在完整空间TIA状态上执行，能改善mAP–实际时间前沿，且联合分配优于等算量单维/static策略。反证：只减少MAC没有加速、全结构drop更好、或三维叠加加剧边界误差。

H5：额外compute主要应分配给“对任务重要且预测不可靠”的位置，而非单纯高视觉变化/动作分数的位置。反证：calibration失败，uncertainty只反映噪声，或稀有事件召回下降。

不主张固定K无条件保持mAP。两个视频在所有保留的高成本观察与廉价scout可访问表示上相同、却在省略的高分辨率/深层证据中有不同动作时，学生不可区分。当前scout已看过全时间低成本图像，不能把未进重骨干的帧称作物理上完全没观察。可建立的是有限数据/明确分布上的经验Pareto前沿，以及经过校准的风险–成本曲线；测试集选模会破坏未经观察测试估计。Conformal类理论仅作为需要独立校准假设的未来方向。[18]

## 5. 第一代模型：保留H65/BMCR，恢复原时间轴

### 5.1 计算与表示合同

输入为`[B,1,3,768,H,W]`，candidate步幅4。frame router输出单candidate集合S，保持现有exact-K及短窗`min(K,valid_length)`唯一观察数。K384仍以24个计算块送原VideoMAE，得到插值前`Z:[B,C,192]`；现有`encode()`输出已是rank384，必须在backbone后处理插值前暴露native，不可把rank384下采样伪装原native。[P2,P5]

每个anchor记录`source_indices [B,192,2]`、两个真实frame时间、contributor_valid、中心、pair跨度、packed clip/tubelet位置、空间执行质量、最后heavy层。中心只是支持区域摘要。原查询q为384个原tubelet的物理中心；当GT以candidate坐标表示时必须明确转换，而不混用秒/原frame/candidate。

已配对两帧都被选，不保证teacher原tubeletfeature已被“真实观察”：配对/attention/TIA上下文不同。四态00/01/10/11可当query输入，但不是feature同一性的证明。恢复目标是所有原轴位置的TAD表示转换。

### 5.2 恢复器

令`I(Z,c,q)`为原物理中心上的插值底座。Query由底座、q位置、gap、pair/selection状态、已有scout上下文构成；Memory为anchors及来源编码。第一版2层、width192、3头cross-attention，query之间不self-attend，便于分块decode及降低开销；B也先用192宽作为效率原型，官方B decoder384宽初始化实验必须另列匹配架构。

`F_hat(q)=I(Z,c,q)+W_out D(Q(q),K(Z),V(Z))`，W_out零初始化。初值仅等于插值+指定head模式。特征norm/variance/scale桥接显式建模，不能因为通道相同就认定兼容。scout上下文先stop-gradient复用，不加额外MobileNet；是否放开scout由后续消融决定。

官方VideoMAE decoder预训练权重只作为一个初始化对照：检查完整pretrain checkpoint的keys、维度、depth、norm与出处；丢RGB1536 head、新建latent head与时间桥。self-attention迁成cross-attention需参数转换表并与相同转换后结构的random版本比较。当前未下载这些权重，不宣称已有恢复先验收益。[21]

### 5.3 检测与教师

固定official global teacher生成`F_T:[B,C,384]`，学生恢复后经official384→768插值送original detector。官方teacher的参数/最终检测器参数冻结，但学生feature到GT loss必须有autograd。旧rank detector作为单独不可覆盖baseline；可在joint时添加旧路径GT辅助约束，但其计算成本须计入。

第一版teacher为外部official TAD知识；后续同一个学生完整K/全门分支作self-teacher才叫self-KD。未完成的修正BMCR成绩不等待假定胜出；模型选择按真实结果，S可保留修正H65而B保留旧BMCR/修正BMCR候选，配方分组不可混写。

## 6. 条件效用：从代理到真实干预

定义状态s下的实际动作a（固定K交换一帧、某stage增加一组贵更新、某空间块升级）：

`Delta(a|s)=L_GT(s)-L_GT(T_a(s))`，并记录真实增量MAC/latency。

固定观察预算的新增必须与删除配对；不能用单独插入收益替代等成本交换。先小步trust-region调整S0，记录S0→S1总变化、单交换预测和实际重解码收益；若组合交互强，降低同时交换数或在新状态重估。

便宜代理`Delta_repair`通过同格点teacherfeature替换再跑head得到，只用于表示敏感度诊断。抽样真实干预获取`Delta_action`，学习二者偏差；最终utilityhead输出带符号cls/reg收益及不确定性。统计校准在训练视频分组内out-of-fold完成；最后仍用全部200视频训练，不强制160/40。

对固定feature位移delta，可用精确恒等式：

`L(F)-L(F+delta) = - integral_0^1 <grad L(F+alpha delta), delta> d alpha`。

这是数学推导，不是新的已证明mAP定理。它解释teacher端绝对梯度为什么可能错：符号丢失、参考状态不同、有限变化非线性。1–2点有符号近似是可选廉价代理，需计head查询成本，与真实动作标签对照。[16]

BMCR16-slot伙伴先保留完成fidelity实验；在新utility阶段对比局部16、跨cell近邻和覆盖受限全局交换。取消T24不是删除局部先验的证据。候选pool必须含selected与unselected、动作/背景、稀有boundary邻域，防止只监督高score自确认。[P3,P4]

推理只用RGB/scout/feature/物理metadata，不读GT。几何max-gap、uniform anchors可作覆盖约束；真实短动作起止只用于训练标签和评估分组。uncertainty应校准实际任务后果，不能只画feature方差当安全保证。

## 7. 深度与空间：在同一packed-K状态上实现，不重启弱模型

### 7.1 统一状态保留

在原有K384骨干保留`X:[B,192,10,10,C]`完整中间状态。深度门作用于带frame来源的native位置；空间门作用于这些位置的空间token或2×2块。Temporal frame selector仍是单candidate；不增加整clip路由。

第一版前8层全算，后4层选择部分位置的attention/FFN贵残差；其他位置保持identity或小低秩残差预测。每次attention/FFN后scatter回完整空间grid，然后执行**原global-TIA192**。它保留已有权重的时间合同；原轴global384作为后续独立图改动，而不是现在只改temporal_size。[P6,P7]

全门模式须回到原H65/BMCR同权重同图。预测器关闭/重路径全开时不得额外改原输出。Adapter返回值本身含残差，不能再做`x += adapter(x)`。跳过贵block并不等于该位置停止所有计算：TIA、KV、light路径仍应逐层计数。

### 7.2 深度门：先固定退出控制，再动态

先比较静态删后2/4层、PBD式渐进选层、全部浅出口、条件最后heavy更新深度。所有方法使用相同额外训练/teacher预算，深度输出映射至同一个native恢复器；从早层读取feature不能直接冒充最终层。第一轮只8/12两stage；4/8/12属于扩大范围，不一开始引入每层任意退出。

预测器训练在真实学生输入状态上，监督剩余贵残差或最终feature差值。多层目标为可选，feature+GT主干稳定后才加；不默认层数越多监督越好。早退状态继续通过TIA提供上下文，允许以后研究重新进入heavy，但第一版使用嵌套预算保持执行简单。[1,4,7,11]

### 7.3 空间门：FFN先行，attention另测

第一版只在进入深层的状态上稀疏FFN，轻预测所有未heavy位置，heavy覆盖其自身残差。原100空间token保留；低attention不是可删证据。按空间块结构化和按token非结构化分别测相同实际MAC与真实时间，不预言2×2一定更快。还应有规则160→128分辨率/同训练控制，避免用复杂router获得连简单缩放都不能胜过的收益。[12]

若进一步稀疏attention：选中q个Query，但K/V由完整当前light状态产生。对同一个输入和无dropout，这是原attention相应query行的精确计算；进入下一层后学生状态不同，不能称全网络dense等价。真正省Q投影需拆qkv权重做`Q(selected),K(all),V(all)`，不可先全qkv再gather后还按少Q记账。

### 7.4 训练图与梯度

C2：主干贵算子可全前向，但进入下一层的学生状态必须按route选真实/预测残差；GT loss来自该状态。tokenwise MLP同输入dense-mask与compact可一致；attention丢K/V时不一致，所以使用同K/V定义并单测。

J3：真正compact执行训练，用同损失适配系统部署路径。离散router用实际干预监督，或者明确STE/Gumbel近似；直接加GT不会使hard top-k可微。若复用transportRGB代理梯度，保留原forward精确硬选择并单独审计梯度贡献。[P5]

训练dense官方teacher和global学生可能要双前向，不能把同一个dense feature缓存冒充不断演化的学生后层。节省训练梯度成本也不能当推理加速。[4,15]

## 8. 损失和课程：从最少必要项开始

主配置先使用`L_GT + lambda_F L_feature`：GT包含原cls/reg；feature采用masked/valid加权SmoothL1和小cosine项。合法边界加权沿用gt_boundary_validity。feature时间差分和Bernoulli输出KD作为独立增量消融，不同时默认必需。

分类KD匹配独立20类sigmoid，可用温度Bernoulli KL/BCE；回归是左右距离/真实端点，不硬套分布KL。通过foreground/background分层和valid权重避免背景数量吞没短动作。teacher feature stop-gradient；student→head→loss不可no_grad。[P2,P4]

共享full-gate学生self-KD与external officialKD分开：其满分支也需GT约束，防止teacher跟学生一起漂移。多teacher冲突先固定一个主teacher，不能把不同norm/rank语义的latent直接平均。EMA在线/评估分别比较；末层零使首步上游梯度为零属正常，不把整个decoder全零。

建议顺序：基线fidelity → 新head插值基线 → 固定selector恢复器 → GT/feature监督消融 → router真实干预校准 → 深度 → 空间 → 联合小范围budget训练。主研究阶段K384；K320/K256只在恢复效果成立后测试，且当前代码budget依赖16倍数，必须校验配置与checkpoint合同。

每个新阶段建议先技术2步、训练集200更新检查、500–1000更新机制筛选，再固定课程做完整200视频训练与211完整测试；这些是资源建议，不是已经接受或已完成的训练计划，不以试验不足的早期排名证明最终优劣。公平比较要对齐成功更新、完整教师查询、学生实际计算、额外训练时长，不能只写epoch。

## 9. 实验矩阵与最小判别顺序

`plans/experiments.json`包含逐项实现、依赖、匹配因素与证伪结果。没有任何新实验预填mAP。

| 组 | 必要对照 | 可以回答 | 必须披露 |
|---|---|---|---|
| B | official、旧H65/BMCR、修正H65、待做修正BMCR | 执行fidelity与已有锚点 | 老/新配方，peak/terminal，外部teacher |
| R | 旧rank头；相同anchors插值+original头；简单残差MLP；坐标cross-decoder | 头/时间轴变化与decoder增益 | R插值不是旧BMCR恒等 |
| L | feature-only；GT+feature；再加输出KD或差分 | 任务loss、KD真实增量 | 相同更新/teacher查询，损失权重 |
| I | 同架构random vs official初始化；2层vs4层另外测 | 预训练权重复用是否有用 | 不混结构/宽度变化 |
| U | actionness/absgrad/errorproxy/repairproxy/calibrated-action | 真实条件效用是否更好 | actual action与representation repair分开 |
| D | 静态drop/PBD式；8层；动态8/12 | 深度动态是否值得 | 同训练/teacher/实际FLOPs |
| S | 分辨率128；token/2×2FFN；optional sparseQ | 空间收益和硬件收益 | 实际Q/KV/MLP输入与完整栅格成本 |
| J | T、TD、TS、TDS；等成本静态vs条件预算 | 三维协同而非简单叠加 | 可用同校准checkpoint做推理factorial |

2×2×2八种T/D/S策略先在一个见过这些预算的checkpoint上评估，不默认训练八个模型。对应K768 internal模式只在训练覆盖/配置支持时纳入，且不叫official。用差分中的差分衡量交互，但显著性和不同比例成本必须对齐；不同FLOPs的八格只说明条件响应，不证明最优协同。

同checkpoint uniform/random推理控制是合法且便宜的，但不是独立uniform384完整训练。比较路由可用共同route augmentation校准接口，再固定结构分别适配同更新；要注明这仍不等于从头各自完整课程。

主结果保留200/211、S/B。3种子只建议用于胜出少数方法，不强迫所有网格；成对video bootstrap重算AP，抽中同视频多次时重命名ID，不能平均单视频AP。现有测试peak选择必须在表注写出；bootstrap不能消除反复看测试的乐观偏差。进一步泛化建议一个独立数据集/场景，需要视频资源与成本核对，不能用仅THUMOS14的结果写“普适所有TAD”。

## 10. 系统计量与成本模型

原clip每层N=800，channel C，重Query q、重MLP m，matmul MAC为：

`C_layer = 2*N*C^2 + 2*q*C^2 + 2*q*N*C + 8*m*C^2`。

以上适用于q>0的计算块，依次是full K/V、selected Q/output、QK/AV、MLP；某块q=0且没有其他KV消费者时，KV也可完全不算，实际计数优先。另加patch embedding、完整TIA、低秩/light、router、decoder、detector。没有full K/V需求的其他实现要另定义，不可套公式。q=m=N退化原attention+FFN；q=m=N/2仍大于其一半，因为KV不减。

按固定160输入与2MAC定义独立算式得到：S 1173.947019264 GMAC，B 4041.077354496 GMAC，与旧完整参考矩阵计数一致。本包CPU算式测试验证该总数。S TIA约3.267%，last4FFN仅降25%/50%在新模块前最多省官方3.8586%/7.7173%；深度已跳过的FFN不能再次计空间节省。[P1,P8]

新模型每个window记录source有效观察数、physical执行槽位、每层clip与Q/K/V/MLP/light计数、TIA原序列长度、decoderquery/memory数量。完整768、部分503、短253窗口都测；这些不是测试集平均延迟。短窗`min(K,L)`和odd pair padding不得标为K真实观察。[P5,P10]

测量四个scope：准备route后的骨干；GPU常驻RGB完整模型；decode/CPU preprocess/H2D；window/video NMS与完整dataset walltime。使用同卡交错A/B顺序、多次重复、均值/median/p95/离散度和原始样本，不删除outlier装加速；CUDA event计device区段与同步walltime计全scope配合。建议每条件充分热身后至少100次主窗口重复并额外重复完整dataset，次数是测量建议不是硬审批。

纯MAC计数排除softmax/LayerNorm/激活/索引/内存访问等必须列明；大实际开销可能来自Python循环、CPU同步、解码与NMS。代码优化先精确向量化scout condition和重复几何，保持same outputs并另列system-only贡献，避免将相同算法的实现修复冒充模型创新。

缓存分别记录teacher缓存创建、bytes、读取、命中、CPU/GPU存储。固定原clip的local独立性不能推广global192；跨窗只有crop、frame集合、位置、相位、上下文均一致才有精确重用资格。否则只能视近似算法并评价误差。推理全量解码但稀疏重算时不能声称减少全部RGB读取。

## 11. 论文图表与可视化论证

`FIGURES.md`与`plans/figures.csv`给出6个主图组、12个补充图组、4个主表及数据合同。主图不应堆满全部实现细节，应把“为何需要—机制—是否有效—为何有效—是否真快”连接起来。

所有经验图必须从verified records生成，未来成绩留null而非0。时序/空间可视化必须读取真实对应视频和模型trace；不能用生成图替代观测，更不能选择一个最好案例当普适证据。案例选择规则先固定，至少展示失败、短动作、同类重复和长背景各类。

架构图属于概念示意，可用TikZ矢量画；数值图和真实feature/route图来自数据。每张图包含checkpoint、commit、recipe、视频window ID、预算、teacher来源、scope和selection_policy的旁路manifest。训练集归因图和测试集诊断图分别标注，不反向用测试诊断调超参后称unseen test。

PCA/t-SNE最多作描述。更强证据是：paired real intervention、GT detector loss、逐tIoU AP、真实边界/重复/漏检、输出排序变化与执行计数。DETAD式oracle错误修正效应可能重叠，不能相加为总mAP缺口。[17]

CVPR2026模板为本包版式参考：正文8页含图表；实际author-kit总宽6.875in、单栏3.28125in。图内建议8–9pt文字、300dpi或更高frame栅格、其余矢量PDF/SVG、嵌入字体，最终以`\columnwidth`（LaTeX中写作`\columnwidth`）输出检查，不擅改官方字号/栏宽。未来投稿届规则另核查，不宣称“CVPR规格”能保证录用。[22,23]

## 12. 论文故事与结论写法

建议三项贡献均写成需证据支撑的陈述：

C1. 一种保留单帧路由成果、显式建模不规则anchor来源与物理时间的TAD特征回写接口。

C2. 一种区分“修复表示”与“改变真实执行”的条件效用校准机制，约束局部标签到组合route的偏差。

C3. 一种保持完整空间/时间通信状态、真实压紧贵算子的三维预算实现与分层自蒸馏训练方式。

不能宣称首次MAE latent预测、首次稀疏query、首次TAD层drop或首次动态PEFT。更准确的新意可能在上述三者的耦合与严谨干预方法；最终需要同机制强基线验证，而不是靠新缩写证明。[1–4,9]

8页叙事建议：1页问题+现有实证动机；0.75页相关方法和差异；2.25页定义、回写、干预、训练/执行；3页结果、主消融、速度；1页误差、限制、结论。页数为组织建议，references按官方要求。正文优先图1/2和核心表；长loss/实现细节放supp，禁止缩字体塞满。

结果分三级：若只有MAC降，写compute reduction而非acceleration；若均值mAP恢复但0.7/短动作掉，写trade-off而非性能保持；若跨种子、跨预算与独立场景都支持且系统时延改善，才写更强generalization/efficiency。无法达到三维同时保性能时，可报告一个更可信的二维前沿及第三维失败机制，不为论文故事隐藏负结果。

## 13. 实施交付与当前完成状态

本包提供机器可读计划、各agent职责与命令合同、可运行CPU原型与13项合成测试、历史真实数据绘图脚本及来源清单。它不包含训练好的新模型，也不假装已经集成OpenTAD production trainer或访问服务器。

优先顺序：先补同配方修正BMCR；并行完成native提取与旧图恒等；固定K384训练回写分支并测插值/新head真实起点；随后才校准router，再做深度与空间。最先值得争取的不是任意1pp门槛或一口气75%省算，而是证明“恢复feature与实际TAD收益耦合，条件执行比等成本静态分配更好”，并让实际系统测量支持这些结论。
