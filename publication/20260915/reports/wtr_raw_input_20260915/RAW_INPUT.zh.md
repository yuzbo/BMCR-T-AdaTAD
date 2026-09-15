# WTR原始视频采样：时机、输入合同与评审记录

> 后续接口补充见[Raw-v1进一步思考与合同](RAW_V1_CONTRACT.zh.md)：episode先定、共同preview、有界proposal、显式tubelet及三坐标。新增两份原文已完整保存，本文保留为上一轮记录。

日期：2026-09-15。两份建议已完整阅读并原样保存。本轮只核对本机相关代码、阅读解码文档、记录设计与意见；没有修改模型/数据配置、运行实验、恢复旧课程或部署任务。

**结论：可以从原始视频的可用时间戳出发进行智能heavy采样，不需要把OpenTAD的固定候选帧生成器当作模型不可改变的前提。** 但这不等于没有初始观测、不等于删除全部数据/评测协议，也不必然意味着改为16帧micro-clip或获得I/O加速。建议以保持episode与query lattice的Raw输入分支研究；两份原文不能全部原样采纳。

## 1. 两份材料及主要分歧

- [第一份：保留episode、改用raw acquisition，并提出micro-clip](inputs/01_Raw_Episodes_and_Microclip_Proposal.txt)。原附件 `65e1fb1e-4bc5-4da4-aa0f-f4adbaedb31b/pasted-text.txt`。
- [第二份：原始时间戳、区域分层选帧与时间编码](inputs/02_Raw_Timestamps_and_Hierarchical_Selection.txt)。原附件 `92f1aa61-0830-4c2f-9adb-5a8b5df6f095/pasted-text.txt`。

|方面|第一份|第二份|当前取舍|
|---|---|---|---|
|输入变化|移除固定768候选限制，保留episode|统一raw physical-time候选，采用区域预算|接受输入接口与候选域扩展|
|heavy动作|倾向选center后获取16帧局部clip|继续选择原始时间戳上的帧|Raw-v1继续逐帧；clip是独立动作空间实验|
|THUMOS|先保留相同物理窗口，整视频后置|提出整视频Scout＋regional budgets|先保持窗口与可见上下文，避免同时改变模型视野|
|时间编码|主要依靠局部clip保存时序结构|提出physical-time PE/relative bias|独立验证，不宣称任一方案必然修复|
|Graph/动态预算|作为Raw完整候选的组成|认为Raw更需要Graph，预算与采样联合|两者继续可选，不作为Raw可行性的前提|
|系统实现|先预解码算法验证，再on-demand decode|直接提出按所选timestamp解码|采用两阶段验证，分别报告算法与系统成本|

第一份对阶段与解码边界的区分值得采纳；第二份的逐帧raw timestamp方向更接近已经确认的Core。两者可以提供互补实验，但不能合并成“必须同时去滑窗、换clip、加时间PE、Graph和动态预算”的单一改动。

## 2. 当前已经少算的部分与真正要改变的部分

本机 `h65/paper/encoder.py:95–107` 先按selection执行真实RGB gather，再将选中的观察按16打包；`:129–132` 才调用heavy engine。**当前H65已经只heavy编码选中的RGB，不是先heavy编码768帧再丢掉一半。**

当前数据侧先形成固定候选RGB输入，Scout与选择受这个候选集合约束。Raw分支的增量是：

1. 将heavy可选观察从固定候选网格扩展到原始视频的合法帧/时间戳。
2. 让选择发生在固定高分辨率候选张量完整构造之前。
3. 在系统版中，只向heavy路径提交实际所选RGB，并测量是否减少了真实读取、预处理与传输开销。

它可以改变信息获取范围和数据流，但不能再次把已有selected-RGB heavy执行记为新的全部节省。

## 3. “官方采样”需要分四层理解

|层次|Raw第一版的处理|
|---|---|
|Benchmark评测|保留数据划分、类别/GT定义、绝对时间预测与官方AP计算|
|Episode与训练/推理配方|暂时保留THUMOS物理窗口范围、训练crop与GT处理，ANet保留整视频episode|
|候选观察生成|可以替换固定stride/resize候选网格，改用raw-time提议与获取|
|Heavy观察单位|继续逐帧作为Core默认；micro-clip与可变span另作研究|

THUMOS滑窗、ANet固定resize和具体输入长度主要是这套AdaTAD/OpenTAD方法配方，不应统称为benchmark绝对不可变规则。保留它们的episode/输出范围是当前为了归因与复现作出的控制选择。

不同方法可以在同一benchmark报告最终结果，但输入视野、训练配方或采样方式改变时必须标明；其总分差不能全部归因于Value Router。同raw接口下的强uniform等控制仍然必要。

## 4. 代码核对到的输入协议

THUMOS目标配置为window_size=768、scale_factor=1；训练random_trunc、测试sliding window。基础feature_stride=4、sample_stride=1，因此候选原始frame index间隔为4。

ANet目标配置为resize_length=192、scale_factor=4；整视频采768个RGB候选，映射到192长度的检测栅格。两者的768并不代表相同秒数或相同原始时间间隔。

数据流水线中，PrepareVideoInfo构造视频引用，LoadFrames生成frame_inds，随后DecordDecode才解码，再Resize/Crop/FormatShape。模型内gather发生在这之后。因此Raw并非删除一个sampler函数就自动实现；需要把前置cheap读取、模型选择和后置heavy读取分成明确阶段。

这些不同配方不妨碍原来的VoC数学定义；使用统一physical-time接口主要改善支持描述和候选获取方式，不意味着此前无法统一定义价值。

## 5. 建议的Raw-v1输入时序

1. Dataset返回episode描述、视频引用、时间/帧索引信息、GT及固定query定义。
2. 先获取低分辨率/低频cheap观测，运行共享Scout，保存时间和小网格空间特征。
3. 从cheap证据选择合法容量和raw-frame proposal，执行集合条件交换；所有决策只使用当时可见信息。
4. 将选定时间映射到实际有效frame IDs，记录去重、有效性和来源；由reader获取这些heavy RGB。
5. 进行一次patch embedding、dense prefix、D/S refinement和TIA。
6. 用真实contributors恢复到本次episode原有query lattice，交给原head、后处理及评测。

数据与模型的调度入口可以协调CPU reader和GPU选择；无需要求普通DataLoader在生成完完整heavy tensor后才能调用模型，也不应在forward里隐式重复解码而不记账。

三个轴必须分开：cheap观测索引、heavy acquisition的原始frame IDs/时间戳、detector query索引。它们各自有valid mask和映射，不能继续假设同一个768数组同时表示三者。

Raw的candidate空间仍须有限、可枚举或通过有界分层提议产生。预测连续时间后最终也要落到视频中存在的帧。统一接口不等于取消所有离散候选、初始采样或成本边界。

## 6. 哪些cheap输入设计合理，哪些保证过强

低分辨率＋低FPS、固定数量的全episode观测，都可以作为受控起点。低FPS的神经成本随时长增长；固定数量的观测在长视频上分辨率下降。两者应根据测量取舍，不能同时宣称任意长视频成本固定且短动作信息完整。

粗cells/region quotas可以缓解全局竞争，但不能保证发现短动作。动作可能落在cheap采样间隙，或者被粗网络遗漏；覆盖配额不等于语义召回保证。

模型可以根据cheap/邻域证据预测某个尚未heavy观察位置的期望收益；不能无观测地保证知道其唯一内容。boundary与Graph信息仍是预测线索，不是GT答案或计算价值本身。

Raw并不必然更需要Graph；连续时间插值、局部/多尺度上下文也是候选。Graph、动态预算和FVD均独立于Raw输入是否成立，第一阶段不同时加入。

## 7. 为什么不直接接受“改成16帧micro-clip更好”

去掉固定候选网格与改变heavy动作单位是两件事。当前Core保留individual-frame acquisition；micro-clip建议单列，不能自动并入主线或恢复已撤销的旧课程。

`384/16=24`只确定打包数量。若分辨率、token形状、D/S容量和算子路径全部相同，某些encoder矩阵/卷积分项可以相同；它并不自动建立完整模型或系统成本匹配。

至少需要区分：encoder处理的观察次数、实际独立帧数、clip重叠、物理span、帧间隔、TIA配置、controller/recovery及解码成本。局部clips可能重复覆盖相同帧；重复帧可减少某些解码，却未必减少依赖clip上下文的heavy计算。

尤其THUMOS该配方的候选stride为4。连续16张原始帧与16个按该配方采样的观察，时间跨度不同；VideoMAE使用16个输入观察也不意味着预训练只见过连续原始帧。不能只固定24 chunks，就把精度差单独归因为temporal inductive bias。

Uniform local-clips胜过irregular-frame方案只能首先说明这两个完整acquisition方案不同。若要解释packing/时间编码的因果作用，还需尽量固定同一观察集合、其他mask和输出协议，单独比较表示方式；checkpoint内干预与独立适配训练分开。

## 8. Encoder时间信息需要验证，而非预先承诺修复

当前普通engine在 `engine.py:88–94` 对patch tokens加原pos_embed；真实时间另进入geometry、anchors、queries，以及启用时的Graph geometry（`:182–187`）。普通路径未在这里显式编码不规则Δt，这构成可检验的表示失配假设，但不能据此声称模型必然错误或已证明掉点原因。

可以测试在当前token/adapter中加入真实秒数、相对episode时间、contributor跨度及相邻时间间隔，或单独研究relative bias。时间标记能描述已有观测的间隔，却不能恢复从未获取的像素/运动，也不自动逆转tubelet混合产生的信息损失。

物理时间编码应在同支持、相同其余执行条件下比较，并记录新增成本。它是Raw候选的一个表示机制，不是Raw-v1必须预设有效的前提。

## 9. 保留head需要保留完整几何合同

第一阶段恢复到THUMOS 768、ANet 192的query lattice是合理选择，但只保留长度不够。还需保持每个query对应的物理时间、mask、window offset、时间尺度和GT映射。

保留原始frame IDs/时间戳、fps、duration、total_frames、episode起止、crop变换、contributor及support span；normalized time与真实秒数同时存在。选择坐标与检测坐标明确分开，不把packed rank当时间。

THUMOS局部GT涉及window起点与stride换算；ANet以duration归一到192-grid。统一raw acquisition不能顺带删掉这些任务适配、类别映射或数据集后处理。原轴query完整也不等于dense语义信息无损。

## 10. 系统成本与公式修正

建议把神经计算明确记为：

`Cneural = Cscout + Ccontroller + Cencoder(selected evidence, D/S plan) + Crecovery + Chead`。

Cencoder已经包含实际heavy/light attention、FFN、TIA和prefix。若Cclip定义为完整VideoMAE clip成本，就不能再额外加一份完整C_D/S，否则可能重复计费。48→24 chunks也不能直接称全系统省50%。

解码/传输/I/O另报真实时间、像素/帧数与缓存条件。选择16帧不意味着只需解码16帧；FFmpeg文档说明精确seek通常仍需解码最近seek point至目标之间的内容。[FFmpeg seek文档](https://ffmpeg.org/ffmpeg.html#Main-options)

低分辨率Scout只保证其神经输入更小，不自动保证底层codec工作同比减少。预先生成proxy或缓存原始帧的成本也必须声明；单次运行和多次复用的摊销不能混报。CPU/GPU阶段可能重叠，端到端时延应直接实测，不把各阶段时间机械相加。

原来的容量上界仍可约束声明范围内的神经计算。Raw的seek/解码还依赖位置、GOP及读取历史，不能从只含KT/slots的成本表推出端到端时间保证。

## 11. 算法验证与系统实现分开

Phase A可以使用预解码帧或高密度缓存，为模型提供按原始时间查询的接口。它验证raw候选域与选择策略，不宣称I/O加速；缓存本身不允许controller偷看未授权的heavy特征或GT。

Phase B在算法方案固定后，实现cheap stream→selection→on-demand reader→heavy inference。保持同一所选内容、预处理和checkpoint，以检查算法行为一致，再测解码、传输、GPU和端到端性能。

第一版THUMOS的Scout也保持同一物理episode内的可见上下文。整视频Scout即使heavy仍分窗口，已经增加了上下文，应另列实验。可以复用读取/缓存，但不能把跨窗口新增证据当作单纯I/O优化。

Whole-video THUMOS还会改变时间视野、正负分布、query尺度、head适配和窗口合并。将heavy分批运行不自动等价于保留原global TIA语义。这项扩展后置，不与Raw候选域替换混在一次比较中。

## 12. 最小研究顺序与对照

WTR-Standard继续保留，全数据characterization与固定预算Core的优先级不变。Raw作为独立分支记录，第一步固定预算和局部D/S策略，不要求同时训练DB、Graph或FVD。

建议先形成候选域与选择规则的受控比较：

|候选域|简单控制|学习规则|
|---|---|---|
|官方固定候选网格|Uniform|Value selection|
|Raw合法原始时间戳|Raw-Uniform|Raw-Value selection|

保持相同物理episode、尽可能相同cheap证据、heavy预算/分辨率、D/S、recovery/head和训练起点。每个域的标签辅助参考仅作headroom诊断，不叫mAP数学oracle。若cheap采样也改变，单独说明和控制其贡献。

随后依次研究同支持的时间编码、单列的frame-packed/local-clip动作空间、真正按需解码，最后才考虑整视频THUMOS。ANet整视频resize是概念上直接的替换对象，但正式开展仍依赖数据和任务资产齐备，不因建议“ANet先做”而阻塞已有可行的窗口内验证。

最终主性能全部重算dataset AP，按视频cluster重采样；完整保留输入协议和成本范围。可以同benchmark比较Standard与Raw的总成绩，但对Value学习增益的归因应依靠各自接口内的matched控制。

建议展示：raw-time/coarse/heavy/query四条时间轴；候选域×selector的完整mAP—计算曲线；支持缺口/短动作分析；独立帧数与重复观察次数；解码—传输—神经执行的实测分项。没有实测数据时不画假想加速点。

## 13. 接受意见

**可以采纳：** 把采样决策前移，基于廉价证据从原始视频时间轴获取heavy观察；保留episode与原轴head作为首阶段控制；先算法、后按需解码；持续使用条件任务收益监督。

**不原样采纳：** Raw必然更需要Graph、regional quota保证短动作、不规则输入一定导致模型错误、时间PE必然修复、24个clip自动匹配全成本、Cclip与D/S成本重复相加，以及只保留query长度就实现全部数据集无关化。

**当前不切换：** 逐帧Core不自动改为micro-clip，主实验不立即取消THUMOS窗口，未完成的Standard证据不被Raw故事替代。

对用户问题的明确回答是：**能从原始视频出发智能采样；值得作为WTR的输入获取扩展；但两份建议不是完全正确，必须分清候选域、观察单位、时间协议和系统成本。**

## 14. 核对依据

本机根：`C:/Users/skywalker/Documents/ChatGPT/H65/h65_clean_adatad/graph_tad_20260914`。

- `h65/paper/encoder.py:95`：先gather所选RGB再packing；`:113`：选帧容量控制TIA/native长度。
- `h65/paper/engine.py:88`：patch/pos embedding；`:182`：Graph使用时间geometry。
- `upstream/configs/adatad/thumos/e2e_thumos_videomae_s_768x1_160_adapter.py`及其dataset base：768窗口、stride与train/test采样。
- `upstream/configs/adatad/anet/e2e_anet_videomae_s_192x4_160_adapter.py`：768观察与192检测栅格。
- `upstream/opentad/datasets/transforms/end_to_end.py:244`：LoadFrames采样与GT映射；配置中的DecordDecode负责解码。
- `h65/paper/runtime.py:11`、`:90`：当前metadata及数据长度合同。

本报告只记录本轮设计判断，不代表Raw输入管线已存在或性能已验证。
