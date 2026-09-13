# 论文绘图生产规格

6个主图组与12个补充图组；每个panel单独输出，LaTeX排版组合。下表是计划而非已产生的新方法可视化。

## F01 · Pareto前沿 (main)

**检验命题**：H1,H4。

**实验来源**：J02,K01,M01。

**原始输入**：results.json + profile raw samples。

**绘制内容/单位**：mAP(%) vs GFLOPs; mAP(%) vs model/E2E ms，各自一图。

**聚合/不确定性**：每骨干/相同计时cohort分开；seed与videoCI分别标。

**禁止的推断**：不是同卡/scope禁止速度连线；不以未来预测点补曲线。

## F02 · 训练与推理架构 (main)

**检验命题**：H1,H3,H4。

**实验来源**：C01,R03,D02,S02。

**原始输入**：source contracts + execution trace。

**绘制内容/单位**：真实路径/预测路径、stop-grad、坐标、tensor shapes。

**聚合/不确定性**：概念图可TikZ，不生成fake feature heatmap。

**禁止的推断**：训练mask图与推理compact图若不同，明确画出。

## F03 · 全时间轴案例与来源 (main)

**检验命题**：H1,H5。

**实验来源**：R01,R03,A01。

**原始输入**：real frames + annotation + per-window trace。

**绘制内容/单位**：时间秒；selected candidate ticks；pair supports；GT/detections。

**聚合/不确定性**：固定规则选短动作/重复/失败/普通各类。

**禁止的推断**：恢复feature看起来平滑不是正确；标明decoder不是RGB还原。

## F04 · 条件效用可靠性 (main)

**检验命题**：H2。

**实验来源**：U01,U02。

**原始输入**：interventions.parquet。

**绘制内容/单位**：预测gain vs真实gain；bin signed mean与CI；rank corr。

**聚合/不确定性**：同窗口paired、训练视频out-of-fold；按动作类型分图。

**禁止的推断**：oracle repair高相关仍需actual action验证。

## F05 · 任务误差与恢复关联 (main)

**检验命题**：H1,H3。

**实验来源**：R03,R04,R07,A01。

**原始输入**：feature_errors + detections + GT。

**绘制内容/单位**：gap/short/boundary分层；feature误差与GT/AP变化。

**聚合/不确定性**：按视频聚合避免长背景支配；paired CI。

**禁止的推断**：feature MSE改善但AP不变必须显示。

## F06 · 三维执行和系统收益 (main)

**检验命题**：H4。

**实验来源**：D02,S02,J02,M01。

**原始输入**：execution.parquet + timeline.json + latency samples。

**绘制内容/单位**：每层Q/KV/FFN/light counts；成本/时间拆分。

**聚合/不确定性**：从actual operator hooks统计；单独绘图再LaTeX拼版。

**禁止的推断**：少训练参数不等于少执行；不乘三保留率。

## S01 · online/EMA与初始化轨迹 (supp)

**检验命题**：H1,H3。

**实验来源**：R03,R06。

**原始输入**：train logs + checkpoint eval。

**绘制内容/单位**：updates vs feature/GT loss/mAP，各自一图。

**聚合/不确定性**：同route和底座比较online/EMA，terminal/peak都标。

**禁止的推断**：EMA初值系数不是输出/mAP占比。

## S02 · 时间来源和空洞分布 (supp)

**检验命题**：H1,H5。

**实验来源**：R03,U04。

**原始输入**：frame ids + anchor contributors。

**绘制内容/单位**：pair跨度、max-gap、动作内部观察率分布。

**聚合/不确定性**：full/partial/short与duration分层。

**禁止的推断**：invalid physical padding不得算real observations。

## S03 · 深层漂移与退出接口 (supp)

**检验命题**：H3,H4。

**实验来源**：D01,D02,D03。

**原始输入**：layer feature captures + op counts。

**绘制内容/单位**：层号vs同轴误差；预测前/后task loss。

**聚合/不确定性**：teacher/current-student状态分别保存。

**禁止的推断**：缓存dense后层不等于递归student状态。

## S04 · 空间证据和heavy图 (supp)

**检验命题**：H4,H5。

**实验来源**：S01,S02,S03。

**原始输入**：real frame + spatial gate + patch trace。

**绘制内容/单位**：真实frame、heavy block、必要小动作证据。

**聚合/不确定性**：同图同尺度；从固定案例库抽取并加失败。

**禁止的推断**：attention热图不是因果重要性。

## S05 · DETAD式误差分解 (supp)

**检验命题**：H1,H5。

**实验来源**：A01。

**原始输入**：full detections + annotation。

**绘制内容/单位**：定位/分类/重复/背景/漏检；oracle修正单独柱。

**聚合/不确定性**：明确IoU阈值、topk/NMS、类别定义。

**禁止的推断**：各oracle效果不可加为总损失。

## S06 · 路由局部性与组合交互 (supp)

**检验命题**：H2。

**实验来源**：U02,U03。

**原始输入**：S0/S1 routes + swap labels。

**绘制内容/单位**：局部16/近邻/全局；单交换总和vs实际S1。

**聚合/不确定性**：相同交换/候选查询预算。

**禁止的推断**：取消T24不证明16cell先验有害。

## S07 · 完整/部分/短窗成本 (supp)

**检验命题**：H4。

**实验来源**：M01。

**原始输入**：profile per-window。

**绘制内容/单位**：有效数/物理数vsMAC/ms/decodercost。

**聚合/不确定性**：full768/partial503/short253均留原始记录。

**禁止的推断**：单个窗口不是dataset平均。

## S08 · 时延分布与端到端 (supp)

**检验命题**：H4。

**实验来源**：M01。

**原始输入**：raw latency/coldwarm metadata。

**绘制内容/单位**：ECDF、median/p95、E2E拆分各自一图。

**聚合/不确定性**：同卡交错、多轮、保留outliers。

**禁止的推断**：FLOPs小不能代替E2E快。

## S09 · 不确定性—任务风险 (supp)

**检验命题**：H5。

**实验来源**：U02,A01。

**原始输入**：predicted risk + actual gains/errors。

**绘制内容/单位**：reliability/risk-coverage vscompute。

**聚合/不确定性**：校准来自train OOF；test仅固定评估。

**禁止的推断**：不称无条件或conformal保证。

## S10 · 预算泛化与种子 (supp)

**检验命题**：H4,H5。

**实验来源**：K01,G01,G02。

**原始输入**：per-seed per-budget result tables。

**绘制内容/单位**：预算/mAP、paired seed differences、second dataset。

**聚合/不确定性**：模型/teacher/训练成本分组。

**禁止的推断**：只有一个seed时不画虚假seed误差条。

## S11 · 失败与可观测性反例 (supp)

**检验命题**：H1,H5。

**实验来源**：A01。

**原始输入**：predeclared real failure cases。

**绘制内容/单位**：短动作完全缺失、重复合并、遮挡、多主体。

**聚合/不确定性**：含改进和退化，同budget原帧。

**禁止的推断**：不要只展最佳case或生成不存在画面。

## S12 · 等价性、误差和成本审计 (supp)

**检验命题**：H3,H4。

**实验来源**：C01,D03,M01。

**原始输入**：unit tests + tolerances + fp modes。

**绘制内容/单位**：dense-mask vscompact误差；clip/Q/KV counts。

**聚合/不确定性**：toy与real-GPU单独panel/data文件。

**禁止的推断**：CPU toy不是新GPUmAP复测。

## 主表

T1：S/B全部五阈值mAP、平均mAP、真实GFLOPs、model/E2E median和p95、teacher/更新/peak选择脚注。

T2：R/L/U核心匹配消融，逐项变化，不能仅full vs minus-many。

T3：相同FLOPs和相同时间分别匹配的静态/动态T/D/S分配。

T4：训练资源与执行协议：teacher来源、GT、预训练decoder、成功更新、查询teacher数量、GPU小时、显存、有效观察、profile scope。

## 排版

按已核查CVPR2026模板，单栏3.28125in、双栏6.875in。建议图内8–9pt（是可读性建议，不是官方硬阈值），线型/符号区分方法，默认matplotlib循环避免手工颜色暗示。统计图输出PDF/SVG和300dpi PNG；真实视频裁剪建议600dpi但不虚假超分。禁止3D柱图、双y轴混单位、截断轴夸大收益、平滑抹掉训练波动。每图旁存input source与过滤规则json；没有数据就拒绝生成结果图。
