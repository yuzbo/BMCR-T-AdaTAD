# 2026-09-15 交互重分析与跨轴补测

本轮已经完成旧 Atlas 的轴内总损失联合记录重分析，并独立启动跨轴补测。旧轴内统计不能充当 T×S / T×D / D×S 结果。

## 已有数据实际告诉了什么

下表为固定分层顺序、16次局部选择改变的真实联合结果。统一 G=Lbase−Lchanged，O/D/S 相对旧 R 仅在新派生分析中翻转符号，原文件不改。分母仍为原官方dense窗口总损失。视频等权；95% CI为10000次video-cluster bootstrap，token/window不是独立抽样单位。

| 轴内组合 | mean(I) [95% CI] | 平均绝对交互 [95% CI] | 视频/窗口 |
|---|---|---|---|
| O | 5.78e−5 [−5.05e−4, 6.25e−4] | 1.060e−3 [5.85e−4, 1.69e−3] | 211/792 |
| D | 2.56e−6 [−8.40e−8, 6.49e−6] | 1.118e−5 [8.60e−6, 1.50e−5] | 211/792 |
| S | 4.74e−8 [−1.62e−6, 1.95e−6] | 8.327e−6 [6.82e−6, 1.02e−5] | 211/792 |
| T | 4.20e−5 [−3.91e−5, 1.29e−4] | 5.331e−4 [3.80e−4, 7.77e−4] | 186/767 |

T在16次联合改变时还需满足全部交换不冲突，因此覆盖少于单次T交换的187视频；不把不适用记录填0。新数值包含median及2/4/16等尺寸，详见analysis/interaction_within_s/within_axis.json与output/interaction_within_s/numeric_tables.md。

**mean接近0与所有样本接近0不是同一个结论。** 当前数据确实存在正负抵消；但旧联合记录没有匹配的I_null，也没有分别保存cls/loc联合损失。因此这里不报告“显著协同比例”，不事后拆分cls/loc，不用非零E|I|证明跨轴强依赖或Joint Planner必要。

## 新测量已锁定并启动

协议INTERACTION_PROTOCOL.zh.md / interaction_protocol.json。采集源49f74bd5601bc298997921a93193109156adf546，独立目录results/interaction_v1_s，官方S epoch59 EMA、strict499键和原V2 light均冻结。

- 开发技术验收覆盖完整窗口、尾窗口及67个有效帧的真实短视频；新旧T全heavy路径的特征/损失/算子成本差均为0，normalizer与temporal_size恢复正确。
- 完整局部8cell固定算子预算相同；没有合法T时只测D×S四格并明确T不适用。T后的D/S位置是同规则在新support重定的policy-level estimand，保留真实RGB pair及native/空间身份。
- 2026-09-15 17:59:27开始完整211视频/792窗口，GPU1 PID284842。每窗四组局部交换，另有T在S-full/sparse背景下的四格及八次no-op cube。GPU0保留RFV；queue/interaction_pause.json可要求在完整窗口保存后让出。
- 分析源1b3850e6db36fbfaaf43fdc71ea1e39ea22af065，finish观察器PID287713。采集完整后自动CPU统计、四条件dataset AP及10k配对bootstrap、生成新PDF；之后仍需本任务下载、逐页视觉QA。

状态：**INTERACTION_GATE = INCONCLUSIVE**。跨轴正式数据未齐，不报告部分数据方向，也不把技术no-op=0当成科学结果。最终Gate要区分数值可测、任务实际量级、条件决策regret与learned matched-cost收益。

## 已完成的原S恢复配对结论

同一V2 checkpoint/同support的component比较：physical−packed为+3.1068 mAP点，95% CI [2.1563,3.9722]，执行GFLOPs差0；Cross−physical为+0.2544 [−0.7276,1.2136]；MD-on−MD-off为+0.00050 [−0.00520,0.00934]。各差值由相同10000次video draws逐次相减得到，未用单组CI拼差值。物理时间恢复已有明确证据；当前同checkpoint的Cross额外收益与multidepth贡献没有稳定增益证据，不能包装成独立重训后的架构优劣。

数学核验中已确认pair/triple、视频权重和条件两选择regret。AP的原始符号与loss相反，因为AP越高越好：I_MAP=(M11−M01)−(M10−M00)。不能误把它改成loss对比的符号。
