# 交互结果展示补充

完整211视频/792窗口测量和原定主统计已完成后，补全展示口径。采集源49f74bd和原始窗口JSON不改，主统计及四条件AP不重跑，不新增GPU查询、模型或选择策略。

1. 主图增加方向明确的条件选择regret：T→S表示用T改变前的真实S收益选“保持/交换”，在T改变后沿用该选择，与同两候选的新背景最优相比。反向分别展示。它复用原已计算的conditional_regret/reverse_conditional_regret，不是learned router性能，也不是最优决策顺序定理。
2. 全部6336次no-op重放的raw cls/loc最大差为0。图中epsilon=0时将“Above replay”明确标为“Nonzero (replay=0)”，不把几乎100%的非零比例称为显著交互比例。
3. 现有原始loss交互值补充视频聚类统计；主图仍用同cube共同baseline total loss归一化，两种尺度分开列出。
4. 按既定探索项补全layer和baseline坐标GT关系分层，视频内样本等权，视频间等权，10000次bootstrap。不同GT组的条件视频总体不同；所有区间未作多重比较校正，不充当组间差异检验，不据此宣称boundary dominance。部分动作区间可仅在窗口内部分可见，这里标GT关系，不冒充专门的完整动作边界实验。
5. 时间×层图为近零实测点加边框，区分“测得近零”和“未查询的空白”。waterfall标明每项实际数值和统一显示尺度，避免小项被T主效应淹没；原inclusion–exclusion分解不变。

正式输出由原4页扩为5页主图和1页探索附录；每页分别PDF/SVG/PNG，合并interaction_atlas.pdf。追加展示不改变INTERACTION_GATE；最终结构判断仍需独立learned matched-cost及实用等效验证。
