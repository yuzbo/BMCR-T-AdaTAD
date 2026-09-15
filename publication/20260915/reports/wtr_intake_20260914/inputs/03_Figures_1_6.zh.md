# Figure 1–6：完整论文证据线与英文caption草稿

下面均为设计与caption模板，非已有实测图。正文中的find/show/improve只有在相应数据产生后才能改成结果性断言。不先画假想Pareto点；图源必须带source_revision、checkpoint、数据split、预算、teacher与成本边界。

## Figure 1 — What does TAD need: observations, access, or updates?

(a) 完整物理时间轴与逐帧heavy selection示意；明确16只是执行打包。(b) shape-preserving observation干预与input-preserving update干预。(c) 同一个视频的GT、dense预测、observation removal effect、update effect；代表样本按开发集median规则选择并给失败例。(d) state/access/update三个成本层次，未观测节点不是已保留全部信息。

**Caption.** *Figure 1. Separating observations, state access, and state updates in temporal action detection.* We distinguish input-preserving computation interventions from observation interventions that preserve tensor shape and physical timestamps. The same frozen detector and video are used for each paired comparison. Frame-level selection is distinct from the subsequent 16-observation backbone packing. The illustrative example motivates a systematic measurement of task-conditioned necessity; it is not itself evidence that arbitrary video information can be removed.

## Figure 2 — Conditional utility and collective removability

(a) cls/reg signed Δloss分布，正/负/近零分别画；不同near-zero阈值敏感性。(b) no-op和独立重复估计数值噪声。(c) LOO与2/4动作coalition联合删除、interaction分布，强调替代/互补。(d) 若使用Lorenz/Gini，只对positive part并显示负部比例和video-cluster CI；不是信息熵。

**Caption.** *Figure 2. Conditional task utility under individual and joint interventions.* Signed changes in classification and localization loss are reported separately for observation and computation interventions. Null interventions characterize numerical variation, while joint interventions test whether individually low-impact actions remain removable together. Concentration statistics, where shown, apply only to explicitly identified positive effects. Confidence intervals use video-level clusters rather than treating correlated windows or tokens as independent samples. Local surrogate effects are not equated with dataset-level AP.

## Figure 3 — Why does the existing MoD route lose accuracy?

(a) incoming attention vs实际conditional update value scatter/rank，按当前合法action域比较。(b) S40/S60实际plan占比和cap暴露，不能只列nominal budget。(c) score-QK、heavyFFN、light、TIA成本拆解。(d) 同support状态误差attention/preTIA/postTIA与当前state heavy-light差异，任务加权proxy对比norm。已存在结果与待跑结果分面，不混snapshot。

**Caption.** *Figure 3. Diagnosing attention-based update allocation in H65.* We compare incoming-attention scores with measured task improvements from matched-capacity slot exchanges, and report the computation plans actually selected by the deployed router. Operator accounting separates routing overhead from saved heavy computation. State diagnostics distinguish divergence along a frozen same-support reference trajectory from heavy-versus-light residuals evaluated at the student's current state. These measurements test routing-target and state-update hypotheses without attributing all performance changes to a single mechanism.

## Figure 4 — Is adaptive allocation useful at matched cost?

(a) Avg-mAP versus完整operator-measured GFLOPs；dense/native K384 direct/adapted、UniformFull、Static、attention、learnedvalue、label-assisted search均保留。(b) mAP@0.7与相同cost或完整曲线。(c) Global mAP差video-cluster CI，重新排序匹配重算AP。(d) 同预算seed/curriculum/teacher/decoder匹配清单；搜索预算与推理预算分开。

**Caption.** *Figure 4. Allocation quality at matched computational cost.* Full-dataset Avg-mAP and mAP at tIoU 0.7 are plotted against complete measured inference cost. Static, uniform, attention-based, and learned-value policies are compared under documented training and initialization controls. Label-assisted search is shown only as a diagnostic reference: a finite greedy or surrogate-loss search is not a guaranteed upper bound. Paired intervals recompute dataset AP under video-cluster resampling. Both intermediate checkpoints and frozen endpoints are identified explicitly.

## Figure 5 — Preserving localization state under irregular computation

固定selectedframes和heavybackbone，比较Interp、Cross、continuous-time、static edges、dynamic/no-referral/referral。(a) 原始query/contributor/support跨度与edge图。(b) boundary/short-duration误差和AP，但必须先测，不预设boundary必高。(c) degree/referral成本—性能；fullKV投影完整计费。(d) time×depth和spatial utility地图、单轴/联合budget互补，仅对已测轴给结论。

**Caption.** *Figure 5. Physical-time state recovery with sparse heavy evidence.* Under identical selected observations, we compare interpolation, the existing cross-attention decoder, continuous-time reconstruction, and static or dynamically referred sparse access. Source timestamps, contributor validity, and support spans are retained rather than replaced by packed indices. Boundary and duration-stratified analyses test localization effects, while degree and referral sweeps expose the accuracy–access–cost trade-off. Graph topology is a candidate mechanism, not an assumed requirement of the final method.

## Figure 6 — Do future-value targets help?

(a) 同action/support三checkpoint Va,Vp,Vf真实效用rank漂移。(b) noKD/postβ1/EMA/future/commonbudgetextra-update；同候选域teacherforecast。(c) 强制一样teacherquery成本或reportcost曲线的最终mAP改善。(d) 最终冻结T/TD/TS/TDS和跨数据集结果，未完成留白；与旧版PaperModel不换名。

**Caption.** *Figure 6. Evaluating trajectory-based computation-value supervision.* The same actions and physical supports are re-executed at multiple training checkpoints to measure target drift. Frozen-post, EMA, and function-extrapolated teachers are compared on a common candidate support, with beta equal to one serving as the matched non-extrapolating control. Predictive agreement with later measured utility is distinguished from downstream allocation performance. Additional teacher queries and optimization costs are accounted for explicitly. Only validated extensions and final-version transfer experiments support the concluding method claims.

## 论文结构与主张

Introduction：目标而非预设冗余 → controlled characterization → why attention centrality not update value → why original-time state and nonstationary targets matter。

Method：legal action/state definition；conditional exchange value；task-aware cheap routing；current-state repair；physical-time recovery；可删除的future teacher。

Experiments：协议和资源；characterization；MoD诊断；matched主结果；target和recovery消融；最终版本泛化；失败与成本。

候选贡献仅写成待验证命题：任务条件边际计算价值的实测刻画；与真实稀疏执行/原轴支持一致的allocation学习；受actual interventions校准的target构造。没有实验时不写“首次”“显著超越”“无需dense teacher”或“信息无损”。
