# 固定提交后的并行科研实施总控任务书

## 可直接交给总控agent的任务

你负责在 `yuzbo/BMCR-T-AdaTAD@5485c3cb8dfbd823acbb177a0b51292dadc44360` 上实现、验证、分析有限的新模型族，并产出可支持或推翻论文主张的完整证据。先读本包REPORT、ACCESS_MANIFEST、SOURCE_MAP、EXPERIMENTS、FIGURES及原仓库讨论入口。**本任务书是科研与代码规格，不表示本文已经执行服务器部署；实际执行须由作者在目标环境授权。**

H65、BMCR、Cross/FPW为作者内部方法；DS3和原clip选择不再进入实验，单candidate frame选择保留；16-observation内部计算打包不等于clip选择。所有完整配置只运行seed42一次。旧3407等结果/权重保留真实标签，不重标成42。主要评价完整模型计算量—最佳完整TAD mAP，保留终点与选模；latency/memory/E2E/training cost需要报告，但不构成启动或淘汰条件。

保留现有52配置计划，不复制同科学配置重复训练，不恢复已退役55条旧FPW。对已提交课程不得无记录改recipe。识别queued/running/complete/blocked/retired五种状态，不能把配置注册写成完成。新增模型族同时实施与排队，不以另一模型mAP为依赖；代码测试、合法权重/数据与本课程前序checkpoint是允许依赖。资源并发以现场合法上限为准，不依据本文件中的历史job ID取消任何作业。

只实现EXPERIMENTS.json中有限配置和已存在对照。预检不计正式课程初始化或更新：成功后重新从固定资产开始正式seed42课程。重复launch需拒绝或恢复原run；分片/resume不算新run。失败记录异常和已成功更新，禁止悄悄换seed重试择优。

最终交付源码、严格参数与状态合同、完整结果、真实execution trace、teacher/query账本、测试峰值/终点、公开基线与内部消融分表、可复现矢量图、每项主张支持/反证/未决清单。不预填mAP、FLOPs或speedup；未测字段null。若简单模型优于复杂模型，保留该结论并收缩最终方法。

## 1. 并行agent与修改边界

### Agent A — 证据、旧路径和实验合同

负责 `research/paper/review_5485/`、manifest、results导出与tests，不改历史metric。

核实52/217、已有六训练＋两预检状态与现场最新状态，生成“固定快照”和“新增实时”两份清单。完整读取本次未能读取的live_snapshot及原远端receipts，把能够确认的差异追加，不改固定报告。核对FrameModel R01_anchor_head是旧head参数的original-axis读出，不是旧rank预测。

测试旧路径native提取、K768同**继承权重同图**恒等，而非错误要求等于官方不同权重。记录actual source/backbone/scout/head/R03/MAE权重role；`no_external_loss`与无teacher查询分开。对Paper learned_state生成部署所需冻结资产列表或可选合并权重包。

输出 `asset_provenance.json`、`source_contract.json`、`state_transition.csv`、`dependency_inventory.json`。

### Agent B — 上下文/状态保持引擎

负责 `h65/paper/engine.py` 的**默认关闭扩展**、新 `support_targets.py` 与引擎测试。不得改旧recipe的默认操作。

实现两类掩码：`depth_admitted`与`ffn_heavy`。给depth-excluded位置新增独立light更新选项，不能复用只服务admitted-minus-heavy的mask而称完成。attention阶段与FFN阶段分别处理，保持norm、残差、TIA顺序；Adapter已含identity，禁止再加一次输入。

full-KV执行必须拆Q/K/V权重，只对selected Q和output投影做重更新，KV来自完整当前state。compact-KV与full-KV不得用相同算量估值。空query、单token、非均匀per-time、odd-valid和全门恒等全部测试。

新增层级ExposureTrace：每层 `q_ids/kv_ids/ffn_heavy_ids/ffn_light_ids/depth_bypass_ids/age_before_reentry/valid_mask`。最后dense层会使旧last_heavy_depth全部等于D，因此它不能独自描述深度差异。空间质量报告在admitted集合内的conditional比例，并另报全grid比例。

只在完整空间grid上执行global-TIA(K/2)；预算K变更后重建/核对temporal_size，禁止把compact-token序列直接reshape进TIA。

### Agent C — 支持匹配蒸馏与PBD控制

负责 `h65/paper/objectives.py` 新分支、`support_targets.py`、静态/PBD模型控制。

从本课程合法初始encoder建立冻结same-support教师：**同一S_K、同一16打包、同一像素增强、同一padding、D1/S1**。输出指定层TIA前后空间状态、池化状态和轻/重残差target。禁止把original384与selected192逐index回归。

同时区分same-input operator target（heavy FFN在当前student输入上）与same-support trajectory target；先诊断二者，不以dense输入拟合替代实际student分布验证。

实现 `same_support_feature_loss` 与可选relation/normalized residual，先使用单一可解释默认；lambda等超参在训练诊断中固定，不能按完整测试反复搜索。对照无loss时仍保存同支持诊断目标/query成本，或清楚披露query差异。

PBD官方参照固定 `74bce476...`。单独维护faithful reproduction与本框架PBD-style：训练集候选层评价、逐层删除、按原层index对齐、适配head/norm/PEFT、阶段LoRA合并/optimizer重启等记录齐全。若改官方softmax KD为本point head Bernoulli，明确是adaptation。

渐进日程属于同一完整课程。预先定义终点层数/预算和阶段更新分配，不以测试mAP触发反复重训；每轮候选评价的训练样本数、forward数和时间计入总训练开销。

### Agent D — Decoder初始化与采样兼容

负责 `h65/frame/mae_init.py`、`h65/paper/decoder.py` 扩展与公平初始化配置。

审计真实官方pretrain checkpoint：key/shape/深度/heads/norm/bias，确认不是encoder-only fine-tune权重。报告copied/reset/new参数数目；不能只strict=False加载不报缺失。MAE random与pretrained严格同架构和随机新增层初值。

增加fresh Cross不加载R03，对比现有MAE双配置；hot-R03作为单独初始化来源，额外预训练更新数入账。输入统计适配只作用decoder输入支路，保留base feature的native幅度。所有query＋anchor拼接输出在原query ID上解码，不假设visible是原axis子集。

如增加覆盖/跨度采样，仍按单candidate决策；不能把两个candidate强制绑定后仍称单帧对照。保持K、已学scout和GT/readout合同，记录selection变化与是否改变oldBMCR conditional含义。

### Agent E — 路由、干预与校准

负责 `h65/paper/routing.py`、`interventions.py`、`calibration.py`。

保留actual_delta与repair_delta。新增plan-aware FrameRouter：条件含K、D/S、作用层或预算id；同一个真实frame swap在多个计划产生标签，但**总查询数与旧对照匹配**。对潜在多交换先单步，避免把局部效用相加当全局最优。

窗口BudgetRouter继续是15计划有限决策。报告每计划实际曝光、重复坍缩和budget分布。固定菜单控制必须有相同训练计划覆盖的版本，另给独立适配静态版本。若扩充状态，比较固定参数/查询成本，不能只靠更大网络增加收益。

OOF只分离router拟合视频，不声称整个表征未见视频；coverage-after-rescaling不是独立保证。另存校准前后gain误差和actual regret。部署所有route API只接视频/metadata/预算，不接GT。

### Agent F — 训练、部署与资源

负责现有 `tools/paper_*` 与新配置/manifest。禁止以此轮报告直接修改正在运行的课程。

所有完整run seed42，登记source commit、recipe、初始化、父资产、数据ID、epochs/schedule_epochs、eval candidates、成功updates、EMA状态、train-time查询。40轮80日程对照和full前40配对，80最佳/终点另报。

同一模型的阶段推进只依赖自身checkpoint；P/D/R模型族互相没有mAP gate。去掉旧多seed部署条目，不复制同run。数据未READY阻塞仅相关数据集；CPU/真实GPU预检失败阻塞相应引擎接口，不暗中当性能淘汰。

活动上限读取现场政策；当前固定计划max_live_jobs10/max_live_train8是已知配置，不是越权使用许可。控制器单一ownership、幂等提交、slice-resume、日志不可覆盖。仅作者明确授权后实际submit，本文阶段只emit/dry-run。

### Agent G — 图表、统计与论文

负责 `paper_review/analysis` 和 `paper_review/figures`，不得更改模型或过滤失败结果。

按FIGURES中的字段读原始reciepts、predictions、trace并保存每图source manifest。新seed42、旧3407、公开reported三种结果分开。只有同checkpoint可连预算曲线；最佳FLOPs与mAP来自同一权重/路由；只发布有真实值的点。终点必须存在或标未测。

固定模型的视频成对bootstrap重算整体AP；重复抽中video必须改重采样ID。不同epochs/不同configs分布不命名为training-seed variance。单seed不画虚构std。空间/时间案例必须用真实帧，GT只离线join。

交付每个H1/H2/H3命题的支持/反证/未决段落。保留简单TCN、uniform、静态PBD对照；若三维不优于更简单模型，改论文贡献，不能为故事隐藏负结果。

## 2. 关键接口合同（新规格，实施后才能执行）

```python
# 图级动作是显式的，不能只有一个抽象retention ratio。
ExecutionPlan = {
    'frames': int, 'depth_capacity': float, 'space_capacity': float,
    'kv_mode': 'compact|full', 'depth_bypass': 'hold|light',
    'static_keep_original_ids': list[int] | None,
    'exposure_schedule_id': str,
}

# 同支持教师与学生必须共享这些元数据。
SupportKey = {
    'candidate_ids': Tensor, 'source_frame_times': Tensor,
    'contributor_valid': Tensor, 'spatial_shape': tuple,
    'augmentation_id': str, 'pack_order': str,
    'tia_temporal_size': int, 'reference_checkpoint': str,
}

# 不以最后一层的单个整数代替完整计算历史。
ExposureTrace = {
    'query_ids_by_layer': list[Tensor],
    'kv_ids_by_layer': list[Tensor],
    'ffn_heavy_ids_by_layer': list[Tensor],
    'ffn_light_ids_by_layer': list[Tensor],
    'depth_bypass_ids_by_layer': list[Tensor],
    'age_before_reentry_by_layer': list[Tensor],
    'actual_macs_by_component': dict,
}
```

Tensor `[B,K/2,Hg,Wg,C]`属于selected support；`[B,C,384]`属于original query；detector THUMOS `[B,C,768]`，ANet配置按自己的mask长度，不使用同shape等价捷径。

## 3. 必须完成的一致性测试

1. 旧默认recipe逐层/native/head输出不变；数据合同拒绝rank/original混用。
2. all-heavy同权重同图恒等；不得要求不同encoder资产等于官方reference。
3. selected-Q/full-KV在同输入同selected query下与dense该query数值一致；训练多层结果需再验证。
4. dense-mask与compact在相同mask、同context支持、同随机实现下逐层一致；删除KV与仅mask输出不能称相同图。
5. depth-light只更新预定depth-bypass，space-light只更新预定FFN-bypass，索引无重叠双算；统计和实际hook一致。
6. support teacher与student源帧、padding、位置、TIA一致；不满足时明确拒绝state KD或标近似。
7. MAE复制键/shape/输入adapter/test；zero head等于新插值而不是旧BMCR；R03 loaded状态单独测试。
8. teacher frozen、student GT梯度非零；sharedfull stop-gradient和fullGT各路径单独核验。
9. no_external_loss/no_teacher_queries/noR03/recognition-only资产与调用合同分别断言。
10. per-counterfactual loss normalizer不变；没有训练GT注入推理router；多预算标签条件一致。
11. odd/short/重复physical frame有效数与physical执行数各自正确；ANet192detector与RGB轴转换正确。
12. 控制器幂等、seed42单run、resume不重起课程、未测null、每图来源追溯。

## 4. 生产命令与执行边界

下面第一段是建立代码工作树的标准git操作，需要在作者已经拥有的本地repo中运行；**本包没有执行它**。

```bash
BASE=5485c3cb8dfbd823acbb177a0b51292dadc44360
# 先确认该对象已存在；不要默认分支替代。
git cat-file -e "${BASE}^{commit}"
git worktree add --detach ../tad-support-review "$BASE"
cd ../tad-support-review
git switch -c research/support-consistent-5485
```

现有生产工具需先读源码/`--help`，默认只预览：

```bash
python tools/paper_plan.py --dry-run
python tools/paper_train.py --help
python tools/paper_eval.py --help
# 下面只是当前已有配置的技术预览，不启动GPU训练。
python tools/paper_train.py --config configs/paper/thumos_s_point_full_seed42.json --dry-run
```

新增工具为**要求agents实现的CLI，不是仓库已有命令**：

```bash
python tools/paper_review_plan.py --spec research/paper/review_5485/EXPERIMENTS.json --emit-only
python tools/paper_support_check.py --config <new_config.json> --device cpu
python tools/paper_review_analyze.py --manifest <completed_runs.json> --output paper_review/analysis
python tools/paper_review_plot.py --manifest <completed_runs.json> --figures <figure_spec.json> --output paper_review/figures
```

`paper_review_plan.py`必须写入完整可执行JSON配置、展开技术DAG、检查52既有配置去重、禁止score/mAP依赖，不直接submit。作者后续授权后，总控复用当前OWN dispatcher及其现场验证过的参数提交，**不提供猜测的sbatch/scancel命令或历史job ID**。

每个新增正式任务的机器记录必须含：`config_id, source_commit, seed=42, runs=1, stage, success_updates, initialization_contract, support_target, actual_execution, external_query_count, shared_full_count, support_teacher_count, eval_candidates, selection_rule, best_state, terminal_state, status`。
