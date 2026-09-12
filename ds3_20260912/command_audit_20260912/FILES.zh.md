**保存的命令文件、任务书与模板逐文件索引**

本次登记197份物理文件实例，按来源区分。包含脚本、任务书、模板、测试及单列的依赖工具；不等于197项实验。

当前活动根目录为`h65_clean_adatad/ds3_20260912`。外部原件与暂停草稿的绝对链接只在本地有效，原件内容未因此公开上传。
参数为静态提取，不导入或执行脚本；无参数列表不代表已验证可直接运行。同名脚本在不同工作树中可能采用不同配方。

**ASFormer_reference：5份。**

|文件|类别/作用|源码声明的参数|
|---|---|---|
|[batch_gen.py](<C:/Users/skywalker/Documents/ChatGPT/H65/h65_clean_adatad/references/ASFormer/batch_gen.py>)|Adapted from https://github.com/yabufarha/ms-tcn|`无argparse参数/见源码`|
|[eval.py](<C:/Users/skywalker/Documents/ChatGPT/H65/h65_clean_adatad/references/ASFormer/eval.py>)|dependency_tool_or_module|`--dataset --split --result_dir`|
|[grid_sampler.py](<C:/Users/skywalker/Documents/ChatGPT/H65/h65_clean_adatad/references/ASFormer/grid_sampler.py>)|This file is a implementation of Time Series Wrapper.|`无argparse参数/见源码`|
|[main.py](<C:/Users/skywalker/Documents/ChatGPT/H65/h65_clean_adatad/references/ASFormer/main.py>)|dependency_tool_or_module|`--action --dataset --split --model_dir --result_dir`|
|[model.py](<C:/Users/skywalker/Documents/ChatGPT/H65/h65_clean_adatad/references/ASFormer/model.py>)|dependency_tool_or_module|`无argparse参数/见源码`|

**BMCR_review1：4份。**

|文件|类别/作用|源码声明的参数|
|---|---|---|
|[check_actual_scout.py](<C:/Users/skywalker/Documents/ChatGPT/H65/h65_clean_adatad/reviews/20260911_external_bmcr_review/check_actual_scout.py>)|Read-only CPU probe of the actual published scout and optimizer function.|`--repo --out`|
|[check_records_and_math.py](<C:/Users/skywalker/Documents/ChatGPT/H65/h65_clean_adatad/reviews/20260911_external_bmcr_review/check_records_and_math.py>)|Independently recompute review arithmetic from local experiment records.|`无argparse参数/见源码`|
|[external_script_rerun/independent_checks.py](<C:/Users/skywalker/Documents/ChatGPT/H65/h65_clean_adatad/reviews/20260911_external_bmcr_review/external_script_rerun/independent_checks.py>)|外部报告纯算术检查，不是模型训练/独立mAP复测|`无argparse参数/见源码`|
|[inputs/package/independent_checks.py](<C:/Users/skywalker/Documents/ChatGPT/H65/h65_clean_adatad/reviews/20260911_external_bmcr_review/inputs/package/independent_checks.py>)|外部报告纯算术检查，不是模型训练/独立mAP复测|`无argparse参数/见源码`|

**BMCR_review2：2份。**

|文件|类别/作用|源码声明的参数|
|---|---|---|
|[checks_rerun/checks.py](<C:/Users/skywalker/Documents/ChatGPT/H65/h65_clean_adatad/reviews/20260911_external_bmcrt_review_v2/checks_rerun/checks.py>)|第二份外部报告纯算术检查，不是模型训练/独立mAP复测|`无argparse参数/见源码`|
|[inputs/package/checks.py](<C:/Users/skywalker/Documents/ChatGPT/H65/h65_clean_adatad/reviews/20260911_external_bmcrt_review_v2/inputs/package/checks.py>)|第二份外部报告纯算术检查，不是模型训练/独立mAP复测|`无argparse参数/见源码`|

**active_ds3：30份。**

|文件|类别/作用|源码声明的参数|
|---|---|---|
|[bmcr_fidelity_20260912/README.md](<../../bmcr_fidelity_20260912/README.md>)|experiment_plan_or_receipt|`无argparse参数/见源码`|
|[docs/BMCR_THREE_DIMENSION_RESEARCH_PROMPT.zh.md](<../../docs/BMCR_THREE_DIMENSION_RESEARCH_PROMPT.zh.md>)|research_prompt|`无argparse参数/见源码`|
|[docs/EXTERNAL_REVIEW_PROMPT.md](<../../docs/EXTERNAL_REVIEW_PROMPT.md>)|research_prompt|`无argparse参数/见源码`|
|[ds3_20260912/DEPLOYMENT.md](<../DEPLOYMENT.md>)|experiment_plan_or_receipt|`无argparse参数/见源码`|
|[ds3_20260912/PLAN.md](<../PLAN.md>)|experiment_plan_or_receipt|`无argparse参数/见源码`|
|[ds3_20260912/site/run_job.sh](<../site/run_job.sh>)|站点实际执行包装器；须按所在实验目录区分|`无argparse参数/见源码`|
|[fidelity_20260911/DEPLOYMENT_20260911.md](<../../fidelity_20260911/DEPLOYMENT_20260911.md>)|experiment_plan_or_receipt|`无argparse参数/见源码`|
|[fidelity_20260911/PLAN.md](<../../fidelity_20260911/PLAN.md>)|experiment_plan_or_receipt|`无argparse参数/见源码`|
|[resources.example.sh](<../../resources.example.sh>)|资源环境变量示例，需实际站点配置|`无argparse参数/见源码`|
|[tools/build_command_catalog_20260912.py](<../../tools/build_command_catalog_20260912.py>)|本次静态命令索引生成，不运行实验|`无argparse参数/见源码`|
|[tools/build_research_context_20260912.py](<../../tools/build_research_context_20260912.py>)|固定提交外部研究上下文生成|`无argparse参数/见源码`|
|[tools/ds3_dispatch.py](<../../tools/ds3_dispatch.py>)|当前73阶段调度与资源上限管理|`--once`|
|[tools/ds3_eval.py](<../../tools/ds3_eval.py>)|单policy全211视频评测与三个profile场景|`--backbone --policy --epoch --profile-only --metrics-only`|
|[tools/ds3_preflight.py](<../../tools/ds3_preflight.py>)|S/B dense/compact/稀疏MLP/冻结/真实更新/EMA GPU预检|`--backbone`|
|[tools/ds3_select.py](<../../tools/ds3_select.py>)|T24A总5..80每5轮的峰值选模|`无argparse参数/见源码`|
|[tools/ds3_site.py](<../../tools/ds3_site.py>)|建立本项目数据、官方TAD、MobileNet资源链接|`无argparse参数/见源码`|
|[tools/ds3_train.py](<../../tools/ds3_train.py>)|80轮D1辅助训练，pilot100步属于同一轨迹|`--backbone --pilot --workers`|
|[tools/fidelity_compare.py](<../../tools/fidelity_compare.py>)|修正H65峰值/终点与旧/官方结果比较|`无argparse参数/见源码`|
|[tools/fidelity_dispatch.py](<../../tools/fidelity_dispatch.py>)|修正H65专用24阶段调度；没有修正BMCR注册|`--once`|
|[tools/fidelity_finalize.py](<../../tools/fidelity_finalize.py>)|修正H65最终证据整理|`无argparse参数/见源码`|
|[tools/fidelity_progress.py](<../../tools/fidelity_progress.py>)|修正H65中间结果汇总|`无argparse参数/见源码`|
|[tools/fidelity_select.py](<../../tools/fidelity_select.py>)|修正H65总25..60测试峰值选择|`--backbone`|
|[tools/full_audit.py](<../../tools/full_audit.py>)|训练来源128窗口反事实/贡献效用审计与尺度校准|`--backbone`|
|[tools/full_compare.py](<../../tools/full_compare.py>)|旧Phase2结果比较，不能直接代替修正BMCR的选模/汇总|`--backbone`|
|[tools/full_eval.py](<../../tools/full_eval.py>)|完整测试与profile；早期版本只终点，当前版本支持milestone|`--backbone --variant --profile-only --milestone --metrics-only`|
|[tools/full_train.py](<../../tools/full_train.py>)|H65/BMCR训练入口；配方与CLI细节依源码版本区分|`--backbone --phase --variant --preflight --workers`|
|[tools/retrospective_20260912.py](<../../tools/retrospective_20260912.py>)|历史完整结果回顾与图表生成|`无argparse参数/见源码`|
|[tools/run_full.sh](<../../tools/run_full.sh>)|训练脚本包装器；根目录为旧站点版，工作树内为通用模板|`无argparse参数/见源码`|
|[tools/summarize_audit.py](<../../tools/summarize_audit.py>)|效用审计结果整理|`--backbone`|
|[tools/verify_results.py](<../../tools/verify_results.py>)|已生成实验记录复核|`无argparse参数/见源码`|

**active_tests：9份。**

|文件|类别/作用|源码声明的参数|
|---|---|---|
|[tests/test_attention_macs.py](<../../tests/test_attention_macs.py>)|Check fused-attention matrix arithmetic against explicitly executed products.|`无argparse参数/见源码`|
|[tests/test_counterfactual_labels.py](<../../tests/test_counterfactual_labels.py>)|Real upstream focal assignment checks retention/insertion utility semantics.|`无argparse参数/见源码`|
|[tests/test_ds3.py](<../../tests/test_ds3.py>)|test_suite_source|`无argparse参数/见源码`|
|[tests/test_fidelity.py](<../../tests/test_fidelity.py>)|Failures that would invalidate the two requested H65 recipe corrections.|`无argparse参数/见源码`|
|[tests/test_fidelity_selection.py](<../../tests/test_fidelity_selection.py>)|test_suite_source|`无argparse参数/见源码`|
|[tests/test_full.py](<../../tests/test_full.py>)|Checks for failures that would invalidate the full-training comparison.|`无argparse参数/见源码`|
|[tests/test_profiler_hooks.py](<../../tests/test_profiler_hooks.py>)|A profiler must preserve tensor/tuple outputs while counting nested modules.|`无argparse参数/见源码`|
|[tests/test_scout.py](<../../tests/test_scout.py>)|test_suite_source|`无argparse参数/见源码`|
|[tests/test_transport.py](<../../tests/test_transport.py>)|Detect budget, gradient and physical-time failures before GPU experiments.|`无argparse参数/见源码`|

**current_heartbeat：1份。**

|文件|类别/作用|源码声明的参数|
|---|---|---|
|[automation.toml](<C:/Users/skywalker/.codex/automations/h65-ds3-80/automation.toml>)|automation_configuration|`无argparse参数/见源码`|

**draft_tests：6份。**

|文件|类别/作用|源码声明的参数|
|---|---|---|
|[tests/test_attention_macs.py](<C:/Users/skywalker/Documents/ChatGPT/H65/h65_clean_adatad/mainline_20260911/tests/test_attention_macs.py>)|Check fused-attention matrix arithmetic against explicitly executed products.|`无argparse参数/见源码`|
|[tests/test_counterfactual_labels.py](<C:/Users/skywalker/Documents/ChatGPT/H65/h65_clean_adatad/mainline_20260911/tests/test_counterfactual_labels.py>)|Real upstream focal assignment checks retention/insertion utility semantics.|`无argparse参数/见源码`|
|[tests/test_full.py](<C:/Users/skywalker/Documents/ChatGPT/H65/h65_clean_adatad/mainline_20260911/tests/test_full.py>)|Checks for failures that would invalidate the full-training comparison.|`无argparse参数/见源码`|
|[tests/test_profiler_hooks.py](<C:/Users/skywalker/Documents/ChatGPT/H65/h65_clean_adatad/mainline_20260911/tests/test_profiler_hooks.py>)|A profiler must preserve tensor/tuple outputs while counting nested modules.|`无argparse参数/见源码`|
|[tests/test_scout.py](<C:/Users/skywalker/Documents/ChatGPT/H65/h65_clean_adatad/mainline_20260911/tests/test_scout.py>)|test_suite_source|`无argparse参数/见源码`|
|[tests/test_transport.py](<C:/Users/skywalker/Documents/ChatGPT/H65/h65_clean_adatad/mainline_20260911/tests/test_transport.py>)|Detect budget, gradient and physical-time failures before GPU experiments.|`无argparse参数/见源码`|

**fidelity_completed：19份。**

|文件|类别/作用|源码声明的参数|
|---|---|---|
|[docs/EXTERNAL_REVIEW_PROMPT.md](<C:/Users/skywalker/Documents/ChatGPT/H65/h65_clean_adatad/fidelity_20260911/docs/EXTERNAL_REVIEW_PROMPT.md>)|research_prompt|`无argparse参数/见源码`|
|[fidelity_20260911/DEPLOYMENT_20260911.md](<C:/Users/skywalker/Documents/ChatGPT/H65/h65_clean_adatad/fidelity_20260911/fidelity_20260911/DEPLOYMENT_20260911.md>)|experiment_plan_or_receipt|`无argparse参数/见源码`|
|[fidelity_20260911/PLAN.md](<C:/Users/skywalker/Documents/ChatGPT/H65/h65_clean_adatad/fidelity_20260911/fidelity_20260911/PLAN.md>)|experiment_plan_or_receipt|`无argparse参数/见源码`|
|[fidelity_20260911/deployment.json](<C:/Users/skywalker/Documents/ChatGPT/H65/h65_clean_adatad/fidelity_20260911/fidelity_20260911/deployment.json>)|experiment_plan_or_receipt|`无argparse参数/见源码`|
|[fidelity_20260911/site/run_job.sh](<C:/Users/skywalker/Documents/ChatGPT/H65/h65_clean_adatad/fidelity_20260911/fidelity_20260911/site/run_job.sh>)|站点实际执行包装器；须按所在实验目录区分|`无argparse参数/见源码`|
|[fidelity_20260911/site/setup_resources.py](<C:/Users/skywalker/Documents/ChatGPT/H65/h65_clean_adatad/fidelity_20260911/fidelity_20260911/site/setup_resources.py>)|历史修正H65资源链接准备|`无argparse参数/见源码`|
|[resources.example.sh](<C:/Users/skywalker/Documents/ChatGPT/H65/h65_clean_adatad/fidelity_20260911/resources.example.sh>)|资源环境变量示例，需实际站点配置|`无argparse参数/见源码`|
|[tools/fidelity_compare.py](<C:/Users/skywalker/Documents/ChatGPT/H65/h65_clean_adatad/fidelity_20260911/tools/fidelity_compare.py>)|修正H65峰值/终点与旧/官方结果比较|`无argparse参数/见源码`|
|[tools/fidelity_dispatch.py](<C:/Users/skywalker/Documents/ChatGPT/H65/h65_clean_adatad/fidelity_20260911/tools/fidelity_dispatch.py>)|修正H65专用24阶段调度；没有修正BMCR注册|`--once`|
|[tools/fidelity_finalize.py](<C:/Users/skywalker/Documents/ChatGPT/H65/h65_clean_adatad/fidelity_20260911/tools/fidelity_finalize.py>)|修正H65最终证据整理|`无argparse参数/见源码`|
|[tools/fidelity_progress.py](<C:/Users/skywalker/Documents/ChatGPT/H65/h65_clean_adatad/fidelity_20260911/tools/fidelity_progress.py>)|修正H65中间结果汇总|`无argparse参数/见源码`|
|[tools/fidelity_select.py](<C:/Users/skywalker/Documents/ChatGPT/H65/h65_clean_adatad/fidelity_20260911/tools/fidelity_select.py>)|修正H65总25..60测试峰值选择|`--backbone`|
|[tools/full_audit.py](<C:/Users/skywalker/Documents/ChatGPT/H65/h65_clean_adatad/fidelity_20260911/tools/full_audit.py>)|训练来源128窗口反事实/贡献效用审计与尺度校准|`--backbone`|
|[tools/full_compare.py](<C:/Users/skywalker/Documents/ChatGPT/H65/h65_clean_adatad/fidelity_20260911/tools/full_compare.py>)|旧Phase2结果比较，不能直接代替修正BMCR的选模/汇总|`--backbone`|
|[tools/full_eval.py](<C:/Users/skywalker/Documents/ChatGPT/H65/h65_clean_adatad/fidelity_20260911/tools/full_eval.py>)|完整测试与profile；早期版本只终点，当前版本支持milestone|`--backbone --variant --profile-only --milestone --metrics-only`|
|[tools/full_train.py](<C:/Users/skywalker/Documents/ChatGPT/H65/h65_clean_adatad/fidelity_20260911/tools/full_train.py>)|H65/BMCR训练入口；配方与CLI细节依源码版本区分|`--backbone --phase --variant --preflight --workers`|
|[tools/run_full.sh](<C:/Users/skywalker/Documents/ChatGPT/H65/h65_clean_adatad/fidelity_20260911/tools/run_full.sh>)|训练脚本包装器；根目录为旧站点版，工作树内为通用模板|`无argparse参数/见源码`|
|[tools/summarize_audit.py](<C:/Users/skywalker/Documents/ChatGPT/H65/h65_clean_adatad/fidelity_20260911/tools/summarize_audit.py>)|效用审计结果整理|`--backbone`|
|[tools/verify_results.py](<C:/Users/skywalker/Documents/ChatGPT/H65/h65_clean_adatad/fidelity_20260911/tools/verify_results.py>)|已生成实验记录复核|`无argparse参数/见源码`|

**historical_diagnosis：11份。**

|文件|类别/作用|源码声明的参数|
|---|---|---|
|[compare_boundary_targets.py](<C:/Users/skywalker/Documents/ChatGPT/H65/h65_clean_adatad/diagnostics/h65_65_gap_20260911/compare_boundary_targets.py>)|CPU comparison of actual historical/current auxiliary target functions; no training.|`无argparse参数/见源码`|
|[sources/acquisition.py](<C:/Users/skywalker/Documents/ChatGPT/H65/h65_clean_adatad/diagnostics/h65_65_gap_20260911/sources/acquisition.py>)|diagnosis_or_source_snapshot|`无argparse参数/见源码`|
|[sources/historical_boundary_target.py](<C:/Users/skywalker/Documents/ChatGPT/H65/h65_clean_adatad/diagnostics/h65_65_gap_20260911/sources/historical_boundary_target.py>)|diagnosis_or_source_snapshot|`无argparse参数/见源码`|
|[sources/probe.py](<C:/Users/skywalker/Documents/ChatGPT/H65/h65_clean_adatad/diagnostics/h65_65_gap_20260911/sources/probe.py>)|diagnosis_or_source_snapshot|`--config --out-dir --device --epochs --batch-size --num-workers --lr --seed --probe-model --scout-spatial-size --mobilenet-sizes --tcn-variants --official-action-seg-backends --matrix-model-ids --matrix-model-tier --matrix-include-optional --matrix-temporal-hidden-dim --matrix-video-clip-len --matrix-video-anchor-stride --matrix-pretrained --no-matrix-pretrained --matrix-freeze-backbone --no-matrix-freeze-backbone --matrix-continue-on-model-error --mobilenet-weights-path --probe-checkpoint --mobilenet-pretrained --no-mobilenet-pretrained --freeze-backbone --no-freeze-backbone --coverage-only --coverage-budget-fraction --coverage-budget --boundary-radius --sample-jsonl --max-train-batches --max-val-batches --val-every-epochs --early-stop-patience --early-stop-min-epochs --early-stop-min-delta --early-stop-metric --early-stop-mode --log-every-batches --ann-file --class-map --train-data-path --val-data-path --test-data-path --train-subset-name --val-subset-name --test-subset-name --eval-window-overlap-ratio --eval-include-all-windows --fast-lowres-pipeline --probe-window-size --save-checkpoint`|
|[sources/score42_backend.py](<C:/Users/skywalker/Documents/ChatGPT/H65/h65_clean_adatad/diagnostics/h65_65_gap_20260911/sources/score42_backend.py>)|diagnosis_or_source_snapshot|`无argparse参数/见源码`|
|[sources/score42_recovery.sh](<C:/Users/skywalker/Documents/ChatGPT/H65/h65_clean_adatad/diagnostics/h65_65_gap_20260911/sources/score42_recovery.sh>)|历史65+恢复训练脚本快照，不属于当前新训练入口|`无argparse参数/见源码`|
|[sources/score42_selector.py](<C:/Users/skywalker/Documents/ChatGPT/H65/h65_clean_adatad/diagnostics/h65_65_gap_20260911/sources/score42_selector.py>)|diagnosis_or_source_snapshot|`无argparse参数/见源码`|
|[sources/score42_vit_adapter.py](<C:/Users/skywalker/Documents/ChatGPT/H65/h65_clean_adatad/diagnostics/h65_65_gap_20260911/sources/score42_vit_adapter.py>)|diagnosis_or_source_snapshot|`无argparse参数/见源码`|
|[sources/selector.py](<C:/Users/skywalker/Documents/ChatGPT/H65/h65_clean_adatad/diagnostics/h65_65_gap_20260911/sources/selector.py>)|diagnosis_or_source_snapshot|`无argparse参数/见源码`|
|[sources/stage1.py](<C:/Users/skywalker/Documents/ChatGPT/H65/h65_clean_adatad/diagnostics/h65_65_gap_20260911/sources/stage1.py>)|diagnosis_or_source_snapshot|`无argparse参数/见源码`|
|[sources/stage2.py](<C:/Users/skywalker/Documents/ChatGPT/H65/h65_clean_adatad/diagnostics/h65_65_gap_20260911/sources/stage2.py>)|diagnosis_or_source_snapshot|`无argparse参数/见源码`|

**legacy_root：21份。**

|文件|类别/作用|源码声明的参数|
|---|---|---|
|[EXPERIMENT_PLAN.md](<C:/Users/skywalker/Documents/ChatGPT/H65/h65_clean_adatad/EXPERIMENT_PLAN.md>)|experiment_plan_or_receipt|`无argparse参数/见源码`|
|[STATE.md](<C:/Users/skywalker/Documents/ChatGPT/H65/h65_clean_adatad/STATE.md>)|experiment_plan_or_receipt|`无argparse参数/见源码`|
|[phase2_20260910/PLAN.md](<C:/Users/skywalker/Documents/ChatGPT/H65/h65_clean_adatad/phase2_20260910/PLAN.md>)|experiment_plan_or_receipt|`无argparse参数/见源码`|
|[phase2_20260910/deployment.json](<C:/Users/skywalker/Documents/ChatGPT/H65/h65_clean_adatad/phase2_20260910/deployment.json>)|experiment_plan_or_receipt|`无argparse参数/见源码`|
|[tools/check_full_cpu.py](<C:/Users/skywalker/Documents/ChatGPT/H65/h65_clean_adatad/tools/check_full_cpu.py>)|旧正式路径CPU机制检查|`无argparse参数/见源码`|
|[tools/check_tia_connectivity.py](<C:/Users/skywalker/Documents/ChatGPT/H65/h65_clean_adatad/tools/check_tia_connectivity.py>)|非零Adapter跨clip连接与跨视频隔离检查|`无argparse参数/见源码`|
|[tools/final_report.py](<C:/Users/skywalker/Documents/ChatGPT/H65/h65_clean_adatad/tools/final_report.py>)|初期诊断报告生成|`无argparse参数/见源码`|
|[tools/full_audit.py](<C:/Users/skywalker/Documents/ChatGPT/H65/h65_clean_adatad/tools/full_audit.py>)|训练来源128窗口反事实/贡献效用审计与尺度校准|`--backbone`|
|[tools/full_compare.py](<C:/Users/skywalker/Documents/ChatGPT/H65/h65_clean_adatad/tools/full_compare.py>)|旧Phase2结果比较，不能直接代替修正BMCR的选模/汇总|`--backbone`|
|[tools/full_dispatch.py](<C:/Users/skywalker/Documents/ChatGPT/H65/h65_clean_adatad/tools/full_dispatch.py>)|旧Phase2的16阶段调度；已完成的历史控制器|`--once`|
|[tools/full_eval.py](<C:/Users/skywalker/Documents/ChatGPT/H65/h65_clean_adatad/tools/full_eval.py>)|完整测试与profile；早期版本只终点，当前版本支持milestone|`--backbone --variant --profile-only`|
|[tools/full_train.py](<C:/Users/skywalker/Documents/ChatGPT/H65/h65_clean_adatad/tools/full_train.py>)|H65/BMCR训练入口；配方与CLI细节依源码版本区分|`--backbone --phase --variant --preflight --workers`|
|[tools/gpu_probe.py](<C:/Users/skywalker/Documents/ChatGPT/H65/h65_clean_adatad/tools/gpu_probe.py>)|初期单视频/两窗口GPU诊断|`--steps --windows`|
|[tools/inspect_runtime.py](<C:/Users/skywalker/Documents/ChatGPT/H65/h65_clean_adatad/tools/inspect_runtime.py>)|资源/配置/模型运行环境检查|`无argparse参数/见源码`|
|[tools/one_swap_audit.py](<C:/Users/skywalker/Documents/ChatGPT/H65/h65_clean_adatad/tools/one_swap_audit.py>)|初期固定K单帧交换诊断|`无argparse参数/见源码`|
|[tools/preflight_full.py](<C:/Users/skywalker/Documents/ChatGPT/H65/h65_clean_adatad/tools/preflight_full.py>)|旧完整实验的S/B真实GPU预检|`--backbone`|
|[tools/prepare_publication.py](<C:/Users/skywalker/Documents/ChatGPT/H65/h65_clean_adatad/tools/prepare_publication.py>)|旧实验公开仓库整理工具|`无argparse参数/见源码`|
|[tools/run_full.sh](<C:/Users/skywalker/Documents/ChatGPT/H65/h65_clean_adatad/tools/run_full.sh>)|训练脚本包装器；根目录为旧站点版，工作树内为通用模板|`无argparse参数/见源码`|
|[tools/run_gpu_probe.sh](<C:/Users/skywalker/Documents/ChatGPT/H65/h65_clean_adatad/tools/run_gpu_probe.sh>)|初期4090探针的站点sbatch包装器，历史已完成|`无argparse参数/见源码`|
|[tools/summarize_audit.py](<C:/Users/skywalker/Documents/ChatGPT/H65/h65_clean_adatad/tools/summarize_audit.py>)|效用审计结果整理|`--backbone`|
|[tools/verify_saved_probe.py](<C:/Users/skywalker/Documents/ChatGPT/H65/h65_clean_adatad/tools/verify_saved_probe.py>)|初期已保存权重严格重载核验|`无argparse参数/见源码`|

**legacy_tests：7份。**

|文件|类别/作用|源码声明的参数|
|---|---|---|
|[tests/test_attention_macs.py](<C:/Users/skywalker/Documents/ChatGPT/H65/h65_clean_adatad/tests/test_attention_macs.py>)|Check fused-attention matrix arithmetic against explicitly executed products.|`无argparse参数/见源码`|
|[tests/test_counterfactual_labels.py](<C:/Users/skywalker/Documents/ChatGPT/H65/h65_clean_adatad/tests/test_counterfactual_labels.py>)|Real upstream focal assignment checks retention/insertion utility semantics.|`无argparse参数/见源码`|
|[tests/test_dispatch_scheduling.py](<C:/Users/skywalker/Documents/ChatGPT/H65/h65_clean_adatad/tests/test_dispatch_scheduling.py>)|Completed checkpoints can be tested while two training jobs keep running.|`无argparse参数/见源码`|
|[tests/test_full.py](<C:/Users/skywalker/Documents/ChatGPT/H65/h65_clean_adatad/tests/test_full.py>)|Checks for failures that would invalidate the full-training comparison.|`无argparse参数/见源码`|
|[tests/test_profiler_hooks.py](<C:/Users/skywalker/Documents/ChatGPT/H65/h65_clean_adatad/tests/test_profiler_hooks.py>)|A profiler must preserve tensor/tuple outputs while counting nested modules.|`无argparse参数/见源码`|
|[tests/test_scout.py](<C:/Users/skywalker/Documents/ChatGPT/H65/h65_clean_adatad/tests/test_scout.py>)|test_suite_source|`无argparse参数/见源码`|
|[tests/test_transport.py](<C:/Users/skywalker/Documents/ChatGPT/H65/h65_clean_adatad/tests/test_transport.py>)|Detect budget, gradient and physical-time failures before GPU experiments.|`无argparse参数/见源码`|

**mainline_draft：10份。**

|文件|类别/作用|源码声明的参数|
|---|---|---|
|[docs/EXTERNAL_REVIEW_PROMPT.md](<C:/Users/skywalker/Documents/ChatGPT/H65/h65_clean_adatad/mainline_20260911/docs/EXTERNAL_REVIEW_PROMPT.md>)|research_prompt|`无argparse参数/见源码`|
|[phase3_20260911/PLAN.md](<C:/Users/skywalker/Documents/ChatGPT/H65/h65_clean_adatad/mainline_20260911/phase3_20260911/PLAN.md>)|experiment_plan_or_receipt|`无argparse参数/见源码`|
|[resources.example.sh](<C:/Users/skywalker/Documents/ChatGPT/H65/h65_clean_adatad/mainline_20260911/resources.example.sh>)|资源环境变量示例，需实际站点配置|`无argparse参数/见源码`|
|[tools/full_audit.py](<C:/Users/skywalker/Documents/ChatGPT/H65/h65_clean_adatad/mainline_20260911/tools/full_audit.py>)|训练来源128窗口反事实/贡献效用审计与尺度校准|`--backbone`|
|[tools/full_compare.py](<C:/Users/skywalker/Documents/ChatGPT/H65/h65_clean_adatad/mainline_20260911/tools/full_compare.py>)|旧Phase2结果比较，不能直接代替修正BMCR的选模/汇总|`--backbone`|
|[tools/full_eval.py](<C:/Users/skywalker/Documents/ChatGPT/H65/h65_clean_adatad/mainline_20260911/tools/full_eval.py>)|完整测试与profile；早期版本只终点，当前版本支持milestone|`--backbone --variant --profile-only`|
|[tools/full_train.py](<C:/Users/skywalker/Documents/ChatGPT/H65/h65_clean_adatad/mainline_20260911/tools/full_train.py>)|H65/BMCR训练入口；配方与CLI细节依源码版本区分|`--backbone --phase --variant --preflight --workers`|
|[tools/run_full.sh](<C:/Users/skywalker/Documents/ChatGPT/H65/h65_clean_adatad/mainline_20260911/tools/run_full.sh>)|训练脚本包装器；根目录为旧站点版，工作树内为通用模板|`无argparse参数/见源码`|
|[tools/summarize_audit.py](<C:/Users/skywalker/Documents/ChatGPT/H65/h65_clean_adatad/mainline_20260911/tools/summarize_audit.py>)|效用审计结果整理|`--backbone`|
|[tools/verify_results.py](<C:/Users/skywalker/Documents/ChatGPT/H65/h65_clean_adatad/mainline_20260911/tools/verify_results.py>)|已生成实验记录复核|`无argparse参数/见源码`|

**original_DS3_pack：23份。**

|文件|类别/作用|源码声明的参数|
|---|---|---|
|[COMMANDS_zh.md](<C:/Users/skywalker/Documents/ChatGPT/H65/h65_clean_adatad/reviews/20260912_ds3/bundle/H65_DS3_research_20260911/COMMANDS_zh.md>)|command_document|`无argparse参数/见源码`|
|[README_zh.md](<C:/Users/skywalker/Documents/ChatGPT/H65/h65_clean_adatad/reviews/20260912_ds3/bundle/H65_DS3_research_20260911/README_zh.md>)|command_document|`无argparse参数/见源码`|
|[agents/00_coordinator.md](<C:/Users/skywalker/Documents/ChatGPT/H65/h65_clean_adatad/reviews/20260912_ds3/bundle/H65_DS3_research_20260911/agents/00_coordinator.md>)|agent_task_book|`无argparse参数/见源码`|
|[agents/01_audit.md](<C:/Users/skywalker/Documents/ChatGPT/H65/h65_clean_adatad/reviews/20260912_ds3/bundle/H65_DS3_research_20260911/agents/01_audit.md>)|agent_task_book|`无argparse参数/见源码`|
|[agents/02_time_frontend.md](<C:/Users/skywalker/Documents/ChatGPT/H65/h65_clean_adatad/reviews/20260912_ds3/bundle/H65_DS3_research_20260911/agents/02_time_frontend.md>)|agent_task_book|`无argparse参数/见源码`|
|[agents/03_dense_aux_depth.md](<C:/Users/skywalker/Documents/ChatGPT/H65/h65_clean_adatad/reviews/20260912_ds3/bundle/H65_DS3_research_20260911/agents/03_dense_aux_depth.md>)|agent_task_book|`无argparse参数/见源码`|
|[agents/04_spatial_runtime.md](<C:/Users/skywalker/Documents/ChatGPT/H65/h65_clean_adatad/reviews/20260912_ds3/bundle/H65_DS3_research_20260911/agents/04_spatial_runtime.md>)|agent_task_book|`无argparse参数/见源码`|
|[agents/05_integration_training.md](<C:/Users/skywalker/Documents/ChatGPT/H65/h65_clean_adatad/reviews/20260912_ds3/bundle/H65_DS3_research_20260911/agents/05_integration_training.md>)|agent_task_book|`无argparse参数/见源码`|
|[agents/06_eval_profile.md](<C:/Users/skywalker/Documents/ChatGPT/H65/h65_clean_adatad/reviews/20260912_ds3/bundle/H65_DS3_research_20260911/agents/06_eval_profile.md>)|agent_task_book|`无argparse参数/见源码`|
|[agents/07_independent_review.md](<C:/Users/skywalker/Documents/ChatGPT/H65/h65_clean_adatad/reviews/20260912_ds3/bundle/H65_DS3_research_20260911/agents/07_independent_review.md>)|agent_task_book|`无argparse参数/见源码`|
|[bin/bootstrap.sh](<C:/Users/skywalker/Documents/ChatGPT/H65/h65_clean_adatad/reviews/20260912_ds3/bundle/H65_DS3_research_20260911/bin/bootstrap.sh>)|原DS3包旧锚点worktree准备器|`无argparse参数/见源码`|
|[bin/run_agent.sh](<C:/Users/skywalker/Documents/ChatGPT/H65/h65_clean_adatad/reviews/20260912_ds3/bundle/H65_DS3_research_20260911/bin/run_agent.sh>)|原DS3包Codex任务书启动器|`无argparse参数/见源码`|
|[bin/run_experiment.py](<C:/Users/skywalker/Documents/ChatGPT/H65/h65_clean_adatad/reviews/20260912_ds3/bundle/H65_DS3_research_20260911/bin/run_experiment.py>)|原DS3包的manifest启动框架；不自带模型实现|`manifest --execute`|
|[bin/source_audit.py](<C:/Users/skywalker/Documents/ChatGPT/H65/h65_clean_adatad/reviews/20260912_ds3/bundle/H65_DS3_research_20260911/bin/source_audit.py>)|原DS3包的旧锚点/源码来源审计|`--repo --out`|
|[bin/verify_bundle.py](<C:/Users/skywalker/Documents/ChatGPT/H65/h65_clean_adatad/reviews/20260912_ds3/bundle/H65_DS3_research_20260911/bin/verify_bundle.py>)|原DS3包文件/语法/数学参考检查|`无argparse参数/见源码`|
|[contracts/cli_v1.md](<C:/Users/skywalker/Documents/ChatGPT/H65/h65_clean_adatad/reviews/20260912_ds3/bundle/H65_DS3_research_20260911/contracts/cli_v1.md>)|command_document|`无argparse参数/见源码`|
|[contracts/route_v1.json](<C:/Users/skywalker/Documents/ChatGPT/H65/h65_clean_adatad/reviews/20260912_ds3/bundle/H65_DS3_research_20260911/contracts/route_v1.json>)|plan_or_manifest_template|`无argparse参数/见源码`|
|[experiments.json](<C:/Users/skywalker/Documents/ChatGPT/H65/h65_clean_adatad/reviews/20260912_ds3/bundle/H65_DS3_research_20260911/experiments.json>)|plan_or_manifest_template|`无argparse参数/见源码`|
|[reference/reference_ops.py](<C:/Users/skywalker/Documents/ChatGPT/H65/h65_clean_adatad/reviews/20260912_ds3/bundle/H65_DS3_research_20260911/reference/reference_ops.py>)|原DS3包数学参考实现，不是当前模型执行器|`无argparse参数/见源码`|
|[reference/test_reference_ops.py](<C:/Users/skywalker/Documents/ChatGPT/H65/h65_clean_adatad/reviews/20260912_ds3/bundle/H65_DS3_research_20260911/reference/test_reference_ops.py>)|原DS3包数学参考测试，不是实际4090测试|`无argparse参数/见源码`|
|[templates/gate.template.json](<C:/Users/skywalker/Documents/ChatGPT/H65/h65_clean_adatad/reviews/20260912_ds3/bundle/H65_DS3_research_20260911/templates/gate.template.json>)|plan_or_manifest_template|`无argparse参数/见源码`|
|[templates/launch.template.json](<C:/Users/skywalker/Documents/ChatGPT/H65/h65_clean_adatad/reviews/20260912_ds3/bundle/H65_DS3_research_20260911/templates/launch.template.json>)|plan_or_manifest_template|`无argparse参数/见源码`|
|[templates/run_ds3.sbatch](<C:/Users/skywalker/Documents/ChatGPT/H65/h65_clean_adatad/reviews/20260912_ds3/bundle/H65_DS3_research_20260911/templates/run_ds3.sbatch>)|原DS3包站点模板，不是当前已部署run_job.sh|`无argparse参数/见源码`|

**original_H65_pack：17份。**

|文件|类别/作用|源码声明的参数|
|---|---|---|
|[AGENTS.md](<C:/Users/skywalker/Documents/ChatGPT/H65/h65_clean_adatad/phase2_20260910/inputs/reference_pack/H65_agents_pack/AGENTS.md>)|command_document|`无argparse参数/见源码`|
|[README.md](<C:/Users/skywalker/Documents/ChatGPT/H65/h65_clean_adatad/phase2_20260910/inputs/reference_pack/H65_agents_pack/README.md>)|command_document|`无argparse参数/见源码`|
|[SOURCE_LOCK.json](<C:/Users/skywalker/Documents/ChatGPT/H65/h65_clean_adatad/phase2_20260910/inputs/reference_pack/H65_agents_pack/SOURCE_LOCK.json>)|plan_or_manifest_template|`无argparse参数/见源码`|
|[TRAINING_HANDOFF.md](<C:/Users/skywalker/Documents/ChatGPT/H65/h65_clean_adatad/phase2_20260910/inputs/reference_pack/H65_agents_pack/TRAINING_HANDOFF.md>)|command_document|`无argparse参数/见源码`|
|[approvals/implementation.example.json](<C:/Users/skywalker/Documents/ChatGPT/H65/h65_clean_adatad/phase2_20260910/inputs/reference_pack/H65_agents_pack/approvals/implementation.example.json>)|plan_or_manifest_template|`无argparse参数/见源码`|
|[experiment_manifest.example.json](<C:/Users/skywalker/Documents/ChatGPT/H65/h65_clean_adatad/phase2_20260910/inputs/reference_pack/H65_agents_pack/experiment_manifest.example.json>)|plan_or_manifest_template|`无argparse参数/见源码`|
|[prompts/01_source_audit.md](<C:/Users/skywalker/Documents/ChatGPT/H65/h65_clean_adatad/phase2_20260910/inputs/reference_pack/H65_agents_pack/prompts/01_source_audit.md>)|agent_task_book|`无argparse参数/见源码`|
|[prompts/02_geometry_gradient.md](<C:/Users/skywalker/Documents/ChatGPT/H65/h65_clean_adatad/phase2_20260910/inputs/reference_pack/H65_agents_pack/prompts/02_geometry_gradient.md>)|agent_task_book|`无argparse参数/见源码`|
|[prompts/03_literature_transfer.md](<C:/Users/skywalker/Documents/ChatGPT/H65/h65_clean_adatad/phase2_20260910/inputs/reference_pack/H65_agents_pack/prompts/03_literature_transfer.md>)|agent_task_book|`无argparse参数/见源码`|
|[prompts/04_design_cost.md](<C:/Users/skywalker/Documents/ChatGPT/H65/h65_clean_adatad/phase2_20260910/inputs/reference_pack/H65_agents_pack/prompts/04_design_cost.md>)|agent_task_book|`无argparse参数/见源码`|
|[prompts/05_adversarial_review.md](<C:/Users/skywalker/Documents/ChatGPT/H65/h65_clean_adatad/phase2_20260910/inputs/reference_pack/H65_agents_pack/prompts/05_adversarial_review.md>)|agent_task_book|`无argparse参数/见源码`|
|[prompts/06_integrated_plan.md](<C:/Users/skywalker/Documents/ChatGPT/H65/h65_clean_adatad/phase2_20260910/inputs/reference_pack/H65_agents_pack/prompts/06_integrated_plan.md>)|agent_task_book|`无argparse参数/见源码`|
|[prompts/AFTER_APPROVAL_implementation.md](<C:/Users/skywalker/Documents/ChatGPT/H65/h65_clean_adatad/phase2_20260910/inputs/reference_pack/H65_agents_pack/prompts/AFTER_APPROVAL_implementation.md>)|agent_task_book|`无argparse参数/见源码`|
|[scripts/bootstrap_sources.sh](<C:/Users/skywalker/Documents/ChatGPT/H65/h65_clean_adatad/phase2_20260910/inputs/reference_pack/H65_agents_pack/scripts/bootstrap_sources.sh>)|原H65包获取固定旧来源的脚本|`无argparse参数/见源码`|
|[scripts/check_sources.py](<C:/Users/skywalker/Documents/ChatGPT/H65/h65_clean_adatad/phase2_20260910/inputs/reference_pack/H65_agents_pack/scripts/check_sources.py>)|原H65包来源检查|`无argparse参数/见源码`|
|[scripts/run_approved_implementation.sh](<C:/Users/skywalker/Documents/ChatGPT/H65/h65_clean_adatad/phase2_20260910/inputs/reference_pack/H65_agents_pack/scripts/run_approved_implementation.sh>)|原H65包未来M0–M2实现模板入口|`无argparse参数/见源码`|
|[scripts/run_research_agents.sh](<C:/Users/skywalker/Documents/ChatGPT/H65/h65_clean_adatad/phase2_20260910/inputs/reference_pack/H65_agents_pack/scripts/run_research_agents.sh>)|原H65包六个研究任务启动器|`无argparse参数/见源码`|

**publication_legacy：9份。**

|文件|类别/作用|源码声明的参数|
|---|---|---|
|[docs/EXTERNAL_REVIEW_PROMPT.md](<C:/Users/skywalker/Documents/ChatGPT/H65/h65_clean_adatad/publication/BMCR-T-AdaTAD/docs/EXTERNAL_REVIEW_PROMPT.md>)|research_prompt|`无argparse参数/见源码`|
|[resources.example.sh](<C:/Users/skywalker/Documents/ChatGPT/H65/h65_clean_adatad/publication/BMCR-T-AdaTAD/resources.example.sh>)|资源环境变量示例，需实际站点配置|`无argparse参数/见源码`|
|[tools/full_audit.py](<C:/Users/skywalker/Documents/ChatGPT/H65/h65_clean_adatad/publication/BMCR-T-AdaTAD/tools/full_audit.py>)|训练来源128窗口反事实/贡献效用审计与尺度校准|`--backbone`|
|[tools/full_compare.py](<C:/Users/skywalker/Documents/ChatGPT/H65/h65_clean_adatad/publication/BMCR-T-AdaTAD/tools/full_compare.py>)|旧Phase2结果比较，不能直接代替修正BMCR的选模/汇总|`--backbone`|
|[tools/full_eval.py](<C:/Users/skywalker/Documents/ChatGPT/H65/h65_clean_adatad/publication/BMCR-T-AdaTAD/tools/full_eval.py>)|完整测试与profile；早期版本只终点，当前版本支持milestone|`--backbone --variant --profile-only`|
|[tools/full_train.py](<C:/Users/skywalker/Documents/ChatGPT/H65/h65_clean_adatad/publication/BMCR-T-AdaTAD/tools/full_train.py>)|H65/BMCR训练入口；配方与CLI细节依源码版本区分|`--backbone --phase --variant --preflight --workers`|
|[tools/run_full.sh](<C:/Users/skywalker/Documents/ChatGPT/H65/h65_clean_adatad/publication/BMCR-T-AdaTAD/tools/run_full.sh>)|训练脚本包装器；根目录为旧站点版，工作树内为通用模板|`无argparse参数/见源码`|
|[tools/summarize_audit.py](<C:/Users/skywalker/Documents/ChatGPT/H65/h65_clean_adatad/publication/BMCR-T-AdaTAD/tools/summarize_audit.py>)|效用审计结果整理|`--backbone`|
|[tools/verify_results.py](<C:/Users/skywalker/Documents/ChatGPT/H65/h65_clean_adatad/publication/BMCR-T-AdaTAD/tools/verify_results.py>)|已生成实验记录复核|`无argparse参数/见源码`|

**resource_handoff：1份。**

|文件|类别/作用|源码声明的参数|
|---|---|---|
|[handoff_4090.md](<C:/Users/skywalker/Documents/ChatGPT/H65/handoff_4090.md>)|resource_command_document|`无argparse参数/见源码`|

**upstream_tools：22份。**

|文件|类别/作用|源码声明的参数|
|---|---|---|
|[model_converters/convert_videomae_finetuned.py](<C:/Users/skywalker/Documents/ChatGPT/H65/h65_clean_adatad/upstream/tools/model_converters/convert_videomae_finetuned.py>)|dependency_tool_or_module|`in_file out_file --arch`|
|[model_converters/convert_videomaev2.py](<C:/Users/skywalker/Documents/ChatGPT/H65/h65_clean_adatad/upstream/tools/model_converters/convert_videomaev2.py>)|dependency_tool_or_module|`in_file out_file`|
|[model_converters/publish_model.py](<C:/Users/skywalker/Documents/ChatGPT/H65/h65_clean_adatad/upstream/tools/model_converters/publish_model.py>)|dependency_tool_or_module|`in_file out_file`|
|[prepare_data/activitynet/download_annotation.sh](<C:/Users/skywalker/Documents/ChatGPT/H65/h65_clean_adatad/upstream/tools/prepare_data/activitynet/download_annotation.sh>)|dependency_tool_or_module|`无argparse参数/见源码`|
|[prepare_data/charades/download_annotation.sh](<C:/Users/skywalker/Documents/ChatGPT/H65/h65_clean_adatad/upstream/tools/prepare_data/charades/download_annotation.sh>)|dependency_tool_or_module|`无argparse参数/见源码`|
|[prepare_data/ego4d/accurate_trim_MQ.py](<C:/Users/skywalker/Documents/ChatGPT/H65/h65_clean_adatad/upstream/tools/prepare_data/ego4d/accurate_trim_MQ.py>)|dependency_tool_or_module|`anno_dir data_dir save_path --short_size --part --total`|
|[prepare_data/ego4d/convert_ego4d_anno.py](<C:/Users/skywalker/Documents/ChatGPT/H65/h65_clean_adatad/upstream/tools/prepare_data/ego4d/convert_ego4d_anno.py>)|dependency_tool_or_module|`data_dir save_path`|
|[prepare_data/ego4d/download_annotation.sh](<C:/Users/skywalker/Documents/ChatGPT/H65/h65_clean_adatad/upstream/tools/prepare_data/ego4d/download_annotation.sh>)|dependency_tool_or_module|`无argparse参数/见源码`|
|[prepare_data/ego4d/ego4d_challenge_submit.py](<C:/Users/skywalker/Documents/ChatGPT/H65/h65_clean_adatad/upstream/tools/prepare_data/ego4d/ego4d_challenge_submit.py>)|dependency_tool_or_module|`config --checkpoint --seed --id --map_sigma --recall_sigma --cfg-options`|
|[prepare_data/epic/convert_epic_kitchens_anno.py](<C:/Users/skywalker/Documents/ChatGPT/H65/h65_clean_adatad/upstream/tools/prepare_data/epic/convert_epic_kitchens_anno.py>)|dependency_tool_or_module|`csv_dir save_path`|
|[prepare_data/epic/download_annotation.sh](<C:/Users/skywalker/Documents/ChatGPT/H65/h65_clean_adatad/upstream/tools/prepare_data/epic/download_annotation.sh>)|dependency_tool_or_module|`无argparse参数/见源码`|
|[prepare_data/epic/epic_action_noun_verb_submit.py](<C:/Users/skywalker/Documents/ChatGPT/H65/h65_clean_adatad/upstream/tools/prepare_data/epic/epic_action_noun_verb_submit.py>)|dependency_tool_or_module|`config_noun config_verb ckpt_noun ckpt_verb --submit --seed --id --pre_nms_topk --max_seg_num --not_eval --cfg-options`|
|[prepare_data/epic_sounds/convert_epic_sounds_anno.py](<C:/Users/skywalker/Documents/ChatGPT/H65/h65_clean_adatad/upstream/tools/prepare_data/epic_sounds/convert_epic_sounds_anno.py>)|dependency_tool_or_module|`csv_dir save_path`|
|[prepare_data/epic_sounds/download_annotation.sh](<C:/Users/skywalker/Documents/ChatGPT/H65/h65_clean_adatad/upstream/tools/prepare_data/epic_sounds/download_annotation.sh>)|dependency_tool_or_module|`无argparse参数/见源码`|
|[prepare_data/epic_sounds/epic_sound_submit.py](<C:/Users/skywalker/Documents/ChatGPT/H65/h65_clean_adatad/upstream/tools/prepare_data/epic_sounds/epic_sound_submit.py>)|dependency_tool_or_module|`config --checkpoint --seed --id --pre_nms_topk --max_seg_num`|
|[prepare_data/fineaction/download_annotation.sh](<C:/Users/skywalker/Documents/ChatGPT/H65/h65_clean_adatad/upstream/tools/prepare_data/fineaction/download_annotation.sh>)|dependency_tool_or_module|`无argparse参数/见源码`|
|[prepare_data/generate_missing_list.py](<C:/Users/skywalker/Documents/ChatGPT/H65/h65_clean_adatad/upstream/tools/prepare_data/generate_missing_list.py>)|dependency_tool_or_module|`anno_file data_dir --prefix --suffix --ext`|
|[prepare_data/hacs/download_annotation.sh](<C:/Users/skywalker/Documents/ChatGPT/H65/h65_clean_adatad/upstream/tools/prepare_data/hacs/download_annotation.sh>)|dependency_tool_or_module|`无argparse参数/见源码`|
|[prepare_data/multi-thumos/download_annotation.sh](<C:/Users/skywalker/Documents/ChatGPT/H65/h65_clean_adatad/upstream/tools/prepare_data/multi-thumos/download_annotation.sh>)|dependency_tool_or_module|`无argparse参数/见源码`|
|[prepare_data/thumos/download_annotation.sh](<C:/Users/skywalker/Documents/ChatGPT/H65/h65_clean_adatad/upstream/tools/prepare_data/thumos/download_annotation.sh>)|dependency_tool_or_module|`无argparse参数/见源码`|
|[test.py](<C:/Users/skywalker/Documents/ChatGPT/H65/h65_clean_adatad/upstream/tools/test.py>)|dependency_tool_or_module|`config --checkpoint --seed --id --not_eval --cfg-options`|
|[train.py](<C:/Users/skywalker/Documents/ChatGPT/H65/h65_clean_adatad/upstream/tools/train.py>)|dependency_tool_or_module|`config --seed --id --resume --not_eval --disable_deterministic --cfg-options`|
