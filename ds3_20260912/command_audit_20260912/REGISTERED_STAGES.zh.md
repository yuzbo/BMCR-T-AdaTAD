**当前DS3调度器登记的全部阶段**

快照：2026-09-13T00:06:17.570042+08:00。共73阶段；状态{'COMPLETED': 14, 'RUNNING': 2, 'WAITING': 57}。WAITING代表依赖/资源尚未满足，不等于模型没实现。

以下只是从收据提取的入口，不是本次重跑命令。GPU入口由本项目站点包装器与现有控制器安排，不能在登录节点直接跑，也不应启动第二个控制器。

|阶段|种类|状态/作业|入口或待解析模板|依赖|
|---|---|---|---|---|
|preflight|preflight|COMPLETED/1287657|`tools/ds3_preflight.py --backbone both`||
|s_D768G|test|COMPLETED/1287659|`tools/ds3_eval.py --backbone s --policy D768G`|preflight|
|s_D768L|test|COMPLETED/1287701|`tools/ds3_eval.py --backbone s --policy D768L`|preflight|
|s_Z16|test|COMPLETED/1287768|`tools/ds3_eval.py --backbone s --policy Z16`|preflight|
|s_Z24|test|COMPLETED/1287762|`tools/ds3_eval.py --backbone s --policy Z24`|preflight|
|s_Z36|test|COMPLETED/1287806|`tools/ds3_eval.py --backbone s --policy Z36`|preflight|
|s_ZR24|test|COMPLETED/1287866|`tools/ds3_eval.py --backbone s --policy ZR24`|preflight|
|s_pilot|pilot|COMPLETED/1287766|`tools/ds3_train.py --backbone s --pilot`|preflight, s_D768G, s_D768L, s_Z24|
|s_train|train|RUNNING/1287797|`tools/ds3_train.py --backbone s`|s_pilot|
|s_T24A_05|test|COMPLETED/1287831|`tools/ds3_eval.py --backbone s --policy T24A --epoch 5 --metrics-only`|preflight|
|s_T24A_10|test|COMPLETED/1287883|`tools/ds3_eval.py --backbone s --policy T24A --epoch 10 --metrics-only`|preflight|
|s_T24A_15|test|COMPLETED/1287909|`tools/ds3_eval.py --backbone s --policy T24A --epoch 15 --metrics-only`|preflight|
|s_T24A_20|test|COMPLETED/1287912|`tools/ds3_eval.py --backbone s --policy T24A --epoch 20 --metrics-only`|preflight|
|s_T24A_25|test|COMPLETED/1287928|`tools/ds3_eval.py --backbone s --policy T24A --epoch 25 --metrics-only`|preflight|
|s_T24A_30|test|COMPLETED/1287929|`tools/ds3_eval.py --backbone s --policy T24A --epoch 30 --metrics-only`|preflight|
|s_T24A_35|test|WAITING/—|`tools/ds3_eval.py --backbone s --policy T24A --epoch 35 --metrics-only`|preflight|
|s_T24A_40|test|WAITING/—|`tools/ds3_eval.py --backbone s --policy T24A --epoch 40 --metrics-only`|preflight|
|s_T24A_45|test|WAITING/—|`tools/ds3_eval.py --backbone s --policy T24A --epoch 45 --metrics-only`|preflight|
|s_T24A_50|test|WAITING/—|`tools/ds3_eval.py --backbone s --policy T24A --epoch 50 --metrics-only`|preflight|
|s_T24A_55|test|WAITING/—|`tools/ds3_eval.py --backbone s --policy T24A --epoch 55 --metrics-only`|preflight|
|s_T24A_60|test|WAITING/—|`tools/ds3_eval.py --backbone s --policy T24A --epoch 60`|preflight|
|s_T24A_65|test|WAITING/—|`tools/ds3_eval.py --backbone s --policy T24A --epoch 65 --metrics-only`|preflight|
|s_T24A_70|test|WAITING/—|`tools/ds3_eval.py --backbone s --policy T24A --epoch 70 --metrics-only`|preflight|
|s_T24A_75|test|WAITING/—|`tools/ds3_eval.py --backbone s --policy T24A --epoch 75 --metrics-only`|preflight|
|s_T24A_80|test|WAITING/—|`tools/ds3_eval.py --backbone s --policy T24A --epoch 80`|preflight|
|s_selected_profile|selected_profile|WAITING/—|`tools/ds3_eval.py --backbone s --policy T24A --epoch <T24A选中epoch> --profile-only`|s_train, s_T24A_05, s_T24A_10, s_T24A_15, s_T24A_20, s_T24A_25, s_T24A_30, s_T24A_35, s_T24A_40, s_T24A_45, s_T24A_50, s_T24A_55, s_T24A_60, s_T24A_65, s_T24A_70, s_T24A_75, s_T24A_80|
|s_selected_PONLY|selected_test|WAITING/—|`tools/ds3_eval.py --backbone s --policy PONLY --epoch <T24A选中epoch>`|s_train, s_T24A_05, s_T24A_10, s_T24A_15, s_T24A_20, s_T24A_25, s_T24A_30, s_T24A_35, s_T24A_40, s_T24A_45, s_T24A_50, s_T24A_55, s_T24A_60, s_T24A_65, s_T24A_70, s_T24A_75, s_T24A_80|
|s_selected_T24U|selected_test|WAITING/—|`tools/ds3_eval.py --backbone s --policy T24U --epoch <T24A选中epoch>`|s_train, s_T24A_05, s_T24A_10, s_T24A_15, s_T24A_20, s_T24A_25, s_T24A_30, s_T24A_35, s_T24A_40, s_T24A_45, s_T24A_50, s_T24A_55, s_T24A_60, s_T24A_65, s_T24A_70, s_T24A_75, s_T24A_80|
|s_selected_D8|selected_test|WAITING/—|`tools/ds3_eval.py --backbone s --policy D8 --epoch <T24A选中epoch>`|s_train, s_T24A_05, s_T24A_10, s_T24A_15, s_T24A_20, s_T24A_25, s_T24A_30, s_T24A_35, s_T24A_40, s_T24A_45, s_T24A_50, s_T24A_55, s_T24A_60, s_T24A_65, s_T24A_70, s_T24A_75, s_T24A_80|
|s_selected_DAD|selected_test|WAITING/—|`tools/ds3_eval.py --backbone s --policy DAD --epoch <T24A选中epoch>`|s_train, s_T24A_05, s_T24A_10, s_T24A_15, s_T24A_20, s_T24A_25, s_T24A_30, s_T24A_35, s_T24A_40, s_T24A_45, s_T24A_50, s_T24A_55, s_T24A_60, s_T24A_65, s_T24A_70, s_T24A_75, s_T24A_80|
|s_selected_S75|selected_test|WAITING/—|`tools/ds3_eval.py --backbone s --policy S75 --epoch <T24A选中epoch>`|s_train, s_T24A_05, s_T24A_10, s_T24A_15, s_T24A_20, s_T24A_25, s_T24A_30, s_T24A_35, s_T24A_40, s_T24A_45, s_T24A_50, s_T24A_55, s_T24A_60, s_T24A_65, s_T24A_70, s_T24A_75, s_T24A_80|
|s_selected_S50|selected_test|WAITING/—|`tools/ds3_eval.py --backbone s --policy S50 --epoch <T24A选中epoch>`|s_train, s_T24A_05, s_T24A_10, s_T24A_15, s_T24A_20, s_T24A_25, s_T24A_30, s_T24A_35, s_T24A_40, s_T24A_45, s_T24A_50, s_T24A_55, s_T24A_60, s_T24A_65, s_T24A_70, s_T24A_75, s_T24A_80|
|s_selected_F000|selected_test|WAITING/—|`tools/ds3_eval.py --backbone s --policy F000 --epoch <T24A选中epoch>`|s_train, s_T24A_05, s_T24A_10, s_T24A_15, s_T24A_20, s_T24A_25, s_T24A_30, s_T24A_35, s_T24A_40, s_T24A_45, s_T24A_50, s_T24A_55, s_T24A_60, s_T24A_65, s_T24A_70, s_T24A_75, s_T24A_80|
|s_selected_F001|selected_test|WAITING/—|`tools/ds3_eval.py --backbone s --policy F001 --epoch <T24A选中epoch>`|s_train, s_T24A_05, s_T24A_10, s_T24A_15, s_T24A_20, s_T24A_25, s_T24A_30, s_T24A_35, s_T24A_40, s_T24A_45, s_T24A_50, s_T24A_55, s_T24A_60, s_T24A_65, s_T24A_70, s_T24A_75, s_T24A_80|
|s_selected_F011|selected_test|WAITING/—|`tools/ds3_eval.py --backbone s --policy F011 --epoch <T24A选中epoch>`|s_train, s_T24A_05, s_T24A_10, s_T24A_15, s_T24A_20, s_T24A_25, s_T24A_30, s_T24A_35, s_T24A_40, s_T24A_45, s_T24A_50, s_T24A_55, s_T24A_60, s_T24A_65, s_T24A_70, s_T24A_75, s_T24A_80|
|s_selected_F110|selected_test|WAITING/—|`tools/ds3_eval.py --backbone s --policy F110 --epoch <T24A选中epoch>`|s_train, s_T24A_05, s_T24A_10, s_T24A_15, s_T24A_20, s_T24A_25, s_T24A_30, s_T24A_35, s_T24A_40, s_T24A_45, s_T24A_50, s_T24A_55, s_T24A_60, s_T24A_65, s_T24A_70, s_T24A_75, s_T24A_80|
|s_selected_F111|selected_test|WAITING/—|`tools/ds3_eval.py --backbone s --policy F111 --epoch <T24A选中epoch>`|s_train, s_T24A_05, s_T24A_10, s_T24A_15, s_T24A_20, s_T24A_25, s_T24A_30, s_T24A_35, s_T24A_40, s_T24A_45, s_T24A_50, s_T24A_55, s_T24A_60, s_T24A_65, s_T24A_70, s_T24A_75, s_T24A_80|
|b_D768G|test|RUNNING/1287932|`tools/ds3_eval.py --backbone b --policy D768G`|preflight|
|b_D768L|test|WAITING/—|`tools/ds3_eval.py --backbone b --policy D768L`|preflight|
|b_Z16|test|WAITING/—|`tools/ds3_eval.py --backbone b --policy Z16`|preflight|
|b_Z24|test|WAITING/—|`tools/ds3_eval.py --backbone b --policy Z24`|preflight|
|b_Z36|test|WAITING/—|`tools/ds3_eval.py --backbone b --policy Z36`|preflight|
|b_ZR24|test|WAITING/—|`tools/ds3_eval.py --backbone b --policy ZR24`|preflight|
|b_pilot|pilot|WAITING/—|`tools/ds3_train.py --backbone b --pilot`|preflight, b_D768G, b_D768L, b_Z24|
|b_train|train|WAITING/—|`tools/ds3_train.py --backbone b`|b_pilot|
|b_T24A_05|test|WAITING/—|`tools/ds3_eval.py --backbone b --policy T24A --epoch 5 --metrics-only`|preflight|
|b_T24A_10|test|WAITING/—|`tools/ds3_eval.py --backbone b --policy T24A --epoch 10 --metrics-only`|preflight|
|b_T24A_15|test|WAITING/—|`tools/ds3_eval.py --backbone b --policy T24A --epoch 15 --metrics-only`|preflight|
|b_T24A_20|test|WAITING/—|`tools/ds3_eval.py --backbone b --policy T24A --epoch 20 --metrics-only`|preflight|
|b_T24A_25|test|WAITING/—|`tools/ds3_eval.py --backbone b --policy T24A --epoch 25 --metrics-only`|preflight|
|b_T24A_30|test|WAITING/—|`tools/ds3_eval.py --backbone b --policy T24A --epoch 30 --metrics-only`|preflight|
|b_T24A_35|test|WAITING/—|`tools/ds3_eval.py --backbone b --policy T24A --epoch 35 --metrics-only`|preflight|
|b_T24A_40|test|WAITING/—|`tools/ds3_eval.py --backbone b --policy T24A --epoch 40 --metrics-only`|preflight|
|b_T24A_45|test|WAITING/—|`tools/ds3_eval.py --backbone b --policy T24A --epoch 45 --metrics-only`|preflight|
|b_T24A_50|test|WAITING/—|`tools/ds3_eval.py --backbone b --policy T24A --epoch 50 --metrics-only`|preflight|
|b_T24A_55|test|WAITING/—|`tools/ds3_eval.py --backbone b --policy T24A --epoch 55 --metrics-only`|preflight|
|b_T24A_60|test|WAITING/—|`tools/ds3_eval.py --backbone b --policy T24A --epoch 60`|preflight|
|b_T24A_65|test|WAITING/—|`tools/ds3_eval.py --backbone b --policy T24A --epoch 65 --metrics-only`|preflight|
|b_T24A_70|test|WAITING/—|`tools/ds3_eval.py --backbone b --policy T24A --epoch 70 --metrics-only`|preflight|
|b_T24A_75|test|WAITING/—|`tools/ds3_eval.py --backbone b --policy T24A --epoch 75 --metrics-only`|preflight|
|b_T24A_80|test|WAITING/—|`tools/ds3_eval.py --backbone b --policy T24A --epoch 80`|preflight|
|b_selected_profile|selected_profile|WAITING/—|`tools/ds3_eval.py --backbone b --policy T24A --epoch <T24A选中epoch> --profile-only`|b_train, b_T24A_05, b_T24A_10, b_T24A_15, b_T24A_20, b_T24A_25, b_T24A_30, b_T24A_35, b_T24A_40, b_T24A_45, b_T24A_50, b_T24A_55, b_T24A_60, b_T24A_65, b_T24A_70, b_T24A_75, b_T24A_80|
|b_selected_PONLY|selected_test|WAITING/—|`tools/ds3_eval.py --backbone b --policy PONLY --epoch <T24A选中epoch>`|b_train, b_T24A_05, b_T24A_10, b_T24A_15, b_T24A_20, b_T24A_25, b_T24A_30, b_T24A_35, b_T24A_40, b_T24A_45, b_T24A_50, b_T24A_55, b_T24A_60, b_T24A_65, b_T24A_70, b_T24A_75, b_T24A_80|
|b_selected_T24U|selected_test|WAITING/—|`tools/ds3_eval.py --backbone b --policy T24U --epoch <T24A选中epoch>`|b_train, b_T24A_05, b_T24A_10, b_T24A_15, b_T24A_20, b_T24A_25, b_T24A_30, b_T24A_35, b_T24A_40, b_T24A_45, b_T24A_50, b_T24A_55, b_T24A_60, b_T24A_65, b_T24A_70, b_T24A_75, b_T24A_80|
|b_selected_D8|selected_test|WAITING/—|`tools/ds3_eval.py --backbone b --policy D8 --epoch <T24A选中epoch>`|b_train, b_T24A_05, b_T24A_10, b_T24A_15, b_T24A_20, b_T24A_25, b_T24A_30, b_T24A_35, b_T24A_40, b_T24A_45, b_T24A_50, b_T24A_55, b_T24A_60, b_T24A_65, b_T24A_70, b_T24A_75, b_T24A_80|
|b_selected_DAD|selected_test|WAITING/—|`tools/ds3_eval.py --backbone b --policy DAD --epoch <T24A选中epoch>`|b_train, b_T24A_05, b_T24A_10, b_T24A_15, b_T24A_20, b_T24A_25, b_T24A_30, b_T24A_35, b_T24A_40, b_T24A_45, b_T24A_50, b_T24A_55, b_T24A_60, b_T24A_65, b_T24A_70, b_T24A_75, b_T24A_80|
|b_selected_S75|selected_test|WAITING/—|`tools/ds3_eval.py --backbone b --policy S75 --epoch <T24A选中epoch>`|b_train, b_T24A_05, b_T24A_10, b_T24A_15, b_T24A_20, b_T24A_25, b_T24A_30, b_T24A_35, b_T24A_40, b_T24A_45, b_T24A_50, b_T24A_55, b_T24A_60, b_T24A_65, b_T24A_70, b_T24A_75, b_T24A_80|
|b_selected_S50|selected_test|WAITING/—|`tools/ds3_eval.py --backbone b --policy S50 --epoch <T24A选中epoch>`|b_train, b_T24A_05, b_T24A_10, b_T24A_15, b_T24A_20, b_T24A_25, b_T24A_30, b_T24A_35, b_T24A_40, b_T24A_45, b_T24A_50, b_T24A_55, b_T24A_60, b_T24A_65, b_T24A_70, b_T24A_75, b_T24A_80|
|b_selected_F000|selected_test|WAITING/—|`tools/ds3_eval.py --backbone b --policy F000 --epoch <T24A选中epoch>`|b_train, b_T24A_05, b_T24A_10, b_T24A_15, b_T24A_20, b_T24A_25, b_T24A_30, b_T24A_35, b_T24A_40, b_T24A_45, b_T24A_50, b_T24A_55, b_T24A_60, b_T24A_65, b_T24A_70, b_T24A_75, b_T24A_80|
|b_selected_F001|selected_test|WAITING/—|`tools/ds3_eval.py --backbone b --policy F001 --epoch <T24A选中epoch>`|b_train, b_T24A_05, b_T24A_10, b_T24A_15, b_T24A_20, b_T24A_25, b_T24A_30, b_T24A_35, b_T24A_40, b_T24A_45, b_T24A_50, b_T24A_55, b_T24A_60, b_T24A_65, b_T24A_70, b_T24A_75, b_T24A_80|
|b_selected_F011|selected_test|WAITING/—|`tools/ds3_eval.py --backbone b --policy F011 --epoch <T24A选中epoch>`|b_train, b_T24A_05, b_T24A_10, b_T24A_15, b_T24A_20, b_T24A_25, b_T24A_30, b_T24A_35, b_T24A_40, b_T24A_45, b_T24A_50, b_T24A_55, b_T24A_60, b_T24A_65, b_T24A_70, b_T24A_75, b_T24A_80|
|b_selected_F110|selected_test|WAITING/—|`tools/ds3_eval.py --backbone b --policy F110 --epoch <T24A选中epoch>`|b_train, b_T24A_05, b_T24A_10, b_T24A_15, b_T24A_20, b_T24A_25, b_T24A_30, b_T24A_35, b_T24A_40, b_T24A_45, b_T24A_50, b_T24A_55, b_T24A_60, b_T24A_65, b_T24A_70, b_T24A_75, b_T24A_80|
|b_selected_F111|selected_test|WAITING/—|`tools/ds3_eval.py --backbone b --policy F111 --epoch <T24A选中epoch>`|b_train, b_T24A_05, b_T24A_10, b_T24A_15, b_T24A_20, b_T24A_25, b_T24A_30, b_T24A_35, b_T24A_40, b_T24A_45, b_T24A_50, b_T24A_55, b_T24A_60, b_T24A_65, b_T24A_70, b_T24A_75, b_T24A_80|
