# 复现、源码身份与下载

本次发布不运行新模型。主Git分支含当前RFV完整实现、配方、工具、测试、报告、图和重要数值；Atlas/Raw保持独立科学来源，以精确源码包补充。历史已发布内容继续保留。

## Release资产

|文件|大小(MiB)|内容|
|---|---:|---|
|[atlas-s-numerical-evidence.zip](https://github.com/yuzbo/BMCR-T-AdaTAD/releases/download/wtr-rfv-evidence-20260915/atlas-s-numerical-evidence.zip)|47.85|完整S/interaction统计、AP缓存、action manifests及回执/现有日志|
|[atlas-publication-figures.zip](https://github.com/yuzbo/BMCR-T-AdaTAD/releases/download/wtr-rfv-evidence-20260915/atlas-publication-figures.zip)|10.78|已交付Atlas最终PDF/PNG/SVG、图注和精确数值表|
|[atlas-source-65e14a8.zip](https://github.com/yuzbo/BMCR-T-AdaTAD/releases/download/wtr-rfv-evidence-20260915/atlas-source-65e14a8.zip)|24.22|对应不可变SHA的完整当前源码；私有资源/重复legacy runtime排除项列在manifest|
|[raw-source-27d557e.zip](https://github.com/yuzbo/BMCR-T-AdaTAD/releases/download/wtr-rfv-evidence-20260915/raw-source-27d557e.zip)|23.06|对应不可变SHA的完整当前源码；私有资源/重复legacy runtime排除项列在manifest|
|[raw-existing-evidence-and-logs.zip](https://github.com/yuzbo/BMCR-T-AdaTAD/releases/download/wtr-rfv-evidence-20260915/raw-existing-evidence-and-logs.zip)|4.22|只读回收的Raw已有JSON/log与protocol|
|[rfv-evidence-and-course-logs.zip](https://github.com/yuzbo/BMCR-T-AdaTAD/releases/download/wtr-rfv-evidence-20260915/rfv-evidence-and-course-logs.zip)|4.74|RFV/R1/Reverse/RISE/D-S证据、审阅、绘图、所有当前可用课程日志|

## 不可变科学身份

|部分|SHA|
|---|---|
|rfv_implementation|72459727f61454194f0d84865f31a16ed4e10448|
|atlas_closeout|65e14a8df792a5ea9f1ffdc7737981dd9c880b17|
|raw|27d557ee3bd38a52aec8caaf16860a95957b44b6|
|D_S_science|40552945ad5d56b7404f833cd86f998e8558e8b1|
|rfv_bank|2ca4d4b5f2ad657410bfc5c2c4cebd3bc6a11a32|
|r1|b644d870d1845abbc1e4fd5ab7780f29ff96a53a|
|reverse_capture_rise_export|800bcd10a66c69d0d57142ac02389adbc45d2975|
|reverse_fit|4aa1ca242c37b3f7e2aa727b05419628a74e4e90|
|ds_fixed_diagnostic|72459727f61454194f0d84865f31a16ed4e10448|
|atlas_measure|a50b84db1bcdafd86ceec4d9737156807f9daa9a|
|interaction_measure|49f74bd5601bc298997921a93193109156adf546|
|interaction_analysis|1b3850e6db36fbfaaf43fdc71ea1e39ea22af065|
|interaction_display_analysis|0a9380a6560a789bd9b58a534a8c8329f7fb55f6|
|interaction_renderer|5d2402a11873f81ddccb29c861c4be80c9289378|

文档/整理提交不回填为旧运行source。源码包包含h65、tools、configs、tests、第三方源码及适用研究协议；源码包不是包含私有资源配置的完整机器镜像。主GitHub tag另外提供GitHub自动生成的Source code下载。

## 只重绘本次RFV和课程图

在所需Python环境中使用Matplotlib/NumPy，以下命令仅读已保存JSON，不加载detector、不创建作业：

~~~bash
python tools/rfv_plot_publication_courses.py --snapshot research/wtr_rfv_release_20260915/courses/SNAPSHOT.json --output output/publication_courses
python tools/rfv_plot_reverse.py --reverse-report research/rfv_sprint_20260915/evidence/reverse_4aa1ca2/REVERSE_MINI/REVERSE_MINI.json --rise-report research/rfv_sprint_20260915/evidence/reverse_800bcd1/RISE_CHOICES/RISE_CHOICES.json --output output/reverse
~~~

Atlas跨轴数值和图可在对应Atlas源码包中配合numerical-evidence的analysis/interaction_v1_s重绘。全部已导出PDF/PNG/SVG已提供；重绘某些原Atlas视觉案例需要原始window/thumbnail记录，重新执行完整AP则需要数据、权重和对应冻结执行协议。不要用最终统计JSON冒充原始RGB/模型预测。

## 实验复现与资产

GPU重现实验应checkout其实际science SHA，使用对应数据/初始化资源和明确的bank/action manifest，而不是任意新文档HEAD。入口为tools/rfv_run.py、rfv_value_r1.py、rfv_reverse_mini.py、rfv_forecast.py及rfv_ds_diagnostic.py。保存的命令、分区、配置、normalization来源、query counts、原始失败和完成回执可直接核对。

原始视频、官方大权重、训练checkpoint和机器凭据不包含在此发布。模型训练依赖的初始化来源和实际checkpoint路径保存在运行metadata；需要自行取得相应资产。重要已有数值和日志按发布清单完整保留，未声称上传所有视频、缓存或模型权重。

ops目录是实际使用的采集/部署/归档脚本来源记录，包含原机器路径；它们不是绘图复现的自动入口。当前四条旧D/S课程仍按快照记录自身运行，RFV和Atlas自动跟进保持暂停。

## 历史记录

[前一完整历史快照](https://github.com/yuzbo/BMCR-T-AdaTAD/releases/tag/wtr-snapshot-20260915)继续保留。history/及raw/design_history中的原文带有历史日期、旧PID和本机路径，不是当前运行命令或新模型状态。当前入口以本目录STATUS_AND_FINDINGS和METHOD_AND_IMPLEMENTATION为准。
