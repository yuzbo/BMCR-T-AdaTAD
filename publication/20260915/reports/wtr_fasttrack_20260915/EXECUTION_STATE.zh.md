# 当前执行与后续交接

日期：2026-09-15。本文件区分已完成的实现/审阅、真实运行与尚未完成的科学路线。当前计划见EXPERIMENT_PLAN.zh.md，逐实验审阅见REVIEW_STATUS.zh.md及../wtr_code_review_20260915_578bab2/REVIEW.zh.md。

## 最新11:33进度覆盖

完整报告见progress_20260915_1100/REPORT.zh.md。D-V/D-U已到epoch10、1000 updates，04:51/04:55因首次eval的JSON日志序列化故障停止；科学模型未改，已保留checkpoint并部署独立复核通过的评测记录修复6e2fc7f（EVALUATION_REVISION单记，模型Science SHA仍405）。当前续跑1290651/1290652 Priority排队，尚无新Core mAP。原有前10轮不重训。

A100原S248205/248206确认长期PENDING后已取消并迁至4090，1290654/1290655 Priority排队。现4090唯一owner PID279794（操作前仍须核对最新deployment）。A100 S owner在共享取消状态下退出；不要恢复旧排队申请造成重复。

Raw1290331已于04:07:41完成（9分53秒）。6视频19窗口一致性、24视频346swap和诊断均完成，replay error=0；Value尚未拟合。R+180动作中仅24次off-grid，38/45配对状态动作相同，holdout4视频8状态全部同动作；候选池实际每状态扩展576个frame。域/学习gate未通过，不直接扩6400。

Atlas T两模型完整211/792及10000次video bootstrap已有结果：8组S的CF−Uniform为+3.507pp [2.337,4.692]；B为+1.392pp [−0.058,2.594]。额外GT/查询开销分列。11:25 S-D509/792、B population321/792正常推进。V2-B、Uniform S/B的旧80轮终点均已完成。下方早期运行/PID快照保留为历史，当前状态以本节及新报告为准。

## 用户当前约束

按Fast-Track推进完整80轮及全量评测；边训边测，不拆大量独立评测任务；每个实验实现完成后交叉审核与讨论。用户随后授权取消价值较低的4090旧课程。Atlas继续，其他成员作业不属于本次取消范围。所有方法选择使用training/development；publication不能反向调参。

## 源码和已完成工作

- Core工作树：C:/Users/skywalker/Documents/ChatGPT/H65/h65_clean_adatad/wtr_fasttrack_20260915；分支codex/wtr-fasttrack；科学代码40552945ad5d56b7404f833cd86f998e8558e8b1。
- Raw工作树：C:/Users/skywalker/Documents/ChatGPT/H65/h65_clean_adatad/wtr_raw_v1_20260915；分支codex/wtr-raw-v1；bank/Value修正版27d557ee3bd38a52aec8caaf16860a95957b44b6。
- 新代码实现精确packed D配额、守恒native-time S配额、真实D/S交换监督、统一T Value接口、固定容量正式配置、fresh-instance reload和内联评测。D/S目前使用当前状态、pack均值及几何，尚不是最终共享Scout/联合预算全部实现。
- Raw前置preview、公开Episode、Raw-Expand、真实帧读取、tubelet重排、384→192→384→768桥、mini/full bank及Value fit入口已实现。1259d25在4090完成一个训练视频3窗口的RGB/预测/官方AP缓存一致性；后续Raw改动未改reader几何。
- 独立交叉审阅发现并修正：Core Value label holdout隔离；TDS D/S标签保持当前T支持；Raw完整bank准入；评测点元数据；Raw收益排序。Temporal descriptor同步传入实际D/S容量。
- Code/protocol回执为精确版本、逐配置D-U/D-V/S-U/S-V PASS。仅覆盖代码与协议，GPU启动及科学收益另判。回执已复制到Core research/wtr_fasttrack/reviews；paper_course/admission会消费。
- Value监督仅160 fit，20 calibration/20 holdout只作稳定checkpoint re-query；detector仍200。称router-label holdout，不称全模型未见视频。排序统一actual cls+loc，RMS仅影响回归conditioning。

## 4090当前部署

SSH配置 C:/Users/skywalker/Documents/ChatGPT/refine-tad/V3/motivation/ssh_config，别名bcr-4090；Python /data/run01/sczc063/yuzibo/conda_envs/opentad/bin/python。

Core远端 /data/run01/sczc063/yuzibo/wtr_fasttrack_20260915。使用原paper_20260913 owner，未另建竞争队列。owner目录 /data/run01/sczc063/yuzibo/h65_clean_adatad_20260910/paper_20260913；最新launch的owner PID2094632（操作前读deployment确认，不盲用PID）。新D-V/D-U作业1290328/1290329，405版本已通过两次更新/fresh reload并进入训练。当前精确进度读取Core research/wtr_fasttrack/current_status.json或各run/train.jsonl。

04:04:14 +0800快照：D-V=227 updates/epoch3，D-U=213 updates/epoch3；D-V累计23条Value标签，labels_outside_fit=0。两条本版本preflight均2更新/fresh reload通过。Raw1290331已RUNNING@g0066。此快照不是实时状态，也不是科学收益结论。

Raw修正版远端 /data/run01/sczc063/yuzibo/wtr_raw_v1_20260915/revision_27d557e；复用上一级assets只读。作业1290331低优先级在同一allocation内执行六训练视频验证→384上限mini-bank→仅诊断统计，遇失败停止，不自动扩6400或跑publication。入口raw_course_4090.sh，输出results/r2_six、results/mini_bank、results/mini_diagnostics。旧A100 Raw排队248202已取消以避免重复。

## A100部署

同一SSH配置的bcr-a100；Python /HOME/pxyai/pxyai_0057/HDD_POOL/yzb/geosparse_tad_20260907/envs/opentad/bin/python。

Core目录 /HOME/pxyai/pxyai_0057/HDD_POOL/yzb/wtr_fasttrack_20260915（真实路径映射到/XYAIFS00/HDD_POOL/pxyai/pxyai_0057/yzb/...）。复用paper_dispatch的A100分区支持，独立管理S-V/S-U课程。新owner PID1121267、host=ln302；gateway会在ln301/ln302间切换，因此不能在另一个登录节点盲按/proc/PID停止。全局dispatcher.lock及共享deployment状态用于确认唯一owner。

新S pair读取405版本和对应code review回执；排队/运行状态以squeue和research/paper/deployment.json为准。旧578候选S作业248203/248204已取消。Raw assets已部署两端。

当前新S-V/S-U作业为248205/248206，04:04左右均PENDING(Priority)。

## 已退役及保留

按用户授权，停止9个旧H65活动作业并关闭384个旧阶段。取消范围为旧Full-V1、PBD、旧Graph组合及额外factor评测；保留V2-B和Uniform S/B收尾。V2-S已有80轮及20/40/60/80 checkpoint，可用于后续受控诊断。详细回执receipts/FASTTRACK_RETIREMENT.json，现有结果和checkpoint未删除。

旧Core 578bab2候选D作业1290324/1290325因router-label划分修订保存后归档，目录Core research/paper/review_candidates/578bab2_20260915_033052。其技术通过不冒充405科学结果。原owner3506502、后续1912190/1955978均不再是当前owner；旧source_revision不回填。

## Atlas及剩余工作

AutoDL入口ssh -p44909 root@connect.nmb1.seetacloud.com；Atlas目录/root/autodl-tmp/wtr_characterization_20260915，两张4080 SUPER由既有Atlas owner使用。原Atlas task=01a0a089-9fc9-7412-9a04-6fa1ee51cdbb；本次按用户指定持续通信的讨论/审阅task=01a0a0b2-3d36-76b1-839b-bdb98ae48c0e。Atlas未取消、未重复采集。

当前仍未完成：新模型全量mAP；Raw六视频/完整mini-bank的实际完成与信号判断；稳定checkpoint上的Core calibration/holdout re-query及科学gate；逐配置T/DS/TDS准入；Graph G0/G1、RISE-A/B、DB的科学runner和实测证据。不能将配置登记或code PASS解释为这些路线完成。DS/TDS、Graph/FVD、Raw matched train及B/ANet扩展按计划的证据依赖继续。

Graph G0需Cross_existing/Cross_fresh及训练预算匹配；G1需同bank Plain/Graph/参数匹配MLP。RISE-A真实value drift与RISE-B同一raw state的完整routing-function forecast分开。DB须先有trained-tier支持。持久fixed-action bank必须绑定checkpoint/support/policy/recovery/candidates；在线即时CF日志不能直接改名为另一版本的bank。

## 复现和维护

Core使用tools/paper_course.py（review准入、丢弃两次技术更新、fresh reload后进入80轮）、paper_train.py（checkpoint/EMA/RNG/内联10/20/40/60/80评测）、原paper_dispatch（time-slice续跑）。不要修改正在运行的科学文件或把旧checkpoint当新版本resume；实质变更须留旧回执并按新版本重开对应课程。

本地Torch DLL不可用，纯语法/合同可在本机执行；Torch CPU回归使用既有4090环境。未改共享Python环境。FastCtx工具用于本地文件和bash；它们不在functions.exec的tools对象里，须直接调用。SSH偶尔connection closed；复制/读取可重试，提交作业不确定时先核对回执，避免重复。

FINAL_MODEL_LEDGER.csv目前均WAITING/Final=no，记录科学组件准入而非代码审阅。原文完整保存在inputs/；CSV已按电子表格技能生成并渲染检查，无需重复生成。
