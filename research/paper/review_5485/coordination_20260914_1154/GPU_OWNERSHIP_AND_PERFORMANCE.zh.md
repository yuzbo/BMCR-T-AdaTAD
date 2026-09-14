# GPU所有权与完整性能只读报告

核验截点：2026-09-14 11:54:50 UTC+8。只读Slurm元数据、本任务ledger和已有完整结果；没有取消、暂停、提交或重启作业，没有读取其他项目的代码/训练内容。

**本任务仍无活跃自有GPU作业，ID列表=[]，当前没有可通过本任务让出的GPU。** V3/DST两个运行目录的实时作业匹配为空，DST ledger仍为HOLD_FOR_FPW_RESOURCE_PRIORITY、training_dispatcher_started=false。旧E1/E2收尾与所有准入已完成。

## 1. 1289618的归属与状态

- JobName：raw-half-distill-t；当前PENDING(AssocGrpGRES)，RunTime=00:00:00，AllocTRES=(null)。
- 请求2个节点、每节点2张GPU，合计申请4张；**当前实际分配0张，不能把它当作正在占用4张GPU的任务。**
- WorkDir：`/data/run01/sczc063/wangruofan/projects/project1/train/code/DIVA_Code/~custom/vibe/8_raw_half_distillation_t_baseline`。
- Command：上述目录中的`run_un2CLIP_irrgb_half_distill_sbatch.sh`。

这不是本任务V3/DST或协调方paper目录的作业。本任务没有它的提交或使用记录。具体Codex任务/人员归属不由共享账号或目录标签进一步猜测，也未读取该项目的训练结果。

## 2. 账户当前16张已分配GPU

按RUNNING作业的AllocTRES统计，未用PENDING请求数充当实际占用。

| 作业 | 已分配GPU | 元数据所示归属 |
|---|---:|---|
| 1289570–1289575 | 6（各1） | 协调方paper的六个核心训练 |
| 1289505 scripted_gpu | 1 | `/data/run01/sczc063/wangruofan/projects/project1`，主命令sleep |
| 1287130 scripted_gpu | 2 | 同上，主命令sleep |
| 1288541 seed-cure | 3 | `/data/run01/sczc063/huangjian/SEED/SEED-main`；启动run_seed_gpu_hold_14d_3gpu.sbatch |
| 1287823 interactive | 4 | `/data/home/sczc063`，主命令/bin/bash；仅此元数据不能确定具体项目归属 |
| 本任务V3/DST | **0** | 无活跃作业 |

1287823没有本任务提交/使用记录，不列为本任务可取消资源。`sleep`或`/bin/bash`主命令也不证明allocation内没有其他计算，不能据此称空占或自行取消。

协调方的PBD 1289576/77、Dense复测1289578/79及ANet1289684均为PENDING(AssocGrpGRES)。此外1289618也在等4张，尚未分配。用户对本任务低性能实验让位的授权不扩展到上述其他项目或归属不明的interactive。

## 3. 本任务已完成、同预算配对的mAP与计算量

全部为THUMOS完整200训练/211测试、seed42、60轮、固定epoch59 EMA，按每5轮完整评估。主K16；K20是同一最终checkpoint的预算曲线，不能改称主预算或逐epoch峰值。

计算量为单个官方768采样帧、160输入窗口的总神经网络FLOPs，含cheap/selector、heavy、传播、接口和检测器，补计SDPA；2 FLOPs/MAC，排除解码与NMS。表中GFLOPs=10^9 FLOPs。

| 已完成方法 | K | 平均mAP B / S | GFLOPs B / S |
|---|---:|---:|---:|
| PVR-ASFormer，最佳学习派生 | 16 | 59.1071 / 55.8110 | 2799.302274 / 855.887994 |
| PVR-ASFormer，最佳学习派生 | 20 | 61.0295 / 57.8431 | 3471.391976 / 1050.375085 |
| E1-heavy，最佳Uniform控制 | 16 | 61.0934 / 57.5514 | 2774.360487 / 830.831978 |
| E1-heavy，最佳Uniform控制 | 20 | 63.0069 / 59.2391 | 3446.488061 / 1025.328568 |

原联合目标仍未达到：B/S平均mAP至少64.026/62.127，且各自官方总FLOPs不超过50%。上述成本满足减半，精度不足；Uniform不算学习路由创新成绩。DST只有工程准入与成本结果，正式训练0、mAP=null。

协调方本次提供Full-V1-S epoch10全211/792为63.2251083%、代表窗1170.152664G，BMCR-B80新峰68.3447613%、4077.475688G。本任务最强控制K20精度分别仍低约3.9860和5.3378点，学习派生更低；没有超过所提供的精度。但成本、训练配方、轮次/峰值选择不相同，不把差值解释为严格等预算、等配方的因果比较。本次没有重跑协调方模型或复核其完整算量边界。

性能来源：[完整性能与成本记录](C:/Users/skywalker/Documents/ChatGPT/refine-tad/V3/RESOURCE_PERFORMANCE_REPORT_20260913.md:15)、[已完成阶段结果](C:/Users/skywalker/Documents/ChatGPT/refine-tad/V3/RESEARCH_STAGE_FINAL_RESULTS_20260913.md:3)。原报告的旧运行作业栏不作为当前资源状态使用。

## 4. ActivityNet新阶段

协调方已定位原发现逻辑只接收mp4，漏掉真实存在的MKV/WebM（其报告raw目录有736个MKV、16个WebM，ZIP也有）。修复与新作业1289684均由协调方执行，本任务未重复实现或提交。当前权威回执已实际读取为job_id=1289684、phase=resume_all_observed_video_formats；Slurm为PENDING(AssocGrpGRES)。继续单一准备owner及只读同步，不重新下载数据。

原始所有权证据：[resource_ownership_audit_20260914.json](C:/Users/skywalker/Documents/ChatGPT/refine-tad/V3/OpenTAD/reports/coordination/resource_ownership_audit_20260914.json)。分配汇总：[resource_ownership_summary_20260914.json](C:/Users/skywalker/Documents/ChatGPT/refine-tad/V3/OpenTAD/reports/coordination/resource_ownership_summary_20260914.json)。
