# 双线Batch-1：实施接手记录

> 接手后更新：任务“实现 Raw-v1 并部署并行实验”（01a0a12c-241a-7772-986d-387138bce4d6）报告已从149fc54建立codex/wtr-raw-v1，CPU四项合同通过，已写Raw接口、GPU gate和384-swap mini-bank CLI，正准备只读资产传输与Slurm提交。这里是接手方进度报告，不是本线程独立复验；下文“无Raw实现”是接手前库存快照。本线程不修改Raw源文件。

> Atlas负责的Codex任务已核实为“报告首轮实验与验证目标”，taskId=01a0a089-9fc9-7412-9a04-6fa1ee51cdbb，hostId=local。02:43:59远端读取：S-T allocation 697/792，B-T 153/792，均未完成，failed.json不存在。没有自动可用的GPU时隙；任何Raw短验收需由Atlas owner明确协调，不能在阶段切换时自行占卡。

日期：2026-09-15。接手任务：`01a0a12c-241a-7772-986d-387138bce4d6`。本文件由讨论任务整理，仅做只读核验与文档交接；未修改Raw/Atlas代码、未提交或取消作业、未重复采集。

## 1. 最新用户决定覆盖旧FIRST_BATCH的顺序

**Batch-1A：Core scientific gate。** Atlas继续提供现有合法算子空间的D/S测量与Uniform/Attention/CF参考，优先D，再S。数据足够后独立决定是否推进Core-D/Core-S，不等待Raw。

**Batch-1B：Evidence Acquisition Prototype。** Raw-v1独立开发EpisodePublic、公共preview、有界Raw proposals、Unseen Descriptor、RawSelection与tubelet映射，然后进行冻结detector的O/R实验与共享Temporal Value pilot，不替代Core科学主线。

旧FIRST_BATCH中的“1A=Raw零更新、1B=Raw小头拟合”已被新命名覆盖；Raw内部使用R0/R1等阶段即可。P1改称Evidence Acquisition Prototype，P2为Computation Refinement Core。

四条最新硬规则：

1. Atlas/Core与Raw并行，互不阻塞，不重复建立已有D/S采集体系。
2. Raw proposal、preview、descriptor和search参数只由training/development确定；publication/test CF不反向调参。
3. 先做300–500个swap的mini-bank，检查重放、噪声、候选重复和pair重组，再决定是否扩大到约6400。
4. router holdout ranking/regret通过简单proxy/Uniform控制后，才运行O-V/R+-V全211视频/792窗口。

192×64×64 preview是pilot，不是永久架构；不马上扫描多个preview密度。主shared Value控制不输入实验domain one-hot或语义等价的条件ID；真实几何、membership、有效性和观测来源仍可用。

Raw GPU顺序：先一个完整THUMOS视频所有窗口的reader→prediction/AP桥接，再6个training/dev视频，随后mini-bank。记录N_changed_pairs、N_changed_packs和pair-span变化。其相关性不能单独证明收益由packing而非内容造成。

## 2. 权威设计文件

- [Master Plan](MASTER_PLAN.zh.md)
- [旧FIRST_BATCH实施细目](FIRST_BATCH.zh.md)，顺序与mini-bank范围由本文件及用户最新消息覆盖
- [Raw-v1接口](../wtr_raw_input_20260915/RAW_V1_CONTRACT.zh.md)
- [Core概念设计](../wtr_design_20260915/DESIGN_BASELINE.zh.md)
- [Atlas当前STATE](https://github.com/yuzbo/BMCR-T-AdaTAD/blob/df2b168ff109dda8838ef38bf5dbee048d14c70b/research/atlas_20260915/STATE.md)
- [Atlas冻结PROTOCOL](https://github.com/yuzbo/BMCR-T-AdaTAD/blob/df2b168ff109dda8838ef38bf5dbee048d14c70b/research/atlas_20260915/PROTOCOL.zh.md)

Atlas publication协议不因新Raw设计改变。新Value训练只读取training/development记录；不能把publication输出回流训练或调阈值。未来FVD/Graph可独立诊断，但不进入Raw-v1主配置。

## 3. 代码状态快照

Atlas路径：`C:/Users/skywalker/Documents/ChatGPT/H65/h65_clean_adatad/wtr_characterization_20260915`，分支`codex/wtr-characterization-20260915`，本次核对HEAD=`149fc54e5c673f9afc25cfae0c9a886abc22b100`。此前报告的50a49e已不是最新HEAD；实际测量源码另看CODE_REVISION。

dirty：`research/atlas_20260915/STATE.md`有更新；CODE_REVISION/PLOT_REVISION、3个部署tar包、source_resources.local.json未跟踪。它们是现有Atlas运行/交接区，不要回退。该区由现有Atlas任务管理，本线程未修改它。

Graph路径：`C:/Users/skywalker/Documents/ChatGPT/H65/h65_clean_adatad/graph_tad_20260914`，HEAD=`1111e53da91e591e5ad702e1c4e499f905337841`。未跟踪旧characterization原型、tools/characterization_*.py及Graph图/monitor；不把它们冒充已验证Raw实现。

本次库存快照未发现Raw工作树或Raw源实现。本讨论线程此前只写reports，没有创建Raw代码，也没有Raw GPU入口；接手任务后续新建内容不在这个库存结论范围内。

## 4. AutoDL/ssh服务器

连接：`ssh -T -o BatchMode=yes -o ConnectTimeout=15 -p 44909 root@connect.nmb1.seetacloud.com`。

- 实验根：`/root/autodl-tmp/wtr_characterization_20260915`
- Python：`/root/autodl-tmp/envs/opentad/bin/python`
- 官方源码：`/root/autodl-tmp/OpenTAD`
- 数据：`/root/autodl-tmp/thumos14/{train,test,annotations}`，正式200训练/211测试视频。
- 资产根：实验根下`assets/`；含adatad_s_ema.pth、adatad_b_ema.pth、v2_s_epoch040_light.pth、v2_b_epoch040_light.pth及initial_source重构材料。

V2导出包包含完整EMA和初始差异等重构信息，并非只有light层。`h65/atlas/recovery.py::build_probe`按需构造initial_source，避免并发重复写共享资产。S/B恢复预检均已有回执。

02:31:59–02:32:19快照：owner PID32242；GPU0 allocation_s_T PID40846，606/792；GPU1 allocation_b_T PID80493，99/792。两卡RTX4080 SUPER各32760MiB，分别约55%/3670MiB和68%/6544MiB，两卡均被Atlas任务使用。

队列为16 COMPLETED、2 RUNNING、12 WAITING；training_updates=0。queue/launch登记source_revision为5236ae4，启动时间01:12:24；CODE_REVISION=`a50b84db1bcdafd86ceec4d9737156807f9daa9a`，PLOT_REVISION=`2c217ea8dddff02b4ff0b2dc0e9627f6c1314296`。两种版本含义不同，不覆盖各科学回执的来源。

## 5. Atlas可用记录与Go/No-Go边界

相对远端实验根：

- `results/calibration_s_D/`、`calibration_s_S/`、`calibration_b_D/`、`calibration_b_S/`：均development，32视频/32窗口，各有`windows/`32个记录、manifest_0.json、progress_0.json、accounting_0.json、shard_0_done.json。
- `results/allocation_s_T/`与`allocation_b_T/`：publication，目标211/792，当前未完成。
- population_s/b、allocation_*_D/S、完整recovery、analysis/figures尚等待。

Dense-S完整结果：Avg-mAP68.9767271877%，@0.7为48.2622113548%，2347.894038528G/窗口；科学来源75ba823，完成00:52:41。

Dense-B完整结果：Avg-mAP71.1416637624%，@0.7为49.5514356021%，8082.154708992G/窗口；科学来源a50b84db，完成01:50:20。两者均通过既定0.10pp复现容差。

S/B开发完整视频、15组AP复算、恢复预检及T/D/S校准已完成。当前仍没有完整D/S publication分配曲线，不能从单点/组loss或calibration accounting编造CF–Uniform mAP。可以先复用development数据做明确范围的分析；完整科学gate需相应真实执行结果。

Atlas D动作包含其冻结协议定义的attention/FFN更新，不能不核对动作语义就改名为新Core纯attention admission标签。新策略条件slot-exchange需要在现有框架上扩展，不重复重写整套collector。

## 6. 4090 Slurm资源

连接：`ssh -F 'C:/Users/skywalker/Documents/ChatGPT/refine-tad/V3/motivation/ssh_config' bcr-4090`。

- Host：ssh.cn-zhongwei-1.paracloud.com:22；User：sczc063@BSCC-N16R4。
- Python：`/data/run01/sczc063/yuzibo/conda_envs/opentad/bin/python`，CUDA11.8。
- H65数据：`/data/run01/sczc063/yuzibo/thumos14/{train,test,annotations}`。
- 原始训练视频：`/data/run01/sczc063/yuzibo/raw/Validation Data/validation`。
- 原始测试视频：`/data/run01/sczc063/yuzibo/raw/Test Data/TH14_test_set_mp4`，按标注选211，不使用额外文件。
- 共享工作根：`/data/run01/sczc063/yuzibo`；既有motivation目录与其他任务源码/结果仅作各自已有资源，不写Raw进去。

只读squeue快照：12运行、2排队。运行：1290311@g0066、1290286@g0014、1290284@g0059、1290283@g0044、1290282@g0043、1290280@g0087、1290164与1290163@g0059、1289883@g0014、1289827@g0043、1288541@g0022、1287130@g0006。排队1290136(AssocGrpGRES)、1290312(Priority)。这些是账户作业，不能据ID默认都归本任务可调整。

GPU必须通过Slurm allocation，不能在登录节点或其他任务allocation中计算。节点/配额状态执行前再查，不取消其他项目作业。

## 7. A100资源

连接：同ssh_config中的`bcr-a100`。Host为ssh.cn-zhongwei-1.paracloud.com:2222，User为pxyai_0057@GUANGZHOUXY-A100。本轮已实际只读连接成功。

- 工作根：`/HOME/pxyai/pxyai_0057/HDD_POOL/yzb`
- Python：`/HOME/pxyai/pxyai_0057/HDD_POOL/yzb/geosparse_tad_20260907/envs/opentad/bin/python`
- 既有镜像：`/HOME/pxyai/pxyai_0057/HDD_POOL/yzb/bcr_tad_v3_motivation`
- 原始训练：`.../geosparse_tad_20260907/assets/raw/Validation Data/validation`
- 原始测试：`.../geosparse_tad_20260907/assets/raw/Test Data/TH14_test_set_mp4`
- 标注：`.../geosparse_tad_20260907/assets/thumos14/annotations/{thumos_14_anno.json,category_idx.txt}`
- 官方B权重：`.../geosparse_tad_20260907/official_adatad/assets/official_b_checkpoint.pth`；motivation/checkpoints/official_b_checkpoint.pth为其symlink。

02:35:48本账户squeue为空，但不代表GPU空闲。a100x中an103/104/105/109/110/113为MIXED；scontrol确认每个节点8张A100均已分配，其他列出节点为draining/drain。新任务须经Slurm申请/排队，不能依据空用户队列或剩余CPU直接使用节点。

S及V2资产需由实施owner按已核对来源同步/核验，本线程没有修改这些资源。数据原始路径已确认存在，但正式仍以200/211标注集合验证。

## 8. 可复用入口与尚不存在的Raw入口

已有：`tools/atlas_run.py`、`atlas_validate.py`、`atlas_analyze.py`、`atlas_plot.py`；唯一队列owner为`tools/atlas_queue.py`。状态读`queue/status.json`和`queue/launch.json`；不要再启动一个--launch。

复用函数：`CharacterizationData/window_metadata`、`FrozenReference`、`recovery.build_probe/recovery_window`及已验证AP/profile。注意Atlas temporal.py的index_select只读取官方pipeline生成的data['inputs']，不是原始任意frame ID reader。

Raw最小模块尚需实现：前置preview、bounded raw proposal、Unseen Descriptor、raw frame读取、独立坐标/GT转换、确定性tubelet、共享Temporal Value与专用评测入口。没有可宣称“现已可跑”的Raw命令。

关键桥接：THUMOS K384→192 heavy anchors→384 Cross配对query→readout映射768 detector位置。不得简单改decoder length为768。Raw-v1保持D/S、query与head，Graph/FVD/DB/时间PE/micro-clip不并入首批。

原资源详情见[handoff_4090](C:/Users/skywalker/Documents/ChatGPT/H65/handoff_4090.md)及[RESOURCE_LOCATIONS](C:/Users/skywalker/Documents/ChatGPT/refine-tad/V3/motivation/RESOURCE_LOCATIONS.md)。没有复制或发送任何私钥/密码内容。

本交接已通过Codex任务消息发送给实施owner；本线程保持Raw文件不动，现有Atlas作业继续由其owner管理。
