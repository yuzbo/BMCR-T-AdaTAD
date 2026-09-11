# 实验协议与重跑入口

## 数据、初始化和日程

THUMOS14：训练200视频；测试211视频、792滑动窗口。官方文件命名中的Validation原始视频被用于训练，标注内部subset为training；Test视频对应评测subset validation。不要因这些历史命名把测试视频当成开发验证集。后续调参应从200训练视频中按视频划分开发集，保持211测试集封存。

新模型：K400识别预训练；新Adapter与检测器；VideoMAE主干非Adapter参数冻结；seed3407；batch2；200视频每epoch各一次标准随机窗口；100更新/epoch。每骨干共享20轮均匀预热，H65-C/BMCR-T各40轮联合；两个分支从同一预热末轮EMA初始化，重置优化器和调度器。共20000次实际更新，记录的六阶段训练时间合计21.22 GPU小时，不含排队、预检、审计或测试。

预热会训练Adapter、检测器，以及CNN/ASFormer动作辅助任务与转变监督；scout不控制重骨干采样。联合阶段采用20轮过渡＋20轮完整联合，检测器反馈在667次更新后开始并用1333次更新爬升。AdamW、梯度范数裁剪1、EMA0.999；S/B Adapter LR分别2e-4/1e-4，检测器1e-4。详细优化器分组与课程见 `h65/full/runtime.py`、`objectives.py` 和各run的 `config.json`。

官方基线严格加载交接中已有的 `state_dict_ema`；S/B检查点epoch字段59/51。不从该字段推断原始训练总长度、最优轮次或选模历史。本轮不重新训练官方模型，也不依据本次测试替换其权重。新模型主结果统一取末轮EMA。

两骨干各做训练来源128窗口审计：118窗口/75视频可测，390次交换，310拟合和80留出（实际60/15视频）。目标缩放只用拟合部分；审计不是测试集结果，也没有证明最终模型的效用排序可靠。

## 评测口径

精度：原OpenTAD完整视频汇总、类别SoftNMS和五阈值mAP，211视频全部覆盖。每组公开压缩预测含422000条候选记录。测试不能改变训练checkpoint或超参数。

MAC：实际前向矩阵/卷积形状计数，含12次融合FlashAttention的QK与AV，未解析矩阵算子为空。主窗口固定为 `video_test_0000007`、start0、测试index2、768有效候选；另保存503与253有效候选窗口。FLOPs=2MAC仅针对矩阵/卷积，归一化、softmax等非MAC算子单列，不声称统计全部算术。

时延：RTX4090，BF16骨干/scout、FP32检测器，batch1，GPU驻留输入，5次预热＋20次CUDA同步测量。含scout、路由、骨干、Adapter、检测器、NMS前回映；不含解码和NMS。保留全部样本，报告均值/中位数/范围。不同方法是独立作业，可能不同节点，未测组件耗时，不能量化归因某模块或称端到端服务加速。

## 环境与资源

已运行环境：Linux，Python3.10.20，PyTorch2.0.1/CUDA11.8，torchvision0.15.2，mmcv2.0.1，mmaction2 1.1.0，mmengine0.10.7。更多版本见[环境记录](ENVIRONMENT.json)。这是已有环境记录，不是已验证的全新安装锁文件。OpenTAD依赖和编译扩展见其[固定版本安装文档](https://github.com/sming256/OpenTAD/tree/346d09d19e2091372cec48172dbe40f7b28bdee6/docs)与 `upstream/requirements.txt`；custom ops必须针对实际PyTorch/CUDA编译。

默认资源布局（可使用同名符号链接），原视频与权重不在Git仓库内：

```text
resources/
  thumos14/annotations/thumos_14_anno.json
  thumos14/annotations/category_idx.txt
  thumos14/videos/validation/*.mp4
  thumos14/videos/test/*.mp4
  checkpoints/videomae_s_k400.pth
  checkpoints/videomae_b_k400.pth
  checkpoints/adatad_s_ema.pth
  checkpoints/adatad_b_ema.pth
```

在独立环境中 `source resources.example.sh` 后配置 `H65_RESOURCE_ROOT`、`H65_RUNS_DIR`，识别和官方权重路径也可分别覆盖。默认新输出位于 `runs/`；公开历史证据位于 `phase2_20260910/runs/`，不要将其作为新训练输出目录。

GPU入口保留原实验的独立Slurm allocation与4090要求。集群账户、partition、qos和环境激活由操作者设置；仓库不包含原调度器、SSH配置或账户专属队列规则。例如在已激活的环境下：

```bash
sbatch --gres=gpu:1 --cpus-per-task=6 --time=12:00:00 tools/run_full.sh tools/full_train.py --backbone s --phase warm
```

每个骨干的依赖顺序如下；这些是完整运行命令，不是发布时自动执行的步骤：

```bash
# 1. 共用预热，成功完成后才执行下面步骤
python tools/full_train.py --backbone s --phase warm
# 2. 训练来源审计及目标缩放
python tools/full_audit.py --backbone s
python tools/summarize_audit.py --backbone s
# 3. 两个独立40轮分支
python tools/full_train.py --backbone s --phase joint --variant h65
python tools/full_train.py --backbone s --phase joint --variant bmcr
# 4. 固定终点模型完整测试；官方仅测试已有权重
python tools/full_eval.py --backbone s --variant official
python tools/full_eval.py --backbone s --variant h65
python tools/full_eval.py --backbone s --variant bmcr
```

以上GPU命令应分别通过自己的Slurm作业运行；B将 `--backbone s` 改为 `b`。不要在登录节点直接执行这些GPU命令。训练自动从本输出目录latest checkpoint恢复；`--preflight`仅两步，用于新环境运行核查，不能作为完整训练。

## 发布检查

发布副本的19项CPU测试已通过；六组全部压缩预测/指标/profile与20000条成功更新的一致性检查通过。命令和输出见[发布验证](../validation/README.md)。原正式实验已经完成真实GPU更新、严格EMA加载和完整测试；发布过程中没有重训或重复整套精度测试。CPU检查不会证明所有科学假设成立。
