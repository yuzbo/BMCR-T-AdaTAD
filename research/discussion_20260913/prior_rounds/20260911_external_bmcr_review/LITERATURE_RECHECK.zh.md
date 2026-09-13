# 文献与外部源码逐项复核

复核日期：2026-09-11。编号P/X沿用外部原报告。这里核验报告所引用的机制与任务边界，不声称复现外部方法，也不从他任务结果推断BMCR-T能提升多少mAP。外部作者仓库链接多为浮动分支，本次只核对所述关键片段，不将其视为与BMCR固定提交同范围的完整审计。

| 项目 | 评价 | 本轮证据与应保留的限定 |
|---|---|---|
| P1 AdaTAD | 接受 | [原论文](https://arxiv.org/abs/2311.17241v2)与[作者仓库](https://github.com/sming256/AdaTAD)支持CVPR2024、TIA/参数高效端到端TAD的身份。不能借其训练显存收益为“RGB观察减半无损加速”背书。 |
| P2 Uni-AdaFocus | 接受，迁移为假设 | [原文](https://arxiv.org/html/2412.11228v1)采用廉价全序列观察与重局部编码，讨论深特征采样监督及梯度干扰。任务是视频识别；ActivityNet数据集名称不能把它变成TAD边界证据。 |
| P3 PointRend | 接受 | [论文](https://arxiv.org/abs/1912.08193)为实例/语义分割。X1代码确实先插值logits再计算不确定性；省的是精细预测头点计算，不是整个已运行图像骨干。 |
| P4 Deep Feature Flow | 接受 | [论文](https://arxiv.org/abs/1611.07715)与[代码](https://github.com/msracver/Deep-Feature-Flow)支持视频检测/分割中的稀疏关键帧重编码与光流传播。把此机制迁移到global-TIA窗口特征缓存仍需上下文一致性条件。 |
| P5 CoLT5 | 接受 | [论文](https://aclanthology.org/2023.emnlp-main.309.pdf)表3的Base均值为CoLT5 42.4、LongT5 43.1；整体速度还受decoder/MQA等配置影响。light-all/heavy-selected可借鉴，不能称所有规模无损，更不能推断VideoMAE-S必然获益。 |
| P6 Sparse DETR | 接受 | [论文](https://arxiv.org/abs/2111.14330)、[作者代码](https://github.com/kakaobrain/sparse-detr)支持COCO检测与decoder相关token的稀疏encoder更新。减少的主要是encoder更新，前置dense CNN成本仍在。 |
| P7 Focus-DETR | 接受 | [原论文](https://arxiv.org/abs/2307.12612)明确评分同时考虑定位与类别语义，舍弃背景query并加强有效query交互；官方入口为[huawei-noah](https://github.com/huawei-noah/noah-research/tree/master/Focus-DETR)。本轮没有逐行认证整个实现。 |
| P8 Soft Teacher | 接受 | [论文](https://arxiv.org/abs/2106.09018)与X2支持EMA、弱/强视图对应、分类置信度和框抖动可靠性。它是半监督检测，迁移为有标签TAD互补视图KD是新假设。 |
| P9 Be Your Own Teacher | 接受 | [原文](https://arxiv.org/abs/1905.08094)摘要明确把网络分段，将深层知识传给浅层。主要分类证据不等同于时序定位改善。 |
| P10 data2vec 2.0 | 接受 | [原论文§3.1–3.3](https://proceedings.mlr.press/v202/baevski23a/baevski23a.pdf)明确EMA teacher、无mask教师、多mask学生共用上下文化目标以摊薄教师成本。当前二分类actionness不能直接与20类检测输出做同分布KL，是本项目的额外接口判断。 |
| P11 CrossKD | 接受，改写迁移表述 | [论文](https://mftp.mmcheng.net/Papers/24CVPR-CrossKD.pdf)、[作者代码](https://github.com/jbwang1997/CrossKD)的核心是把student head中间表示送入teacher head作跨头预测，缓解GT/KD目标冲突。“借鉴梯度隔离”可作为设计建议，不能替代原方法完整机制描述。 |
| P12 On the Efficacy of KD | 接受 | [原论文](https://arxiv.org/abs/1910.01348)明确更大、更准确的教师不一定更会教，容量失配会使小学生难以模仿。只能否定“更强teacher必更好”的通则，不能据此归因本项目S下降。 |
| P13 SampleNet | 接受 | [论文](https://arxiv.org/abs/1912.03663)及X3支持训练软投影、推理硬近邻/补齐。官方registration实现确有CPU NumPy往返，已直接读到；没有理由因早期访问未成功而删除这一正确描述。原论文任务不等同于3D检测。 |
| P14 SAMBLE | 接受 | [论文](https://arxiv.org/abs/2504.19581)、[作者代码](https://github.com/stevenczwu/SAMBLE)支持针对过度sharp-edge采样偏置，在形状相关分配中平衡局部细节与全局覆盖，分类/分割为主要证据。把它类比为TAD边界/覆盖是合理假设，不是收益保证。 |
| P15 Length-Adaptive Transformer | 接受 | [论文](https://aclanthology.org/2021.acl-long.508/)及X4支持中间drop、末端restore，以保留span/token任务的输出身份。主代理直接核对了按原token索引scatter保存最近有效hidden的实现。 |
| P16 TOKEE | 接受 | [论文](https://aclanthology.org/2021.acl-long.16.pdf)明确退出token不再向其他token发起高层注意力，但仍能被其他token读取。任务为序列标注；不能当成事件抽取已经得到验证。 |
| P17 IA-SSD | 接受主要机制，保留实现范围 | [论文](https://arxiv.org/abs/2203.11139)、[代码](https://github.com/yifanzhang713/IA-SSD)支持实例感知前景下采样和3D检测/中心定位。原XYZ/投票细节与点集表示相容，但本轮未遍历其全部算子和坐标更新，不称完整源码认证。 |
| P18 MSDR-Mamba | 接受论文所述事实，源码仍有限 | [出版页面](https://www.mdpi.com/2079-9292/15/17/3797)直接请求遇429，随后检索工具返回了该出版社页面的§3.5–3.7、算法1、实验讨论和数据可用性正文。双路径均执行、每FPN层全局标量gate、冻结特征输入以及router增加时延/内存均有原文支持；不是RGB稀疏执行。源码未取得；作者写可合理请求获得，不能说源码不存在。 |
| P19 Ada3D | 接受主要机制，保留实现范围 | [论文](https://arxiv.org/abs/2307.08209)、[作者代码](https://github.com/A-suozhang/ada3d)支持voxel/BEV自适应稀疏和3D检测计算/内存收益。将其概括为保持空间布局的稀疏表示是合理的；本轮未独立运行或完整核验其归一化/硬件实现。 |
| P20 Deformable DETR | 接受 | [论文](https://arxiv.org/abs/2010.04159)和X5支持围绕reference points的少量跨尺度采样；value projection仍作用于全部输入特征。因此不能把它当作RGB骨干自动少算的证据。 |

## 关键外部源码证据

- X1：[PointRend point_features.py](https://raw.githubusercontent.com/facebookresearch/detectron2/main/projects/PointRend/point_rend/point_features.py)，`get_uncertain_point_coords_with_randomness`中先`point_sample(coarse_logits, ...)`再`uncertainty_func(point_logits)`，源码还给出了两步不可交换的反例。支持报告P3的细节。
- X2：[SoftTeacher soft_teacher.py](https://raw.githubusercontent.com/microsoft/SoftTeacher/main/ssod/models/soft_teacher.py)，作者实现提供teacher冻结、同样本视图对应与坐标变换路径；本轮只审查该机制层面，未复现训练。
- X3：[SampleNet registration/src/samplenet.py](https://raw.githubusercontent.com/itailang/SampleNet/master/registration/src/samplenet.py)，推理分支先KNN，随后`.cpu().detach().numpy()`，`nn_matching`后再转CUDA Tensor；训练分支使用`self.project`。此处的CPU往返是可见执行事实，不只是从`import numpy`猜测。
- X4：[LAT modeling_bert.py](https://raw.githubusercontent.com/clovaai/length-adaptive-transformer/master/length_adaptive_transformer/modeling_bert.py)，`BertEncoder.forward`更新`remain_indices`，用`restored_hidden_states.scatter`写回当前位置的新hidden，最后返回恢复后的序列。删除位置保留的是其最近被更新的状态，不是重建出未进行计算的深层上下文。
- X5：[Deformable-DETR ms_deform_attn.py](https://raw.githubusercontent.com/fundamentalvision/Deformable-DETR/main/models/ops/modules/ms_deform_attn.py)，`value_proj(input_flatten)`与query生成offset/weight是不同成本，不能将后者的稀疏点数当作前者不执行。
- X6：[Deep Feature Flow作者仓库](https://github.com/msracver/Deep-Feature-Flow)支持视频检测的关键帧/传播结构；报告针对BMCR global-TIA缓存的限制是项目结构推断，不是DFF原文对本项目的结论。

总体上，未发现这些文献被整体张冠李戴。应继续保留报告已有的任务迁移、硬件执行和未取得源码的限定。无需为了“不同意一部分”而把已经能由原文支持的BYOT、data2vec、SampleNet或LAT描述误判为错误。
