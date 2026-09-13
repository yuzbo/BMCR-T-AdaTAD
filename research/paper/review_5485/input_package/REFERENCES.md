# 一手文献与官方代码索引

阅读日期：2026-09-13。以下不是相关工作穷尽清单；不因列入文献便宣称已验证其在TAD上的迁移。没有下载或独立加载官方大权重。链接中的固定版本优先于浮动主页。

## 原论文

1. **L01 — Chen et al., Temporal Action Detection Model Compression by Progressive Block Drop, CVPR 2025.** [arXiv v1正文](https://arxiv.org/html/2503.16916v1)，[PDF](https://arxiv.org/pdf/2503.16916)。已读方法、实验、附录相关段并视觉核查PDF第5/6页。部分CVF直连403，由论文同源arXiv补读。直接TAD压缩对手；THUMOS主要比较@0.5，不与五阈值均值混用。
2. **L02 — Gadhikar et al., Attention Is All You Need For Mixture-of-Depths Routing, arXiv:2412.20875v1.** [正文](https://arxiv.org/html/2412.20875v1)。已读方法/训练/DETR附录。attention token重要性启发；DETR中非普遍保精度。未完成作者运行代码审计。
3. **L03 — Rethinking Patch Dependence for Masked Autoencoders (CrossMAE), arXiv:2401.14391v2.** [正文](https://arxiv.org/html/2401.14391v2)。已读cross-only decoder、inter-block fusion与实验相关段。预训练机制，不是本项目下游回写现成模型。
4. **L04 — Zhao et al., Dynamic Tuning Towards Parameter and Inference Efficiency for ViT Adaptation (DyT), NeurIPS 2024; arXiv:2403.11808v2.** [正文](https://arxiv.org/html/2403.11808v2)。已读dispatcher/self-distillation及不同子层dispatch比较。当前轮未对其完整作者代码再审，不沿用历史代码结论替代新核查。
5. **L05 — Tong et al., VideoMAE: Masked Autoencoders Are Data-Efficient Learners for Self-Supervised Video Pre-Training, NeurIPS 2022.** [论文](https://arxiv.org/abs/2203.12602)。decoder和mask合同以O05/O06固定源码核查；不能把90%空间tube mask转成时间删除性能保证。
6. **L06 — Masked Video Distillation: Rethinking Masked Feature Modeling for Self-supervised Video Representation Learning, CVPR 2023.** [原论文页](https://arxiv.org/abs/2212.04500)。本轮以原论文摘要/方法入口确认teacher latent任务；没有独立复测，也未审完整实现。
7. **L07 — Chen et al., Efficient Video Action Detection with Token Dropout and Context Refinement (EVAD), ICCV 2023.** [v3正文](https://arxiv.org/html/2304.08451v3)。已读关键帧保留、context refinement相关段；实证为时空动作检测，不能沿用AVA数值作为THUMOS结果。
8. **L08 — Tang et al., Dynamic Token Pruning in Plain Vision Transformers for Semantic Segmentation (DToP), ICCV 2023.** [原论文](https://arxiv.org/abs/2308.01045)。本轮核查原论文页/机制概述，未声称完整源码逐行复核。语义分割早退出/类别context参照。
9. **L09 — Liang et al., Expediting Large-Scale Vision Transformer for Dense Prediction without Fine-tuning, NeurIPS 2022.** [原论文](https://arxiv.org/abs/2210.01035)。本轮读原论文摘要及作者机构页面；用于clustering/reconstruction下游模式证据，不据此推导具体kernel等价或TAD新成绩。
10. **L10 — Ainslie et al., CoLT5: Faster Long-Range Transformers with Conditional Computation, EMNLP 2023.** [v2正文](https://arxiv.org/html/2303.09752v2)。已读light/heavy、Q/KV/FFN路由与预训练设置；训练条件不是纯D1。
11. **L11 — Ye et al., TR-BERT: Dynamic Token Reduction for Accelerating BERT Inference, NAACL 2021.** [ACL原文入口](https://aclanthology.org/2021.naacl-main.463/)。原论文页与任务定义核查，非本轮完整实现复现；span/QA保位置参照。
12. **L12 — Cai et al., Once-for-All: Train One Network and Specialize it for Efficient Deployment, ICLR 2020.** [原论文](https://arxiv.org/abs/1908.09791)。progressive shrinking的弹性训练思想；不把其分类精度结论移植为TAD。
13. **L13 — Zhang et al., Point-M2AE: Multi-scale Masked Autoencoders for Hierarchical Point Cloud Pre-training, NeurIPS 2022.** [原论文](https://arxiv.org/abs/2205.14401)。多尺度位置/跨层恢复原理；本轮原论文页核查，未审全部实现。
14. **L14 — Uni-AdaFocus: Spatial-temporal Dynamic Computation for Video Recognition, arXiv:2412.11228v1.** [正文](https://arxiv.org/html/2412.11228v1)。已读机制、同成本三维表、Fig.9/10说明；PDF43.7MB和部分图像无法读取，未声称视觉复核全部图。ActivityNet分数为视频识别mAP而非TAD tIoU mAP。
15. **L15 — V-JEPA 2.1, arXiv:2603.14482v3.** [正文](https://arxiv.org/html/2603.14482v3)。已读dense predictive loss/deep supervision相关段；2026新文献只作为机制线索，大规模预训练条件不能忽略。
16. **L16 — Yang et al., Adapting Short-Term Transformers for Action Detection in Untrimmed Videos (ViT-TAD), CVPR 2024.** [原论文](https://arxiv.org/abs/2312.01897)。本轮原文摘要/CVF信息核查，短期ViT跨snippet上下文参照，未重训。
17. **L17 — Liu et al., End-to-End Temporal Action Detection with 1B Parameters Across 1000 Frames (AdaTAD), CVPR 2024.** [作者页面](https://zhao-chen.com/publication/adatad/)。TIA和训练效率定位核查；项目实际数值来自固定源码记录而非网页四舍五入表。
18. **L18 — Liu et al., ETAD: Training Action Detection End to End on a Laptop, CVPR Workshops 2023.** [原论文](https://arxiv.org/abs/2205.07134)。训练梯度/提案采样，不作为稀疏RGB推理保精度证据。
19. **L19 — Zeng et al., Benchmarking the Robustness of Temporal Action Detection Models Against Temporal Corruptions, CVPR 2024.** [原论文](https://arxiv.org/abs/2403.20254)，[IEEE正文入口](https://doi.org/10.1109/CVPR52733.2024.01729)。核查摘要/开篇：定位易受损，动作中部腐蚀也可能最严重。不是断言本项目中部损伤必然最大。FrameDrop/TRC作为额外简单机制控制，而不是推理节省自身。
20. **L20 — Zhao et al., Re²TAL: Rewiring Pretrained Video Backbones for Reversible Temporal Action Localization, CVPR 2023.** [原论文](https://arxiv.org/abs/2211.14053)。训练activation-memory基线，非少forward计算的直接竞争。
21. **L21 — Bachmann et al., MultiMAE: Multi-modal Multi-task Masked Autoencoders, ECCV 2022.** [作者项目](https://multimae.epfl.ch/)，[原论文](https://arxiv.org/abs/2204.01678)。task query与多模态输出；任务decoder迁移需区分新头和预训练重建器。
23. **L23 — CVPR 2026 Author Guidelines.** [官方规范](https://cvpr.thecvf.com/Conferences/2026/AuthorGuidelines)。正文8页含图表、引用另列等按该届；实际投稿年份变更后重新核查。图中文字建议并非官方任意数值门槛。

## 本轮深入核查的作者实现

- **O01 — PBD README**：[`readme_en.md`](https://github.com/nanxiaolu/Progressive_Block_Drop-TAD/blob/74bce4764bbce49d6996df0bd6b6b1ab97fe3510/readme_en.md)，注意文件名大小写；不能因不存在README.md便认定作者无代码。
- **O02 — PBD课程**：[`tools/train_distillation_auto.py`](https://github.com/nanxiaolu/Progressive_Block_Drop-TAD/blob/74bce4764bbce49d6996df0bd6b6b1ab97fe3510/tools/train_distillation_auto.py)，已读1–430行；候选评估、循环压缩、恢复与teacher选择。
- **O03 — PBD学生任务头**：[`opentad/models/detectors/actionformer_student.py`](https://github.com/nanxiaolu/Progressive_Block_Drop-TAD/blob/74bce4764bbce49d6996df0bd6b6b1ab97fe3510/opentad/models/detectors/actionformer_student.py)，已读1–190行；GT、分类KD、回归KD与优化分组。
- **O04 — PBD骨干**：[`opentad/models/backbones/vit_longlora_student.py`](https://github.com/nanxiaolu/Progressive_Block_Drop-TAD/blob/74bce4764bbce49d6996df0bd6b6b1ab97fe3510/opentad/models/backbones/vit_longlora_student.py)，已读390行后；保留层编号、特征损失及冻结逻辑。
- **O05 — 官方VideoMAE**：[`modeling_pretrain.py`](https://github.com/MCG-NJU/VideoMAE/blob/5cefe18cecab25e3cce8de88fad0b42e6ce858a7/modeling_pretrain.py)，80–335行；mask、projection、decoder输出和S/B模型定义。
- **O06 — 官方预训练配置**：[`PRETRAIN.md`](https://github.com/MCG-NJU/VideoMAE/blob/5cefe18cecab25e3cce8de88fad0b42e6ce858a7/PRETRAIN.md)；K400/SSv2脚本、mask与decoder depth。

**权重边界**：O05/O06和C18足以定义应该加载哪些参数，不足以证明某一大权重文件在本运行环境已打开。本报告不声称有checkpoint tensor统计或迁移性能。
