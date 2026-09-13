# 一手证据与阅读边界

项目主快照：`yuzbo/BMCR-T-AdaTAD@239d098cd899936c35989fae6243c70259a85adb`。本包以该快照而非默认分支作判断。检索截至2026-09-12；2026年的文献已联网核查。下列论文是机制来源，不代表项目已有对应实现或实验收益。全文阅读包含正文方法与限制；PDF图表阅读使用了页面截图。作者实现只有明确列出的文件/片段作了源码核查；没有声称全面复现所有文献。

## 项目源码与记录

[P1] 固定上下文、课程和已完成成绩。https://github.com/yuzbo/BMCR-T-AdaTAD/blob/239d098cd899936c35989fae6243c70259a85adb/ds3_20260912/research_context_20260912/CONTEXT.zh.md

[P2] FormalH65、route、encode、raw_route。https://github.com/yuzbo/BMCR-T-AdaTAD/blob/239d098cd899936c35989fae6243c70259a85adb/h65/full/model.py

[P3] FormalScout与16-slot局部交换上下文。https://github.com/yuzbo/BMCR-T-AdaTAD/blob/239d098cd899936c35989fae6243c70259a85adb/h65/full/scout.py

[P4] 有符号反事实、分类指派、含漏检惩罚的定位成本。https://github.com/yuzbo/BMCR-T-AdaTAD/blob/239d098cd899936c35989fae6243c70259a85adb/h65/full/utility.py

[P5] 真实硬选择、局部transport代理梯度、时间插值。https://github.com/yuzbo/BMCR-T-AdaTAD/blob/239d098cd899936c35989fae6243c70259a85adb/h65/transport.py

[P6] 原VideoMAE Adapter、Block、patch embedding。https://github.com/yuzbo/BMCR-T-AdaTAD/blob/239d098cd899936c35989fae6243c70259a85adb/upstream/opentad/models/backbones/vit_adapter.py

[P7] DS3教师与compact算子，仅参考底层实现，不恢复整clip路由。https://github.com/yuzbo/BMCR-T-AdaTAD/blob/239d098cd899936c35989fae6243c70259a85adb/h65/ds3/model.py

[P8] 旧模型原始完整比较。https://github.com/yuzbo/BMCR-T-AdaTAD/blob/239d098cd899936c35989fae6243c70259a85adb/phase2_20260910/comparison.json

[P9] 修正H65完整比较。https://github.com/yuzbo/BMCR-T-AdaTAD/blob/239d098cd899936c35989fae6243c70259a85adb/fidelity_20260911/FINAL_COMPARISON.json

[P10] 模型与全流程测量入口。https://github.com/yuzbo/BMCR-T-AdaTAD/blob/239d098cd899936c35989fae6243c70259a85adb/tools/full_eval.py

## 论文：原问题、可借鉴机制、证据边界

[1] Chen et al. **Temporal Action Detection Model Compression by Progressive Block Drop**, CVPR 2025, arXiv:2503.16916v1。全文及方法/结果页截图核查；渐进去block、跨深度参数高效对齐。直接TAD压缩基线；论文MAC、@0.5与本项目全模型MAC/五阈值均值不可直接混合。https://arxiv.org/abs/2503.16916

[2] Fu et al. **Rethinking Patch Dependence for Masked Autoencoders (CrossMAE)**, arXiv:2401.14391v2, 2025-04-10。全文核查；独立query只cross-attend可见latent、多层特征融合。支持解码结构，不直接证明下游缺帧TAD恢复。https://arxiv.org/html/2401.14391v2

[3] Liang et al. **Expediting Large-Scale Vision Transformer for Dense Prediction without Fine-tuning**, NeurIPS 2022, arXiv:2210.01035。全文与方法页截图核查；先有完整浅特征，再聚类、精炼、重构写回；不可当作完全没观察RGB的证据。https://arxiv.org/abs/2210.01035

[4] Zhao et al. **Dynamic Tuning Towards Parameter and Inference Efficiency for ViT Adaptation**, arXiv:2403.11808v2, 2024-10-16。全文核查；full计算masked更新、轻adapter、完整分支self-distillation。训练并非纯冻结D1。https://arxiv.org/html/2403.11808v2

[5] Bachmann et al. **MultiMAE: Multi-modal Multi-task Masked Autoencoders**, ECCV 2022。作者项目/论文核查；任务query、多模态稀疏latent；迁移时换任务head，不把预训练decoder保留等同于推理回写。https://multimae.epfl.ch/ ; https://arxiv.org/abs/2204.01678

[6] Wang et al. **Masked Video Distillation**, CVPR 2023, arXiv:2212.04500v2。全文核查；image/video latent targets。支持目标选择，AVA空间动作检测迁移不是THUMOS少RGB推理保精度。https://arxiv.org/html/2212.04500v2

[7] **V-JEPA 2.1: Unlocking Dense Features in Video Self-Supervised Learning**, arXiv:2603.14482v3，2026。全文核查；dense predictive loss、多深度监督。正文消融存在密集任务/分类任务取舍，不能声称所有监督总能同时改善；大规模预训练不可换算成200视频效果。https://arxiv.org/html/2603.14482v3

[8] Baevski et al. **data2vec: A General Framework for Self-supervised Learning in Speech, Vision and Language**, ICML 2022。原论文页面/摘要核查；完整上下文EMA latent目标跨语音/视觉/语言，不是下游低成本部署保证。https://proceedings.mlr.press/v162/baevski22a.html

[9] Ainslie et al. **CoLT5: Faster Long-Range Transformers with Conditional Computation**, EMNLP 2023, arXiv:2303.09752v2。全文核查；全token轻路径与重Q/KV/FFN分离，条件计算参与预训练与微调。https://arxiv.org/html/2303.09752v2

[10] Ye et al. **TR-BERT: Dynamic Token Reduction for Accelerating BERT Inference**, NAACL 2021。ACL原文入口核查；token级动态深度、QA等下游；结构类比不能直接给TAD边界保证。https://aclanthology.org/2021.naacl-main.463/

[11] Tang et al. **Dynamic Token Pruning in Plain Vision Transformers for Semantic Segmentation (DToP)**, ICCV 2023。CVF原文页面核查；easy位置早退、保留类别上下文，适用于密集输出。https://openaccess.thecvf.com/content/ICCV2023/html/Tang_Dynamic_Token_Pruning_in_Plain_Vision_Transformers_for_Semantic_Segmentation_ICCV_2023_paper.html

[12] Chen et al. **SparseViT: Revisiting Activation Sparsity for Efficient High-Resolution Vision Transformer**, CVPR 2023。CVF原文页面核查；结构化窗口稀疏与稀疏适配，不能保证plain VideoMAE任意token gather同样快。https://openaccess.thecvf.com/content/CVPR2023/html/Chen_SparseViT_Revisiting_Activation_Sparsity_for_Efficient_High-Resolution_Vision_Transformer_CVPR_2023_paper.html

[13] Dutson et al. **Eventful Transformers**, ICCV 2023。CVF原文页面核查；状态缓存/变化驱动重计算。跨窗原位置/上下文不一致时不可免费复用。https://openaccess.thecvf.com/content/ICCV2023/html/Dutson_Eventful_Transformers_Leveraging_Temporal_Redundancy_in_Vision_Transformers_ICCV_2023_paper.html

[14] Zhang et al. **Point-M2AE: Multi-scale Masked Autoencoders for Hierarchical Point Cloud Pre-training**, NeurIPS 2022, arXiv:2205.14401v2。全文核查；几何一致多尺度mask、跨级skip与重建，重建器预训练和下游使用要分别核查。https://arxiv.org/html/2205.14401v2

[15] **ETAD: Training Action Detection End to End on a Laptop**, arXiv:2205.07134。原文入口核查；梯度/提案采样是训练效率，不是删RGB推理保精度。https://arxiv.org/abs/2205.07134

[16] Maier and Rannacher. **A duality-based optimization approach for model adaptivity in heterogeneous multiscale problems**, arXiv:1611.09437。原文阅读；goal-oriented残差误差估计。仅迁移“按目标误差分配精度”的思想，不迁移PDE误差界到mAP。https://arxiv.org/abs/1611.09437

[17] Alwassel et al. **Diagnosing Error in Temporal Action Detectors (DETAD)**, ECCV 2018。作者原文页面核查；定位、分类、重复、背景、漏检及数据因素分析。https://www.humamalwassel.com/publication/detad/ ; https://arxiv.org/abs/1807.10706

[18] Angelopoulos et al. **Conformal Risk Control**, ICLR 2024。原文入口核查；依赖校准数据、交换性与损失条件；本项目测试集选模下不具备现成无偏风险保证。https://arxiv.org/abs/2208.02814

[19] Agrawal et al. **Scaling Action Detection: AdaTAD++ with Transformer-Enhanced Temporal-Spatial Adaptation**, ICCV 2025。CVF原文页面核查；时空adaptation与训练资源策略，不与推理采样自动同义。https://openaccess.thecvf.com/content/ICCV2025/html/Agrawal_Scaling_Action_Detection_AdaTAD_with_Transformer-Enhanced_Temporal-Spatial_Adaptation_ICCV_2025_paper.html

[20] Yang et al. **Masked Generative Distillation**, ECCV 2022, arXiv:2205.01529v2。全文核查；mask学生特征、生成teacher目标是蒸馏分支，不能仅据生成损失便宣称下游推理采用生成后的特征。https://arxiv.org/html/2205.01529v2

[21] Tong et al. **VideoMAE**, NeurIPS 2022。官方预训练decoder源码固定到`5cefe18cecab25e3cce8de88fad0b42e6ce858a7`核查；RGB patch1536维，常规self-attention，不是现成TAD latent decoder。https://arxiv.org/abs/2203.12602 ; https://github.com/MCG-NJU/VideoMAE/blob/5cefe18cecab25e3cce8de88fad0b42e6ce858a7/modeling_pretrain.py

[22] CVPR 2026 Author Guidelines，检索2026-09-12；正文8页含图表，references规则以官方为准。本包将其作为版式参考，不声称未来届规则不变。https://cvpr.thecvf.com/Conferences/2026/AuthorGuidelines

[23] CVPR author-kit `12909ae437f6dbc7435069cfdb4ca44c18e6a02f`，`cvpr.sty`230–270行：总宽6.875in，栏间0.3125in，因此单栏3.28125in。https://github.com/cvpr-org/author-kit/blob/12909ae437f6dbc7435069cfdb4ca44c18e6a02f/cvpr.sty

## 未完成的访问与不应扩大之处

没有下载任何项目大权重/原视频，也没有验证官方decoder checkpoint实际可下载或包含何种keys；需agents本地审计。PBD作者库的`README.md`提交查询返回空，故本包依据论文而非声称已完整审过作者实现。固定项目archive本地下载尝试DNS失败，因此本包CPU测试运行的是自写独立reference机制，不是完整OpenTAD源码。部分CVF PDF路径返回403，使用arXiv原论文或CVF官方HTML替代。没有将这些限制解释成方法不可行。
