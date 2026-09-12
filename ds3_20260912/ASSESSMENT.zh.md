# 对DS3建议的判断与实际采用范围

核心方向合理：全时间廉价视觉表示、最终特征对齐、完整clip前缀早退、先稀疏逐token的heavy MLP，均比直接拼接不同层特征或把非连续帧重组为16帧clip更容易核验。FastBERT支持深层教师向浅层出口自蒸馏；Listen to Look支持廉价视觉/音频信息预测昂贵clip表示；DToP说明密集预测可以配合条件退出。三者均不保证本模型的TAD边界精度、任意组合等价或真实加速。

原始报告基于04c35，不能直接当成本项目当前源码审计。当前纯净OpenTAD的Adapter在构造时使用total_frames/tubelet_size：dense768为384，刚完成的H65 K384为192；S/B均从S配置继承12个TIA。报告对局部TIA8的描述可用于旧锚点，但“选中clip可以独立前向”在当前全局TIA下不成立。DS3-L会明确改变这一运行语义，冻结提供的官方权重，先独立量化global→local dense转换代价，不冒称官方等价或隐藏在稀疏收益里。

也不采纳报告中的额外授权开关、参数哈希、强制内部开发集和多种子启动要求作为阻断：用户已授权实现/部署和80轮；既有要求为全200训练、211测试、单种子和中间峰值。原ZIP和回复已原样保存在父项目reviews/20260912_ds3/originals，执行包仅作设计资料，未运行其中的启动器或bootstrap。

主方法D1确实训练全部clips、patches和层的完整前向，aux在完整教师特征上学习；不会把见过mask的C2校准重命名为D1。低分辨率preview仍观察全窗口，因此首版只主张减少昂贵骨干执行，不主张减少全部视频解码。1.3倍端到端加速和近无损精度是待测目标，不是已达到或必须通过才报告的硬门槛。

新训练上限按80轮预先设置，并保留60/80结果；不在旧60轮学习率降至零后盲目续跑。第一版只部署可核验的0/8/12和后半层MLP稀疏，后续C2/J3、attention merge和deformable以独立实验命名、单列成本。

依据：[FastBERT](https://aclanthology.org/2020.acl-main.537/)、[Listen to Look](https://openaccess.thecvf.com/content_CVPR_2020/html/Gao_Listen_to_Look_Action_Recognition_by_Previewing_Audio_CVPR_2020_paper.html)、[DToP](https://openaccess.thecvf.com/content/ICCV2023/html/Tang_Dynamic_Token_Pruning_in_Plain_Vision_Transformers_for_Semantic_Segmentation_ICCV_2023_paper.html)；本地upstream/opentad/models/backbones/vit_adapter.py:405-422、backbone_wrapper.py:75-124、官方S配置:66-110及B配置对S的继承。
