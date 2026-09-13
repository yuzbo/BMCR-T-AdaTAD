**主代理补核与证据边界，2026-09-13**

主代理完整阅读两份粘贴原文、两份报告、两份实施交接及当前PLAN/IMPLEMENTATION/DECISION/CONTEXT；子代理分别定位BMCR源码、DS3源码、最新回执和文献。主代理回到关键源码点验，执行两个已完整阅读的CPU脚本，并独立核算已有metrics。以下内容补充、必要时覆盖子代理初次检索中的限制；原始附件未改。

**源码点验。** 当前工作树 `ds3_20260912` 的HEAD为0bceec1；审查开始时已跟踪h65/tools相对239d098无diff。交付检查时另有并行工作的命令目录生成器及文档暂存，审查涉及的模型/训练/评测源码未变化，本任务未修改该并行工作。`tools/full_train.py:33–42`的真实warm/scale初始化确被preflight绕过；`h65/full/model.py:45–56,102–109`确为S0标签与S1重采样；`h65/full/scout.py:63–97`确认伙伴tie及集合特征；`h65/ds3/routes.py:57–88`确认uniform嵌套分支，但注册策略未使用该不等预算uniform组合；`model.py:175–234`确认T24A路径及四格尚未解耦；`losses.py:41–79`确认非混合D1；`auxiliary.py:31–39`确认tubelet内均值；`runtime.py:64–87`确认EMA浮点buffer与online加载。

需要修正子代理摘要的一处泛化：只有teacher被强制eval；`DS3.train`会使aux进入训练模式，`ds3_train.py:22`明确调用train。因此H0的BN在训练时可更新。本文的两帧交换不变性限定于确定性eval；不声称aux一直eval。原报告把BN当待测因素是合理的。

**DyT原文与固定代码。** [论文§3.2式5/6](https://arxiv.org/html/2403.11808v2#S3.SS2)明确完整计算后mask残差，后续学生状态已经受mask影响。不能把它写成“训练下游只见dense输出”。[固定分割Block](https://raw.githubusercontent.com/NUS-HPC-AI-Lab/Dynamic-Tuning/d1744f0b9366f79ad9b78f586e479af34e81807a/dense_tasks/Segmentation/backbone/segmentation_vision_transformer_IN21K.py)的252–271行先全MLP再乘mask，无compact eval条件分支；这一点支持第二份的细化，不能外推作者所有入口都未加速。MLP逐token可在同输入下等价压紧；删K/V的attention另有图变化。

**TR-BERT。** 主代理打开了[ACL正式PDF](https://aclanthology.org/2021.naacl-main.463.pdf)：§3.1说明停止token保留退出层状态，§3.2说明任务微调→策略RL→联合优化以及启发式warmup/KD。原建议机制正确，子代理初次仅查到摘要的限制已解决。

**DToP。** [固定作者README](https://raw.githubusercontent.com/zbwxp/Dynamic-Token-Pruning/2baefffc879622b6514504dff44cf04d7e0e7504/README.md)22–33行列出base训练、从checkpoint剪枝训练和eval；原建议的适配边界正确。两份采用v1/v2，核心机制判断不因这个版本差异冲突。

**ETAD。** [作者论文§3.2/3.3](https://arxiv.org/html/2205.07134v2)明确所有snippet eval前向→检测器学习并取得特征梯度→只抽取部分梯度重放更新encoder。因此两份将其限定为训练效率证据是正确的；不能据此证明少RGB推理保精度。

**Temporal Corruptions。** [作者论文](https://arxiv.org/html/2403.20254v1)摘要及结论明确动作中部腐坏往往造成最大下降。支持保留动作内部证据这一研究动机；其腐坏协议不等于本项目Z24或T24A，不能搬用其因果比例或性能数字。

其余文献参见两个literature证据文件。它们主要做机制级定向核查，部分依据摘要；本次没有完整复现作者模型，没有做2025–2026排他性新颖性检索，也未把链接可访问说成全库审计。子代理文献文件中的初步未核准项及DyT泛化以本文件为准。

**运行验证。** `tools/verify_materials.py`核对2份原文/2份ZIP副本和10个解包文件直接字节相等，运行两份CPU机制脚本，结果与包内JSON在浮点容差内一致；12份现有DS3完整测试metrics的五阈值均值及211/792/seed和完成回执相符。没有重新读取大权重、视频或原始预测，没有项目GPU复测。图形仅据已完成回执生成，已目视检查。

**快照边界。** 完整测试表来自2026-09-13 00:03:39部署回执，包含S第5至30轮；00:10:19进度补核确认S已保存37轮/3700更新，B无D1 progress。训练第38轮的部分日志不作为完整轮。远端本次发布树没有bmcr_fidelity准备文件；本地准备目录和修正warm审计记录存在。DS3登记回执无修正BMCR阶段；未据缺目录声称整个账户不存在任何别处BMCR作业。
