**Full-V1 / Full-V2 / Simple强控制实施状态**

当前独立工作树为support_review_20260914，分支codex/support-review-20260914。保留现有52个完整课程，复审18项加最新Full-V2 S/B、late/统一深度控制和完整shared-full移除，共23项新增、75个不同完整配置；全部seed42一次。重复的C2/C3、C6/C7用同一课程别名，不重复训练。

P0核心顺序：Full-V1 S/B、Full-V2 S/B、现有Uniform Full S/B、PBD-style S/B；Static S/B随后补位。其余按P1机制/decoder、可用泛化和P2补充依次进入空槽，不使用mAP启动门槛或早期成绩淘汰。现有Uniform Full仍是原来的动态D/S架构中的uniform frame控制，不能叫纯T-only；纯T控制另用独立axes课程。

Full-V2：保持原H65/BMCR时间策略；前4层dense，路由层原ID4/6/8/10（0 based），最后一层dense；selected-Q/full-KV，独立depth-attention/depth-FFN轻残差，空间heavy/light，冻结初始encoder的同选帧支持目标，原轴Cross、可训练head、原有shared-full与外部目标。

PBD/Static按最新Final-Simple要求采用空间容量1，关闭动态空间；终点保留9/12层。PBD在同课程0/13/26 epoch处基于固定8个训练样本的候选loss逐次删层并恢复，重建optimizer但保持80轮LR前缀。使用本框架Adapter/norm/head，不冒充官方LoRA配方faithful复现。Static和PBD终点同层数/全容量算子成本匹配，不能声称与动态N00有严格相同FLOPs。

实现已包含同支持TIA前后/attention状态、独立depth-bypass、实际曝光、支持参考成本、plan-conditioned单次frame refinement、fresh Cross、MAE输入统计适配、严格无外部查询、公共K400初始化稀疏/dense配对、完整训练和续跑、官方dense同口径复测、配对bootstrap及五张决定性图的生成入口。

CPU结果：16项旧/新合同测试全部通过（47.779秒）；另以checkpoint重算和直接反传比较中间状态loss梯度，1项通过（26.947秒）。真实权重下11种代表性模型构造通过，公共S识别预训练163 keys且无TAD Adapter/scout/head，官方MAE S/B资源可用。冻结参考采用从同一资产重新构造，避免框架包装器不可deepcopy的问题。

GPU技术问题仍须完成：旧paper课程在两步更新后的推理一致性检查失败；专用诊断1288834已提交，尚需取回inference_contract.json以区分非有限输出、重复推理波动或路由差异，不放宽条件掩盖问题。TadTR缺失单进程distributed初始化已修正。新P0完整课程尚不能宣称已经成功启动；统一队列注册脚本已实现，等待连接恢复及相应技术核验。

网络状态：ANet计算节点不能访问HTTPS，已改为桌面中转；val.09的4194304000字节已成功传入独立修复目录，但后续SSH提交受阻。SSH入口现返回nginx网页而非SSH banner，HTTPS也出现协议握手异常；未改全局网络设置。已提出网络变化询问并继续本地工作。数据覆盖仍以真实READY标记为准，不把传输完成当作解压和全数据准备完成。

绘图：已生成并视觉检查Full-V2网络结构的PNG/SVG/PDF；新实验性能图在无结果时明确列为等待，不填点。现有69条历史完整测试已读取归集。需在连接恢复后同步最新结果，不能将历史Cross或BMCR成绩作为Full-V2成果。

主要入口：tools/paper_review_plan.py、paper_support_check.py、paper_review_assets.py、paper_review_register.py、paper_review_diagnostics.py、paper_review_analyze.py、paper_review_plot.py、paper_public_dense.py。已实现与待部署状态分开，不以本文件替代实际Slurm回执。
