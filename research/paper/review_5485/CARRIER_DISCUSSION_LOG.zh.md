# 跨线程讨论记录

2026-09-14。本文件记录本次实际收到的答复，不把未完成实验或新建议记为已有实现。

## 方法线程的关键答复

来源：`01a084a0-88c4-7730-9c2b-e351aaef33c9`，通过线程消息收到。

> 当前PVR carrier保留空间grid，C_l为[B,384,5,5,128]（768输入帧、tubelet=2），不是只有[B,384,128]的向量序列。Proxy是12层各自独立P_l，不共享：BCRFrontend构造ModuleList，每层CarrierLayer的P=LN128→Linear128→TIA_dim→GELU；B的TIA_dim=192，S=96。P后bilinear 5×5→10×10，Q前adaptive_avg_pool 10×10→5×5。

> 当前PVR实际是精确native_id的硬index_copy selected replacement；软门控/一致性是迁移到不规则单帧pack时的新候选，不是现实现。L6/9/12少数coupling、保留当前Cross起点是新消融；当前PVR为全部12层。零初始化宜只放新增残差映射最后Linear，门控初值保持非零，避免双零锁住学习。当前Q末Linear weight/bias=0；官方TIA Up=0、gamma=1，而不是Up与gamma都为0。空间grid及完整P/Q、低维TIA、pool/resize/gather成本必须计入。

> 没有可靠的“同support、同预算、同配方Full-V2逐层carrier vs仅末端Cross”已完成对照。已有U-prog>U-static是另一个问题，不能替代这个证据。

上面是对方消息中的直接节选。随后已读取其指定的 `bricks/bcr_carrier.py:65–98` 与 `backbones/progressive_carrier_vit.py:152–192`，核实空间网格、替换、更新与读出逻辑。

## 发回方法线程的进一步讨论

从已训练Full-V2出发，直接将selected-support TIA换成全轴TIA时，旧Up已可能非零；仅将Q置零不足以保留旧函数。建议保留旧TIA，用新的附加残差引入full-axis correction，仅新输出最后Linear归零，gate非零；Cross新增carrier输入投影同理。原已训练Up不重置。已向对方发送该边界请其核对。

## 实验线程的关键答复

来源：`01a09654-63fa-7831-9eef-af21a5a73ac1`，通过线程消息收到。

对方确认当前科学模型结构相对da26af3未变，运行模型版本db9c749；Full-V2配置仍使用R03 Cross，官方MAE decoder为消融；D/S只在原block ID[4,6,8,10]启用。Full-V2-S/B epoch10完整测试mAP为64.2351/68.3708%，两者所有测试窗口均为D100预算，因此不能解释为已验证的稀疏深度收益。

这些状态已通过 `monitor_20260914_1235/UPDATE.zh.md` 点验。

## 已发送的成果

已将 `CARRIER_TRANSFER_RECOMMENDATIONS.zh.md`、新主图路径、三个候选范围、物理时间对齐要求、零初始化边界、全模型成本要求与D100证据限制发送给实验线程。明确本轮不替换既有模型、不恢复整clip选择、不因这份建议取消现有作业。

方法线程的完整短答已保存为 `CARRIER_PEER_REPLY.zh.md`，并已完整阅读。对方进一步确认：Cross新增零投影若读取完整非零C_L，可以先学习读出再训练上游；若读取(C_L-C0)，同时Q与输出投影都零，则没有其他监督路径时，该差分支路无法启动学习。已补入建议并同步给实验线程。

回复还给出U-static→U-prog的B56.1546→59.4518%、S53.1331→55.6313%，以及对E1证据边界和非完整成本小计的说明。这些均为原clip支持协议内证据，不记作Full-V2单帧迁移的结果。
