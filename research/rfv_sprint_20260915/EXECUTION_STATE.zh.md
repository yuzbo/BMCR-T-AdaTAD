# RFV-T 当前执行状态

2026-09-15。本文件已按三份最新审阅更新。完整当前命令见 EXPERIMENT_PLAN.zh.md；这次主修改是反向一致性 mini，不是扩充 Graph/FVD 模块矩阵。

当前 scientific branch 为 codex/wtr-rfv-20260915。b644d87 的 Value-R1 已完成18个head并经独立复核，结论 LEARNABILITY_FAIL 保持。full inner10 的 regret 改善CI跨0；within R1 regret0.001598842、R0 0.001519199、STOP0.001413537，held rho0.00540。fit rho约0.944，说明它拟合了已见数据。

原 within split 的fit200全部位于tubelet局部位置5，held200为位置1/2。几何范围检查没有发现先前疑似的常量列突变；不能把失败简单归为一般同位置插值失败，也不能唯一归因于packing。三个新评论审阅的是更早6d1f924；采用建议不重写实验发生顺序。

主线：完成真实 S/S′/S 合同后，以相同407、原fit normalization、原8/8、2000updates×3seed运行P1与P2。P1-bi直接复用P1 raw权重零新增训练。shared bias在P2输出差分中相消；反向membership仍为(1,0)。反向signed distance原尺度可达−65至−75σ，将记录优化状况。覆盖held4和partner96拟合均暂缓。

并行：RISE仅导出已完成B0的共同state分数、STOP、top2 margin与选择变化，β1.1冻结，逐视频复现旧结果。D/S先定位原405的真实checkpoint并回收现有grad_norm，再做独立固定checkpoint路由/梯度诊断；不改课程、不做optimizer更新。旧四条4090课程继续，最后核实进度仍以16:40回执为准，不能把新40评测当作已完成。

AutoDL R1 PID283745已完成。Atlas owner报告interaction_v1_s全部792窗于19:10:33完成，GPU PID284842退出，两张卡均已释放，但新部署仍先读取实际占用。Atlas CPU287713统计收尾；新interaction与旧S Atlas的source分别记录。A100和4090已通过b644d87的14项CPU合同，新反向源待独立核验。

T-local-CF的cal20/46窗增量仍为+0.528413pp，视频CI[+0.082440,+0.714687]；这是有限GT辅助机会，不是廉价Value已学会。Graph旧mini不稳定，历史RISE有drift但Future未优于Post/EMA。T/Graph/FVD长训均未解锁，最终路线科学可行性尚未证明。epoch80 primary、test不选模型、outer20封存继续执行。

已发布快照保持6d1f924及旧release不变。新代码、合同、结果将以新不可变SHA逐项发布；未提交实现不能被称为已经部署或科学PASS。所有完整新实验完成后独立交叉审核。
