# D/S 固定参数诊断登记

2026-09-15。复用 science405 的 D-V、S-V epoch40/4000updates 原始 learned 参数；明确不使用EMA、不换checkpoint比较路由，不修改或更新旧四条课程。资产只抽取原checkpoint的learned/metadata，不复制optimizer来续训。当前D/S执行内核、encoder、decoder、readout、训练目标与405的git diff为空；fasttrack仅torch import位置变化，T修订不在此路径执行。

路由诊断：原 calibration20 按ID排序前4视频，每视频确定性的中间官方窗。对同raw40模型完整执行Value，再仅将当前D或S策略改为Uniform，复用同preview和T support，核对每层每pack重算quota。报告两个任务loss分量及实际路由差、包括Value开销的成本。此项是4视频开发诊断，不称为full mAP或单独task gate。

梯度诊断：原fit160，训练dataset顺序前4个符合fit的视频，原seed42、epoch39的确定性训练增强。2组×2个microbatch，匹配旧accumulate2；第一个microbatch模拟有CF的更新，sequence0/1，覆盖D或S层4/6。每组从同raw40参数/buffers重置，bf16 autocast，无GradScaler，无optimizer.step。分别对task、0.1self_feature、0.1Value辅助除以2后求梯度并按参数组累加；只做一次训练路径的前向，按既定loss图计算分量。比较同一task/feature梯度加入或移除Value辅助时全局maxnorm1系数。两种系数并非两个重新采样的课程。

记录Value梯度是否完全限于router；即使detector的直接Value梯度为0，全局裁剪仍可能缩放它的任务梯度。实际记录全局/分组norm、clip系数、同detector梯度的系数比及真实CF标签。若两组未显示额外缩放，则不把旧日志普遍裁剪归因于Value。即使显示缩放，也只证明这个固定更新的耦合，不宣称解释训练最终mAP。不得据本开发诊断悄悄修改运行中课程的clip或loss权重。

输出 `DS_DIAGNOSTIC.json`、每模型JSON、source/config/Slurm回执和日志；旧测试milestone仅作描述性比较，epoch80仍primary。完整实验后交叉审核。
