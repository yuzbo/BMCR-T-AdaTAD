# Agent 02：完整 clip 前端与 native 时间网格

所有权：拟新增 opentad/models/ds3/contracts.py、clip_frontend.py、native_grid.py，以及 tests/test_ds3_time_*.py。共享wrapper/detector只交patch建议，不直接写。

实现独立的 original-clip subset 路线：根据原始候选窗口分组16，不把任意稀疏帧按rank重新拼成clip。返回压紧后的[B*K,3,16,H,W]、original clip ids、frame indices、native centers、有效mask。selected clip仍全空间全深度，第一版不送旧global-rank canonical correction。

提取backbone native spatial-pooled[B,K,8,C]，在真实tubelet centers上恢复[B,48,8,C]，再复用原768插值。不能只生成clip级单向量。实现Z0均匀36/24/16、固定seed随机24、单clip/短窗/端点；所有case strict budget、端点策略、padding独立声明。

提供插值baseline，已观测位置严格保持；valid_mask是完整有效timeline，observed_mask另存，禁止把missing当padding。逆映射必须保留原GT/seconds，没有selected-axis GT remap。

CPU tests应覆盖全选identity、实际时间而非rank插值、批间异长、严格升序/重复拒绝、部分clip支持、无观测拒绝。为GPU提供完整clip feature equivalence测试与forward hook计数脚本，若未运行标NOT_RUN。测试毒化未选中高分辨率帧且preview固定时结果不变。

交付接口文档和A05集成patch建议。纯gather dense teacher输出仅可用作等价验证/训练模拟，不可作为部署实现。

## 所有角色的共同边界

源代码锚点是 `04c35a3b76897e6c1569eeede41ed3aecaf7f854`。先读取本地 AGENTS.md、RTK.md，再读取研究包 REPORT_zh.md、contracts/route_v1.json、experiments.json。若本地 HEAD 更晚，记录 diff，不把较晚代码伪装成锚点，也不自动 reset/stash/clean。不得改旧配置和旧门禁来让新路线通过。

本轮只完成当前角色的实现/检查，不启动 GPU 训练或正式评估、不提交 Slurm、不联网下载模型、不升级环境、不 push。GPU 环节交付待执行命令和阻塞项。不得读取/复制/输出仓库或环境中的代理密码、SSH 私钥、tokens。只记录必要的路径与哈希。输出到传入外部 RUN_ROOT，不把 checkpoint、数据、服务器日志、图、压缩包放源码库。

Z0/D1/C2/J3 必须始终分开。D1 训练中所有 clip、patch、主干层全前向，禁止 masked/packed/跳层训练。C2 才允许稀疏辅助校准。冻结参数不代表禁止梯度穿过该算子。推理禁用 GT、dense teacher、离线 dense feature 与 raw prediction cache。

每次输出：修改文件列表、接口摘要、已运行命令及退出码、未运行项、证据文件路径、发现的反例/风险、下一角色依赖。没有运行的测试标 NOT_RUN；材料缺失标 BLOCKED/HOLD；不得填造 PASS、mAP 或时延。代码修改仅在新研究分支；完成 CPU 检查后可以提交清晰的本地 commit，不自动合并他人分支。
