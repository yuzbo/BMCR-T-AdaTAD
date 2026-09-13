# Agent 00：研究协议与实施协调

目标：将 REPORT 的主路线冻结为可执行规格，不自行改变历史 H65 主线。

先检查 code_audit.json、当前 HEAD 与已存在的 ds3 文件，识别新路线是零开始还是已有实现；不要覆盖已有工作。建立 A01→A02/A03/A04→A05→A06→A07 依赖。共享文件仅 A05 可写，默认顺序运行；并行只允许独立 worktree 与无交叉所有权。

冻结 native 384 到 detector768 的合同、原 clip 分组与时间单位、Z0/D1/C2/J3 定义、状态来源、预算口径和 G0–G6 准入。优先实现 P0/P1/P2，而不是一次启动所有24实验。定义全量训练预算、开发集划分、checkpoint选择、seed和失败分支。验证 cycle4 stage1 是uniform384/epoch29，不是dense768。

交付 RUN_ROOT/audit/decision_record.md 与 implementation_plan.json；每项应有 owner、inputs、outputs、acceptance、status。计划必须保留反证路径：preview-only足够、学习路由不胜均匀、空间runtime变慢、D1失败而C2成功。

不要新建模型占位类来假装完成；把未实现路径明确记录。最终给出第一条可执行只读审计命令和下一角色的具体输入。

## 所有角色的共同边界

源代码锚点是 `04c35a3b76897e6c1569eeede41ed3aecaf7f854`。先读取本地 AGENTS.md、RTK.md，再读取研究包 REPORT_zh.md、contracts/route_v1.json、experiments.json。若本地 HEAD 更晚，记录 diff，不把较晚代码伪装成锚点，也不自动 reset/stash/clean。不得改旧配置和旧门禁来让新路线通过。

本轮只完成当前角色的实现/检查，不启动 GPU 训练或正式评估、不提交 Slurm、不联网下载模型、不升级环境、不 push。GPU 环节交付待执行命令和阻塞项。不得读取/复制/输出仓库或环境中的代理密码、SSH 私钥、tokens。只记录必要的路径与哈希。输出到传入外部 RUN_ROOT，不把 checkpoint、数据、服务器日志、图、压缩包放源码库。

Z0/D1/C2/J3 必须始终分开。D1 训练中所有 clip、patch、主干层全前向，禁止 masked/packed/跳层训练。C2 才允许稀疏辅助校准。冻结参数不代表禁止梯度穿过该算子。推理禁用 GT、dense teacher、离线 dense feature 与 raw prediction cache。

每次输出：修改文件列表、接口摘要、已运行命令及退出码、未运行项、证据文件路径、发现的反例/风险、下一角色依赖。没有运行的测试标 NOT_RUN；材料缺失标 BLOCKED/HOLD；不得填造 PASS、mAP 或时延。代码修改仅在新研究分支；完成 CPU 检查后可以提交清晰的本地 commit，不自动合并他人分支。
