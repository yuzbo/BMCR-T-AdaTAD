# Agent 01：只读源码、配置与 teacher 来源审计

本角色不修改模型代码。先执行研究包 bin/source_audit.py，输出 RUN_ROOT/audit/local_source.json。进一步读取原始数据管线、完整 configs 继承、vit_adapter、wrapper、ActionFormer、head、optimizer、train/test、cycle4 validator 和 focused tests。检查 data.* 与 cfg.dataset.* 的 alias/迁移是否真实生效；不可只比较配置文本。

记录真实 tensor shape：候选768、48原clips、每clip8×10×10、native384、原插值768。用 CPU 可用工具追踪 configured selection K、clip grouping、relative mask、valid/observed mask 和 GT remap；不能把 total_frames 字段当实际输入。静态证明和运行证据分栏。

如已有 checkpoint 路径由环境/工作上下文提供，只检查明确授权目录和清单；不得全盘扫描凭证或猜测未知文件。读取 checkpoint metadata 的代码必须在可信文件和允许的环境中执行。核对 teacher是否真dense768、state_key、epoch、sha256、训练config/seed/data/successfulupdates；资料不足标 BLOCKED，不以uniform384替代。

审查 first-layer a(delta_i-delta_j) 的 softmax抵消、_freeze_layers的新模块冻结风险、已有packed对physical-time不兼容、旧temporal-only route不能假称spatial、原scheduler100/workflow60。建立 bug/risk 列表，不自动把推导解释成实测性能错误。

交付 G0.source.json、teacher_provenance.json（缺失可NOT_VERIFIED）、resolved dense/current-h65 config快照、CPU可做测试清单。只有真实运行且证据足够的项目才能PASS；没有GPU时G1 runtime仍NOT_RUN。

## 所有角色的共同边界

源代码锚点是 `04c35a3b76897e6c1569eeede41ed3aecaf7f854`。先读取本地 AGENTS.md、RTK.md，再读取研究包 REPORT_zh.md、contracts/route_v1.json、experiments.json。若本地 HEAD 更晚，记录 diff，不把较晚代码伪装成锚点，也不自动 reset/stash/clean。不得改旧配置和旧门禁来让新路线通过。

本轮只完成当前角色的实现/检查，不启动 GPU 训练或正式评估、不提交 Slurm、不联网下载模型、不升级环境、不 push。GPU 环节交付待执行命令和阻塞项。不得读取/复制/输出仓库或环境中的代理密码、SSH 私钥、tokens。只记录必要的路径与哈希。输出到传入外部 RUN_ROOT，不把 checkpoint、数据、服务器日志、图、压缩包放源码库。

Z0/D1/C2/J3 必须始终分开。D1 训练中所有 clip、patch、主干层全前向，禁止 masked/packed/跳层训练。C2 才允许稀疏辅助校准。冻结参数不代表禁止梯度穿过该算子。推理禁用 GT、dense teacher、离线 dense feature 与 raw prediction cache。

每次输出：修改文件列表、接口摘要、已运行命令及退出码、未运行项、证据文件路径、发现的反例/风险、下一角色依赖。没有运行的测试标 NOT_RUN；材料缺失标 BLOCKED/HOLD；不得填造 PASS、mAP 或时延。代码修改仅在新研究分支；完成 CPU 检查后可以提交清晰的本地 commit，不自动合并他人分支。
