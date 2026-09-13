# Agent 05：集成、配置、训练协议和启动manifest

唯一所有权：共享模型注册、wrapper/detector接入、tools入口最小扩展、新ds3配置和validator。先核对A02/03/04模块接口，拒绝合并占位实现或不一致时间单位。

创建opt-in新wrapper/子类，旧路径默认完全不变；不要给当前ActionFormer传其不接受的参数。先提供dense768和Z0配置，再D1 aux/time/depth/spatial。允许旧state_dict核心权重严格匹配，仅新增aux缺失白名单。dense detector长768、native384、无旧selected-axis GT remap、旧token_compressor禁用/显式不适用。

审查freeze/optimizer：newaux不能被forward内_freeze_layers重置；B,D eval与requires_grad策略正确；taskloss梯度穿过冻结D；参数分组、非零LR与实际update计数形成测试。D1 dense training graph的执行计数写入日志，不以布尔配置代替证据。

新增configs/adatad/thumos/ds3/中experiments.json需要的文件，未支持实验保持明确UNIMPLEMENTED，禁止生成只改名字的空config。为每个真实支持配置导出resolved snapshot，检查data/dataset aliases、scheduler和budget。新研究协议不能借关掉旧guard偷偷跑历史sealed arm。

创建 tools/bata/ds3_validate.py：子命令source、config、checkpoint、gates，CLI写入README；每项独立NOT_RUN/PASS/BLOCK。创建run manifest，先review_status UNREVIEWED，实测门禁后由人工批准。manifest记录teacher路径/sha/statekey/epoch、configsha、codecommit、datasetids/hash、budget与checkpoint规则。

运行CPU测试、py_compile和旧focused tests（环境允许时）。交付新GPU smoke命令（至少2成功非零LR更新）、全量train/test/profile命令；GPU不执行。不要改历史cycle4的epoch29/59合同。

## 所有角色的共同边界

源代码锚点是 `04c35a3b76897e6c1569eeede41ed3aecaf7f854`。先读取本地 AGENTS.md、RTK.md，再读取研究包 REPORT_zh.md、contracts/route_v1.json、experiments.json。若本地 HEAD 更晚，记录 diff，不把较晚代码伪装成锚点，也不自动 reset/stash/clean。不得改旧配置和旧门禁来让新路线通过。

本轮只完成当前角色的实现/检查，不启动 GPU 训练或正式评估、不提交 Slurm、不联网下载模型、不升级环境、不 push。GPU 环节交付待执行命令和阻塞项。不得读取/复制/输出仓库或环境中的代理密码、SSH 私钥、tokens。只记录必要的路径与哈希。输出到传入外部 RUN_ROOT，不把 checkpoint、数据、服务器日志、图、压缩包放源码库。

Z0/D1/C2/J3 必须始终分开。D1 训练中所有 clip、patch、主干层全前向，禁止 masked/packed/跳层训练。C2 才允许稀疏辅助校准。冻结参数不代表禁止梯度穿过该算子。推理禁用 GT、dense teacher、离线 dense feature 与 raw prediction cache。

每次输出：修改文件列表、接口摘要、已运行命令及退出码、未运行项、证据文件路径、发现的反例/风险、下一角色依赖。没有运行的测试标 NOT_RUN；材料缺失标 BLOCKED/HOLD；不得填造 PASS、mAP 或时延。代码修改仅在新研究分支；完成 CPU 检查后可以提交清晰的本地 commit，不自动合并他人分支。

实现入口时必须遵循研究包 `contracts/cli_v1.md` 的具体参数及验证/正式运行分离。不要自行更名导致 COMMANDS_zh.md 不可执行；必要变更由协调者同时更新全部任务、manifest和命令。
