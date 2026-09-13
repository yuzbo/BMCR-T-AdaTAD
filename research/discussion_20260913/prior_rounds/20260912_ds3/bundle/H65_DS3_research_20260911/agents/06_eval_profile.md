# Agent 06：独立评估、实际计算计数与性能测量

所有权：拟新增 tools/bata/ds3_profile.py、ds3_metrics.py、tests/test_ds3_evidence_*.py；外部manifest/evidence。不要改方法结构来获得好看结果。

复用官方dataset/evaluator/NMS，记录每tIoU mAP、avgmAP、高tIoU、短动作AP/Recall、boundary error、background false positives和video coverage。短动作阈值在train数据预先定义，不用测试分位数调参。检查官方train/development/test视频ID不交叠，目录名validation不代表调参集。

profile至少三口径：backbone-only、GPU model-only（含preview/router/recon/head）、decode到NMS全链路。CUDA warmup、同步、p50/p95、batch1/部署batch、peakmemory、精度/SDPAbackend/环境全部记录。冷缓存/稳态分开，训练teacher-cache禁止用于部署计时。复用只允许同窗口实际算过的信息并计数。

输出每层真实heavy FFN token数、attention token数、active clips、executed blocks、preview解码帧数、完整重观察帧数、pack/scatter成本。展示预算理论计数与实际counter差异；不直接将T*S*D作为实际加速。

完成G1–G6测试和profiler CLI代码，本轮仅CPU测试与dry-run；GPU测试交给明确Slurm allocation里的用户命令。未测mAP/latency填null/NOT_RUN，不填0。最终result schema必须包含code/config/checkpoint/data/evaluator哈希、seed、训练总成本、运行口径。

将P1→P5的最小实验序列生成独立manifest，预算未固定或门禁缺失就BLOCK；不要自动把所有24实验全部提交。

## 所有角色的共同边界

源代码锚点是 `04c35a3b76897e6c1569eeede41ed3aecaf7f854`。先读取本地 AGENTS.md、RTK.md，再读取研究包 REPORT_zh.md、contracts/route_v1.json、experiments.json。若本地 HEAD 更晚，记录 diff，不把较晚代码伪装成锚点，也不自动 reset/stash/clean。不得改旧配置和旧门禁来让新路线通过。

本轮只完成当前角色的实现/检查，不启动 GPU 训练或正式评估、不提交 Slurm、不联网下载模型、不升级环境、不 push。GPU 环节交付待执行命令和阻塞项。不得读取/复制/输出仓库或环境中的代理密码、SSH 私钥、tokens。只记录必要的路径与哈希。输出到传入外部 RUN_ROOT，不把 checkpoint、数据、服务器日志、图、压缩包放源码库。

Z0/D1/C2/J3 必须始终分开。D1 训练中所有 clip、patch、主干层全前向，禁止 masked/packed/跳层训练。C2 才允许稀疏辅助校准。冻结参数不代表禁止梯度穿过该算子。推理禁用 GT、dense teacher、离线 dense feature 与 raw prediction cache。

每次输出：修改文件列表、接口摘要、已运行命令及退出码、未运行项、证据文件路径、发现的反例/风险、下一角色依赖。没有运行的测试标 NOT_RUN；材料缺失标 BLOCKED/HOLD；不得填造 PASS、mAP 或时延。代码修改仅在新研究分支；完成 CPU 检查后可以提交清晰的本地 commit，不自动合并他人分支。

profiler与gate入口必须实现研究包 `contracts/cli_v1.md`。纯synthetic timing与真实视频end-to-end timing分栏，缺数据时不得将前者作为后者。
