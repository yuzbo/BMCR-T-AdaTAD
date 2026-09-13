# Agent 04：空间稀疏重MLP与注意力压缩对照

所有权：拟新增 ds3/spatial_runtime.py、spatial_policy.py、spatial_surrogate.py 与test；不改旧PackedTubeletRuntimeRoute含义。

主线先只稀疏heavy FFN：注意力保持dense，在attention后对token选m，heavy FFN只执行gather token，cheap surrogate补其余残差，scatter回原[t,h,w]再执行TIA。D1训练时teacher FFN和surrogate都看全部dense tokens、不masked前向。首轮只后半层，比例0.75和0.5，ratio1严格调用dense identity path。

明确保留QKV/attention/TIA成本，不能按保留率缩放整block FLOPs。记录heavy_ffn_token_count、cheap_token_count、dense_attention_token_count、pack/scatter耗时。dense mask乘零不算成功。

探索支线可实现每tubelet内local merge/unmerge，保存token mass、坐标与映射；以原raster identity+unmerge residual供TIA。先禁止跨时间合并。KV集合变化会影响保留token输出，不得拿dense feature gather当精确模拟。该支线失败不阻塞MLP主线。

测试shape/order、full raster恢复、allselected identity、每tubelet空间覆盖、未选中heavy FFN不调用、梯度只流入预期surrogate/gate。提供GPU actual-forward与latency验证脚本但本角色不启动GPU。

与A03约定depth早退和空间route同时开启时的来源/误差累积。新增参数不能被旧_freeze_layers或backbone.lr0静默冻结。

## 所有角色的共同边界

源代码锚点是 `04c35a3b76897e6c1569eeede41ed3aecaf7f854`。先读取本地 AGENTS.md、RTK.md，再读取研究包 REPORT_zh.md、contracts/route_v1.json、experiments.json。若本地 HEAD 更晚，记录 diff，不把较晚代码伪装成锚点，也不自动 reset/stash/clean。不得改旧配置和旧门禁来让新路线通过。

本轮只完成当前角色的实现/检查，不启动 GPU 训练或正式评估、不提交 Slurm、不联网下载模型、不升级环境、不 push。GPU 环节交付待执行命令和阻塞项。不得读取/复制/输出仓库或环境中的代理密码、SSH 私钥、tokens。只记录必要的路径与哈希。输出到传入外部 RUN_ROOT，不把 checkpoint、数据、服务器日志、图、压缩包放源码库。

Z0/D1/C2/J3 必须始终分开。D1 训练中所有 clip、patch、主干层全前向，禁止 masked/packed/跳层训练。C2 才允许稀疏辅助校准。冻结参数不代表禁止梯度穿过该算子。推理禁用 GT、dense teacher、离线 dense feature 与 raw prediction cache。

每次输出：修改文件列表、接口摘要、已运行命令及退出码、未运行项、证据文件路径、发现的反例/风险、下一角色依赖。没有运行的测试标 NOT_RUN；材料缺失标 BLOCKED/HOLD；不得填造 PASS、mAP 或时延。代码修改仅在新研究分支；完成 CPU 检查后可以提交清晰的本地 commit，不自动合并他人分支。
