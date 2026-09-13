# Agent 03：D1 廉价表示、出口蒸馏和真实条件深度

所有权：拟新增 ds3/preview.py、exit_heads.py、depth_runtime.py、losses.py，相关test。依赖A02合同；不直接改核心注册/旧optimizer。

实现H0：利用低成本RGB preview/已有粗probe可复用的合法特征，预测每原clip的8个最终native384维特征，不重复一个向量。原probe若输出grid不同，显式记录转换与成本。H8第一版、H4可选；输入为完整clip prefix的spatial-pooled tubelet表示，输出final-feature接口。

D1辅助训练：B和D冻结eval；所有48clip、全部patch和12block完整前向，捕获中间特征。teacher no_grad目标，学生D(F_hat)不得no_grad；训练全F0和全F8图而非masked异质混合图。feature SmoothL1/cosine/temporal derivative，边界权重仅train合法GT。分类KD按head实际sigmoid定义，回归KD只在对齐正样本，拒绝全background L1。

实现depth runtime嵌套bucket：0/8/12先，4/8/12后。active tensor真正compact，退出clip不调用后层；出口后恢复原clip/tubelet顺序。gate只能读取preview或已经执行prefix，不能偷看深层teacher。先uniform分配，再预测dense error/task敏感性proxy，明示proxy不是真counterfactual。

测试：全12depth identity；H8 prefix输入一致；后层hook调用数；B/D哈希不变、aux有grad/LR/paramdelta；D1训练计数始终dense；batch1和异budget可用。递归freeze层风险交A05。提供所有新增参数的白名单和学习率建议，不自动启动GPU。

## 所有角色的共同边界

源代码锚点是 `04c35a3b76897e6c1569eeede41ed3aecaf7f854`。先读取本地 AGENTS.md、RTK.md，再读取研究包 REPORT_zh.md、contracts/route_v1.json、experiments.json。若本地 HEAD 更晚，记录 diff，不把较晚代码伪装成锚点，也不自动 reset/stash/clean。不得改旧配置和旧门禁来让新路线通过。

本轮只完成当前角色的实现/检查，不启动 GPU 训练或正式评估、不提交 Slurm、不联网下载模型、不升级环境、不 push。GPU 环节交付待执行命令和阻塞项。不得读取/复制/输出仓库或环境中的代理密码、SSH 私钥、tokens。只记录必要的路径与哈希。输出到传入外部 RUN_ROOT，不把 checkpoint、数据、服务器日志、图、压缩包放源码库。

Z0/D1/C2/J3 必须始终分开。D1 训练中所有 clip、patch、主干层全前向，禁止 masked/packed/跳层训练。C2 才允许稀疏辅助校准。冻结参数不代表禁止梯度穿过该算子。推理禁用 GT、dense teacher、离线 dense feature 与 raw prediction cache。

每次输出：修改文件列表、接口摘要、已运行命令及退出码、未运行项、证据文件路径、发现的反例/风险、下一角色依赖。没有运行的测试标 NOT_RUN；材料缺失标 BLOCKED/HOLD；不得填造 PASS、mAP 或时延。代码修改仅在新研究分支；完成 CPU 检查后可以提交清晰的本地 commit，不自动合并他人分支。
