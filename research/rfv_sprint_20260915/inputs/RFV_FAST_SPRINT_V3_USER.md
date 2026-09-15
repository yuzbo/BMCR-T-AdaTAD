建议把原来的推进命令**正式改成“RFV-T 主线优先、D/S 后台保留、Atlas 收尾、Raw 暂缓扩张”**。你附件里的版本已经非常接近最终可执行形态，我会再收紧成下面这份，可直接交给 agents/实验 owner 执行。

# 更新后的实验推进命令：RFV Fast Sprint v3

> **立即生效。当前主科学目标从“继续扩展 T/D/S 模块”调整为：先证明 Temporal Value 可学，再验证 Graph 是否提供额外关系信息、RISE 是否能预测未来 Value。D/S 现有课程继续完成，但不再新增变体；Atlas 只做收尾；Raw 暂停长训练。所有新 Graph/FVD 长训必须由预注册 Gate 解锁。**

## 一、总优先级

从现在起按以下顺序执行：

$$
\boxed{
T\text{-local-CF}
\rightarrow
T\text{-Value}
\rightarrow
{\text{Graph-T},\text{RISE-T}}\_{\parallel}
\rightarrow
T\text{-V-G},T\text{-V-F}
\rightarrow
T\text{-V-G-F}
}
$$

后台保留：

$$
D/S\text{ existing courses}
$$

$$
Atlas\text{ finishing}
$$

$$
Raw\text{ coverage diagnostics only}
$$

不得再让 D/S/Raw 阻塞 RFV-T 主线。

---

# 二、立即冻结版本

新建独立分支：
```text
codex/wtr-rfv-20260915
```

完成下列修正后生成：
```text
WTR_RFV_SCIENCE_SHA=<immutable commit>
```

不得复用当前 D/S science SHA 冒充新 RFV 版本。

旧结果保持各自 source revision：

- D/S：原 science SHA；
- Raw：原 Raw SHA；
- Atlas：原 Atlas SHA；
- RFV：新 SHA。

---

# 三、Phase 0：4 个硬修复先完成

正式 T 长训前必须完成以下四项。

### P0.1 Temporal short-window mapping

修复 duplicate physical frame → invalid padding slot 风险。

强制：
```python
assert candidate_mask.gather(
    1, selection.indices
)[selection.valid].all()
```

所有有效 selection 必须映射到 valid candidate slot。

### P0.2 Temporal Value safe initialization

禁止随机 Value head 从 step 0 改变 Uniform support。

默认采用：
```text
offline CF prefit
→ load Value
→ online routing
```

无预训练权重时，最终输出层 zero-init，使初始策略严格退化为 Uniform/STOP。

### P0.3 Test checkpoint protocol

删除：
```text
best_full_test_mAP
```

作为 primary selection。

统一：
```text
epoch80 = primary endpoint
epoch10/20/40/60 = trajectory diagnostics
```

Graph/FVD/β/threshold/model selection 不得依据 official test milestone。

### P0.4 Full-video integration smoke

每个新 RFV recipe 在正式训练前必须跑：
```text
1 complete video
→ all windows
→ postprocess
→ JSON
→ full metric computation
```

不能只靠 2-update preflight。

---

# 四、Phase 1：先做 T-local-CF Gate

不要直接训练 T-V。

严格使用当前实际部署动作空间：
```text
Uniform K384
+ max 4 swap rounds
+ max 16 proposals per round
+ positive-gain STOP
```

GT-assisted Local-CF 必须使用**完全相同的 proposal/action generator**。

目标：

$$
\Delta\_{\rm local}
===================

## mAP\_{\rm LocalCF}

mAP\_{\rm Uniform}
$$

第一阶段只在 training/calibration 跑，不看 publication test 调 action space。

### PASS

如果：

$$
\Delta\_{\rm local}>0
$$

且 video-level paired CI 稳定为正，记录：
```text
T_ACTION_SPACE = HEADROOM_PASS
```

### FAIL

如果接近0：
```text
T_ACTION_SPACE = FAIL
```

先扩大 proposal/action space，不训练 Graph/FVD。

---

# 五、Phase 2：建立统一 T-CF Bank

只生成**一套共享 bank**，Plain / Graph / RISE 全部复用。

保存：
```text
video_id
window_id
checkpoint
support_ids
physical_times
candidate_ids
swap(remove, insert)
delta_cls
delta_loc
cheap_context
actionness
transition
coverage
D/S plan
source_revision
state_hash
candidate_hash
```

GT 只用于生成 label，不进入 predictor input。

第一版：
```text
32 fit videos
10 calibration videos
10 holdout videos
```

先做 500–1500 actions。

信号成立后扩到完整：
```text
160 fit
20 calibration
20 router-holdout
```

---

# 六、Phase 3：Value-R1 最小修订

当前最大问题是：
```text
fit Spearman ≈ 0.94
unseen-candidate Spearman ≈ 0.04
```

所以暂时**不改 descriptor，不改 action space，不加 Graph**。

只改监督形式。

### Plain-R0

当前 absolute regression。

### Plain-R1

同一 architecture：
```text
407 → 128 → 64 → 2
```

改为 state-wise ranking objective。

对每个 state 内候选：

$$
\tilde g\_i=g\_i-\bar g
$$

$$
p\_i^\*=\operatorname{softmax}(\tilde g\_i/\tau\_g)
$$

$$
p\_i=\operatorname{softmax}((q\_i-\bar q)/\tau\_q)
$$

训练：

$$
L=
JS(p^\*,p)
\+
0.1L\_{\rm Huber}.
$$

Primary metric：

$$
\boxed{\text{finite-budget action regret}}
$$

Secondary：
```text
NDCG
Top-K overlap
Spearman
```

---

# 七、Value-R1 离线对照

必须同时跑：
```text
Plain-R0
Plain-R1
Plain-L-R1
```

`Plain-L` 参数量匹配后续 Graph。

每种：
```text
3 head-training seeds
```

同：

- action bank；
- split；
- optimizer steps；
- normalization。

### Value PASS

要求：

$$
Regret\_{R1}\<Regret\_{R0}
$$

且 holdout/unseen candidate 的 ranking 明显改善。

若失败：
```text
VALUE = LEARNABILITY_FAIL
```

暂停 task-level Graph/FVD，优先检查 descriptor/action representation。

---

# 八、Phase 4：Graph-T 和 RISE-A 同时开始

不要等 T-V 80轮结束。

## Graph G1-T

统一比较：
```text
Plain-M R1
Plain-L R1
Static Graph R1
Dynamic Graph R1
```

Graph 只提供：

$$
\text{relational context}
$$

第一版禁止同时开启：
```text
GraphKV
GraphRecovery
Referral-heavy architecture
```

Static graph 先用物理邻居。

Dynamic graph 再学习 content + geometry edge。

### Graph PASS

必须：

$$
Regret\_G\<Regret\_{Plain-L}
$$

且 paired video bootstrap 方向稳定。

若：
```text
Static ≈ Dynamic
```

最终优先 Static。

---

# 九、Phase 5：RISE-A / RISE-B

## RISE-A：先证明真实 Value Drift

固定：
```text
video
window
support
candidate set
action
```

跨：
```text
epoch10
epoch20
epoch40
epoch60
```

真实重执行：

$$
V^{actual}*{10},
V^{actual}*{20},
V^{actual}*{40},
V^{actual}*{60}.
$$

输出：
```text
Spearman drift
sign flip rate
Top-K overlap
future action regret
```

若 drift 不超过 replay/noise：
```text
RISE = FAIL
```

停止 FVD。

---

## RISE-B：同 state 函数外推

必须固定：

$$
s,\mathcal C
$$

计算：

$$
z\_A=Q\_{\Psi\_A}(s,\mathcal C)
$$

$$
z\_P=Q\_{\Psi\_P}(s,\mathcal C)
$$

中心化后：

$$
z\_F=
\bar z\_P+
(\beta-1)(\bar z\_P-\bar z\_A).
$$

β 只在 calibration 选择：

$$
{1,1.05,1.1,1.2}.
$$

比较：
```text
Current
True EMA
Post (β=1)
Future
```

### RISE PASS

要求：

$$
Regret\_F
<
\min(
Regret\_{Current},
Regret\_{EMA},
Regret\_{Post}
)
$$

且 NDCG / Top-K 不恶化。

否则：
```text
FVD = NOT_UNLOCKED
```

---

# 十、Phase 6：T-U / T-V 直接80轮

只要：
```text
T action-space PASS
+
Value-R1 PASS
```

立即启动完全 matched：
```text
T-U
T-V-R1
```

共同：
```text
V2-S epoch40 EMA init
K384
D100/S100
Cross
same optimizer
same schedule
same augmentation
same 200 detector videos
80 epochs
```

唯一差异：
```text
Uniform
vs
Learned Value-R1
```

不再另开10轮 pilot。

---

# 十一、T-V 训练时自动保存 RISE 资产

在：
```text
10 / 20 / 40 / 60 / 80
```

保存：
```text
router weights
EMA router
normalization
descriptor version
candidate generator version
science SHA
fixed action manifest
```

这样 RISE-B 可以边训练边做。

---

# 十二、Phase 7：Graph task model

如果：
```text
Value-R1 PASS
Graph G1 PASS
```

立即解锁：
```text
T-V-GCTX
```

80轮。

只加：
```text
GraphContext
```

不加：
```text
GraphRecovery
GraphKV
FVD
DB
Raw
```

对照：

$$
T\text{-V}
\quad vs\quad
T\text{-V-GCTX}.
$$

---

# 十三、Phase 8：FVD task model

如果：
```text
RISE-A PASS
RISE-B PASS
```

才解锁：
```text
T-V-FVD
```

第一版保持简单：
```text
fixed β
snapshot every 10 epochs
JS distillation
```

不同时加入：
```text
adaptive β
EMA anchor variants
Graph-conditioned teacher
extra ranking loss
```

必须同时有：
```text
β=1 / Post distillation control
```

否则不能证明 extrapolation 本身有效。

---

# 十四、Phase 9：Graph + FVD 2×2

只有 Graph 和 FVD 单独都 PASS，才启动：
```text
T-V-GCTX-FVD
```

最终必须得到四格：

$$
M\_0=T\text{-V}
$$

$$
M\_G=T\text{-V-G}
$$

$$
M\_F=T\text{-V-F}
$$

$$
M\_{GF}=T\text{-V-G-F}.
$$

计算：

$$
I\_{GF}=M\_{GF}-M\_G-M\_F+M\_0.
$$

如果 GF 不优于最佳单项：

> 删除冗余模块。

---

# 十五、D/S 现在的更新指令

当前已运行：
```text
D-V
D-U
S-V
S-U
```

继续。

但：

$$
\boxed{\text{禁止新增 D/S 新变体}}
$$

只完成：

- epoch40完整eval；
- epoch80 endpoint；
- calibration/holdout re-query；
- actual route/cost statistics。

若最终：

$$
V\<U
$$

且 holdout regret也无改善：
```text
D/S = NEGATIVE RESULT
```

不做 DS/TDS。

不要让“三轴”标题反过来逼模型。

---

# 十六、Atlas 更新指令

Atlas 进入：
```text
FINISHING MODE
```

继续：

- S D/S aggregate；
- recovery；
- benign/no-op noise；
- interaction；
- figures。

停止：

- 新 heuristic；
- 新 selector；
- 新 Graph；
- 新 teacher；
- 新 budget sweep。

B 保持暂停。

Atlas 的职责只剩：

$$
\boxed{\text{characterize}}
$$

不是继续 search method。

---

# 十七、Raw 更新指令

Raw 暂停长训。

只做：
```text
Official insert
vs
Off-grid insert
```

的 stratified development diagnostic。

当前禁止：
```text
full 6400-bank expansion
Raw matched 80ep
Graph+Raw
preview sweep
```

等 RFV-T 成败确定后再恢复。

---

# 十八、服务器分配

### 4090

现有：
```text
D-V
D-U
S-V
S-U
```

继续到既定 checkpoint，不抢占。

### AutoDL 两张空闲 GPU

Value-R1 PASS 后：
```text
GPU0 → T-U 80ep
GPU1 → T-V-R1 80ep
```

### A100

优先：
```text
T-CF bank
Graph offline
RISE replay
later T-V-G / T-V-F
```

不要让 A100 等 D/S。

---

# 十九、未来24小时必须产生的结果

不是更多代码文件，而是以下产物：
```text
T_LOCAL_CF.json
VALUE_R1_GATE.json
GRAPH_G1_T.json
RISE_A_T.json
RISE_B_T.json
T_U_train_receipt.json
T_V_train_receipt.json
```

如果这些没有出来，即使实现了20个模块，也不算 Fast-Track 成功。

---

# 二十、Gate 状态统一

`FINAL_MODEL_LEDGER.csv` 改成：
```text
IMPLEMENTED
TECH_PASS
HEADROOM_PASS
LEARNABILITY_PASS
TASK_PASS
FINAL
```

例如当前应该类似：
```text
Temporal action-space: HEADROOM_PASS
Plain Value: WAITING
Graph Context: WAITING
RISE drift: PARTIAL/PASS
RISE forecast: FAIL/WAITING-v2
D Value: TASK_FAIL_so_far
S Value: TASK_FAIL_so_far
Raw: TECH_PASS
```

---

## 最后一段可直接复制给实验 owner

> **从现在起，项目主线切换为 RFV-T。冻结当前 D/S、Raw、Atlas 的 science identity，不修改历史结果。新建独立 RFV science branch。首先修复 Temporal short-window mapping、Value安全初始化和 official-test checkpoint-selection 协议，并运行完整视频端到端 smoke。随后以当前真实部署动作空间完成 T-local-CF headroom Gate。使用统一 T-CF bank，将 Plain-R0、Plain-R1、parameter-matched Plain-L、Static Graph、Dynamic Graph 与 RISE replay 在相同160/20/20 split、相同 action manifest 下并行验证。Plain Value learnability通过后立即用当前空闲两张AutoDL卡启动完全matched的80轮 T-U/T-V；Graph regret稳定优于Plain-L则解锁T-V-G，RISE Future稳定优于Current/TrueEMA/Post则解锁T-V-F，二者分别通过后才解锁T-V-G-F。D/S继续既有课程但不新增变体；Atlas只完成S统计与recovery收尾；Raw仅做off-grid coverage诊断。所有超参数和模型选择只使用training/calibration，holdout只用于最终Value泛化判断，official test不得用于checkpoint或方法选择。**

这就是我建议你现在正式替换旧实验推进命令的版本。

