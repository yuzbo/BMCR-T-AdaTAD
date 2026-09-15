可以。基于目前代码审阅、Atlas 的 T 证据以及你现在希望**更快确认 Graph/RISE 的论文价值**这一目标，我建议把后续正式切成一个 **RFV Fast Sprint**：

$$
\boxed{
\textbf{Grounded T Value}
\rightarrow
\textbf{Relational Value}
\rightarrow
\textbf{Future Value}
\rightarrow
\textbf{RFV}
}
$$

同时 D/S 和 Atlas 继续后台产生证据，但**不再阻塞 RFV-T 主线**。

最重要的执行原则是：

$$
\boxed{
\text{先快速证明 existence / learnability，再直接启动最终80轮课程，不再反复训练短pilot。}
}
$$

---

# 一、先冻结新的实验边界

不要修改当前已经运行的 `4055294` D/S 课程。

它们继续保留自己的 science identity。当前 `6e2fc7f` 也确实只是 evaluator serialization revision，没有改变模型科学执行。

Graph/RISE/T 修正版应该建立**新的科学版本**，例如：

```text
codex/wtr-rfv-20260915
WTR_RFV_SCIENCE_SHA=<new immutable commit>
```

原因是下一步至少会修改：

* Temporal mapping；
* Temporal Value initialization/prefit；
* Graph-conditioned Value；
* FVD；
* checkpoint-selection protocol。

这些都属于科学执行变化，不能继续冒充 `4055294`。

建议最终同时保留：

| 科学版本        | 用途                   |
| ----------- | -------------------- |
| `4055294`   | 当前 D/S FastTrack     |
| `27d557e`   | Raw-v1               |
| 新 `RFV_SHA` | T + Graph + RISE/FVD |

旧结果绝不覆盖。

---

# 二、Phase 0：先做四个必须修复，目标几个小时内完成

这四项在任何 T/Graph/RISE 正式长训之前处理。

## 0A. 修正 T short-window candidate mapping

当前 `to_standard()` 按 physical frame ID 建 dict，而 short window padding 会重复最后一帧，存在把真实 observation 映射到 invalid padding slot 的风险。当前 short-window sampling 合同本身明确是“重复最后一帧但 padding slot invalid”。

修改目标：

```python
valid_selection
    => mapped candidate slot is valid
```

必须新增 hard assertion：

```python
assert candidate_mask.gather(
    1, selection.indices
)[selection.valid].all()
```

并让 physical frame → slot lookup **只在 official valid prefix 中搜索**。

这不是消融，是 correctness fix。

---

## 0B. Temporal Value 不允许随机策略从 step 0 改 Uniform

当前 `TemporalValueHead` 最后一层使用默认初始化，而 router 会立即根据正预测 gain 做 swap。

推荐最终方案：

$$
\boxed{
\text{offline actual-CF prefit}
\rightarrow
\text{online T-Value adaptation}
}
$$

同时最后一层 zero-init，作为没有 prefit 权重时的安全默认。

因此：

* 无预训练 Value → 初始严格 STOP → Uniform；
* 有 prefit → 加载真实 learned routing。

不要让随机 router 决定前几个 epoch 的 support distribution。

---

## 0C. 立即修 checkpoint selection protocol

当前配置仍有：

```text
best_full_test_mAP
```

evaluator metadata 也仍写：

```text
full-test milestone EMA peak
```

。

建议正式改成：

$$
\boxed{
epoch80=\text{primary endpoint}
}
$$

10/20/40/60：

```text
diagnostic trajectory only
```

Gate、β、Graph type、FVD、proposal 等全部只看 training/development。

官方 test 不再参与：

* 早停；
* checkpoint selection；
* Graph static/dynamic selection；
* β selection。

旧 milestone 结果保留，但不能作为 final checkpoint 选择依据。

---

## 0D. 新增完整 end-to-end smoke

两步 optimizer preflight 已经不足够——之前 JSON bug 正是在完整 eval 时才暴露。

以后每个新 RFV recipe 必须跑：

$$
1\text{ complete video}
\rightarrow
all windows
\rightarrow
postprocess
\rightarrow
JSON
\rightarrow
metrics
$$

再启动80轮。

---

# 三、Phase 1：T-local-CF 是第一优先科学 Gate

这是现在最重要的一步。

Atlas 已经证明 broad temporal allocation 有明显 headroom，但 Atlas 的 group action space 与当前 T-V 的：

$$
Uniform\ K384+\le4\ swaps
$$

不是一个 action space。Atlas S 的8-group结果是 +3.507pp，但不能直接赋给当前 T-V。

所以必须测：

$$
\boxed{
M_{\rm LocalCF}-M_{\rm Uniform}
}
$$

其中 LocalCF **严格复用部署时的 T action generator**：

* Uniform K384 initial support；
* 4 rounds；
* 每轮同一个 `geometric_swaps(...,16,round)`；
* actual full TAD loss 对全部16候选真实重执行；
* 选择 best positive gain；
* 更新 support；
* 下一轮重新生成候选；
* 非正收益 STOP。

### 快速两阶段执行

先做 **G0a**：

* calibration 20 videos；
* 每视频 1–2 个固定代表 window；
* 完整4-round/16-candidate真实搜索。

只判断：

$$
\text{loss headroom / regret}
$$

是否真实存在。

如果明显有信号，再做 **G0b**：

* calibration 20 videos 全部 windows；
* 完整 detection AP；
* Uniform vs LocalCF。

### 判定

如果：

$$
LocalCF\approx Uniform
$$

就暂停当前 T-V：

> 先扩大 proposal/action space。

如果：

$$
LocalCF>Uniform
$$

稳定存在：

$$
\boxed{\text{T action-space PASS}}
$$

立即进入 T-Value。

---

# 四、Phase 2：同时建立一份共享 T-CF bank

不要为 Plain、Graph、RISE 分别生成不同的数据。

建立统一：

```text
T_VALUE_BANK_V1
```

每个 state 保存：

* video ID；
* window ID；
* detector checkpoint；
* Uniform/current support；
* physical frame IDs；
* candidate set；
* action pairs；
* actual `[ΔL_cls, ΔL_loc]`；
* cheap preview hidden；
* action/transition logits；
* preview physical timeline；
* plan D/S capacity；
* source revision；
* augmentation/evaluation identity。

**不保存为 Value 输入的：**

* GT；
* unacquired heavy feature；
* future checkpoint output。

GT 只用于生成 label。

---

## 快速 mini-bank

先：

$$
32\text{ fit videos}
+
10\text{ calibration}
+
10\text{ holdout}
$$

每视频约 1–2 个 states，每 state 8–16 actions。

目标约：

$$
500\sim1500
$$

个实际 actions。

这是 learnability probe，不是 publication final bank。

如果信号成立，再扩成完整预注册：

$$
160/20/20.
$$

Detector 仍然使用全部200个训练视频；160/20/20只约束 Value labels，这一点保持不变。当前训练代码已经执行这个合同。

---

# 五、Phase 3：Plain T-Value、Graph 和 RISE-A 三条并行

这是提速最关键的地方。

不要再串行：

```text
T-V → Graph → RISE
```

而应该：

```text
                   ┌── Plain T Value
T CF bank ─────────┼── Graph probe
                   └── RISE-A manifest/replay
```

---

# 六、Plain T-Value：快速建立真正 baseline

直接使用现有：

$$
407\rightarrow128\rightarrow64\rightarrow2
$$

Temporal descriptor/head。当前 descriptor 已包含 cheap context、set summary、时间/gap、provenance 和 D/S capacity。

### Offline prefit

只用160 fit labels。

Normalization：

* input mean/std：fit only；
* target RMS：fit only；
* utility 仍是原始：

  $$
  \Delta L_{cls}+\Delta L_{loc}
  $$

  不使用标准化后的 weighted utility。

Raw 回归代码现在已经遵守这个原则。

### 评价

Calibration 用于：

* STOP threshold；
* training step；
* limited hyperparameters。

Holdout 最终报告：

$$
Spearman,
\quad
NDCG@K,
\quad
TopK,
\quad
Regret.
$$

主指标：

$$
\boxed{\text{action regret}}
$$

不是 Spearman。

---

# 七、Graph G1-T：现在立即实现

我建议第一版 Graph **不要碰 GraphKV，也不要碰 Recovery**。

目标只回答：

$$
\boxed{
V(a|s,G_s)>V(a|s)?
}
$$

---

## G1 的最小实现

新增例如：

```text
h65/paper/graph_value.py
```

构造 **192-node cheap preview graph**。

### Node state

每个 preview node：

$$
[
h_{preview}^{96},
t,
actionness,
transition,
support\_distance,
support\_occupancy
].
$$

全部来自：

* cheap preview；
* physical geometry；
* current support。

不得使用 heavy feature / GT。

---

## Static Graph

固定 physical temporal neighbors，例如：

$$
\pm1,\pm2,\pm4,\pm8,...
$$

degree 16。

---

## Dynamic Graph

可以复用现有 `EdgeRouter/GraphMessage` 的核心思想：

* bounded degree16；
* physical seed；
* content + geometry router；
* indexed gather；
* 无 \(N^2\) affinity。

当前 `edge_ops.py` 已经是真 sparse edge implementation，而且明确 full K/V projection 不等于 sparse projection；G1 只需要 message/context 部分，不要把 GraphKV 一起带进来。

---

## Pair Value

Graph 得到：

$$
g_1,\dots,g_{192}.
$$

按 physical time 插值得：

$$
g_{remove},
\qquad
g_{insert}.
$$

再聚合当前 selected support：

$$
g_S.
$$

最终：

$$
Q_T=
f[
x_{407},
g_r,
g_i,
g_S
].
$$

第一版 graph width 用64足够。

---

# 八、Graph 的四个 offline control

必须同时训练：

$$
Plain\text{-}M
$$

当前407D MLP。

$$
Plain\text{-}L
$$

参数量匹配 Graph 的大 MLP。

$$
StaticGraph
$$

固定边。

$$
DynamicGraph.
$$

全部：

* 相同 bank；
* 相同 optimization steps；
* 相同 seeds；
* 相同 train/cal/holdout。

offline 很便宜，我建议至少 **3个 head-training seeds**，不是只跑一个。

---

## Graph PASS

主要条件：

$$
Regret_{Graph}<Regret_{Plain-L}
$$

而且 paired video bootstrap 的 Graph–Plain-L regret difference 方向稳定。

同时：

$$
NDCG_G\ge NDCG_{Plain-L}.
$$

如果 Dynamic≈Static：

> 最终用 Static，别为了“Graph”复杂化。

只有：

$$
Dynamic>Static
$$

才保留 dynamic/referral 方向。

---

# 九、RISE-A：现在就可以跑，不等 T-V@80

新增例如：

```text
tools/wtr_rise_replay.py
```

输入：

```text
action_manifest
checkpoint_A
checkpoint_P
checkpoint_F
```

manifest 固定：

$$
(video,window,support,candidate\ set,action)
$$

不能不同 checkpoint 自己重新选 action。

---

## 第一轮快速 probe

如果新 T-V checkpoint 还不够，可以先利用已有 V2-S 20/40/60 checkpoint 做：

```text
RISE-A0 historical detector drift probe
```

但明确标成 preliminary。

正式论文 RISE-A 后面仍应使用匹配的新 RFV/T trajectory。

---

## RISE-A 输出

至少：

$$
\rho(V_{10},V_{20}),
\rho(V_{20},V_{40}),
\rho(V_{40},V_{60})
$$

以及：

$$
SignFlipRate,
\quad
TopKOverlap,
\quad
FutureRegret.
$$

并分：

* boundary；
* action interior；
* background；
* short/long actions。

如果 drift 接近 replay/no-op noise：

$$
\boxed{\text{RISE STOP}}
$$

不做 FVD。

---

# 十、T-U / T-V 两条80轮正式课程尽快直接启动

一旦：

1. T short-window correctness PASS；
2. T-local-CF headroom PASS；
3. Plain Value mini holdout 至少显示 learnability；

就直接启动：

$$
\boxed{T\text{-U}}
$$

和：

$$
\boxed{T\text{-V}}.
$$

不是10轮pilot再重启。

直接80轮：

$$
10/20/40/60/80
$$

保存。

---

## T-U 必须 matched adaptation

不能用 frozen C0 与80轮 T-V 比。

两者必须相同：

* V2-S epoch40 EMA initialization；
* Adapter；
* Cross；
* head；
* optimizer；
* schedule；
* augmentations；
* self-feature；
* training videos；
* epochs。

唯一核心区别：

$$
Uniform
$$

vs

$$
Learned Temporal Value.
$$

这样才能说：

$$
\boxed{\text{allocation policy 带来了收益}}
$$

而不是训练时间不同。

---

# 十一、RISE-B：T-V 产生 snapshots 后立即开始

当前 checkpoint 已经保存：

* learned state；
* EMA；
* optimizer；
* scheduler；
* RNG。

所以每个：

$$
10/20/40/60
$$

snapshot 可直接提取完整 Temporal Value function。

固定同一个 detached state：

$$
x,\mathcal C
$$

计算：

$$
z_A=Q_{\Psi_A}(x),
$$

$$
z_P=Q_{\Psi_P}(x),
$$

然后：

$$
\boxed{
z_F=z_P+\beta(z_P-z_A)
}
$$

β：

$$
\{1,1.05,1.1,1.2\}.
$$

只在 calibration 20 视频上选择。

---

## RISE Controls

必须有：

$$
Current
$$

$$
TrueEMA
$$

$$
Post,\ \beta=1
$$

$$
Future,\ \beta>1.
$$

当前 EMA 实现是实际累计 EMA，不是 checkpoint average，因此可以直接作为可信 control。

### FVD Gate

$$
Regret_F
<
\min(
Regret_{Current},
Regret_{EMA},
Regret_{Post}
)
$$

并且 NDCG / TopK 不恶化。

否则：

$$
\boxed{\text{不训练 FVD}}
$$

---

# 十二、Task-level Graph：G1 PASS 后立即启动

只要：

* Plain T-Value learnability PASS；
* Graph offline PASS；

就直接创建正式：

```text
T-V-GCTX
```

80轮。

不要等待：

* D；
* S；
* TDS；
* Raw；
* DB。

---

## 这个课程只加 Graph context

不加：

* GraphRecovery；
* GraphKV；
* FVD；
* DB。

所以：

$$
T\text{-V}
$$

vs

$$
T\text{-V-GCTX}
$$

可以直接归因。

Graph MAC/FLOPs 必须进入完整 cost ledger。

---

# 十三、FVD task course：RISE PASS 后立即启动

新增例如：

```text
h65/paper/future_value.py
```

不要 extrapolate detector。

只 extrapolate：

$$
\boxed{
\text{Value function}
}
$$

第一版建议固定：

```text
snapshot_interval = 10 epochs
FVD_start = epoch20
```

例如：

* e10 = anchor；
* e20 = post；
* e20–30 使用由 e10/e20 构造的 future teacher；
* e30 更新 snapshots；
* 依此类推。

β 从 calibration 冻结。

---

## FVD loss

第一版保持简单：

$$
L_{FVD}
=
Huber(
Q_\theta(x),
\operatorname{sg}(z_F)
).
$$

如果后续需要，可增加 pairwise ranking loss，但第一轮不要同时加两个新目标。

必须保存：

* anchor snapshot SHA；
* post snapshot SHA；
* β；
* teacher-query count；
* FVD loss；
* matched additional MACs。

---

# 十四、FVD 必须有 β=1 控制

否则如果：

$$
T\text{-V-FVD}>T\text{-V},
$$

无法知道是：

> future extrapolation 有效，

还是：

> 多了一个 self-distillation loss 有效。

所以至少有一个：

$$
\boxed{
FVD\text{-}\beta1
}
$$

control。

它不一定第一时间跑完整80轮。

可以先 offline/value-level 或20/40 milestone。

但如果 FVD 成为论文核心贡献，最终必须有这个控制。

---

# 十五、Graph+FVD：只有两个单项都 PASS 才启动

此时训练：

$$
\boxed{
T\text{-V-GCTX-FVD}
}
$$

Graph conditioner 必须包含在 Value snapshot function 中：

$$
\Psi=
\{
GraphConditioner,
ValueHead,
normalization
\}.
$$

RISE 比较时必须：

* 同 graph；
* 同 candidate set；
* 同 physical state；

不能让 anchor/post Graph 本身看到不同节点集合。

---

# 十六、最终做两个 2×2

## Value-level

$$
Q,\ Q_G,\ Q_F,\ Q_{GF}
$$

比较：

* regret；
* NDCG；
* TopK；
* sign；
* calibration。

---

## Task-level

$$
M_0=T\text{-V}
$$

$$
M_G=T\text{-V-GCTX}
$$

$$
M_F=T\text{-V-FVD}
$$

$$
M_{GF}=T\text{-V-GCTX-FVD}.
$$

最终 interaction：

$$
I_{GF}
=
M_{GF}-M_G-M_F+M_0.
$$

这样才真正支持：

$$
\boxed{
V_{\rm future}(a|s,G_s).
}
$$

---

# 十七、D/S 接下来怎么处理

不要停。

但是降为后台证据线：

```text
D-V / D-U → 完成epoch10 eval + holdout re-query
S-V / S-U → 启动并继续
Atlas D/S → 完成当前population
```

不允许它们阻塞：

```text
T-local-CF
Graph-T
RISE-T
```

如果 D/S 后面成功，再把 RFV 机制迁移过去。

如果失败，最终论文仍可以是：

$$
\boxed{\text{RFV-T}}
$$

而不是强迫 TDS。

---

# 十八、Atlas 的明确任务

AutoDL 上 Atlas 继续，但进入收尾模式：

| Atlas任务               | 是否继续     |
| --------------------- | -------- |
| T characterization    | 已完成，冻结   |
| S D/S population      | 继续       |
| recovery/same-support | 继续       |
| noise/benign baseline | 继续       |
| interaction           | 当前必要部分完成 |
| 新 heuristic           | 停        |
| 新 selector            | 停        |
| 新 Graph variants      | 停        |
| 新 distillation        | 停        |

目标：

$$
\boxed{
Atlas=\text{Measure}
}
$$

RFV：

$$
\boxed{
RFV=\text{Learn}
}
$$

---

# 十九、Raw 接下来怎么处理

Raw 不停，但只做两件事：

1. 修 correctness / interface；
2. 改进 off-grid diagnostic query coverage。

不要现在扩大到 full 6400 bank。

先做：

$$
Official\ insert
$$

vs

$$
Offgrid\ insert
$$

stratified development queries。

如果仍没有 Raw headroom：

> 暂停 Raw。

如果有：

> RFV-T 稳定后再将 Graph/Value 前端接到 Raw。

而且当前 Raw contributor identity 与 GraphRecovery 的 candidate-index contract 还不能直接组合，所以现在更不应抢跑 Raw+Graph。

---

# 二十、GPU 资源怎么排最快

## 如果只有2张主GPU

GPU-0：

```text
T-U 80ep
```

GPU-1：

```text
T-V 80ep
```

Graph/RISE offline 尽量 CPU / 空闲 GPU / A100 queue。

当 T-U/T-V 中一条释放：

```text
T-V-GCTX
```

再：

```text
T-V-FVD
```

最后：

```text
T-V-GCTX-FVD.
```

---

## 如果临时能再增加2张4090/A100

我建议直接增加。

四卡可以：

```text
GPU0 T-U
GPU1 T-V
GPU2 T-V-GCTX   (G1 PASS后)
GPU3 T-V-FVD    (R1/R2 PASS后)
```

Graph/FVD 两项都通过后：

```text
first free GPU → T-V-GCTX-FVD
```

这是目前最直接降低 wall-clock 的办法。

---

# 二十一、每条正式 course 的固定验收

任何新的 RFV course：

```text
CPU unit tests
→ dry-run
→ full-video end-to-end smoke
→ 2 optimizer update preflight
→ fresh-instance strict reload
→ no-GT inference
→ 80ep formal course
```

不要再给每个 checkpoint 单独提交作业。

同一 allocation：

```text
train
→ epoch10 eval
→ continue
→ epoch20 eval
→ ...
→ epoch80
```

现有 FastTrack trainer 已经支持这种模式。

---

# 二十二、未来 24 小时我希望必须产生的输出

不是“实现了多少文件”，而是下面这些真正的判别结果：

| 输出                  | 必须回答                               |
| ------------------- | ---------------------------------- |
| `T_LOCAL_CF.json`   | 当前≤4-swap action space到底有无headroom |
| `T_VALUE_MINI.json` | Plain Value是否能学                    |
| `GRAPH_G1_T.json`   | Graph是否降低holdout regret            |
| `RISE_A_T.json`     | Actual value是否漂移                   |
| `T_U/epoch10`       | matched Uniform trajectory         |
| `T_V/epoch10`       | learned T trajectory               |
| `RISE_B_T.json`     | Future是否胜Current/EMA/Post          |

如果这七个产物有了，论文结构基本就会变得清楚。

---

# 二十三、未来48小时目标

若 G1/R1/R2 全部有正信号：

直接确保下面四条正式课程已启动：

$$
T\text{-V}
$$

$$
T\text{-V-G}
$$

$$
T\text{-V-F}
$$

$$
T\text{-V-G-F}.
$$

不要继续等 TDS。

---

# 二十四、明确停止规则

这一点决定我们是不是真的“Fast”。

### T-local-CF 不优于 Uniform

暂停 T-V。

扩 action space，而不是增加 Graph/RISE。

### Plain T-Value holdout 不优于 proxy/uniform

暂不做 task-level Graph/FVD。

先修 Value learnability。

### Graph 不优于 parameter-matched MLP

Graph Context FAIL。

可以继续 GraphRecovery，但不进入 Value 主线。

### Actual value 无 drift

RISE FAIL。

不训练 FVD。

### Future 不胜 Current/EMA/Post

FVD FAIL。

### Graph/FVD individually PASS，但 GF 不胜最佳单项

最终模型只保留最佳单项。

### D/S 不通过

不强行做 TDS。

---

# 二十五、FINAL_MODEL_LEDGER 继续作为唯一入场口

现在所有 optional components 仍应保持 WAITING；当前公开 ledger 就是这么做的。

建议增加两个层次：

```text
Value-level status
Task-level status
```

例如：

| Mechanism     | Value Gate | Task Gate | Final |
| ------------- | ---------- | --------- | ----- |
| T Value       | PASS       | PASS      | yes   |
| Graph Context | PASS       | WAITING   | no    |
| FVD           | PASS       | WAITING   | no    |
| Graph+FVD     | WAITING    | WAITING   | no    |

只有 Task Gate PASS 才进入 final model。

---

# 二十六、给实验 owner 的直接执行指令

1. **从 `6e2fc7f` 建独立 RFV branch，不修改已有405 D/S结果目录。**
2. **修 T short-window valid mapping、Value safe initialization、test checkpoint protocol，并增加 full-video smoke。**
3. **立即实现当前部署 action space 的 T-local-CF development runner，这是 RFV 主线 Gate 0。**
4. **同时构建共享 Standard-T actual-CF bank，bank 必须额外保存 cheap preview graph inputs。**
5. **同一 bank 并行训练 Plain-M、Plain-L、StaticGraph、DynamicGraph；Graph 主指标用 video-level action regret。**
6. **立即实现 RISE-A fixed-manifest replay；历史V2 trajectory可先做 preliminary probe，新T trajectory作为正式结果。**
7. **T-local-CF + Plain learnability 通过后，T-U/T-V 从共同 V2-S epoch40 初始化直接跑完整80轮。**
8. **T checkpoints 出现后立即运行 RISE-B same-state Current/EMA/Post/Future，不等80轮结束。**
9. **Graph PASS 即启动 T-V-GCTX；RISE PASS 即启动 T-V-FVD；二者都PASS即启动 T-V-GCTX-FVD。**
10. **D/S 与 Atlas继续后台，Raw只做coverage diagnostic；任何一条都不得阻塞RFV-T。**
11. **任何参数选择只看training/calibration；holdout只作最终Value判断；official test只作冻结后的报告。**
12. **如果资源允许，临时增加2张同级GPU，把 T-U/T-V/T-G/T-F 并行跑，这是当前最直接的加速手段。**

---

## 最终目标

这一轮不再以：

> “实现完整 T/D/S/Graph/RISE/Raw/DB”

作为短期成功标准。

而是尽快回答三个核心命题：

$$
\boxed{\textbf{1. Computation value can be learned.}}
$$

$$
\boxed{\textbf{2. Computation value benefits from relational context.}}
$$

$$
\boxed{\textbf{3. Computation value evolves predictably during learning.}}
$$

如果三者全部成立，最终主方法自然就是：

$$
\boxed{
\textbf{Relational Future Marginal Computation Value}
}
$$

随后 D/S、Raw、DB 只是把这条原则扩展到更多 action domains；而不是反过来让这些附加实验拖慢主创新的确认速度。
