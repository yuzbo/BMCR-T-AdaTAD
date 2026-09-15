结合你刚给出的最新结果，我建议现在把项目从“继续扩模块”切换成一个非常明确的 **RFV Fast Sprint v2**。目标不是再多实现几个组件，而是在最短时间内回答：

$$
\boxed{
\text{Value能否学会}
\rightarrow
\text{Value是否需要Graph关系信息}
\rightarrow
\text{Value的未来变化是否可预测}
}
$$

当前证据已经足够支持这个转向：T 的当前部署动作空间在 cal20 上仍有 `+0.528pp` 的 Local-CF headroom；但现有 Value 在同视频新候选上的泛化几乎失效，D/S learned Value 在 epoch20 也都落后 Uniform；Graph 目前只有不稳定点估计；RISE 确实测到 value drift，却没有降低 future-action regret。因此现在最该修的不是 backbone，而是 **Value 表示/学习目标本身**。

原始 WTR 研究计划本来就要求先证明真实边际价值、再让 Graph 和 future teacher “挣得”进入最终模型的资格，而不是把模块一起堆上去。 最终设计中 Graph 是关系条件，RISE 是训练期目标增强，真实任务边际收益才是共同中心，这一点也已经在完整设计稿里明确。

---

# 一、从现在开始冻结四条实验线

建议服务器职责立即固定为：

| 线路             | 当前任务                        | 未来 24–48h 目标                 |
| -------------- | --------------------------- | ---------------------------- |
| **RFV-T 主线**   | T-local-CF / Value revision | 拿到稳定 T-U vs T-V              |
| **Graph**      | Value-level relation gate   | Plain-L vs Static vs Dynamic |
| **RISE**       | drift + same-state forecast | Current/EMA/Post/Future      |
| **Operator后台** | D/S + Atlas收尾               | 不阻塞 RFV                      |

Raw 暂时不扩 full bank；DB 暂停；B/ANet 等第一个真正成立的 Core/RFV candidate 出现再启动。

这也符合原 agents 的原则：真实反事实、local value、统计和唯一 owner 是基础；Graph、future teacher 都应有独立 gate。

---

# 二、第一件事不是 Graph，而是冻结一个“最小 Value Revision”

当前结果已经说明，仅仅“多收集一些 pair”不是最有希望的修复：

$$
Spearman\_{\rm fit}\approx0.94,
\qquad
Spearman\_{\rm same-video/unseen-candidates}\approx0.044.
$$

这更像：

$$
\boxed{
\text{当前Value学到了已见action/state的拟合，
而没有学到稳定的candidate-relative ranking。}
}
$$

所以我建议只改 **Value learning objective**，暂时不改 descriptor、不改 action space、不加 Graph。

称为：

$$
\boxed{\text{Value-R1}}
$$

保留当前：

$$
407\rightarrow128\rightarrow64\rightarrow2
$$

Temporal head。

但不要再把主要任务定义成逐个 action 的绝对 gain regression。

---

# 三、Value-R1：改成“状态内相对排序”

对一个固定 decision state (s)，真实候选：

$$
\mathcal C\_s={a\_1,\ldots,a\_n}.
$$

仍保存：

$$
g\_i^{cls},
\qquad
g\_i^{loc}.
$$

用冻结尺度形成：

$$
g\_i=
w\_c\frac{g\_i^{cls}}{s\_c}
\+
w\_l\frac{g\_i^{loc}}{s\_l}.
$$

然后在**同一竞争域内中心化**：

$$
\tilde g\_i
===========

g\_i-\frac1n\sum\_jg\_j.
$$

构造真实排序分布：

$$
p\_i^{\*}
=========

\operatorname{softmax}
\left(
\frac{\tilde g\_i}{\tau\_g}
\right).
$$

Value head 输出：

$$
q\_i.
$$

学生分布：

$$
p\_i
====

\operatorname{softmax}
\left(
\frac{q\_i-\bar q}{\tau\_q}
\right).
$$

主损失建议第一版只用：

$$
\boxed{
L\_{\rm rank}
=============

JS(p^{\*},p)
}
$$

再保留一个很小的当前 regression 辅助：

$$
L\_{\rm value}
==============

L\_{\rm rank}
\+
0.1L\_{\rm Huber}.
$$

这样不会完全失去 gain 的正负/STOP 信息，但优化重点从：

> “精确预测0.0017还是0.0021”

转向：

> “这个状态下究竟应该选哪几个 action”。

这也更自然地与后面的 Graph 和 RISE 对接。

---

# 四、为什么这是现在最值得做的最小修改

因为 router 最终并不关心全局 MSE。

它实际执行：

$$
a^\*=\arg\max\_a \hat V(a|s).
$$

所以真正风险是：

$$
\boxed{\text{decision regret}}
$$

而不是 value calibration error。

建议统一定义：

$$
R(s)
====

\max
\left(0,\max\_{a\in\mathcal C\_s}V(a)\right)
--------------------------------------------

V(\hat a).
$$

如果正确动作应该 STOP，则：

$$
V(\text{STOP})=0.
$$

以后 Plain、Graph、RISE 都以这个 regret 为第一指标。

这会显著简化整篇论文。

---

# 五、未来 4 小时：完成 Value-R1 离线 Gate

不要马上启动新 detector。

复用当前真实 CF bank。

固定三个模型：

$$
\text{Plain-R0}
$$

当前 regression head；

$$
\text{Plain-R1}
$$

同 architecture，只换 ranking objective；

$$
\text{Plain-L-R1}
$$

增宽到与后续 Graph 参数量近似。

对每个模型建议离线训练：

$$
3\text{ seeds}.
$$

所有模型：

- 同 fit actions；
- 同 calibration；
- 同 holdout；
- 同 optimizer updates；
- 同 normalization；
- 不重新查询 GT。

评价：

$$
Regret,
NDCG\@K,
TopK,
Spearman.
$$

如果：

$$
R\_{\rm R1}
<
R\_{\rm R0}
$$

并且 same-video unseen candidate 的 ranking 明显恢复，那么：

$$
\boxed{\text{Value-R1 PASS}}
$$

成为新的 RFV baseline。

如果仍然几乎为零相关：

> 暂停 Graph/RISE task training，转而检查 descriptor / action representation。

不要让 Graph 去掩盖 Plain Value 根本学不会的问题。

---

# 六、Graph 和 Plain-R1 可以从同一小时开始并行

不需要等 Plain-R1 detector 80轮。

Graph Machine 最值得迁移到这里的不是“把 Transformer 换成 GM”，而是它明确区分了：

$$
\text{state}
,\quad
\text{sparse address}
,\quad
\text{continuous edge weight}.
$$

其 referral 用稀疏边在保留大状态空间的同时动态改变可访问关系；原论文也明确说明 hard sparsification 后只有保留下来的连续边权承担梯度。

对于 WTR，我们首先只借它做：

$$
\boxed{\text{Relational Value Context}}
$$

不要第一版就上 GraphKV。

---

# 七、Graph G1-T 的最小实现

建议新模块：
```text
h65/paper/rfv_graph.py
```

输入使用现有 cheap temporal timeline。

每个时间节点：

$$
x\_t=
[
c\_t,
t\_{\rm norm},
actionness\_t,
transition\_t,
coverage\_t,
distance(t,S)
].
$$

第一轮图只做：

### Static

物理时间邻居：

$$
\pm1,\pm2,\pm4,\pm8
$$

控制最大 degree。

### Dynamic

同样 bounded candidate set，但基于：

$$
f(c\_i,c\_j,\Delta t)
$$

重新权重/选择。

输出：

$$
g\_t\in\mathbb R^{64}.
$$

对 swap：

$$
i\rightarrow j
$$

读取：

$$
g\_i,\qquad g\_j
$$

以及 selected-set pooled relation：

$$
g\_S.
$$

最后：

$$
Q\_T=
MLP[
x^{pair}\_{407},
g\_i,
g\_j,
g\_S
].
$$

---

# 八、Graph G1 必须固定四路对照

同一个 CF bank：

| 模型               | 作用         |
| ---------------- | ---------- |
| Plain-M R1       | 当前容量       |
| Plain-L R1       | 参数量匹配      |
| Static Graph R1  | 关系信息是否有用   |
| Dynamic Graph R1 | 动态边是否进一步有用 |

第一版都跑：

$$
3\text{ head seeds}.
$$

Primary：

$$
\boxed{\text{holdout finite-budget regret}}
$$

Secondary：

- NDCG；
- Top-K overlap；
- Spearman。

### Graph 的真正 PASS

不是 Static 点估计好看。

而应该至少满足：

$$
R\_G\<R\_{\rm Plain-L}
$$

且 video-level paired CI 方向稳定。

如果：

$$
Static>Plain-L
$$

但：

$$
Dynamic\approx Static,
$$

就直接用 Static。

不要为了 Graph Machine 故事强留 Dynamic/referral。

如果 Static/Dynamic 都不稳定：

$$
\boxed{\text{Graph Context FAIL}}
$$

Graph 仍可作为 recovery 分支继续研究，但不进入 Value 核心。

---

# 九、RISE 接下来必须改变比较对象

目前：

- actual value 有 drift；
- (\rho\_{20,60}\approx0.776)；
- sign flip≈14.4%；
- 但 Future regret = Current/Post；
- 真 EMA 反而略好。

这意味着：

$$
\boxed{
\text{non-stationarity存在，
但当前raw-value extrapolation并没有利用它。}
}
$$

所以不要继续用完全相同的 raw scalar extrapolation 重跑。

改成：

$$
\boxed{\text{candidate-set centered logit extrapolation}}
$$

这和 RISE 原论文的 function/logit-space formulation 更接近。原 RISE 的核心也是在共同输出空间中外推 anchor→post displacement，再把 extrapolated teacher **蒸馏回来**，而不是直接采用外推模型；论文还证明了直接采用 extrapolated policy 并没有带来主要收益。

---

# 十、RISE-B v2 的准确实现

固定一个 raw decision state：

$$
s
$$

和共同 candidate set：

$$
\mathcal C.
$$

取得：

$$
z\_A=Q\_{\Psi\_A}(s,\mathcal C),
$$

$$
z\_P=Q\_{\Psi\_P}(s,\mathcal C).
$$

先分别中心化：

$$
\bar z\_A=z\_A-\operatorname{mean}(z\_A),
$$

$$
\bar z\_P=z\_P-\operatorname{mean}(z\_P).
$$

然后：

$$
\boxed{
z\_F=
\bar z\_P
\+
(\beta-1)
(\bar z\_P-\bar z\_A)
}
$$

第一轮不要 adaptive β。

只在 calibration 上测试：

$$
\beta\in
{1,1.05,1.1,1.2}.
$$

β=1 必须严格等价于 post。

然后比较：

$$
Current,
TrueEMA,
Post,
Future.
$$

Teacher distribution：

$$
p\_F
====

softmax(z\_F/\tau).
$$

Principal metric：

$$
\boxed{\text{future actual-action regret}}
$$

而不是 logit MSE。

---

# 十一、RISE 现在有两道硬门

### RISE-A

真实 action value 是否漂移？

当前答案是：

$$
\boxed{\text{YES}}
$$

14.4% sign flip 已经不能叫完全 stationary。

所以 existence 部分有信号。

### RISE-B

训练轨迹方向是否**可预测地**指向 later better action？

当前答案是：

$$
\boxed{\text{NO EVIDENCE YET}}
$$

因为 Future 尚未优于 Current/EMA。

只有 RISE-B v2：

$$
R\_F<
\min(R\_{Current},R\_{EMA},R\_{Post})
$$

才解锁 FVD。

否则不要做长训。

这和 RISE 原论文强调的逻辑一致：外推只有在被可靠改进方向 grounding 时才安全，而且安全 β 范围会随训练收敛而变窄。

---

# 十二、当前 T-local-CF 已经允许继续做 T-Value，但不要夸大

现在：

$$
+0.528\text{ pp}
$$

CI：

$$
[+0.082,+0.715]
$$

说明当前 ≤4-swap action space 在 cal20 / 46 windows 上确实存在**有限但非零** headroom。

这已经足够让我们继续：

$$
T\text{-Value}.
$$

但这个 headroom 比 Atlas broad group action space 的数个百分点要小很多。

所以论文里必须分：

$$
\boxed{
\text{broad temporal headroom}
}
$$

和：

$$
\boxed{
\text{deployable local-swap headroom}.
}
$$

Figure 4 本来也要求 label-assisted search 只作为 finite diagnostic reference，不称为 oracle。

---

# 十三、T-U / T-V 现在可以直接准备正式80轮

只要 Value-R1 offline gate 通过，就不要再做独立10轮 pilot。

启动两条完全 matched：

$$
\boxed{T-U}
$$

和：

$$
\boxed{T-V-R1}.
$$

共同：

- V2-S epoch40 EMA init；
- K384；
- D100/S100；
- Cross；
- same optimizer；
- same schedule；
- same augmentation；
- same200 detector videos；
- same80 epochs；
- same milestone storage。

唯一核心差异：

$$
Uniform
$$

vs

$$
Value-R1 swap refinement.
$$

---

# 十四、Temporal Value 建议采用 offline prefit + online sparse refresh

这一步我非常推荐。

不要 step0 让随机 head 控制 support。

流程：
```text
CF bank
   ↓
offline Value-R1 prefit
   ↓
load Value weights
   ↓
formal T-V 80ep
   ↓
low-frequency current-policy CF refresh
```

offline prefit 只用160 fit。

online CF 也只在160 fit 上产生 Value loss。

cal20：

- τ；
- STOP；
- β；
- Graph mode；

holdout20：

- 不参与任何选择；
- 最终 Value/Graph/RISE 泛化判断。

这与当前科学协议完全一致。

---

# 十五、T-V 训练期间必须把 RISE 数据自动收集掉

不要等 T-V 完成以后才想到 RISE。

在：

$$
10,20,40,60,80
$$

保存：
```text
router raw weights
normalization
EMA router
descriptor revision
candidate generator revision
science SHA
```

同时保存一份冻结 action manifest。

这样 epoch20 出现时就能运行：

$$
10\rightarrow20\rightarrow future40
$$

预测测试。

epoch40 出现后：

$$
20\rightarrow40\rightarrow future60.
$$

RISE-B 不需要等80轮。

---

# 十六、Graph task-level training 的启动条件

如果：

$$
Plain\text{-R1 PASS}
$$

且：

$$
Graph\text{-G1 PASS},
$$

马上启动：

$$
\boxed{T\text{-V-G}}
$$

80轮。

不用等：

- D；
- S；
- TDS；
- DB；
- Raw。

但只加：

$$
GraphContext.
$$

不要同时加 GraphRecovery。

这样才能比较：

$$
T\text{-V-R1}
$$

vs

$$
T\text{-V-R1+Graph}.
$$

---

# 十七、FVD task-level training 的启动条件

如果：

$$
RISE\text{-B PASS},
$$

才启动：

$$
\boxed{T\text{-V-FVD}}.
$$

建议第一版：

$$
\beta
$$

在 calibration 冻结。

FVD teacher snapshot 每10 epochs 更新一次即可。

Distillation loss建议用：

$$
JS(
p\_{\rm student},
sg[p\_F]
).
$$

不要第一版同时加入：

- regression FVD；
- ranking FVD；
- adaptive β；
- EMA anchor；
- Graph-conditioned teacher。

一次只验证一个东西。

---

# 十八、FVD 至少需要一个 Post control

如果最终 FVD 有收益，必须区分：

$$
\text{普通self-distillation}
$$

和：

$$
\text{真正future extrapolation}.
$$

所以最终至少有：

$$
T\text{-V+Post}
$$

或等价 β=1 control。

这和原 RISE 论文一样：真正收益来自“grounded extrapolated teacher + distillation”，不能简单归因于多一次优化。原 RISE 还专门做了 compute-matched extra-step control。

如果 FVD 最终成为主贡献，WTR 也应增加一个：

$$
\text{same extra optimization / beta=1}
$$

control。

---

# 十九、Graph+FVD 只有单项都通过才启动

最终：

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

Graph conditioner 必须属于 snapshot function：

$$
\Psi=
{
\text{descriptor},
\text{Graph},
\text{ValueHead},
\text{normalization}
}.
$$

然后：

$$
z\_A,z\_P
$$

必须在**同一 raw state / same candidate set**上获得。

这一步通过，才有资格写：

$$
\boxed{
V\_{\rm future}(a|s,G\_s)
}
$$

---

# 二十、D/S 现在如何处理最省时间

你现在 D/S epoch20 都是：

$$
V\<U.
$$

不要停止已经运行的四条课程，因为它们已经推进到 epoch25–40，而且预登记 endpoint 是80。

但从现在开始：

$$
\boxed{\text{不新增D/S变体。}}
$$

执行：

- D-V/U 取得 epoch40完整评测；
- S-V/U 继续40；
- 冻结40 checkpoint；
- 做 router-label holdout re-query；
- 如果 Value 仍输 Uniform 且 regret/rank也不强，则 D/S 不进入 RFV 主模型。

不用为了三轴故事继续 DS/TDS。

原研究计划本来就允许“只有T有效时写 temporal-focused 论文并报告 D/S 负结果”。

---

# 二十一、Atlas 继续，但只做收尾

Atlas-S GPU measurement 已经齐，这是好消息。

接下来只完成：

- S D/S population CPU aggregate；
- recovery；
- no-op/benign noise；
- interaction；
- publication figures。

不要继续新增 Atlas experiment。

Figure 1–6 的证据规划强调 Atlas 的任务是 characterization、matched headroom 和机制诊断，而不是变成另一个新方法搜索平台。

B 保持暂停是合理的。

---

# 二十二、Raw 暂停长训

Raw 当前最大问题仍然是 query coverage。

所以只做：

$$
Official\ insert
$$

和：

$$
Offgrid\ insert
$$

的 stratified development diagnostic。

不要现在：

- 扩6400；
- matched retrain；
- Graph+Raw；
- new preview sweep。

等 RFV-T 结果清楚，再决定 Raw 是否值得重新进入主线。

---

# 二十三、现在服务器应该怎么分

你现在实际资源非常适合加速。

### 4090 四条现有课程

继续：

- D-V
- D-U
- S-V
- S-U

不要抢占。

### AutoDL GPU0

一旦 Value-R1 Gate 通过：

$$
\boxed{T-U\ 80ep}
$$

### AutoDL GPU1

同时：

$$
\boxed{T-V-R1\ 80ep}
$$

这两张卡当前空闲，应该立即利用。

### A100

优先做：

1. T-CF bank generation；
2. Graph offline fit/eval；
3. RISE actual-value replay；
4. 后续 G/FVD task course。

不要让 A100 空等 D/S。

---

# 二十四、建议的未来24小时墙钟计划

| 时间         | 必须完成                                                      |
| ---------- | --------------------------------------------------------- |
| **0–2h**   | 新 RFV science SHA；T mapping/zero-init/test-endpoint fixes |
| **0–4h**   | Value-R1 offline；Plain-M/R1/L三路                           |
| **2–6h**   | Static/Dynamic Graph离线3 seeds                             |
| **2–8h**   | RISE-A/B v2 replay                                        |
| **4–8h**   | 冻结 Plain/Graph/RISE Gate                                  |
| **8h**     | 若 Plain PASS，AutoDL同时启动 T-U/T-V                           |
| **8–16h**  | Graph PASS则准备 T-V-G；RISE PASS则准备 T-V-F                    |
| **16–24h** | 第一批T训练日志/route/value质量；Graph/FVD preflight                |

不要用“有没有新 mAP”判断24小时是否成功。

真正成功是拿到：
```text
VALUE_R1_GATE.json
GRAPH_G1_T.json
RISE_A_T.json
RISE_B_T.json
T_U/train receipt
T_V/train receipt
```

---

# 二十五、48小时目标

如果 Graph 和 RISE 都成功：

必须至少已经有四条 recipe 能运行：
```text
T-U
T-V
T-V-G
T-V-F
```

并且：
```text
T-V-G-F
```

已经通过 CPU/GPU gate 或开始排队。

如果 Graph FAIL：

不启动 G/GF。

如果 RISE FAIL：

不启动 F/GF。

这才叫 Fast-Track。

---

# 二十六、代码组织建议

建议不要再往当前 Temporal 文件里不断堆逻辑。

建立：
```text
h65/rfv/
    value.py
    graph.py
    future.py
    bank.py
    replay.py
    metrics.py
```

其中：

### `value.py`

- candidate-set centered value；
- listwise ranking；
- STOP；
- gain calibration。

### `graph.py`

- static graph；
- dynamic graph；
- graph-conditioned pair value。

### `future.py`

- snapshot；
- true EMA；
- centered logit extrapolation；
- β=1/post；
- FVD teacher。

### `bank.py`

- immutable state/action bank；
- fit/cal/holdout；
- hashes。

### `metrics.py`

- regret；
- NDCG；
- rank；
- video bootstrap。

Core encoder仍然复用 `h65/paper`，不要复制执行内核。

---

# 二十七、建议新配置命名
```text
rfv_t_uniform_s42
rfv_t_value_r1_s42
rfv_t_value_graph_s42
rfv_t_value_post_s42
rfv_t_value_fvd_s42
rfv_t_value_graph_fvd_s42
```

如果 Dynamic Graph 没有通过，就不要在名字里区分 Graph 类型；最终只保留获胜版本。

---

# 二十八、正式训练命令仍用现有 owner

建议沿用已有训练入口，不写第二套 trainer。

模板：
```bash
python tools/paper_train.py \
  --config configs/rfv/rfv_t_value_r1_s42.json \
  --resources research/paper/resources.local.json \
  --output research/rfv/runs/rfv_t_value_r1_s42 \
  --preflight \
  --slice-hours 1
```

通过后：
```bash
python tools/paper_train.py \
  --config configs/rfv/rfv_t_value_r1_s42.json \
  --resources research/paper/resources.local.json \
  --output research/rfv/runs/rfv_t_value_r1_s42 \
  --slice-hours 10
```

续跑：
```bash
python tools/paper_train.py \
  --config configs/rfv/rfv_t_value_r1_s42.json \
  --resources research/paper/resources.local.json \
  --output research/rfv/runs/rfv_t_value_r1_s42 \
  --resume \
  --slice-hours 10
```

Raw/Graph/RISE 离线 probe 可使用独立 CLI，但 GPU 长训仍归唯一 owner 管理。这和 A7 的部署原则一致。

---

# 二十九、必须统一输出的状态文件

每条 RFV experiment 都输出：
```text
manifest.json
config.json
source_revision.txt
checkpoint/
predictions/
metrics.json
value_metrics.json
route_stats.json
cost.json
```

Graph：
```text
graph_metrics.json
```

RISE：
```text
snapshot_manifest.json
future_metrics.json
```

并继续更新：
```text
FINAL_MODEL_LEDGER.csv
```

状态建议：
```text
IMPLEMENTED
TECH_PASS
HEADROOM_PASS
LEARNABILITY_PASS
TASK_PASS
FINAL
```

---

# 三十、最终 Gate 规则现在就写死

| 机制                 | PASS条件                                    |
| ------------------ | ----------------------------------------- |
| **T action space** | local-CF 稳定优于 Uniform                     |
| **Plain Value**    | holdout regret 优于 proxy/STOP              |
| **Graph**          | Graph regret 优于 parameter-matched Plain-L |
| **RISE drift**     | actual value drift 超过 replay/noise        |
| **RISE forecast**  | Future regret 优于 Current/EMA/Post         |
| **T Value task**   | T-V 全数据 mAP 在matched cost下优于T-U           |
| **Graph task**     | T-V-G 优于T-V且成本计入                          |
| **FVD task**       | T-V-F优于β=1/Post control                   |
| **GF interaction** | T-V-G-F 至少优于最佳单项，否则删冗余模块                  |

Graph/RISE 不是因为论文需要就 PASS。

---

## 最终执行指令

如果要压缩成实验 owner 可以直接采用的一段，我建议：

> **冻结当前 D/S 和 Raw science identity，不修改已运行结果。新建独立 RFV science branch。首先仅修改 Temporal Value 的学习目标为 candidate-set conditioned ranking，并完成 short-window、initial routing 和 test-checkpoint protocol 修复。使用现有真实CF建立共享 T bank，Plain-M、Plain-L、Static Graph、Dynamic Graph 以及 RISE replay 全部在相同160/20/20 split和相同action manifest上并行评测。Plain Value learnability通过后立即占用当前空闲两张AutoDL GPU，以完全matched的80轮 T-U/T-V 课程开始主训练。Graph Value gate通过即解锁 T-V-G，RISE drift+forecast两关通过即解锁 T-V-F；二者均通过才训练 T-V-G-F。D/S继续现有课程和holdout复核，但不新增变体；Atlas只完成S统计/recovery收尾；Raw只做off-grid coverage诊断。所有方法选择只看training/calibration，holdout只作Value泛化判断，official test不参与checkpoint或超参数选择。**

这会是我现在认为**最快确认 Graph/RISE 是否真的能成为 WTR 论文核心创新，同时又不会让它们掩盖 Value 本身失败**的执行方案。

