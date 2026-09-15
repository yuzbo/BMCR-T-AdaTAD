我认真看完了这两份审阅。总体判断是：

$$
\boxed{\textbf{值得吸收，而且这轮建议总体质量很高。}}
$$

它没有推翻我们此前的 WTR/RFV 主线，而是根据新出现的负结果，把路线从“Graph/RISE 尽快加入”进一步收紧成：

$$
\boxed{
\text{先确认CF机会是否可预测}
\rightarrow
\text{修正Value本身}
\rightarrow
\text{再判断Graph/RISE有没有额外价值}
}
$$

这是合理的证据更新。当前审阅最重要的贡献，不是提出了一个“更复杂的新模型”，而是识别出：**目前真正卡住论文的已经不是 headroom，而是 predictability / learnability gap。** 

不过，我不会把其中所有建议原样执行。我认为应该分成“立即吸收”“修改后吸收”和“暂不接受”三类。

---

# 一、我最认同的新判断：现在要区分 CF headroom 和可预测 headroom

这是这份报告里最值得吸收到论文理论框架的一点。

我们之前容易隐含：

$$
\text{CF能找到更优动作}
\Rightarrow
\text{cheap Value应该能学会}.
$$

实际上不成立。

报告给出的区分非常好：

$$
H_{\rm CF}(Z)
=
\mathbb E[\max_a g(a)\mid Z],
$$

而部署时真正能够达到的是：

$$
H_{\rm cheap}(Z)
=
\max_a\mathbb E[g(a)\mid Z].
$$

因此：

$$
\boxed{
H_{\rm CF}(Z)\ge H_{\rm cheap}(Z).
}
$$



这对我们目前的数据解释极其重要。

现在已经知道：

### T action space

存在：

$$
+0.528\text{ pp}
$$

的 Local-CF headroom。

但 Value：

$$
\rho_{\rm fit}\approx0.94,
\qquad
\rho_{\rm unseen}\approx0.044.
$$

这说明：

$$
\boxed{
\text{Opportunity exists}
\neq
\text{Opportunity is predictable from current cheap evidence}.
}
$$

这其实可以进一步形成论文里的一个新概念：

# **Allocatable / Predictable Computation Headroom**

我们以后不要只问：

> 有没有更好的 allocation？

而应该问：

> **其中有多少可以在执行前根据廉价状态可靠预测？**

这个问题比单纯 CF-Uniform gap 更深。

---

# 二、D/S 当前负结果的解释，我同意

报告正确提醒：

$$
D\text{-V}-D\text{-U}<0,
\qquad
S\text{-V}-S\text{-U}<0
$$

不能简单解释成：

> Value mask 本身直接导致了全部掉点。

因为 D-V 与 D-U 是两条分别适配过的训练 trajectory。

最终差异混合了：

$$
\text{routing}
+
\text{adapter/light adaptation}
+
\text{decoder/head adaptation}
+
\text{optimization trajectory}.
$$



所以我非常赞成加入：

## Fixed-checkpoint policy crossover

冻结同一个 D-V checkpoint，比较：

$$
D\text{-V checkpoint}+Value route
$$

和：

$$
D\text{-V checkpoint}+Uniform route.
$$

S 同理。

它虽然仍有 route distribution mismatch，但至少能够把：

> 当前 mask identity 本身是不是直接有害

与：

> 整条训练 trajectory 已经不同

区分开。

**这项实验值得立即加。**

---

# 三、global gradient clipping 这个新发现也非常值得吸收

这一点我认为是很好的源码审阅发现。

虽然 Value state 已经：

$$
\operatorname{detach},
$$

但训练时仍然对所有可训练参数一起：

```python
clip_grad_norm_(all_trainable_parameters, 1.0)
```

那么 Value gradient 可能改变 global clipping coefficient：

$$
\alpha
=
\min
\left(
1,
\frac{c}{
\sqrt{\|g_{\rm task}\|^2+
\lambda^2\|g_{\rm value}\|^2}
}
\right).
$$

于是 detector 的 task gradient 实际变成：

$$
\tilde g_{\rm task}
=
\alpha g_{\rm task}.
$$

Uniform course 没有同样的 Value gradient。

因此 Value 即使无法通过 detached state 回到 backbone，也可能通过**global clipping**改变 detector 的优化。

但报告的态度也正确：

$$
\boxed{\text{先测，不要现在修改旧课程。}}
$$

我建议立即记录：

* task-only gradient norm；
* Value-only gradient norm；
* total norm；
* clip coefficient；
* CF update step vs ordinary update step；
* D-V vs D-U / S-V vs S-U。

如果：

$$
P(\alpha<1)
$$

很低，这条线可以排除。

如果 Value course 明显更频繁 clipping，下一代 RFV training 才应该考虑：

$$
\text{separate controller/task clipping}.
$$

旧 D/S 课程不要中途改。

---

# 四、bank 只有 round-0，这个问题值得保留，但现在不是首要原因

我赞同报告的判断。

当前部署 T policy 是：

$$
S_0
\rightarrow S_1
\rightarrow S_2
\rightarrow S_3
\rightarrow S_4,
$$

每次 support 都变化。

但训练 bank 主要是：

$$
S_0
$$

上的 actions。

所以最终存在：

$$
\text{offline-state distribution}
\neq
\text{deployed-policy state distribution}.
$$

这是典型 on-policy shift。

不过：

$$
\boxed{
\text{现在连round-0未见candidate都泛化失败，
因此不能把当前失败归因给round1–4。}
}
$$

这一点我完全同意。

所以顺序应该是：

1. 先让 single-state candidate ranking 学会；
2. 再少量采集 round1–3；
3. 最后才做 DAgger-like online refresh。

而不是现在扩大数千条 multi-round bank。

---

# 五、我认为“可逆 T swap 的反向一致性”非常值得做

这是这轮报告里我最支持的**具体模型修订**。

它不是“再造一个新模块”，而是利用真实 task function 本身必须满足的结构。

对于：

$$
S'=S-\{i\}+\{j\},
$$

在：

* 同 checkpoint；
* 同 episode；
* 同 augmentation；
* 同 normalizer；
* 同 D/S/recovery；
* 确定性重执行；

条件下：

$$
g(S,i\to j)
=
L(S)-L(S'),
$$

反向：

$$
g(S',j\to i)
=
L(S')-L(S),
$$

因此精确满足：

$$
\boxed{
g(S,i\to j)
=
-g(S',j\to i).
}
$$



当前自由的 407D pair MLP 没有被强制遵守这个约束。

这确实可能解释为什么：

$$
fit\text{ 很好}
$$

但：

$$
new\ candidate\text{ 很差}.
$$

模型可以利用 action/state identity 记忆，而不是学习 exchange structure。

---

# 六、因此这个“反向一致性 mini”我建议放在我们原来的 Value-R1 之前

这里是它与**我们之前实验计划唯一明显的战术冲突**。

我们之前建议首先尝试：

$$
\boxed{\text{candidate-set listwise ranking Value-R1}}
$$

把 Huber regression 改成 state-relative JS/ranking。

新的审阅建议则是：

$$
\boxed{
\hat g_{\rm sym}
=
\frac{
f(x^+)-f(x^-)
}{2}.
}
$$

这两个方向**并不矛盾**，但绝对不要第一轮同时加。

否则如果结果变好，我们不知道来自：

* relative ranking loss；
* reverse descriptor；
* antisymmetry；
* 双向 head evaluation。

所以我现在会修改我们的旧计划：

$$
\boxed{
\textbf{先 Reverse-Consistency Mini，
再 Listwise-Ranking Mini。}
}
$$

为什么反向一致性优先？

因为它：

* 来自真实函数的精确代数结构；
* 不增加视觉信息；
* 不增加 Value 参数；
* 不引入 temperature 等新超参数；
* 可以大量复用现有 bank；
* 失败时解释非常明确。

而 listwise ranking 属于更一般的 learning objective 假设。

因此：

### Plan A

Antisymmetric exchange。

若成功，再看是否需要 ranking。

### Plan B

若 antisymmetry 不解决泛化，再单独测试我们原先的 candidate-set ranking。

这样证据最干净。

---

# 七、不过我建议对报告的 P1/P2 control 再加强一点

这里我不会完全原样采用。

报告定义：

### P1

两个方向分别监督：

$$
f(x^+)\to g,
\qquad
f(x^-)\to-g.
$$

### P2

预测：

$$
\frac{f(x^+)-f(x^-)}2.
$$

这个控制已经不错。

但 P2 推理时看到了：

$$
x^+,\quad x^-,
$$

而普通 P1 推理只用：

$$
x^+.
$$

所以如果 P2 更好，仍有一部分可能来自：

> 多看了一次 reverse-conditioned descriptor，

而不纯粹是 antisymmetric parameterization。

因此我建议再保存一个极便宜的：

### P1-bi

P1 同样执行两次 head，在 inference 用两个独立估计做对称集成：

$$
\hat g_{\rm bi}
=
\frac{
f(x^+)-f(x^-)
}{2},
$$

但训练仍然是两个独立 Huber 项。

那么：

* P1 vs P1-bi：双向 inference / variance reduction；
* P1-bi vs P2：训练时结构约束本身。

不会增加 detector 长训，只是小头离线控制。

这样因果归因更完整。

---

# 八、反向描述子实现要求，我完全认同

不能简单：

> remove/insert 互换。

必须真正重建：

$$
S'=S-i+j.
$$

比如：

$$
\bar h_{S'}
=
\bar h_S+
\frac{h_j-h_i}{K}.
$$

然后重新计算：

* support gap；
* membership；
* distance；
* provenance；
* coverage；
* Graph occupancy（未来若用 Graph）。



否则你测的根本不是：

$$
\phi(S',j\to i).
$$

CPU test 应明确做：

$$
S
\rightarrow S'
\rightarrow S
$$

的 round-trip。

而且真实 counterfactual 要验证：

$$
g^++g^-
$$

落在 replay/no-op noise 内。

这应该成为 hard contract。

---

# 九、Graph 的新建议和我们之前计划没有战略冲突，但优先级确实应暂时后移

我们之前强调：

> Graph 和 RISE 对论文完整性重要，要尽早验证。

这点**仍然不变**。

但现在已经真正验证过：

* Static Graph 有较好的 point estimate；
* 但相对 Plain-L CI 跨0；
* seed不稳定；
* 相对 random legal swap 几乎没有稳定优势。



这意味着：

$$
\boxed{
\text{Graph目前没有获得进入task-level长训的资格。}
}
$$

所以新报告建议：

> 先让 Plain Value 具有 candidate generalization，再复测一个固定 Graph。

我是赞成的。

这不是“Graph 降级”。

而是：

$$
\boxed{
\text{Graph从架构主角变成待验证的representation hypothesis。}
}
$$

如果 Plain 根本学不会，Graph 相比 Plain-L 的差异很难解释。

---

# 十、Graph 当前 Static vs Dynamic 也确实存在一个设计混淆

报告指出：

> Static 与 Dynamic 不仅 referral 不同，initial neighborhood 也不同。

所以当前结果不能解释成：

$$
\text{Dynamic referral}
\quad vs\quad
\text{no referral}.
$$

这点值得立即吸收到后续 Graph 设计。

以后如果 Graph 再进入：

必须有：

$$
\boxed{
\text{same initial edge candidates}
}
$$

然后：

* referral off；
* referral on。

否则不能把收益归因于 Graph Machine 最核心的 referral。

当前不要扩 degree/referral sweep。

---

# 十一、RISE 的新分析我也基本完全赞同

目前已经有两个不同事实：

### Actual Value

确实有：

$$
\rho_{20,60}\approx0.776
$$

和约：

$$
14.4\%
$$

sign flips。

所以：

$$
\boxed{\text{Value并非完全stationary。}}
$$

### Forecast

但：

$$
R_{Future}=R_{Current/Post}
$$

且：

$$
R_{EMA}<R_{Future}.
$$

因此：

$$
\boxed{\text{目前没有证据支持future extrapolation改善决策。}}
$$



所以我们此前：

> RISE drift → forecast → FVD

这条逻辑完全没错。

只是实验现在已经告诉我们：

$$
\boxed{
Drift=PASS,
\quad
Forecast=FAIL.
}
$$

自然结果就是：

> FVD 不解锁。

这与之前计划不是冲突，而是计划中的 Gate 真正发挥作用了。

---

# 十二、我非常赞成下一步先看 RISE “到底有没有改变选择”

这是一项几乎免费的诊断。

对每个 state 导出：

* anchor logits；
* post logits；
* EMA logits；
* future logits；
* top-1 action；
* runner-up；
* STOP；
* top1-top2 margin；
* extrapolation displacement；
* argmax 是否变化。

因为现在：

$$
R_F=R_P
$$

有两种完全不同的情况。

### 情况 A

$$
\arg\max z_F
=
\arg\max z_P
$$

几乎总是一样。

那么：

> extrapolation 太弱，根本没改变决策。

### 情况 B

选择发生变化，但：

$$
g(a_F)=g(a_P).
$$

那么：

> extrapolation 改了 policy，却没有获得更好 actual utility。

两者对后续方向非常不同。

报告这个建议值得立即执行。

但是：

**不要因为发现没改 argmax，就立即扫 β=1.3/1.5/2.0 强迫它变化。**

这会变成 outcome chasing。

---

# 十三、RISE-B0 的“回顾式 forecast”边界必须吸收进论文

报告指出，当前 β 使用 later-calibration 标签选择，所以它是：

$$
\boxed{\text{retrospective cross-video diagnostic}}
$$

而不是严格：

> epoch40 时真的不知道 epoch60，然后预测未来。



这个边界非常重要。

如果未来想正式声称：

> future-value prediction，

需要重新设计：

* β 由更早 development trajectory 冻结；
* 或预注册 β；
* future checkpoint 不参与参数选择。

当前 B0 结果只能是：

> retrospective forecast probe.

这一点我建议完全吸收。

---

# 十四、对于“更多数据不一定解决问题”，我同意谨慎版，不同意强版本

审阅写得比较谨慎，我赞成：

$$
\boxed{
\text{现在不支持盲目扩 bank。}
}
$$

因为同一视频内部：

$$
\rho_{\rm fit}\approx.94
\rightarrow
\rho_{\rm held}\approx.044.
$$

这说明问题已经超越“视频没见过”。

但这绝不能推成：

$$
\boxed{
\text{更多数据无用。}
}
$$

可能当前 400 fit actions 只覆盖非常窄的 states。

所以现在应该：

> 暂停单纯扩大规模，先测试结构/表示假设。

如果结构修订成功，最后还是需要扩大数据做最终训练。

---

# 十五、关于 interaction 实验：新审阅没有推翻，仍然值得保留

我们上一轮讨论的：

$$
I_{TD},
I_{TS},
I_{DS},
I_{TDS}
$$

依然重要。

但现在它应该是：

$$
\boxed{\text{Atlas characterization / later joint-allocation evidence}}
$$

而不是下一轮 RFV mini 的主任务。

如果已有 joint records：

> 直接 CPU 重新分析 distribution、\(\mathbb E|I|\)、sign fraction、null-normalized interaction。

不要新开大批 GPU interaction experiments。

先解决：

$$
\boxed{\text{单个 Value 能不能泛化。}}
$$

否则联合 Value 更没有意义。

---

# 十六、因此和我们上一版实验计划的关系是这样的

## 没有冲突的部分

仍然成立：

* T 是主科学 carrier；
* D/S 继续后台；
* Atlas收尾；
* Raw暂缓；
* Graph/RISE必须独立 gate；
* official test 不参与方法选择；
* outer20 保持封存；
* Final Model Ledger 证据入场。

---

## 需要更新的部分

上一版：

$$
T\text{-localCF}
\rightarrow
Value\text{-R1 ranking}
\rightarrow
Graph/RISE.
$$

现在我建议：

$$
\boxed{
T\text{-localCF}
\rightarrow
Reverse\text{-Consistency Mini}
\rightarrow
\begin{cases}
\text{成功：确认/扩展}\\
\text{失败：再做 Listwise Value-R1}
\end{cases}
}
$$

Graph/RISE：

> 保留研究优先级，但暂不启动新 task-level 长训。

---

# 十七、我建议现在正式采用的新执行顺序

### P0 — 保留既有 D/S

继续到预注册 endpoint。

不改 optimizer/clipping。

---

### P1 — 固定模型诊断

马上做：

* Value-route vs Uniform-route crossover；
* task/value gradient norms；
* clip coefficient；
* current route statistics。

这是 D/S negative result 的解释。

---

### P2 — T reverse-consistency contract

CPU + 少量 GPU：

1. 真实 forward swap；
2. reverse swap；
3. 验证：

   $$
   g^++g^-\approx0;
   $$
4. S→S'→S；
5. descriptor rebuild；
6. no-op。

---

### P3 — 三个小头，不是 detector course

建议最终是：

* P0：历史 Plain；
* P1：bidirectional supervision；
* P1-bi：bidirectional inference control；
* P2：antisymmetric Value。

3 seeds。

Primary：

$$
regret.
$$

不调 STOP。

---

### P4 — 决策

如果 P2：

$$
R_{P2}<R_{P1-bi}
$$

并且胜：

* STOP；
* random legal；
* old Plain；

则：

$$
\boxed{\text{antisymmetric Value PASS}}
$$

再上新的独立 video/action holdout。

---

### P5 — 只有 Value 有泛化后

再：

* 4-round closed loop；
* dev full AP；
* matched T-U/T-V；
* fixed Graph retest；
* prospective RISE trajectory。

---

# 十八、我还建议增加一个非常明确的新论文诊断量

既然现在最核心的问题已经变成：

> CF opportunity 能不能从 cheap evidence 预测？

我建议以后报告：

$$
\boxed{
\textbf{Predictability Ratio}
=
\frac{
G_{\rm learned}-G_{\rm uniform}
}{
G_{\rm CF}-G_{\rm uniform}
}
}
$$

只在分母正且同 action space/matched protocol 下解释。

例如：

### 0

说明：

> CF有机会，但 learned policy 没吃到。

### 1

说明：

> 已经吃到 reference 找到的全部有限 headroom。

超过1也可能出现，因为 CF只是有限搜索，不是 oracle，因此不要叫 percentage-of-oracle。

可以更安全地称：

# **Recovered Diagnostic Headroom**

这个指标很适合 Figure 4。

它会把现在项目的核心矛盾画得非常清楚：

$$
\text{headroom exists}
\quad vs\quad
\text{headroom recovered}.
$$

---

# 十九、这轮建议最大的价值其实是帮助我们“收缩论文假设”

之前我们的故事容易变成：

$$
T+D+S+Graph+RISE+Raw+DB.
$$

现在证据正在迫使论文变得更科学：

### 已有证据

$$
\boxed{\text{T存在非均匀allocation opportunity}}
$$

### 未解决

$$
\boxed{\text{这种opportunity是否从cheap evidence可预测}}
$$

### 候选解释

* exchange structure；
* relational state；
* on-policy distribution；
* future target drift。

这样论文核心反而更尖锐：

> **When is marginal computation value predictable from cheap state?**

如果最后发现：

* antisymmetry解决；
* Graph进一步解决；
* RISE不解决；

最终论文就可以是：

$$
\boxed{
\text{Grounded Structured/Relational Marginal Value}
}
$$

完全不需要强塞 RISE。

---

# 二十、最终评价

我会这样给两份材料定性：

### 当前进度分析

$$
\boxed{\textbf{非常值得采纳。}}
$$

尤其：

* 不把 D/S 中途负结果粉饰；
* CF ≠ predictable CF；
* Graph/RISE 的负 gate；
* gradient clipping coupling；
* round0 bank；
* retrospective RISE 边界。

这些都应该进入正式研究记录。

### 可逆交换 Value 修订

$$
\boxed{\textbf{值得立即做一个mini。}}
$$

因为它结构明确、成本很低、可证伪、且直接对应当前泛化失败。

但它不是“新的最终模型”。

### 与我们以前计划的关系

$$
\boxed{\textbf{战略一致，战术需要更新。}}
$$

最大的更新只有一句话：

> **先不要同时上 ranking/Graph/FVD；先用 T 交换本身的精确可逆结构检验 Value 泛化问题。**

如果这个 mini 仍失败，再回到我们原先的 candidate-set ranking / richer descriptor / Graph hypothesis；如果它成功，我们就获得了一个比“换一个loss”更强、更容易讲清楚的 Value 机制。
