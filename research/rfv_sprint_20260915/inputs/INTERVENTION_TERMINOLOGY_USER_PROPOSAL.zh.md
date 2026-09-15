# 用户术语建议记录（2026-09-15）

本文件保留用户建议的全部术语、公式与论点，按Markdown整理呈现。实际采用的规范与定义限定见[WTR统一术语](C:/Users/skywalker/Documents/ChatGPT/H65/WTR_TERMINOLOGY.zh.md)。

可以。后续我建议**彻底不用“动作 / action”来称呼这类量**，因为它确实会和 TAD 里的人体动作实例、动作类别混淆。

更学术、也更适合论文的统一表述可以改成：

intervention\boxed{\text{intervention}}

中文统一译成：

干预\boxed{\textbf{干预}}

然后根据对象细分。

最推荐的一套术语是：

- `observation intervention` → **观测干预**
- `computation intervention` → **计算干预**
- `depth intervention` → **深度计算干预**
- `FFN intervention` → **FFN 计算干预**
- `frame exchange` → **帧交换干预**
- `joint intervention` → **联合干预**
- `single intervention` → **单项干预**
- `candidate intervention` → **候选干预**
- `intervention set` → **干预集合**
- `intervention effect` → **干预效应**
- `marginal intervention value` → **边际干预价值**
- `task effect` → **任务效应**

这样整篇论文就会非常清楚。

例如原来：

> Actions below tolerance

建议改成：

Fraction of interventions below tolerance

中文：

> **低于效应阈值的干预比例**

---

原来：

> Highest-effect actions (%)

改成：

> **Highest-effect interventions (%)**

中文：

> **任务效应最大的干预比例**

---

原来：

> Jointly changed actions

改成：

> **Number of jointly applied interventions**

中文：

> **联合施加的干预数量**

---

原来：

> Sum of single effects

改成：

> **Sum of individual intervention effects**

中文：

> **单项干预效应之和**

---

原来：

> Low single effect: joint

改成：

> **Joint effect of individually low-effect interventions**

中文：

> **单独效应较低干预的联合效应**

这个表述会严谨很多。

---

## 对你这套 Atlas 图，我建议术语统一成下面这样

第2页几类：

- Observation replacement → **观测替换干预**
- Depth upgrade → **深度计算增强干预**
- FFN upgrade → **FFN 计算增强干预**
- Frame exchange → **帧交换干预**
- Benign control → **良性对照干预**
- No-op control → **空操作对照**

“upgrade”如果你觉得“增强”也容易带方向暗示，可以更中性地叫：

- **深度路径切换干预**
- **FFN 路径切换干预**

这在论文里更稳，因为它不提前暗示“heavy 一定更好”。

---

## 公式里也建议统一

原来如果写：

V(a∣s)V(a|s)

可以保留数学符号 aa，但正文里解释成：

> aa denotes a candidate intervention.

中文：

> aa 表示一个候选干预。

更进一步，甚至可以把符号改成：

V(δ∣s)V(\delta\mid s)

其中：

δ\delta

表示一次干预。

这样视觉上就不会让读者联想到 action。

例如：

V(δ∣s)=L(s)−L(s⊕δ)\boxed{ V(\delta\mid s) = \mathcal L(s)-\mathcal L(s\oplus\delta) }

中文可以定义为：

> 在状态 ss 下施加干预 δ\delta 后，任务损失相对于原始状态的变化，定义为该干预的边际任务价值。

这个表述我很推荐。

---

## 联合干预也可以这样写

原来：

I(a,b∣S)I(a,b|S)

建议改成：

I(δ1,δ2∣s)=V({δ1,δ2}∣s)−V(δ1∣s)−V(δ2∣s)\boxed{ I(\delta_1,\delta_2\mid s) = V(\{\delta_1,\delta_2\}\mid s) - V(\delta_1\mid s) - V(\delta_2\mid s) }

称为：

> **干预交互效应**

而不是“动作交互”。

---

## Graph 那一部分也会更自然

不要写：

> Graph models relations among actions.

改成：

> Graph models relational context among candidate computation states and interventions.

中文：

> Graph 用于建模候选计算状态及其对应干预之间的关系上下文。

然后 Value 写成：

V(δ∣s,Gs)V(\delta\mid s,G_s)

就是：

> **关系条件下的边际干预价值**

---

## RISE 那部分

也不要写：

> future action value.

建议写：

> **future marginal intervention value**

中文：

> **未来边际干预价值**

或者更凝练：

> **未来计算价值**

公式：

Vfuture(δ∣s,Gs).V_{\mathrm{future}}(\delta\mid s,G_s).

这样和 Graph 也统一。

---

## 对整篇论文，我建议固定下面这套中文术语

以后尽量一直用：

> **观测干预**：改变输入证据\
> **计算干预**：改变网络内部一次计算路径\
> **候选干预**：当前允许评估的一个合法变化\
> **单项干预**：一次只改变一个位置/一个计算决策\
> **联合干预**：同时改变多个位置/计算决策\
> **干预效应**：施加干预后任务损失的变化\
> **边际任务价值**：该干预对任务性能的边际贡献\
> **干预交互效应**：多个干预联合效果超出单项效应之和的部分

这样就不会和：

- 动作类别
- 动作实例
- 动作边界

这些 TAD 术语冲突。

我认为最适合你这篇论文的主句可以直接改成：

我们研究的不是“哪些位置看起来重要”，而是“对某个候选干预额外投入计算后，最终任务能获得多少边际收益”。\boxed{ \textbf{我们研究的不是“哪些位置看起来重要”，而是“对某个候选干预额外投入计算后，最终任务能获得多少边际收益”。} }

这会比“action value”清楚得多。
