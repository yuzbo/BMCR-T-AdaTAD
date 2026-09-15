# 可逆帧交换一致性 mini：输入、对照与解释边界

2026-09-15。依据用户提供的三份新审阅材料、实施任务的最新范围说明，以及 b644d870d1845abbc1e4fd5ab7780f29ff96a53a 的现有底座代码。三份材料以更早的 6d1f924 为审阅对象，不能覆盖已经完成的 R1 及后续 packing 发现。

**认可此次唯一结构 mini：P1 双向分别监督，P1-bi 复用 P1 权重做双向差分推理，P2 直接监督差分输出。保留已有 R1 FAIL；不回退重跑 R1；此前共同 held4 覆盖拟合与 partner96 表示修改暂缓。** 这是下一实验的协议判断，不是新实现或科学结果已经通过。

本任务只读核对并记录，未修改模型/划分、启动拟合、查询新标签或调整服务器队列。

## 1. 固定检测函数下的真实结构

对于当前固定 episode、RGB/增强、checkpoint、D/S/Cross/readout 和 loss normalizer 的立即帧交换：

\[
S'=S-\{i\}+\{j\},\qquad
\mathbf g^+=\mathbf L(S)-\mathbf L(S'),\qquad
\mathbf g^-=\mathbf L(S')-\mathbf L(S)=-\mathbf g^+.
\]

分类与定位两个分量分别满足该关系。当前 bank 的 continuation 明确为一次强制 T 交换后完成固定检测函数，不再进行其他 T 交换，见 [bank.py:168](C:/Users/skywalker/Documents/ChatGPT/H65/h65_clean_adatad/rfv_sprint_20260915/h65/rfv/bank.py:168)。

所以此处反对称约束有准确的代数依据，但没有证明它能够解决泛化。该关系不自动覆盖改变剩余预算、执行历史或后续策略时点的其他局部计算选择。

正反向真实执行校验不能只计算“已存 g 加上人工取负的 g”；那只能验证代数恒等式。应使用少量真实 S→S'→S 重放，核对原/返回支持集、固定函数与 normalizer，并将分类/定位残差分别与该执行路径的 no-op/replay 误差比较。

## 2. 反向输入必须来自真实 S'

正确目标为：

\[
x^+=\phi(C,S,i\to j),\qquad
x^-=\phi(C,S',j\to i).
\]

若有原始完整廉价输出，可调用现有 swap(current,i,j,proposal) 后再调用 descriptors(...,selection=changed,pairs=[(j,i)])。合法支持往返由 [contracts.py:148](C:/Users/skywalker/Documents/ChatGPT/H65/h65_clean_adatad/rfv_sprint_20260915/h65/raw/contracts.py:148) 的成员与候选约束定义。不能保留旧 S，也不能在 S' 上仍传 (i,j)，却把它命名为反向描述子。

从已有 407 与公开 metadata 构造时，各部分应如下处理：

|部分|反向处理|
|---|---|
|remove/insert hidden|交换两个 96 维块|
|support mean|按真实 S' 更新；有效支持数为 K_valid 时，m'=m+(h_j−h_i)/K_valid|
|global mean|不变|
|remove/insert time、preview gap、official/preview provenance、actionness、transition|随端点角色交换|
|signed distance|取负|
|四个 support left/right gap|根据 S' 和新的 remove=j、insert=i 重新计算|
|remove/insert membership|仍为 **(1,0)**，因为 j 已在 S'，i 已不在 S'|
|valid fraction、support density、D/S capacity|固定同容量交换下不变|

依据 [value.py:21](C:/Users/skywalker/Documents/ChatGPT/H65/h65_clean_adatad/rfv_sprint_20260915/h65/raw/value.py:21) 及 [value.py:35](C:/Users/skywalker/Documents/ChatGPT/H65/h65_clean_adatad/rfv_sprint_20260915/h65/raw/value.py:35)。membership 是按当前方向的角色定义，不能机械置换为 (0,1)。独立探子的原说明混用了“固定原端点”与“反向端点角色”，主代理已点验原文并更正。

K_valid 是真实有效支持数，不包括重复 padding；其他几何量继续使用原物理 frame ID、episode bounds 和原插值规则。原 407 已保留这次交换所需的两个 hidden 及均值，因此这次反向构造不需要从 cheap192 恢复任意新帧的完整 hidden，也不需要新增感知输入。

均值增量更新存在 FP32 舍入，少量样本应与原廉价输出上直接重建的 descriptor 比较；同时查看原始误差与经过冻结 normalization 后的误差。不能用有损 cheap192 重建再把差异归咎于增量公式。

反向是合法支持集转换，但未必属于部署生成器在 S' 给出的 16 个候选。把它作为 fit 帧交换的派生监督并不要求扩展部署候选域。每对正反向必须整体继承原 fit/held 归属；反向标签不是新增独立 CF 测量，也不能跨方向随机拆分数据。

当前 swap 拒绝 i=j；no-op 保持为单独的重执行/函数测试，不插入合法帧交换候选。差分输出的 no-op=0 要求两次输入完全相同且 head 为同一确定性函数；这不同于声称所有真实执行误差严格为零。

## 3. P1、P1-bi、P2 的准确比较

令 f 输出 raw cls/loc 收益，q 为原 fit 两个分量的 RMS scale，ρ 为当前 masked、component-mean Huber：

\[
\begin{aligned}
L_{\mathrm{P1}}
&=\tfrac12\{\rho[(f(x^+)-g)/q]+\rho[(f(x^-)+g)/q]\},\\
\hat g_{\mathrm{P1}}&=f(x^+),\\
\hat g_{\mathrm{P1-bi}}&=[f(x^+)-f(x^-)]/2,\\
L_{\mathrm{P2}}
&=\rho\{([f(x^+)-f(x^-)]/2-g)/q\}.
\end{aligned}
\]

现有 [TemporalProbe.forward:44](C:/Users/skywalker/Documents/ChatGPT/H65/h65_clean_adatad/rfv_sprint_20260915/h65/rfv/value.py:44) 已乘回 target_scale，且 target 不减均值；[value_loss:28](C:/Users/skywalker/Documents/ChatGPT/H65/h65_clean_adatad/rfv_sprint_20260915/h65/rfv/objectives.py:28) 先平均两个分量，再对合法候选取均值。因此正反方向可以共享原 q，推理与 STOP 继续使用 raw cls+loc，不能直接把标准化分量之和当实际收益。

P1/P2 使用同 407、同共享 head、同初始化/seed、同原 fit normalization、同 state batch 顺序、同 2000 updates、同 valid mask 和 Huber reduction。不要把 reverse 作为另一个 state 再多采样一轮，也不同时叠加已测 R1 的 JS 目标。

**P1-bi 是同一 P1 权重的另一个读出，零新增拟合、零 optimizer/EMA 更新。** 现有评估使用 raw model，故本轮 P1/P1-bi 对照也应锁定同一 raw snapshot 与 normalization；不能混用 P1 raw 与 P1-bi EMA。

P1-bi 与 P2 的推理公式都反对称。准确的比较解释是：

- P1→P1-bi：在同一训练结果上，施加双向差分读出的影响；不能未经测量就全部称为方差降低。
- P1-bi→P2：双向分别监督与直接监督差分输出之间的训练差异；不是“只有 P2 具有反对称推理”。

更具体地，定义归一化奇部 \(a=[f(x^+)-f(x^-)]/(2q)\)、偶部 \(b=[f(x^+)+f(x^-)]/(2q)\)、目标 \(t=g/q\)，则

\[
L_{\mathrm{P1}}=\tfrac12[\rho(a-t+b)+\rho(a-t-b)],
\qquad L_{\mathrm{P2}}=\rho(a-t).
\]

在 Huber 的二次区间，前者额外约束偶部 b。P2 不要求分别校准两个方向的自由输出；共享最终 bias 在差分中相消。因此注册参数数目相同不意味着所有参数梯度相同，也不意味着 P2 保证比 P1-bi 更好。

P1/P2 训练均使用两次 head 前向；P1 推理只需正向，P1-bi/P2 推理需要反向输入构造与两次 head 前向。实际成本分别报告，不以同 K 声称全系统严格同成本。

## 4. 原 fit normalization 的可达影响

当前冻结规则继续遵守，不擅自改成双向重算统计。但它确实会改变反向输入的数值尺度。

[已有支持范围诊断](C:/Users/skywalker/Documents/ChatGPT/H65/reports/rfv_sprint_20260915/R1_WITHIN_SUPPORT_DIAGNOSTIC.json) 中，原 within fit 的 signed_distance 全正，范围 0.00130378–0.00150830，fit scale=0.0000401771。反向取负后，以原正向均值标准化，约落在 −65 至 −75 个原 scale 的范围内。

这是本项目实际数据下可达的数值条件，不能用上一轮“held 的常量列没有变化”的结论排除。它不表示反向 descriptor 错误，也不是 held 泄漏；P1/P2 同受此条件影响。实施方应随 mini 记录反向输入幅度、输出/梯度有限性与 fit 表现，避免把数值条件差异当成已唯一确认的表示或结构根因。若出现非有限值或确切优化失败，应单独报技术问题，不将其包装成结构科学 FAIL；不因此自动扩大归一化网格。

## 5. 判读和当前范围

主比较应包括 P2−P1-bi，同时保留 P1 及 STOP/随机合法交换。报告 raw regret、chosen gain、负收益执行率及逐 seed 方向；ranking 指标为辅。相同原 state/candidate 的视频配对统计保持，不能把正反向当独立样本将置信区间人为缩窄。

原 8/8 的 packing 位置族外推仍存在；旧 inner 已用于开发，outer20 继续封存。mini 的新结果必须单独命名，不修改 R1 FAIL，也不自动解锁 Graph/FVD/T 长训。

RISE 已有 snapshot 选择/margin 导出与冻结 D/S 的策略切换、分组梯度/clip 诊断可由实施任务独立推进；它们不进入此次 Value 的新输入或训练目标，也不修改旧课程。

本轮主代理完整阅读三份研究材料，两个只读子代理分别检查 descriptor/支持集合接口与训练/统计接口；主代理对关键原文点验，并更正 membership 角色混淆。已向“实现 Raw-v1 并部署并行实验”反馈上述重点。后续新增反向模块、GPU重放与拟合结果仍需按其实际交付版本评价。

来源：[原审阅规格](E:/下载/AUDIT_AND_NEXT_MINI.zh.md)、[第一份评论](C:/Users/skywalker/.codex/attachments/5fb005b4-06fb-4894-9589-e0ed92a482e2/pasted-text.txt)、[第二份评论](C:/Users/skywalker/.codex/attachments/7e5f1d48-9f44-4401-afc7-4c51cb0a3fce/pasted-text.txt)。
