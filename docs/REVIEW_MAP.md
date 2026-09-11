# 外部代码审查导航

从实际调用链读取，不要只看旧的 `h65/model.py` 或早期原型测试。

| 入口/符号 | 需要核对的语义 |
|---|---|
| `tools/full_train.py:main` | 识别预训练、20/40轮、共享warm EMA、恢复、成功更新计数、teacher频率 |
| `h65/full/runtime.py:config, optimizer_for, EMA` | 官方配置继承、冻结参数、LR分组、EMA参数/浮点与整数buffer |
| `h65/full/objectives.py:curriculum, auxiliary_losses` | 学习采样与梯度/贡献课程，半开GT动作目标，端点高斯和边界约束 |
| `h65/full/scout.py:FormalScout.forward` | ASFormer动作/转变路径，空间stem detach，辅助replay与RNG |
| `h65/full/scout.py:FormalScout.condition` | 成员身份、16候选cell显式伙伴、集合条件特征、不可行位置 |
| `h65/transport.py:calibrated_rates, sample_rates, gather_with_transport` | K守恒、有序真实观察、短窗口/填充、手工代理梯度、hard forward |
| `h65/full/model.py:route, train_batch` | S0与一次修正、效用标准化/均值、uniform companion、检测和辅助损失 |
| `h65/full/utility.py:counterfactual_targets` | 选中/未选候选，互反配对条件，保留/插入符号，no_grad目标 |
| `h65/full/utility.py:classification_cost, localization_cost` | 分类固定指派，原始时间定位、漏检dummy、固定/重匹配差别 |
| `h65/full/geometry.py:TrueTimeMap` | rank/真实时间端点扩展和回映；不等同于修复tubelet采样几何 |
| `h65/full/model.py:predictions, raw_route` | EMA干预真实前向与推理GT隔离，预测回映时点 |
| `tools/full_eval.py:OfficialRuntime, ArithmeticCounter, profile_windows, main` | 官方权重、211视频测试、NMS流程、融合注意力MAC、固定窗口计时 |
| `tools/full_compare.py` | 六项原始分数与计算量的比值，均值/中位数、比较口径 |
| `tests/` | 已覆盖的合同和语义；不能把局部测试等同于方法全正确 |

现有读取未发现必须立即更改算法的确定性关键错误。需要进一步独立判断的点包括：局部one-swap效用对最终集合的泛化；S0标签与修正集合分布差别；标准化后cls/loc等权与unit gain；局部伙伴和无可行交换位置的零修正；代理梯度与实际离散目标的关系；不规则tubelet/时间坐标的表示损失。这些是待验证问题，不是预先认定的bug。

所有真实测试使用selected-rank路径和纯净global-TIA核心。原型 `h65/model.py` 与旧结构文档不构成本轮结果的运行入口。真实GPU预检的 `runs/preflight_*/completed.json`、四个完整训练日志与六项完整评测记录补充了CPU测试的覆盖范围。
