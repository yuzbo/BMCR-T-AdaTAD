# 本轮完成实验的独立交叉审核

2026-09-15。G0b、G1、RISE-A0/B0与同视频8/8均按不可变capture/consumer版本记录。讨论任务与两个只读子代理独立核验B0、8/8代码及完整JSON；主代理按报告行号点验关键结论，不重复训练。

- **8/8实现正确，未支持扩样条件。** 按物理(insert,remove)排序后交替划分，未用gain选动作；fit8归一化，PlainM 2000steps×seeds42/43/44，共25个有效fit state。held regret .00149207135，对比STOP .00141353726；rho .04444444。held chosen gain的三seed方向不一致，均值为负。报告没有独立held随机控制或其video CI，不能伪称已测。拒绝再跑一遍同样8/8作为下一步。
- **B0实现与FAIL判定一致。** 连续Adam/累计真EMA；完整函数从同s40重新计算；Current=Post；β只在cal选择。β1.1的Future−Post regret及CI都是0，Future−EMA mean+1.4154613e−5、CI[0,4.2463839e−5]。未发现会反转结果的代码或统计错误。只称历史离线跨视频诊断，不能解锁FVD。
- **混capture等价范围明确。** 2ca4d4b与3a7e93f的CF执行差异仅序列身份比较，另有AP末端字段修复；真实JSON roundtrip通过，修改physical frame或support仍拒绝。CAPTURE_EQUIVALENCE.json只适用于这两个版本的label forward，不扩展成AP或FVD通过。
- **G0b范围修正。** 已保存的GT辅助cal20/AP结果有效；旧报告字段`task_course_eligible`只编码headroom。当前代码改名`headroom_gate_passed`，无前向、标签或AP数值变更。没有自动调度消费者依赖旧字段；正式课程仍要Plain可学性和完整技术准入共同通过。
- **固定descriptor诊断已完成。** 讨论任务独占执行，主代理完整读140行脚本并点验结果：seed42已存fit8归一化、原8/8划分；25state/200近邻，无精确碰撞。近邻对随机归一化gain差−.006311，video95% CI[−.029751,+.020092]，p=.334。实现符合冻结协议，没有新训练、模型前向或CF查询。未检出该固定度量下局部一致性；无精确碰撞不证明表示充分，近邻负结果也不证明所有函数不可学。不重复诊断或提前扩32→64。

原始独立报告与精确数字见 `evidence/`；工作任务入口为 codex://threads/01a0a0b2-3d36-76b1-839b-bdb98ae48c0e。任何mini FAIL都只描述该次数据、表示和协议；完整最终路线仍未被证明。
