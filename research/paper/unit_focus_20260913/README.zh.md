**计算量与性能分布图：参考 Uni-AdaFocus 图9/10**

采用当前已保存的47次完整THUMOS测试与4个固定参照。横轴为完整模型的实际GFLOPs/768候选帧完整窗口（2 MAC），纵轴为211视频/792窗口上的mAP@0.3:0.1:0.7。原论文是视频识别，数值不与我们的TAD指标混合；这里只参考呈现方式。

01：全部已测点与低计算区放大。虚线是不同模型/检查点的已观察上包络，不是单个模型的连续预算曲线。

02：严格同checkpoint预算曲线。每个骨干都固定J01 epoch5 EMA、seed3407；每条线固定D/S，仅改变T384→768，共8点。绿色括号均由真实端点计算，不表示延迟加速：S为1.527倍计算量差，低计算点mAP高0.030个百分点；B为1.623倍，低计算点mAP高1.051个百分点。单checkpoint结果不宣称统计显著，也不作为新80轮模型的最终结论。

03：相同计算设置下的checkpoint性能分布。连线沿实际训练epoch，EMA与online区分，没有虚构多seed置信区间。

未测的新论文模型、ANet、InternVideo不补点。旧16候选clip路线不重新纳入。历史H65-S 65.3857%的高分保留在项目协议中，本图不为它填造配对FLOP值。

参考：[Uni-AdaFocus论文图9/10](https://arxiv.org/html/2412.11228v1)。实际查看了PDF第11页。

数据与每点路径：plot_data.json；口径与数值核验：manifest.json。PNG用于展示，SVG和三页PDF用于论文排版。

使用本次固定数据重绘：`python tools/paper_focus_plot.py --records research/paper/unit_focus_20260913/plot_data.json --output research/paper/unit_focus_20260913`
