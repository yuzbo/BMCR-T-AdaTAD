**逐项复核：60项项目判断与吸收方式**

①指第一份`RESEARCH_REPORT.zh.md`，②指第二份`REPORT.zh.md`；A/B/C/D/E/F对应原报告章节，H为实施交接。对重复主张合并判定，但保留两份出处。这里的“确认”表示源码或已有记录支持，并不表示未来实验已取得收益。16项文献单列在[LITERATURE_RECHECK](C:/Users/skywalker/Documents/ChatGPT/H65/h65_clean_adatad/reviews/20260912_external_239d098_dual_review/LITERATURE_RECHECK.zh.md)。

|编号|原建议位置与要点|复核与对比|吸收方式/证据|
|---|---|---|---|
|F01|①A1、②A1：已有成绩和BMCR增益|确认S/B官方、旧H65、旧BMCR及修正H65数字；“BMCR最好”限已完成新模型B|旧结果与修正配方分别列；[旧结果][legacy]、[修正结果][fidelity]|
|F02|①A1、②A1：LR修复不追溯旧权重|正确；15280e5权重仍属旧配方|重新完成匹配BMCR训练，不重命名旧结果|
|F03|①A1、②A1：历史S65.385724|正确，90轮global192合同；不是04c35a3 local8|作为历史参考，不能当本轮复现终点；[当前上下文][context]|
|F04|①A1、②A1：S只有5轮完整测试|对21:56快照正确，现在过时|更新为完整测试5..30轮、保存37轮；[完整回执][live]、[进度][progress]|
|F05|①A5、②A1/A2：global→local损失1.4781pp|确认，同权重改变TIA连接，FLOPs不减|只归因这一路径的图转换，不等于所有全局语义损失|
|F06|①A1/A5、②A1：Z16/24/36预算曲线|确认43.2835/56.1065/62.5804%；零新增训练，均匀集合不必嵌套|完整clip仍可研究；不能和已训选帧直接作单因素结论；[路由][routes]|
|F07|①A1、②A1：Z24优于ZR24|确认当前带锚点随机控制差3.1043pp|支持检查覆盖，不能推断所有随机或adaptive都差|
|F08|①A3、②A3：T24A未测H8/MLP|正确，24/24深度且space1|T24A低分归因不能包含未执行分支；[装配][native]、[执行][engine]|
|F09|①A1/E2、②E2：半FLOPs不代表加速|正确，旧BMCR更慢；Z24模型快但完整流程接近|GPU与端到端口径分别报告；旧51.21ms不与新67.87ms混用|
|F10|①D6、②D5：重点保护短动作/严格IoU|研究动机合理，但新回执也有明显中长动作退化|保留多时长及排序分析；T24A30比Z24中/长召回−13.02/−11.96pp，不预设短动作是唯一原因|

|编号|原建议位置与要点|复核与对比|吸收方式/证据|
|---|---|---|---|
|B01|①A2、②A3：ASFormer LR已修复|源码支持按真实参数身份分组；不是当前未修bug|新预检核真实组即可，不重做无关审计；[runtime][fullruntime]|
|B02|①A2、②A3：裁剪合法端点已修复|loader委托原裁剪再附validity，辅助目标过滤伪端点|保留已修语义，新增数据路径时再查回归；[data][data]、[目标][objectives]|
|B03|①C1/D0、②C1/D0：修正warm可复用|正式joint严格读warm EMA、20轮/2000更新、fidelity_revision|新增joint40，不重复warm；条件隐藏层不全零，只有输出末层零；[初始化][traininit]|
|B04|①A2/H1：preflight绕过真实初始化|确认33–42行`joint and not preflight`同时跳warm及scale；②摘要未突出此点|第一份此项必须保留；共同初始化供正式训练与预检用，不据此声称正式加载失败|
|B05|①A6、②A3：S0单交换与S1多点重解码|确认，标签在S0，正式检测可用S1|先记录S0→交换和S0→S1真损失；[route][bmcrroute]、[label调用][bmcrlabels]|
|B06|①A6、②A3：反事实方向和双边标签|正确；成员/非成员方向不同，互为伙伴才复用相反标签|保留，不重新当缺陷；[utility][utility]|
|B07|①A6、②A3：分类固定目标/定位重匹配|正确；源码已有rematch对照量，不能说完全没有诊断基础|在新audit读取这些量并比较真实训练损失，不将fixed cls直接等同mAP效用|
|B08|①A6、②A3：dummy/cap/FP/排序局限|真实漏检保留在分母，但双方都漏检可饱和；局部代价不完整表示跨视频AP|记录饱和和未匹配FP，不先修改dummy常量；[utility][utility]|
|B09|①A6/D0、②A3/D0：新scale审计|正确，修正warm改变标签分布；旧尺度不适合作新实验依据|用训练视频审计与校准；diagnostic fit/holdout均来自200训练，不改为160/40协议|
|B10|②A3/C1/H：condition向量化与teacher_rows|循环确实存在；tie、selected_mean与hidden梯度须保留；teacher_rows是batch另一半uniform companion|性能优化与算法收益分开；精确后可复用旧checkpoint；[condition][condition]、[train_batch][bmcrlabels]|

|编号|原建议位置与要点|复核与对比|吸收方式/证据|
|---|---|---|---|
|D01|①A7、②A3：D1没有mixed学生GT|确认，GT梯度只在detached teacher feature上求，aux训练为dense拟合|加入独立mixed loss入口；保留当前D1课程；[loss][loss]|
|D02|①A4/A7、②A3/A4：绝对梯度残差非有符号收益|确认，取abs且固定teacher状态，取消通道抵消和方向|作为proxy保留其身份，新route用实际动作标签校准|
|D03|①A7、②A3：0→12不能替代0→8|正确，head0对H0到F12，而DAD第一次是进8层|当前T24A先测0→12，DAD后独立标定0→8、8→12|
|D04|①A7、②A3：MLP近似递归失配|确认训练输入是dense capture，推理晚层输入受前层近似影响|后续训练真实/逐层等价学生状态，不用dense尾缓存冒充；[loss][loss]、[engine][engine]|
|D05|①A7/A8、②A3：H0/H8新头无强初值保证|正确，H0并非Z24初始化；H8残差输出也未必等于F12|新建ResidualCompletion，旧D1模型保留；[aux][aux]|
|D06|②B1：轻状态需顺序/运动信息|具体支持：H0每对帧mean，确定性eval对内交换不变|作为独立轻运动编码实验，不能据结构推算mAP损失；[aux:31–39][aux]|
|D07|①A7、②A3：EMA滞后解释|算术正确；新3000步初值系数4.97%，只凭第5轮60.64%解释已不充分|同route分开online/EMA填充与路由；aux训练时有BN更新；[EMA][ema]|
|D08|①F3、②A3：零logit不等于uniform|DS3 adaptive stable tie下确认`[0..20,28,38,47]`|显式uniform先验；不把反例说成已训路由实测，不混同BMCR连续率sampler|
|D09|①A2/H、②F：uniform嵌套深层随机|分支确实存在，现行注册policy未命中不等深度uniform组合|扩展该对照时定义均匀嵌套；保留历史语义即可，不能解释T24A低分；[routes:79–86][routes]|
|D10|①C2、②C2：local缓存等价|只在同原clip/相位/增强/padding/eval无跨clip耦合下成立|先小规模cache-vs-compact核对，随后减少标签骨干重算；不要求不同GEMM布局逐bit一致|
|D11|①C2、②C2：global与稀疏MLP不能复用dense最终cache|正确，跨clip及递归状态改变|只能缓存合法共享前缀/固定local结果；不引入泛化缓存框架|
|D12|①F3、②H：teacher冻结仍保留feature梯度|当前loss接口支持；teacher强制eval，detector eval分支不更新normalizer|C2复用接口，学生周围不能no_grad；aux本身不是冻结eval；[teacher][teacher]|
|D13|①C2/F、②C2/H：KD匹配sigmoid和距离|正确，头是独立sigmoid与非负左右距离，不是互斥softmax/分布回归|GT先行；新增global/KD单独对照；[anchor-free head][head]|
|D14|①C4/F2、②A2/F2：时间轴和部分窗|官方native384/orig768、BMCRnative192/rank384、DS3local8不可静默互换；有效观察≠物理槽位|复用真实frame_inds；invalid处保留历史padding，元数据不能改变输入合同|
|D15|①C2/F、②C2/H：拆route/execute/reconstruct|当前native耦合route与fill，online loader存在但四格固定route不足|最小扩展装配函数与route输入，不为此重写所有policy；[native][native]、[load_aux][ema]|

|编号|原建议位置与要点|复核与对比|吸收方式/证据|
|---|---|---|---|
|M01|①C1、②C1：强BMCR/H65底座渐进压缩|合理，保持K384/native192/global192/rank384可隔离内部压缩|保留成果线，但不要求所有新模型必须继承旧全部模块|
|M02|①C2、②C2：插值加缺失位置零残差|合理且初值可验证；它继承Z24函数，不继承BMCR高分|首选新原型，重特征位置不被重写；预览额外成本仍计|
|M03|①C3、②C3：全时间浅状态+global重更新|有预算和机制依据，尚无项目精度验证|作为中期候选；全量patch不是充分语义表示，不能承诺保精度|
|M04|①A8/C、②A4/C：全预算等价保证|只限同输入/权重/运算/状态合同；减少重算后不成立|先全重identity，再训练半预算；不把少观察无条件保证当研究目标|
|M05|①C3、②C3/H：原Adapter含identity|确认`return x*gamma+inputs`，Block依次attention、MLP、Adapter|global engine不能再x+adapter(x)；[上游Adapter][adapter]|
|M06|①C3、②C3/H：TIA需要空间池化前状态|确认GELU/投影在每个格点上，mean与非线性不交换|每层回写完整空间格再TIA；不能只对池化native串时间|
|M07|①C3、②C3：selected-Q/full-KV|合理候选；Q和KV角色不同，attention图必须训练/推理一致|优先先MLP/原clip，后Q；不能按Q比例平方计整个attention|
|M08|①C1/C2、②C1/C2：迁移BMCR/scout知识|应迁移同合同权重或物理时间预测/先验；跨native轴直接MSE不成立|先固定底座，clip效用重新校准；避免重复两套昂贵scout|
|M09|①A7/C3、②C3/H：C2/J3命名|报告用法存在参数更新范围的松动，不能只靠名字判断实验一致|每实验明确teacher、学生TIA/head是否训练及学生执行图；global可训练学生不能冒称当前冻结D1|
|M10|①C4、②C0：多种采样粒度|分析合理，但attention支持、配对与观察数常同时改变|先patch之后16/8计算分组，后RGB删除；按等观察/等FLOPs分开|

|编号|原建议位置与要点|复核与对比|吸收方式/证据|
|---|---|---|---|
|E01|①D0、②D0：先修正BMCR|配方对照必要；不意味着所有便宜诊断必须等它完成|P0与P1并行准备，遵守已有资源限制，不重启旧任务|
|E02|①D1、②D1：四格与EMA|第二份固定route语义更精确|先S30，复用已完成Z24/T24A；EMA route/fill分别切，避免一个参数同时换两因素|
|E03|①D2、②D2：D1 vs 残差C2|总体方向赞同，但同时换模块、输入、初值、loss不能归因GT|同参数化同父状态做成对loss对照；不同参数化差异不能全叫初始化收益|
|E04|①D3、②A3/C2：条件utility|①把它列独立阶段更清楚，②补S1与动作细节|合并吸收，填充器固定后再路由；仅训练诊断oracle，推理无GT|
|E05|①D4、②D3：粒度扫描|②16/8起步更省、更易解释|不用第一轮全1/2/4/8/16、重叠和长短段网格；必要时扩展|
|E06|①D5、②C3/D4：128/D8简单对照|第一份在实验排序更突出，第二份仍提但不宜略过|全状态复杂结构前测简单适配对照，近似算量不同要披露|
|E07|①D6、②D5：错误集和bootstrap|正确；top100时长召回已有，无完整遮挡/多主体标签|按视频重算总体AP，不平均视频AP；不声称bootstrap消除训练方差/测试选模偏差|
|E08|①F4、②F4：扩大/停止依据|支持按具体失败改对应因素，不接受无限延训或全组合|当前80轮保留；新支线先短程配对，结论依真检测与成本，不设任意1pp门槛|

|编号|原建议位置与要点|复核与对比|吸收方式/证据|
|---|---|---|---|
|C01|①E1、②E1：S/B MAC公式|两包CPU复算均通过；S1173.9470/B4041.0774 GMAC|bone由公式，head部分沿用记录，不能说全模型已独立profile|
|C02|①E1、②E1：TIA和晚4层MLP份额|S TIA3.2670%；MLP75/50理想省3.8586/7.7173%；①扣近似器后更完整|小机制试验有用，大幅加速不能依赖它|
|C03|①E1、②C3/E1：D8/128/global半重算量|口径一致：D8不含出口67.7202%，含出口67.7251%；128为58.8869%；全局半重约53.21%|都是估算；必须完整计路由/预览/重建，不能填新精度或时延|
|C04|①E3、②E1：三维保留率不相乘|正确，层是否进入、attention二次项、固定TIA/head不能相乘|按实际执行clip/token/层累加，再测完整GPU时间|
|C05|①E4、②E4：状态内存|S/B单BF16空间格28.125/56.25MiB，和池化cache相差100倍|不是训练峰值；不无条件保存12层或所有MLP中间量|
|C06|①E2/E3、②E2/E3：分段计时/p95|正确，20样本尾延迟不稳定，解码/传输/NMS尚无独立归因|同卡/封装交错测，保留原样本；异步组件不可简单相加当E2E|
|C07|①E4、②E4：teacher和跨窗cache成本|正确，local训练查询缓存与推理跨窗复用是两件事|先复用F12训练标签，F8按需；跨窗复用后置且实测上下文与命中收益|

两份toy中的rank-IoU例子分别是0.5→1/6和1/3→1/20，是不同区间/映射，不构成数值矛盾。最大空洞的一个脚本报“中间空clip数”，另一脚本报“相邻clip ID间隔”，相差1也是定义差异。两个脚本不导入项目，不证明GPU执行、hard RGB梯度或检测精度。

[legacy]: C:/Users/skywalker/Documents/ChatGPT/H65/h65_clean_adatad/ds3_20260912/phase2_20260910/comparison.json
[fidelity]: C:/Users/skywalker/Documents/ChatGPT/H65/h65_clean_adatad/ds3_20260912/fidelity_20260911/FINAL_COMPARISON.json
[context]: C:/Users/skywalker/Documents/ChatGPT/H65/h65_clean_adatad/ds3_20260912/ds3_20260912/research_context_20260912/CONTEXT.zh.md
[live]: C:/Users/skywalker/Documents/ChatGPT/H65/h65_clean_adatad/reviews/20260912_external_239d098_dual_review/evidence/live_snapshot.json
[progress]: C:/Users/skywalker/Documents/ChatGPT/H65/h65_clean_adatad/reviews/20260912_external_239d098_dual_review/evidence/progress_snapshot.json
[routes]: C:/Users/skywalker/Documents/ChatGPT/H65/h65_clean_adatad/ds3_20260912/h65/ds3/routes.py:57
[native]: C:/Users/skywalker/Documents/ChatGPT/H65/h65_clean_adatad/ds3_20260912/h65/ds3/model.py:175
[engine]: C:/Users/skywalker/Documents/ChatGPT/H65/h65_clean_adatad/ds3_20260912/h65/ds3/model.py:125
[fullruntime]: C:/Users/skywalker/Documents/ChatGPT/H65/h65_clean_adatad/ds3_20260912/h65/full/runtime.py:101
[data]: C:/Users/skywalker/Documents/ChatGPT/H65/h65_clean_adatad/ds3_20260912/h65/full/data.py:14
[objectives]: C:/Users/skywalker/Documents/ChatGPT/H65/h65_clean_adatad/ds3_20260912/h65/full/objectives.py:17
[traininit]: C:/Users/skywalker/Documents/ChatGPT/H65/h65_clean_adatad/ds3_20260912/tools/full_train.py:33
[bmcrroute]: C:/Users/skywalker/Documents/ChatGPT/H65/h65_clean_adatad/ds3_20260912/h65/full/model.py:45
[bmcrlabels]: C:/Users/skywalker/Documents/ChatGPT/H65/h65_clean_adatad/ds3_20260912/h65/full/model.py:74
[utility]: C:/Users/skywalker/Documents/ChatGPT/H65/h65_clean_adatad/ds3_20260912/h65/full/utility.py:8
[condition]: C:/Users/skywalker/Documents/ChatGPT/H65/h65_clean_adatad/ds3_20260912/h65/full/scout.py:63
[loss]: C:/Users/skywalker/Documents/ChatGPT/H65/h65_clean_adatad/ds3_20260912/h65/ds3/losses.py:41
[aux]: C:/Users/skywalker/Documents/ChatGPT/H65/h65_clean_adatad/ds3_20260912/h65/ds3/auxiliary.py:31
[ema]: C:/Users/skywalker/Documents/ChatGPT/H65/h65_clean_adatad/ds3_20260912/h65/ds3/runtime.py:64
[teacher]: C:/Users/skywalker/Documents/ChatGPT/H65/h65_clean_adatad/ds3_20260912/h65/ds3/model.py:98
[head]: C:/Users/skywalker/Documents/ChatGPT/H65/h65_clean_adatad/ds3_20260912/upstream/opentad/models/dense_heads/anchor_free_head.py:144
[adapter]: C:/Users/skywalker/Documents/ChatGPT/H65/h65_clean_adatad/ds3_20260912/upstream/opentad/models/backbones/vit_adapter.py:57
