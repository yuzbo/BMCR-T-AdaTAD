"""Consolidate preserved experiment records without changing any model or run."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'ds3_20260912/retrospective_20260912'
read=lambda path:json.loads(path.read_text(encoding='utf-8'))
legacy=read(ROOT/'phase2_20260910/comparison.json')
training=read(ROOT/'phase2_20260910/TRAINING_SUMMARY.json')
fixed=read(ROOT/'fidelity_20260911/FINAL_COMPARISON.json')
fixed_training=read(ROOT/'fidelity_20260911/TRAINING_COMPLETION.json')
curve=read(ROOT/'fidelity_20260911/INTERMEDIATE_RESULTS.json')
snapshot=read(OUT/'remote_snapshot.json')
hist=fixed['historical_context']
official={r['backbone']:r for r in legacy['results'] if r['method']=='official'}
old={(r['backbone'],r['method']):r for r in legacy['results']}
new={(r['backbone'],r['selection']):r for r in fixed['results']}
scores=['average_mAP']+[f'mAP@{x}' for x in (.3,.4,.5,.6,.7)]
names={'official':'官方 AdaTAD','h65':'旧 H65-C','bmcr':'旧 BMCR-T'}

def table(headers,rows):
    return '\n'.join(['|'+'|'.join(headers)+'|','|'+'|'.join(['---']*len(headers))+'|']+
                     ['|'+'|'.join(str(x) for x in row)+'|' for row in rows])+'\n'

def metric_check(row):
    m=row['metrics']
    assert abs(m['average_mAP']-sum(m[f'mAP@{x}'] for x in (.3,.4,.5,.6,.7))/5)<1e-12

for row in legacy['results']+fixed['results']+curve['results']:metric_check(row)
assert len(curve['results'])==16 and fixed['training_updates_total']==12000
assert training['actual_successful_optimizer_updates']==20000
total_hours=training['actual_training_gpu_hours']+fixed_training['recorded_single_gpu_training_hours']
assert sum(s['successful_updates'] for s in fixed_training['stages'])==12000

# Independent arithmetic used in the tables and accompanying machine-readable data.
changes={}
for b in ('S','B'):
    ref=official[b]['metrics']['average_mAP'];h=old[b,'h65']['metrics']['average_mAP'];m=old[b,'bmcr']['metrics']['average_mAP']
    terminal=new[b,'epoch60_terminal']['metrics']['average_mAP'];peak=new[b,'test_peak']['metrics']['average_mAP']
    changes[b]=dict(bmcr_gain_vs_old_h65_pp=100*(m-h),old_h65_gap_to_official_pp=100*(h-ref),
                    old_bmcr_gap_to_official_pp=100*(m-ref),old_h65_relative_map_loss_percent=100*(ref-h)/ref,
                    old_bmcr_relative_map_loss_percent=100*(ref-m)/ref,matched60_gain_pp=100*(terminal-h),
                    corrected_peak_extra_gain_pp=100*(peak-terminal),corrected_peak_gap_to_official_pp=100*(peak-ref))

complete_ds3={name:data for name,data in snapshot['artifacts'].items() if 'metrics.json' in data}
ds3_rows=[]
for name,data in complete_ds3.items():
    m=data['metrics.json'];metric_check(m)
    assert m['test_videos']==211 and m['test_windows']==792
    p=data.get('profile.json',{}).get('cases',{}).get('full')
    base=snapshot['artifacts'].get(name[0]+'_D768G',{}).get('profile.json',{}).get('cases',{}).get('full')
    ds3_rows.append([name,f"{m['metrics']['average_mAP']*100:.4f}%",
                     f"{p['matrix_conv_flops']/1e9:.3f}" if p else '尚无',
                     f"{p['total_macs']/base['total_macs']*100:.4f}%" if p and base else '尚无',
                     f"{p['model_timing']['mean_seconds']*1000:.2f} / {p['model_timing']['p50_seconds']*1000:.2f}" if p else '尚无'])

all_rows=[]
for b in ('S','B'):
    for method in ('official','h65','bmcr'):
        r=old[b,method];m=r['metrics']['average_mAP'];ref=official[b]
        all_rows.append([b,names[method],'提供权重' if method=='official' else '第60轮终点',f'{m*100:.4f}%',
                         '—' if method=='official' else f"{100*(m-ref['metrics']['average_mAP']):+.4f}",
                         f"{r['matrix_conv_gflops']:.3f}",f"{100*r['gmacs']/ref['gmacs']:.4f}%",
                         f"{r['latency_ms']:.2f} / {r['latency_median_ms']:.2f}",f"{r['peak_gib']:.4f}"])
    for choice in ('test_peak','epoch60_terminal'):
        if b=='S' and choice=='epoch60_terminal':continue
        r=new[b,choice]
        all_rows.append([b,'修正 H65',f"第{r['total_epoch']}轮"+('峰值' if choice=='test_peak' else '终点'),
                        f"{r['metrics']['average_mAP']*100:.4f}%",f"{r['gap_to_official_pp']:+.4f}",
                        f"{r['matrix_conv_gflops']:.3f}",f"{r['flops_fraction_of_official']*100:.4f}%",
                        f"{r['latency_mean_ms']:.2f} / {r['latency_median_ms']:.2f}",f"{r['peak_gib']:.4f}"])

parts=[]
add=parts.append
add('**H65 / BMCR-T / DS3-L 实验记录完整回顾**\n')
add(f"记录截止：{snapshot['timestamp']}。数值来自已保存的评测JSON、训练完成记录和本次只读远端快照；本次整理没有启动额外训练、改写旧分数或改变在运行的80轮配方。\n")
add('**已经成立的结论是：重骨干计算确实减少，官方精度和实际速度尚未同时保持。** 旧 BMCR-T 在两种骨干上改善了旧 H65 的平均mAP，但收益有限；修正学习率和裁剪边界后，H65有所恢复，仍未达到历史65+。当前DS3-L的密集参考、局部转换与零新增训练Z0对照已有评测，不能把它们当作完成80轮D1辅助训练的成绩。\n')
add(f"已完成的旧配方和保真修正两轮主实验，共有6条完整模型训练路线、10个实际训练阶段、32,000次正式更新、{total_hours:.4f}个记录训练GPU小时，以及22次完整211视频评测。共享预热只计一次；历史90轮、初期诊断、GPU预检、审计、测试、排队及当前DS3都不计入上述训练小时。DS3本次快照另有{len(complete_ds3)}项完整参考评测。\n")

add('**实验名称相近，但每一阶段解决的问题和已完成范围不同。**\n')
add(table(['阶段','模型/任务','实际完成内容','不可混淆的边界'],[
 ['历史H65','VideoMAE-S，30+60轮','收回原始65.385724%终点评测回执与训练日志','历史实验；本项目未重新跑这90轮'],
 ['初期机制验证，09-10','纯净OpenTAD上的H65思想原型','一个训练视频的两个重叠窗口、2步最终诊断权重、连接/梯度/换帧检查','无完整训练、无全测试mAP'],
 ['旧配方完整实验，09-10至09-11','S/B × H65-C/BMCR-T','4条60轮路线，6项官方/新模型完整测试及profile','旧LR分组和裁剪边界缺口仍在这些模型中'],
 ['两轮外部复审，09-11','15280e5固定源码与意见包','58项与45项主体复核，源码/实际参数/算术/文献对照','复审结论不是新模型性能'],
 ['保真修正，09-11至09-12','S/B H65，20+40轮','重建新warm，修LR/边界，16次全测试，峰值/终点profile','没有按修正配方重训BMCR，也未完成独立G/F'],
 ['DS3-L，09-12起','S/B D1，80轮计划','辅助模块、0/8/12执行与MLP稀疏实现；S/B真实GPU预检和参考评测','80轮新模型性能尚未产生；C2/J3未实施']]))

add('**所有完整主实验都使用200条训练视频、211条测试视频。** THUMOS14训练视频在资源目录中叫Validation Data；测试目录是TH14_test_set_mp4。程序中test使用的annotation subset名称为validation，不表示拿训练集评测。实测数据ID集合互不相交，测试滑窗共792个，动作类别20。旧记录中的199条说法已被实际文件、标注及加载器核对纠正为200。\n')
add('每轮对200个训练视频各抽取一次标准随机窗口，batch2，故每轮100个优化器更新；这不等于每轮穷举每个视频的全部滑窗。候选步幅是4个原视频帧，768个候选观察不是连续768张原视频帧。mAP均值取tIoU0.3/0.4/0.5/0.6/0.7五个阈值，不能与单独@0.5混用。\n')
add('“完整训练”指完整课程与完整训练集覆盖，不表示所有VideoMAE参数都训练。旧H65/BMCR和修正H65冻结非Adapter的VideoMAE参数，训练Adapter、ActionFormer检测部分及scout；DS3 D1进一步冻结提供的TAD教师和检测器，只更新辅助模块。冻结不代表跳过前向计算。\n')

add('**预热确实训练ASFormer，但它不是官方AdaTAD的完整768均匀训练复现。** 旧配方及修正H65的20轮warm采用均匀K384；Adapter、检测器以及低分辨率CNN+ASFormer的动作/转变辅助任务都训练。后40轮为20轮过渡与20轮完整联合；检测反馈先预热667更新、再放开1333更新。batch2内的均匀companion来自另一行自己的视频/GT，不是同一视频的双视图输出蒸馏。\n')
add(table(['对象','初始化','训练预算','优化与选择'],[
 ['官方 S/B','提供的TAD EMA，checkpoint epoch字段59/51','本项目不重训','仅严格加载并完整测试；字段不推断原始总轮数'],
 ['旧H65-C/BMCR-T','K400识别预训练；新Adapter/检测器','公共warm20一次/骨干；各joint40；6000步/路线','AdamW，EMA0.999，末轮EMA；没有本轮测试峰值选模'],
 ['修正H65 S/B','重新从识别预训练开始，自己的新warm','20+40，各6000步','同类课程；总25至60每5轮完整测试，按测试峰值选EMA'],
 ['DS3-L D1 S/B','提供的官方TAD权重冻结；H0额外ImageNet MobileNetV3Small','80轮，各8000步；100步pilot就是第1轮','aux AdamW1e-4，500步线性LR预热后余弦至8000；每5轮T24A全测试，保留60/80']]))
add('H65/BMCR的Adapter基础LR为S2e-4、B1e-4；检测器1e-4；scout warm的trunk/action/scorer为5e-5/1e-4/5e-5，joint为1e-5/2e-5/5e-5。warm/joint学习率预热分别2/3轮，之后余弦衰减。旧版实际误把18,816个内部投影参数分到两倍的action LR，后面的“修正H65”才按参数身份改正。旧PLAN中“已经在正式训练前正确分组”的表述与实际运行不符，不能继续引用为事实。\n')

add('**已完成的新模型成绩应同时看精度、计算量和时延。** 下表来自两轮完成报告，使用对应官方参考作为分母；修正S峰值也是第60轮终点。FLOPs单位为GFLOPs，时延是平均/中位ms，显存为该次推理的peak allocated GiB。\n')
add('FLOPs按实际执行的矩阵/卷积2MAC计数，包含融合注意力QK和AV，不包含softmax、归一化及所有逐元素算术。旧实验与修正实验的主profile统一使用完整768候选窗口video_test_0000007、起点0、测试index2；另保留503候选部分窗和253候选短窗。计时为batch1、输入已在GPU、BF16骨干/scout和FP32检测器，5次预热后保留20次同步调用；包含路由/骨干/Adapter/检测及NMS前回映，排除视频读取、解码和NMS。不同作业可能位于不同4090节点，因此不是同卡交错实验或全视频端到端时间。\n')
add(table(['骨干','方法','权重选择','平均mAP','对官方差值pp','GFLOPs','FLOPs/官方','模型时延ms','推理GiB'],all_rows))
add('旧H65/BMCR是末轮EMA，修正H65是按用户要求的中间测试峰值；两类选择口径不同。官方权重的训练课程与新模型不同，故官方差距不是“只更换一个采样模块”的单变量消融。所有新路线为seed3407的单种子结果，不据此宣称统计显著性。\n')
add('![已完成主实验的精度和中位时延对照](overview.png)\n')
add('图中修正H65使用测试峰值，旧模型使用终点；历史90轮和当前未训练完的DS3不并入柱形排名。中位时延也仍高于对应官方参考，计算量减半没有在已完成模型中转化为推理加速。\n')
add('S/B的旧H65各有相同18.5338GMAC的scout开销；它在S总计算中的占比更大，因此即使重骨干clip都减半，总FLOPs比例仍分别为51.5488%和50.4499%。这是计算占比的解释，不是S精度下降更多的因果解释。动态索引、校准、同步等耗时也不能按MAC占比直接分摊。推理显存同样没有减半：修正S约0.8613GiB，B约1.1233GiB，分别只较对应参考降低约7.92%和25.51%。\n')

add('**BMCR-T带来的收益集中在有限的平均mAP改善，不能表述为全面保持检测能力。** 该名称是本项目按提案实现的固定预算反事实时序路由方案，不是已经确认发表论文的官方复现。它在同一K384核心上增加候选、相反成员伙伴与集合上下文，用EMA教师测量带符号的分类/原时间定位交换效用，以Huber监督替换旧贡献分布监督；每8次联合更新尝试可行交换，推理不使用GT。\n')
add(table(['骨干','BMCR−旧H65平均mAP(pp)','BMCR对官方差值(pp)','BMCR相对官方mAP损失','@0.5变化(pp)','@0.7变化(pp)'],[
 [b,f"{changes[b]['bmcr_gain_vs_old_h65_pp']:+.4f}",f"{changes[b]['old_bmcr_gap_to_official_pp']:+.4f}",f"{changes[b]['old_bmcr_relative_map_loss_percent']:.4f}%",
  f"{100*(old[b,'bmcr']['metrics']['mAP@0.5']-old[b,'h65']['metrics']['mAP@0.5']):+.4f}",
  f"{100*(old[b,'bmcr']['metrics']['mAP@0.7']-old[b,'h65']['metrics']['mAP@0.7']):+.4f}"] for b in ('S','B')]))
add('百分点与相对百分比不同。例如旧H65-S低于官方6.1532个百分点，相对mAP损失8.9161%；B低4.5152个百分点，相对损失6.3480%。S的相对差距更大是真实观测，尚不能单独归因于小模型容量、时间表征、LR错误或课程长度。较严格阈值下的损失更明显也不能直接等同于纯回归错误，因为排序与召回同样参与mAP。\n')
add('训练集上的128窗口交换审计，S/B各得到390次可测交换，310次用于拟合/尺度估计，80次来自15个留出训练视频。旧绝对贡献代理在留出交换上的分类/定位Spearman为S−0.0302/−0.0283、B0.1259/−0.2034，弱且不一致；这不是最终BMCR效用头的精度。每个BMCR联合阶段日志各有355次非空事件，可重建1065条教师路由前向；它是前向行数，不是训练FLOPs或耗时百分比。\n')

add('**两项保真修正有实际收益，但没有解释完整历史差距。** LR修复将四个attention内部输出投影归回trunk LR，真正两个动作分类头只有388个参数；裁剪边界修复通过gt_boundary_validity排除随机窗口制造的假起止点。两者同时应用，没有分开的消融，因此不能分摊各自mAP贡献。\n')
add(table(['骨干','旧H65第60轮','修正第60轮','相同终点变化pp','修正峰值','峰值选择额外变化pp'],[
 [b,f"{old[b,'h65']['metrics']['average_mAP']*100:.4f}%",f"{new[b,'epoch60_terminal']['metrics']['average_mAP']*100:.4f}%",
  f"{changes[b]['matched60_gain_pp']:+.4f}",f"{new[b,'test_peak']['metrics']['average_mAP']*100:.4f}%",
  f"{changes[b]['corrected_peak_extra_gain_pp']:+.4f}"] for b in ('S','B')]))
crop_batches=sum(s['batches_with_crop_created_endpoints'] for s in fixed_training['stages'])
add(f"修正训练四阶段共记录{crop_batches}/12000个batch含裁剪制造端点，约{crop_batches/120:.2f}%；这是训练日志中重复样本/窗口的统计，不是独立视频或动作数量。它说明边界有效性问题在真实训练中出现，但仍不能将全部分数差距归给它。\n")
add(table(['总epoch']+[f'{b}平均mAP' for b in ('S','B')],[[e]+[
 f"{next(r for r in curve['results'] if r['backbone']==b and r['total_epoch']==e)['metrics']['average_mAP']*100:.4f}%" for b in ('S','B')]
 for e in range(25,61,5)]))
add('全部16个中间点都完整测试211视频/792窗口。S从55到60轮仍上升0.3261pp，B同期下降0.2302pp，因此“60轮一定不够”尚未被统一证实。按测试峰值选模是用户指定流程，不能称为未见测试集上的无偏泛化成绩。旧BMCR没有采用修正配方重训，不能用旧BMCR与新H65直接回答修正后的BMCR增量。\n')

add('**历史65+是真实回执，但它属于另一条90轮课程。** 固定实现为42dba3f90b37243e7965d18b6707e88e81bf7109，stage2 epoch59 EMA，211视频，平均mAP65.3857244379457%。这里epoch59是第二阶段的第60轮，前面还有30轮warm；不是总60轮。历史日志晚期峰值约65.65是四舍五入日志值，不能代替终点回执65.385724。\n')
add('历史恢复段从joint epoch9继续，原作业1191957最终FAILED来自训练后收据冲突；更新审计仍确认后50轮的5000次成功更新，补评1193610完成。原评估器与当前纯净OpenTAD mAP.py逐字节一致。65.39对应global-TIA192，与当前H65一致；较晚04c35a3采用local-TIA8，不能倒推它生成了65.39。\n')
add(f"旧S终点62.8593比历史65.3857低2.5264pp；修正S63.4094仍低{-hist['s_corrected_gap_to_historical_terminal_pp']:.4f}pp。历史90轮课程中的总60轮日志约64.40，也不是本次20+40课程的等价时点。新DS3的80轮既更换训练对象又更换结构/初始化，不能拿它与旧60轮直接声称测出了纯延长20轮的因果收益；本次DS3内部第60到80轮的变化才是同一新轨迹的训练收益观察。\n")

add('**从原型到当前模型，时间轴和观察预算有几次明确变化。**\n')
add(table(['路线','重RGB/clip预算','native时间长度','检测轴','TIA与填充'],[
 ['官方密集','768候选，48×16','384','原768','全局384；上游edge padding'],
 ['最初两步原型','K384，24×16','192','按物理中心插值到原768','全局192；仅诊断'],
 ['旧/修正H65与旧BMCR','K384，24×16','192','selected-rank384，NMS前回映原轴','全局192；无效packed RGB清零'],
 ['DS3-L密集参考','768，48完整clip','384','原768','局部8；与官方全局转换单测'],
 ['DS3-L T24/DAD等','原clip身份上的0/8/12层条件执行','拼回native384','原768','局部8；保留每clip的8个tubelet；重MLP可稀疏']]))
add('H65的K384是384个RGB物理槽位，不是192帧；192是两个观察组成一个tubelet后的特征数。短窗实际有效观察可少于384，但旧实现仍执行固定24clips。DS3保留完整原clip内部观察次序，尾窗也可能有部分有效clip，不能对所有窗口宣传exact384有效观察。原时间输出回映不等于修复骨干内不规则时间间隔，也不凭空恢复未观察的视觉信息。\n')

add('**当前DS3有新的完整参考结果，但尚未产生训练后模型成绩。**\n')
add(table(['本轮策略','平均mAP','GFLOPs','FLOPs/本轮官方','模型平均/中位ms'],ds3_rows))
if 's_D768L' in complete_ds3:
    g=complete_ds3['s_D768G']['metrics.json']['metrics']['average_mAP'];l=complete_ds3['s_D768L']['metrics.json']['metrics']['average_mAP']
    add(f"S局部密集转换从{g*100:.4f}%降至{l*100:.4f}%，变化{(l-g)*100:+.4f}pp；FLOPs未减少，因为仍执行全部48clips和12层。这项变化发生在稀疏采样/辅助训练之前，必须与后续采样损失分开。它也再次证明：同一权重严格加载、全选局部路径等价，都不意味着local-TIA与官方global-TIA功能等价。\n")
if 's_Z24' in complete_ds3:
    z=complete_ds3['s_Z24']['metrics.json']['metrics']['average_mAP']
    zp=complete_ds3['s_Z24']['profile.json']['cases']['full'];gp=complete_ds3['s_D768G']['profile.json']['cases']['full']
    add(f"零新增训练Z24将48个原clip中均匀选出的24个完整clip送入局部重骨干，再按原中心插值到native384；平均mAP为{100*z:.4f}%，相对局部密集另降{100*(l-z):.4f}pp，相对官方共降{100*(g-z):.4f}pp。在完整窗口上，实际12层均执行24clips，FLOPs为官方的{100*zp['total_macs']/gp['total_macs']:.4f}%。固定窗口GPU平均/中位时延42.60/42.56ms，低于本轮官方67.87/64.77ms，但全数据评测1229.65秒与官方1236.41秒接近，未证明完整流水线的有意义加速。它是精度明显下降的零训练对照，不是已训练D1，更不证明80轮一定能补回这部分损失。\n")
add('本轮官方S的模型平均/中位时间为67.87/64.77ms，与旧官方51.21/51.17ms不是同一次作业测量；当前时延包含DS3评测封装中的特征装配和路由元数据。全数据官方S评测1236.41秒包含解码、CPU变换、传输、模型及NMS，不含AP计算/预测文件写入，缓存状态未控制；不能把这三种时延口径混为一谈。\n')
add('D1训练保留全部48clips、全部12层和全部tokens前向，只优化H0低分辨率预测、H8最终特征对齐、后4层低秩MLP近似及空间评分。推理才压紧clip/深度和后4层heavy MLP输入；attention/TIA仍保持其所需栅格。它与旧BMCR的EMA反事实效用监督不同，不再声称旧方法已经实现检测输出自蒸馏。\n')
add('S/B实际预检1287657均完成2次batch2辅助更新，直接比较确认教师/检测器参数及buffer不变，辅助组确实更新，EMA严格重载和稀疏预测有限。CPU7项检查也通过。当前只读快照中自己的队列为：'+('; '.join(snapshot['own_queue']) or '无活动作业')+'。完整的后续计划为参考/Z0→100步pilot→80轮D1→中间T24A全测试→同一选中EMA的固定策略对照；不因本次报告新增训练。\n')

add('**运行开销应按实际阶段计算，共享预热不能重复计费。**\n')
train_rows=[]
for b in ('s','b'):
    rs=[r for r in training['rows'] if r['backbone']==b];h=next(r for r in rs if r['variant']=='h65')
    train_rows.append(['旧配方',b.upper(),'公共warm',2000,f"{h['warm_seconds']/3600:.4f}",'仅计一次'])
    for r in rs:train_rows.append(['旧配方',b.upper(),r['variant']+' joint',r['joint_updates'],f"{r['joint_seconds']/3600:.4f}",r['slurm_job_id']])
for s in fixed_training['stages']:
    train_rows.append(['修正H65',s['backbone'],s['phase'],s['successful_updates'],f"{s['recorded_training_seconds']/3600:.4f}",s['slurm_job_id']])
add(table(['实验','骨干','阶段','成功更新','记录GPU小时','作业/备注'],train_rows))
add(f"旧配方合计{training['actual_training_gpu_hours']:.4f}小时，修正H65合计{fixed_training['recorded_single_gpu_training_hours']:.4f}小时，总计{total_hours:.4f}小时。它们是训练脚本记录的单卡时间，不是完整Slurm占用或计费账单。初期机制诊断4次作业共79秒，最终权重来自最后2次更新；另一次已更新2步但保存失败，所以初期累计开销含4次成功更新。\n")
add('完整训练课程记录的峰值allocated显存：旧H65-S/B为8.5045/14.3793GiB，旧BMCR-S/B为7.7206/13.5891GiB；修正H65-S/B为8.5046/14.3793GiB。DS3两步预检的4.1200/7.1527GiB覆盖了该次短程检查，不能先写成完整80轮训练的峰值。\n')

add('**原始记录中的失败、勘误与陈旧状态已按时间关系归位，没有删除原件。**\n')
add(table(['记录/问题','实际影响及处理','当前解释'],[
 ['初期ixBrowser computer-use失败','窗口句柄解析失败，改用精确会话ID只读收取最终回复','未声称computer-use成功或读取了当时不可下载的附件'],
 ['初期初始化/BF16算子/JSON保存失败','分别修复；最终1282774成功保存并严格重载','两步诊断，不是完整复现'],
 ['旧配方参数冻结预检失败','在优化器构造前明确冻结，再通过S/B真实预检','不将错误预检冒充正式训练'],
 ['反事实只监督已选候选','正式BMCR训练前改为已选和未选各一个，明确伙伴与反向符号','最终BMCR使用修复后的标签方向'],
 ['曾经训练完成但测试未完成','调度限制与profiling hook错误妨碍测试；修正为2训练+1测试及hook返回None','后续六项旧测试和16项修正测试均完整完成'],
 ['早期FLOPs漏计融合attention QK/AV','修复实际形状计数，补齐full/partial/short窗口','旧不完整profile保留但不进入当前比较'],
 ['此前宣称LR分组正确','实际18,816个内部投影错入action LR','明确撤回；旧分数保留为实际错误配方观测'],
 ['历史65+曾与local8来源混同','锁定成绩提交42dba3f后确认global192','不再用后来的04c35a3状态倒推旧成绩'],
 ['训练阶段文件写full_test_complete=false','生成时间早于后续完整测试','以最终completed/FINAL_VALIDATION为准，不误判缺测试'],
 ['DS3启动1287651/1287655失败','分别为profile未定义环境变量、Slurm脚本spool路径','未进入模型执行；修复后1287657通过'],
 ['个别时延长样本','旧S204.56ms、修正B峰值253.66ms等均保留','同时报告均值/中位，不删除后宣称加速']]))

add('**两份外部优化意见被逐项复审，没有被当成自动执行指令。** 第一份准确指出LR错误，并建议原时间几何、粗信息融合及后续KD/clip；第二份补充zero/edge填充差异与重放特征梯度方案，但遗漏LR问题。确定性LR/真实裁剪边界修复已经实施；取消zero padding、gate、KD、多种子、160/40划分及独立U384不因附件建议自动加入。\n')
add('用户选择全部200训练并跳过独立均匀训练，因此没有U384完整基线，也不能声称“学习采样已被证明胜过同预算均匀训练”。旧20轮均匀warm是模型课程；DS3的Z0/uniform是推理对照，两者都不能冒充新增独立均匀训练。修正配方→原时间检测→粗细融合的独立主线曾开始草稿，随后按用户要求优先诊断历史分数、完成保真修正，再进入DS3；独立G/F没有完整部署结果，C2/J3同样未实现。\n')
add('原始两份最终讨论、H65报告ZIP/DOCX、两套BMCR/BMCRT外部回复包、DS3研究包均按原文件分别保存。公开结果仓库为[yuzbo/BMCR-T-AdaTAD](https://github.com/yuzbo/BMCR-T-AdaTAD)；用户回复附件的保留不等于已公开上传原件。\n')

add('**后续结果的判断需要继续保持这几条边界。** 32k更新和已完成全测试证明实验执行完整，不证明方法假设全部正确；无同配方U、无多种子、没有拆开的LR/边界消融，也没有修正BMCR对照。80轮不能预先保证改善，DS3应分别报告局部转换差距、条件执行后的额外差距、最终mAP、FLOPs及实际时间。下一次有新成绩时直接补同一记录，保留所有阶段与失败历史，不把建议成本估算当成实测。\n')

add('**完整阈值与证据索引附在下面，便于继续复核。**\n')
threshold_rows=[]
for r in legacy['results']:
    threshold_rows.append([r['backbone'],names[r['method']],'提供权重' if r['method']=='official' else '60终点']+[f"{r['metrics'][s]*100:.4f}" for s in scores])
for r in fixed['results']:
    if r['backbone']=='S' and r['selection']=='epoch60_terminal':continue
    threshold_rows.append([r['backbone'],'修正H65',str(r['total_epoch'])+('峰值' if r['selection']=='test_peak' else '终点')]+[f"{r['metrics'][s]*100:.4f}" for s in scores])
for name,d in complete_ds3.items():threshold_rows.append([name[0].upper(),name,'参考']+[f"{d['metrics.json']['metrics'][s]*100:.4f}" for s in scores])
add(table(['骨干','方法','选择','平均','@.3','@.4','@.5','@.6','@.7'],threshold_rows))
add('表中数值单位均为百分数。更完整的16个中间点逐阈值记录见[INTERMEDIATE_RESULTS.md](../../fidelity_20260911/INTERMEDIATE_RESULTS.md)。\n')
add(table(['证据','固定来源/文件'],[
 ['纯净上游','OpenTAD346d09d19e2091372cec48172dbe40f7b28bdee6；新增代码在h65/与tools/，未覆盖上游源码'],
 ['旧完整实验','[EXPERIMENT_REPORT](../../phase2_20260910/EXPERIMENT_REPORT.md)、[comparison.json](../../phase2_20260910/comparison.json)、[TRAINING_SUMMARY](../../phase2_20260910/TRAINING_SUMMARY.json)；方法/结果发布15280e5，prompt更新8a85290'],
 ['保真修正','[FINAL_REPORT](../../fidelity_20260911/FINAL_REPORT.md)、[FINAL_COMPARISON](../../fidelity_20260911/FINAL_COMPARISON.json)、[训练完成](../../fidelity_20260911/TRAINING_COMPLETION.json)；科学代码1c1552a，部署990169a，最终报告e407236'],
 ['历史65+','原42dba3f运行回执；本地h65_clean_adatad/diagnostics/h65_65_gap_20260911/historical_run/terminal_evaluation.json；[本次汇总JSON](record_data.json)保留历史已核对值'],
 ['DS3实现','[PLAN](../PLAN.md)、[IMPLEMENTATION](../IMPLEMENTATION.md)、[预检](../validation/gpu_preflight_s.json)；实现1621ad6，启动修正与GPU证据5cffb8a，官方S参考21ff5ea'],
 ['本次只读快照','[remote_snapshot.json](remote_snapshot.json)，包含时间戳、自己的队列、各阶段收据及已产生评测；不含模型权重'],
 ['新增参考结果验证','[ds3_reference_verification.json](ds3_reference_verification.json)，直接读取局部密集/Z24的全视频预测，复核211 IDs、全量预测数量和实际clip/token算量'],
 ['原文归档','本地h65_clean_adatad/discussions/、phase2_20260910/inputs/、reviews/20260911_external_bmcr_review/、reviews/20260911_external_bmcrt_review_v2/、reviews/20260912_ds3/'],
 ['独立实验目录','本地h65_clean_adatad/；远端/data/run01/sczc063/yuzibo/h65_clean_adatad_20260910；新DS3在其中ds3_20260912独立源目录及嵌套runs/，共享资源只读']]))

OUT.mkdir(parents=True,exist_ok=True)
(OUT/'REPORT.zh.md').write_text('\n'.join(parts),encoding='utf-8')
data=dict(snapshot_time=snapshot['timestamp'],completed_formal_updates=32000,recorded_training_gpu_hours=total_hours,
          legacy=legacy,fidelity=fixed,fidelity_curve=curve,training_legacy=training,training_fidelity=fixed_training,
          derived_changes=changes,ds3_completed_references={k:v['metrics.json'] for k,v in complete_ds3.items()},
          historical_context=hist,aggregation_checks=dict(five_threshold_means_verified=True,legacy_updates=20000,
          fidelity_updates=12000,shared_warm_counted_once=True,ds3_complete_reference_count=len(complete_ds3)))
(OUT/'record_data.json').write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')

plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.spines.left':False,
                     'axes.grid':True,'grid.color':'#dfe4ea','axes.axisbelow':True,'savefig.facecolor':'white'})
fig,axes=plt.subplots(1,2,figsize=(13.8,5.4),gridspec_kw={'width_ratios':[1.15,1]})
y=np.arange(4);labels=['Official AdaTAD','H65-C (legacy, terminal60)','BMCR-T (legacy, terminal60)','H65 corrected (test peak)']
for offset,b,color in ((-.17,'S','#356caa'),(.17,'B','#dc8841')):
    values=[old[b,k]['metrics']['average_mAP']*100 for k in ('official','h65','bmcr')]+[new[b,'test_peak']['metrics']['average_mAP']*100]
    ratios=[old[b,k]['latency_median_ms']/official[b]['latency_median_ms'] for k in ('official','h65','bmcr')]+[new[b,'test_peak']['latency_median_ms']/official[b]['latency_median_ms']]
    for ax,vals,fmt in ((axes[0],values,lambda x:f'{x:.2f}'),(axes[1],ratios,lambda x:f'{x:.2f}x')):
        bars=ax.barh(y+offset,vals,height=.29,color=color,label='VideoMAE-'+b)
        ax.bar_label(bars,labels=[fmt(v) for v in vals],padding=4,fontsize=9)
for ax in axes:ax.set_yticks(y);ax.invert_yaxis();ax.grid(axis='y',visible=False)
axes[0].set_yticklabels(labels);axes[0].set_xlim(0,80);axes[0].set_xlabel('Mean mAP (%)');axes[0].set_title('Complete 211-video test results',loc='left',fontweight='bold')
axes[1].set_yticklabels([]);axes[1].set_xlim(0,2.8);axes[1].set_xlabel('Median model latency / corresponding official');axes[1].axvline(1,color='#7c8795',ls='--',lw=1)
axes[1].set_title('Compute reductions did not yield measured speedup',loc='left',fontweight='bold',fontsize=11)
handles,legend_labels=axes[0].get_legend_handles_labels()
fig.legend(handles,legend_labels,loc='upper left',bbox_to_anchor=(.21,.99),ncol=2,frameon=False)
fig.text(.025,.035,'Seed3407. Corrected H65 uses test-peak selection; legacy models use terminal EMA.\nLatency: fixed full window, GPU-resident input, BF16 feature modules / FP32 detector; independent 4090 jobs; decoding/NMS excluded.',fontsize=9,color='#475467')
fig.subplots_adjust(left=.21,right=.975,top=.84,bottom=.2,wspace=.15)
fig.savefig(OUT/'overview.png',dpi=180);fig.savefig(OUT/'overview.svg');plt.close(fig)
svg=OUT/'overview.svg'
svg.write_text('\n'.join(line.rstrip() for line in svg.read_text(encoding='utf-8').splitlines())+'\n',encoding='utf-8')
print(json.dumps(dict(report=str(OUT/'REPORT.zh.md'),formal_updates=32000,gpu_hours=total_hours,reference_count=len(complete_ds3)),ensure_ascii=False))
