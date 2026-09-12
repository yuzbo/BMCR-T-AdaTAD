"""Publish a dated, evidence-backed context for a fixed-commit external prompt."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'ds3_20260912/research_context_20260912'
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
snapshot=read(OUT/'latest_snapshot.json')
legacy=read(ROOT/'phase2_20260910/comparison.json')
fixed=read(ROOT/'fidelity_20260911/FINAL_COMPARISON.json')
warm=read(ROOT/'bmcr_fidelity_20260912/initial_snapshot.json')['warm']

def table(headers,rows):
 return '\n'.join(['|'+'|'.join(headers)+'|','|'+'|'.join(['---']*len(headers))+'|']+['|'+'|'.join(map(str,row))+'|' for row in rows])+'\n'

parts=[];add=parts.append
add('**当前代码与实验上下文：用于固定提交的外部研究讨论**\n')
add('最新应用实现已核对本地与远端同为`76f763d105cfbf8ee0d092e8d80468828b402939`。本次快照和文字整理没有改变模型/训练代码。讨论prompt将固定到包含本文件的后续提交，而非浮动默认分支。\n')
add(f"实验快照时间：{snapshot['timestamp']}。公开仓库不含大模型权重和原始视频，外部模型可进行源码/记录核查，但不应声称已经独立加载这些私有路径上的权重重跑CUDA或mAP。\n")
add('用户要求以BMCR-T已有成果为下一步研究锚点，统筹帧/clip、空间、深度与密集训练—稀疏推理，避免从弱随机起点丢弃已有性能。这个要求不意味着BMCR在所有表格里最高：B的旧BMCR是已完成新模型路线中的B最高值；S的修正H65更高；官方仍高于两者。修正BMCR实验已经授权但尚未启动，见[bmcr准备记录](../../bmcr_fidelity_20260912/README.md)。\n')
rows=[]
for r in legacy['results']:
 rows.append([r['backbone'],r['method'],'提供EMA' if r['method']=='official' else '旧配方60终点',f"{100*r['metrics']['average_mAP']:.6f}%",f"{(1-r['mac_reduction'])*100:.6f}%",f"{r['latency_ms']:.3f}/{r['latency_median_ms']:.3f}"])
for r in fixed['results']:
 if r['backbone']=='S' and r['selection']=='epoch60_terminal':continue
 rows.append([r['backbone'],'corrected H65',str(r['total_epoch'])+('峰值' if r['selection']=='test_peak' else '终点'),f"{100*r['metrics']['average_mAP']:.6f}%",f"{100*r['flops_fraction_of_official']:.6f}%",f"{r['latency_mean_ms']:.3f}/{r['latency_median_ms']:.3f}"])
add(table(['骨干','模型','选择/配方','平均mAP','FLOPs/对应官方','模型平均/中位ms'],rows))
add('旧BMCR/H65权重的运行来源是15280e5（旧LR分组和裁剪边界缺口）；当前源码已经含保真修正，不能倒推旧权重也是修正版。修正H65科学代码1c1552a，完整报告e407236。BMCR平均增益相对旧H65为S+0.296977pp、B+0.727645pp；S的@0.5和@0.7却更低。当前没有同配方修正BMCR对照，也无多种子显著性证据。\n')
add('完整主实验采用200训练、211测试/792窗口、seed3407。旧方法K400识别预训练→20轮共享warm→40轮各自joint；官方只测提供检查点。修正H65重建了新warm，按用户要求在总25..60每5轮全测试选EMA峰值，并单报60；旧模型用终点EMA。DS3使用提供的TAD权重作为冻结教师，H0另有ImageNet MobileNetV3Small预训练，监督来源和训练预算不同。\n')
add('修正BMCR可严格复用已完成的修正warm20；S/B条件输出末层都为零，但隐藏层并非全零。它仍需要新的训练集效用尺度、真实预检和joint40，不能只改代码就宣称旧权重被修复。\n')

rows=[];reference_rows=[]
base=snapshot['artifacts'].get('s_D768G',{}).get('profile.json',{}).get('cases',{}).get('full')
order=['s_D768G','s_D768L','s_Z16','s_Z24','s_Z36','s_ZR24']
order += sorted(n for n in snapshot['artifacts'] if n.startswith(('s_T24A_epoch_','b_T24A_epoch_')))
order += sorted(n for n in snapshot['artifacts'] if n.startswith('b_') and n not in order and 'metrics.json' in snapshot['artifacts'][n])
for name in order:
 a=snapshot['artifacts'].get(name,{})
 if 'metrics.json' not in a:continue
 m=a['metrics.json'];values=m['metrics'];assert m['test_videos']==211 and m['test_windows']==792
 assert abs(values['average_mAP']-sum(values[f'mAP@{t}'] for t in (.3,.4,.5,.6,.7))/5)<1e-12
 p=a.get('profile.json',{}).get('cases',{}).get('full')
 ref=snapshot['artifacts'].get(name[0]+'_D768G',{}).get('profile.json',{}).get('cases',{}).get('full')
 row=[name,f"{100*values['average_mAP']:.6f}%",f"{100*p['total_macs']/ref['total_macs']:.6f}%" if p and ref else '尚未profile',f"{p['model_timing']['mean_seconds']*1000:.3f}/{p['model_timing']['p50_seconds']*1000:.3f}" if p else '尚未profile']
 rows.append(row);reference_rows.append(dict(name=name,metrics=m,profile=p))
add(table(['DS3阶段策略','全211视频平均mAP','FLOPs/本轮对应官方','模型平均/中位ms'],rows))
progress=[]
for b in ('s','b'):
 a=snapshot['artifacts'].get(b+'_d1',{});p=a.get('progress.json',{})
 progress.append([b.upper(),p.get('completed_epochs','未产生完整epoch回执'),p.get('successful_updates','未产生完整epoch回执'),8000])
add(table(['D1骨干','已保存完成epoch','已保存成功更新','预设总更新'],progress))
add('自己的实时队列：'+('; '.join(snapshot['own_queue']) or '无活动作业')+'。第10轮等仍运行/等待的测试不填入成绩表。snapshot中历史progress.json可能落后于completed.json；已完成与否以完整metrics/completed及各阶段状态为准。\n')
add('D768G→D768L的S损失1.478114pp发生在稀疏之前，且FLOPs相同。Z24相对local dense再下降11.427936pp。Z16/24/36是零新增训练的均匀完整clip抽取与物理中心插值，ZR24是带覆盖锚点的确定种子随机clip控制，均不是16/24/36轮训练。它们的预算曲线不能直接判死刑所有clip方法，也不能与已训练帧选择做单因素归因。\n')
add('第5轮T24A EMA的平均mAP是50.64583163849076%（原始比例0.5064583163849076），只代表500次更新的早期候选。D1尚未完成80轮。EMA0.999初始复制aux，500次更新后初值系数约0.6064；这说明存在滞后，不能仅凭系数断言它解释全部mAP下降。需要与在线权重、初始化/路由和分支质量区分。\n')

add('**当前训练/推理合同**\n')
add('候选步幅4个原视频帧。16-observation clip指16个候选观察，不能写成16张连续原视频帧。官方768候选→48clips→每clip8个tubelet→native384→detector768。H65/BMCR的K384由24个重新组装的clip形成native192，使用selected-rank检测及NMS前回映；global-TIA192。DS3-L选择原clip身份，保留8个时间tubelet并回到native384/原detector768，local-TIA8。\n')
add('D1训练每次完整执行48clips、12层、全部tokens，教师与检测器冻结。ImageNet预训练MobileNetV3Small提供64²视觉预览，新增H0输出/utility、H8残差出口、后4层低秩MLP近似器和空间评分从新初始化学习。损失主要是特征/时间差分拟合、梯度加权残差效用proxy及MLP近似；冻结检测器对teacher feature的GT损失梯度只用于标签，没有对真实混合稀疏学生图进行检测损失优化。\n')
add('推理按原clip ID执行0/8/12层：未执行重骨干的位置由H0估计；8层出口经H8对齐；12层位置用深层特征覆盖。后4层可只让部分空间token经过heavy MLP，其余用近似器，attention和TIA并未因此都跳过。当前H0/出口/近似模块没有保证初始混合输出等于已验证Z24或已训练H65/BMCR；零路由logits也不自动等于uniform24，因为top-k/tie-break仍需定义。\n')
add('当前local dense教师67.5344%是参照，不是严格精度上限；直接复制global权重并改local图不等价。实际高分H65/BMCR的scout/边界先验、任务效用和预测可作为知识来源，但rank特征/检测器与original-grid接口不能无校准直接迁移。训练内clip特征缓存反事实只在相应独立性/同状态条件下可成立，不能自动推广到global-TIA。\n')
add('需要区分Z0零新增训练、D1完整密集前向辅助学习、C2骨干密集但辅助图见过稀疏缺失的校准、J3稀疏路径联合适配。用户要求全量新模型训练并关心性能保持；不要把“主干训练时全部算”与“任何辅助训练都不能见缺失模式”默认为同一个不可讨论的约束。当前80轮D1仍按既定配置运行，C2/J3尚未部署。\n')

add('**计量与已知局限**\n')
add('FLOPs=实际矩阵/卷积2MAC，计入融合QK/AV，不包含所有逐元素操作。旧训练模型计时与本轮DS3封装/节点计时不同，不能混用51.21ms和67.87ms作为同一次基准。完整窗口、部分窗503、短窗253分别保存；推理不读GT。Z24固定窗口GPU更快，但全数据约1229.65秒与官方1236.41秒接近，缓存状态未控制，不能把FLOPs削减直接写成端到端加速。\n')
add('基于S实测算子分项，D8全时域轻出口约67.73% FLOPs、全时域160→128约58.89%，均为估算且无对应新精度；仅后4层MLP保留75%/50%在额外模块前最多省原总量3.86%/7.72%，TIA算术占比约3.27%。不能把时间/空间/深度保留率直接相乘。未做多种子、独立U384完整训练和160/40划分；用户指定完整测试峰值选模，结果不是未见测试估计。\n')
add('历史H65-S65.385724%来自42dba3f的30+60课程/global-TIA192，不能与后来的04c35a3 local8混同；日志约65.65峰值与精确终点区分。现有修正H65只补LR/裁剪边界，原时间检测和独立粗细融合三阶段仍无完整部署结果。\n')

add('**优先阅读入口**\n')
add('- 当前完整历史：[retrospective](../retrospective_20260912/REPORT.zh.md)；旧模型原始汇总：[phase2/comparison](../../phase2_20260910/comparison.json)。\n- 修正H65：[FINAL_COMPARISON](../../fidelity_20260911/FINAL_COMPARISON.json)、[INTERMEDIATE_RESULTS](../../fidelity_20260911/INTERMEDIATE_RESULTS.json)。\n- DS3：[PLAN](../PLAN.md)、[IMPLEMENTATION](../IMPLEMENTATION.md)、[本次原始快照](latest_snapshot.json)。\n- 源码：h65/full/{model,scout,utility,objectives,geometry,runtime,data}.py；h65/ds3/{model,auxiliary,losses,routes,runtime}.py；tools/{full_train,full_eval,ds3_train,ds3_eval}.py。\n- 后续候选仅为建议：[DECISION](../decisions_20260912/DECISION.zh.md)、[算量估算](../decisions_20260912/compute_estimates.json)。\n')
OUT.mkdir(parents=True,exist_ok=True)
(OUT/'CONTEXT.zh.md').write_text('\n'.join(parts),encoding='utf-8')
(OUT/'summary.json').write_text(json.dumps(dict(snapshot_time=snapshot['timestamp'],implementation_commit='76f763d105cfbf8ee0d092e8d80468828b402939',ds3=reference_rows,progress=progress,bmcr_corrected_training_started=False,ema_initial_coefficient_at500=.999**500),indent=2)+'\n',encoding='utf-8')
print(OUT/'CONTEXT.zh.md')
