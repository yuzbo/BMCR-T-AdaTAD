"""Full-test tables and publication figures. Unmeasured results stay absent."""
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from h65.paper.runtime import json_write
from h65.paper.figures import architecture,save
import numpy as np
import matplotlib.pyplot as plt


def collect(folder,family):
    rows=[]
    if not folder.exists():return rows
    for path in sorted(folder.glob('**/completed.json')):
        data=json.loads(path.read_text());cfg=data.get('config',{})
        if 'metrics' not in data or not cfg or not data.get('gflops'):continue
        ds=cfg.get('dataset','thumos');expected=211 if ds=='thumos' else 4728
        if data.get('test_videos')!=expected:raise ValueError('Incomplete evaluation record '+str(path))
        if ds=='thumos' and data.get('test_windows')!=792:raise ValueError('Incomplete THUMOS windows '+str(path))
        row=dict(protocol=family,id=cfg['id'],dataset=ds,backbone=cfg['backbone'],head=cfg.get('head','point'),
                 comparison=cfg.get('comparison',cfg['id']),seed=cfg.get('seed',3407),epoch=data.get('epoch'),state=data.get('checkpoint_state'),
                 average_mAP=float(data['metrics']['average_mAP'])*100,gflops=data['gflops'],metrics=data['metrics'],
                 latency_ms=data.get('latency_ms'),dataset_total_gflops=data.get('dataset_total_gflops'),
                 compute_scope=data.get('compute_scope','fixed first full-window operator profile'),
                 source=str(path),evaluation=path.parent.name,force_plan=data.get('force_plan'),
                 budget_fraction=cfg.get('budget_fraction'),selector=cfg.get('selector'),
                 training_selector=data.get('training_config',cfg).get('selector'))
        rows.append(row)
    return rows


def standard(row):
    if row['state']!='ema' or row['force_plan'] is not None:return False
    if row['selector']!=row['training_selector']:return False
    return row['evaluation'].startswith('eval_') and row['evaluation'][5:8].isdigit() if row['protocol']=='paper' else True


def figures(rows,out):
    groups=defaultdict(list)
    for row in rows:groups[(row['protocol'],row['dataset'],row['backbone'],row['head'])].append(row)
    for (protocol,ds,bb,head),members in groups.items():
        prefix=f'{protocol}_{ds}_{bb}_{head}';fig,axes=plt.subplots(1,2,figsize=(12,4.3),layout='constrained')
        selected=[r for r in members if (standard(r) or 'calibrated_budget' in r['evaluation'])]
        if protocol=='pilot':selected=[r for r in selected if r['id'].startswith(('R01_interpolate','R03_cross','D02_amod50','S02_token48','J01_joint'))]
        else:selected=[r for r in selected if r['comparison'] in ('full','uniform','dense','random')]
        best={}
        for row in selected:
            key=row['id'] if 'calibrated_budget' not in row['evaluation'] else row['id']+row['evaluation']
            if key not in best or row['average_mAP']>best[key]['average_mAP']:best[key]=row
        for key,row in best.items():
            label=row['comparison'].removesuffix('_'+bb)
            if 'calibrated_budget' in row['evaluation']:label=f'calibrated {row["budget_fraction"]:.0%}'
            axes[0].scatter(row['gflops'],row['average_mAP'],s=28)
            if len(best)<=15:axes[0].annotate(label,(row['gflops'],row['average_mAP']),xytext=(4,3),textcoords='offset points',fontsize=8)
        axes[0].set(xlabel='Complete model GFLOPs / full window',ylabel='Average mAP (%)',title='Measured compute–accuracy points')
        curves=defaultdict(list)
        for row in members:
            if standard(row) and row['epoch'] is not None:curves[row['id']].append(row)
        for ident,curve in curves.items():
            if protocol=='pilot' and not ident.startswith(('R03_cross','R04_feature_only','R02_tcn','J01_joint')):continue
            if protocol=='paper' and curve[0]['comparison'] not in ('full','uniform','dense','random'):continue
            curve=sorted(curve,key=lambda r:r['epoch']);axes[1].plot([r['epoch'] for r in curve],[r['average_mAP'] for r in curve],'o-',ms=3,lw=1,label=ident.replace(ds+'_','').replace('_seed',' s'))
        axes[1].set(xlabel='Completed training epochs',ylabel='Full-test average mAP (%)',title='EMA milestone distribution')
        if axes[1].lines:axes[1].legend(fontsize=7,frameon=False)
        fig.suptitle(f'{protocol.upper()} | {ds} | {bb} | {head}',fontsize=12);save(fig,out,prefix+'_performance')
        if protocol=='paper':
            chosen=[r for r in best.values() if r['comparison']=='full'][:6]
            values=[];labels=[]
            for row in chosen:
                file=Path(row['source']).parent/'window_distribution.npz'
                if file.exists():
                    with np.load(file) as data:values.append(data['gflops'].copy())
                    labels.append(row['evaluation'])
            if values:
                fig,ax=plt.subplots(figsize=(9,4),layout='constrained');ax.boxplot(values,showfliers=False)
                ax.set_xticks(range(1,len(labels)+1));ax.set_xticklabels(labels)
                ax.set(ylabel='Actual GFLOPs across all test windows',title='Executed compute distribution, including partial windows');ax.tick_params(axis='x',rotation=20)
                save(fig,out,prefix+'_compute_distribution')
            fixed=[r for r in members if standard(r) and r['epoch']==40 and r['seed']==3407]
            baseline=next((r for r in fixed if r['comparison']=='full'),None)
            if baseline and len(fixed)>1:
                fixed=sorted((r for r in fixed if r['comparison']!='full'),key=lambda r:r['average_mAP'])
                fig,axes=plt.subplots(1,2,figsize=(11,max(4,len(fixed)*.28)),layout='constrained')
                ids=np.arange(len(fixed));axes[0].barh(ids,[r['average_mAP']-baseline['average_mAP'] for r in fixed],color='#63869f')
                axes[0].set_yticks(ids);axes[0].set_yticklabels([r['comparison'] for r in fixed],fontsize=8)
                axes[0].axvline(0,color='.3',lw=.8);axes[0].set(xlabel='mAP change from full model (pp)',title='Matched 40-epoch independent training')
                axes[1].barh(ids,[r['gflops'] for r in fixed],color='#6fa597');axes[1].axvline(baseline['gflops'],color='.3',ls='--',lw=.8)
                axes[1].set_yticks(ids);axes[1].set_yticklabels([]);axes[1].set(xlabel='Complete-model GFLOPs',title='Dashed line: full model at epoch 40')
                save(fig,out,prefix+'_ablations40')
            factors={r['force_plan']:r for r in members if r['epoch']==40 and r['seed']==3407 and r['force_plan'] is not None}
            layout=((0,13,12,14),(1,3,2,4))
            if all(i in factors for row in layout for i in row):
                fig,axes=plt.subplots(1,2,figsize=(11,3.5),layout='constrained')
                for ax,field,title in zip(axes,('average_mAP','gflops'),('Average mAP (%)','Actual complete-model GFLOPs')):
                    values=np.array([[factors[i][field] for i in row] for row in layout]);im=ax.imshow(values,cmap='viridis',aspect='auto')
                    ax.set_xticks(range(4));ax.set_xticklabels(['D1 S1','D.5 S1','D1 S.48','D.5 S.48'],fontsize=9)
                    ax.set_yticks([0,1]);ax.set_yticklabels(['T768','T384']);ax.set_title(title)
                    for (y,x),v in np.ndenumerate(values):ax.text(x,y,f'{v:.2f}',ha='center',va='center',color='white',fontsize=10)
                    fig.colorbar(im,ax=ax,shrink=.8)
                fig.suptitle('Same epoch-40 model: all eight T/D/S configurations',fontsize=12);save(fig,out,prefix+'_three_axis_factors')


def main(args):
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True);architecture(out)
    rows=collect(ROOT/'research/paper/runs','paper')
    if args.pilot_root:rows+=collect(Path(args.pilot_root)/'research/frame/runs','pilot')
    json_write(out/'full_test_records.json',rows);figures(rows,out)
    plan=json.loads((ROOT/'research/paper/plan.json').read_text());path=ROOT/'research/paper/deployment.json'
    deployment=json.loads(path.read_text()) if path.exists() else {'stages':plan['stages']}
    counts=Counter(s.get('status','REGISTERED') for s in deployment['stages'].values())
    best={}
    for row in rows:
        if not standard(row):continue
        key=(row['protocol'],row['dataset'],row['backbone'],row['head'],row['id'])
        if key not in best or row['average_mAP']>best[key]['average_mAP']:best[key]=row
    summary=dict(configurations=len(plan['configs']),stages=len(plan['stages']),status_counts=dict(counts),
        completed_paper_tests=sum(r['protocol']=='paper' for r in rows),completed_pilot_tests=sum(r['protocol']=='pilot' for r in rows),
        best_preregistered_ema=list(best.values()),paper_final_claims_ready=False,
        decision='Full split mAP and actual complete-model FLOPs; latency is descriptive',
        conclusion='Pilot recovery is promising. Final claims require full joint-model courses, matched controls, seeds and generalization results.')
    json_write(out/'summary.json',summary)
    lines=['**完整论文实验：执行与证据状态**','',f'已登记 {len(plan["configs"])} 个训练配置、{len(plan["stages"])} 个阶段。状态：{dict(counts)}。',
        '',f'新论文协议完整测试 {summary["completed_paper_tests"]} 条；旧探索协议完整测试 {summary["completed_pilot_tests"]} 条。',
        '', '以下是各已测配置的预登记 EMA 里程碑最佳值；终点、逐阈值 AP、原始路径和预算分布保存在 full_test_records.json。尚未测量的配置没有填入结果。',
        '', '|协议|配置|最佳 epoch|mAP %|GFLOPs|','|---|---|---:|---:|---:|']
    for row in sorted(best.values(),key=lambda r:(r['protocol'],r['dataset'],r['backbone'],r['head'],-r['average_mAP'])):
        lines.append(f'|{row["protocol"]}|{row["id"]}|{row["epoch"]}|{row["average_mAP"]:.3f}|{row["gflops"]:.2f}|')
    lines+=['','判定条件：以实际完整模型计算量与 TAD 性能判断；延迟、显存、E2E 和训练开销另行呈现。',
        '论文主张须由同课程独立训练对照、三个种子、视频配对 bootstrap、ANet、InternVideo1-MQ、TadTR 和真实干预数据共同支持。',
        '激活相似度只用作冗余代理，不能单凭相似度宣称信息可删除或任务无损。','']
    (out/'REPORT.zh.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps({k:v for k,v in summary.items() if k!='best_preregistered_ema'},ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--pilot-root');p.add_argument('--output',default=str(ROOT/'research/paper/figures'))
    main(p.parse_args())
