"""Figures from completed measurements, with explicit missing-data coverage."""
import argparse
import json
import math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

THRESHOLDS=[f'mAP@{x:.1f}' for x in (.3,.4,.5,.6,.7)]
def numeric(x):return isinstance(x,(int,float)) and math.isfinite(x)
def complete(r):return str(r.get('status','')).lower()=='complete'
def cohort(r):return r.get('dataset','THUMOS14'),r.get('backbone',''),r.get('compute_protocol','full768_2MAC')

def load_records(path):
    path=Path(path)
    if not path.is_dir():return json.loads(path.read_text(encoding='utf-8'))['records']
    rows=[]
    for file in sorted(path.glob('*/completed.json')):
        row=json.loads(file.read_text())
        if row.get('test_videos')!=211 or row.get('test_windows')!=792 or 'metrics' not in row:continue
        row.update(source=str(file),run_id=row['id'],id=f"{row['id']}@{row.get('epoch',0)}_{row.get('checkpoint_state','ema')}",status='complete')
        profile=file.parent/'profile.json'
        if profile.exists():
            data=json.loads(profile.read_text());row.update(gflops=data['matrix_conv_flops']/1e9,latency_ms=data['latency_mean_ms'],profile_data=data)
        rows.append(row)
    return rows

def best_records(records):
    best={}
    for row in records:
        if not complete(row) or not numeric(row.get('metrics',{}).get('average_mAP')) or row.get('checkpoint_state','ema')!='ema':continue
        # Progressive deletion changes compute during its trajectory; keep each
        # distinct retained-layer policy instead of hiding its cheaper points.
        keep=row.get('runtime_policy',{}).get('static_keep')
        key=cohort(row),row.get('run_id',row['id']),tuple(keep) if keep is not None else None
        if key not in best or row['metrics']['average_mAP']>best[key]['metrics']['average_mAP']:best[key]=row
    return list(best.values())

def pareto(records):
    points=[r for r in records if complete(r) and numeric(r.get('gflops')) and numeric(r.get('metrics',{}).get('average_mAP'))]
    return [r for r in points if not any(cohort(q)==cohort(r) and q['gflops']<=r['gflops'] and q['metrics']['average_mAP']>=r['metrics']['average_mAP'] and
        (q['gflops']<r['gflops'] or q['metrics']['average_mAP']>r['metrics']['average_mAP']) for q in points)]

def save(fig,out,name):
    for ext in ('png','svg','pdf'):fig.savefig(out/f'{name}.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)

def analyze(manifest,output,figspec=None,dry_run=False):
    records=load_records(manifest);out=Path(output);best=best_records(records);front=pareto(best)
    coverage=dict(manifest=str(manifest),record_count=len(records),figures={},latency_is_decision_gate=False,selection='best completed full-test EMA; exploratory test selection')
    if not dry_run:out.mkdir(parents=True,exist_ok=True)
    def status(name,n,why):coverage['figures'][name]=dict(status='available' if n else 'missing',points=n,reason=why)
    points=[r for r in best if numeric(r.get('gflops'))];status('map_gflops_pareto',len(points),'all measured best models, including dominated/negative results; separate backbone frontiers')
    if points and not dry_run:
        groups=sorted({cohort(r) for r in points});fig,axes=plt.subplots(1,len(groups),figsize=(6*len(groups),4),squeeze=False)
        for ax,key in zip(axes[0],groups):
            group=[r for r in points if cohort(r)==key];edge=sorted([r for r in front if cohort(r)==key],key=lambda r:r['gflops'])
            ax.scatter([r['gflops'] for r in group],[100*r['metrics']['average_mAP'] for r in group],color='#8598a8')
            ax.plot([r['gflops'] for r in edge],[100*r['metrics']['average_mAP'] for r in edge],'o-',color='#126e82')
            for r in group:ax.annotate(r.get('run_id',r['id']),(r['gflops'],100*r['metrics']['average_mAP']),fontsize=6,xytext=(3,3),textcoords='offset points')
            ax.set(xlabel='Total matrix/conv GFLOPs (2 MAC)',ylabel='Best full-test average mAP (%)',title=f'{key[0]} / VideoMAE-{key[1].upper()}');ax.grid(alpha=.15)
        save(fig,out,'map_gflops_pareto')
    th=[r for r in best if all(numeric(r['metrics'].get(k)) for k in THRESHOLDS)];status('five_threshold_map',len(th),'all five measured tIoU thresholds')
    if th and not dry_run:
        groups=sorted({cohort(r) for r in th});fig,axes=plt.subplots(1,len(groups),figsize=(6*len(groups),4),squeeze=False)
        for ax,key in zip(axes[0],groups):
            for r in th:
                if cohort(r)==key:ax.plot([.3,.4,.5,.6,.7],[100*r['metrics'][k] for k in THRESHOLDS],'.-',label=r.get('run_id',r['id']))
            ax.set(xlabel='tIoU',ylabel='mAP (%)',title=f'VideoMAE-{key[1].upper()}');ax.legend(fontsize=6)
        save(fig,out,'five_threshold_map')
    lat=[r for r in best if numeric(r.get('latency_ms'))];status('latency_cohort',len(lat),'hardware/cohort retained; no latency gating')
    if lat and not dry_run:
        fig,ax=plt.subplots(figsize=(max(7,len(lat)*.35),4));ax.bar(range(len(lat)),[r['latency_ms'] for r in lat],color='#9f779b')
        ax.set_xticks(range(len(lat)),[r.get('run_id',r['id']) for r in lat],rotation=60,ha='right',fontsize=6);ax.set(ylabel='Mean model latency (ms)',title='Latency report, excluded from route selection');save(fig,out,'latency_cohort')
    curves={}
    for r in records:
        if complete(r) and numeric(r.get('epoch')) and r.get('checkpoint_state','ema')=='ema' and numeric(r.get('metrics',{}).get('average_mAP')):curves.setdefault((r.get('backbone',''),r.get('run_id',r['id'])),[]).append(r)
    curves={k:sorted(v,key=lambda r:r['epoch']) for k,v in curves.items() if len(v)>1};status('training_curve',len(curves),'completed full-test milestones; incomplete trajectories remain incomplete')
    if curves and not dry_run:
        groups=sorted({k[0] for k in curves});fig,axes=plt.subplots(1,len(groups),figsize=(6*len(groups),4),squeeze=False)
        for ax,b in zip(axes[0],groups):
            for (backbone,name),rows in curves.items():
                if backbone==b:ax.plot([r['epoch'] for r in rows],[100*r['metrics']['average_mAP'] for r in rows],'o-',label=name)
            ax.set(xlabel='Completed epoch',ylabel='Full-test average mAP (%)',title=f'VideoMAE-{b.upper()}');ax.legend(fontsize=6);ax.grid(alpha=.15)
        save(fig,out,'training_curve')
    status('component_compute',sum('profile_data' in r for r in best),'actual per-scope operator counts')
    for r in best:
        if dry_run or 'profile_data' not in r:continue
        scopes=r['profile_data']['macs_by_component'];fig,ax=plt.subplots(figsize=(6,3));ax.barh(list(scopes),[2*x/1e9 for x in scopes.values()],color='#287d8e');ax.set(xlabel='GFLOPs',title=r['id']);save(fig,out,'compute_'+r['id'].replace('@','_'))
    root=Path(manifest);utility=[];train=[]
    if root.is_dir():
        for file in root.glob('*/interventions.jsonl'):
            utility.extend(r for r in (json.loads(s) for s in file.read_text().splitlines()) if r.get('actual_reencoded'))
        for file in root.glob('*/train.jsonl'):
            rows=[json.loads(s) for s in file.read_text().splitlines() if s.strip()]
            if rows:train.append(dict(id=file.parent.name,recorded_updates=len(rows),last_update=rows[-1]['successful_updates'],teacher_queries=rows[-1]['teacher_queries'],peak_gib=max(r['peak_gib'] for r in rows),loss_quantiles={k:np.quantile([r['losses'][k] for r in rows if k in r['losses']],[.1,.5,.9]).tolist() for k in rows[-1]['losses']}))
    status('utility_repair_vs_actual',len(utility),'actual RGB frame swaps compared with distinct feature-repair proxy')
    if utility and not dry_run:
        fig,axes=plt.subplots(1,2,figsize=(10,4))
        for i,ax in enumerate(axes):ax.scatter([r['repair_delta'][i] for r in utility],[r['actual_delta'][i] for r in utility],s=8,alpha=.3);ax.set(xlabel='Feature repair benefit',ylabel='Actual frame swap benefit',title=['Classification','Localization'][i])
        save(fig,out,'utility_repair_vs_actual')
    masks=list(root.glob('*/execution_full.npz')) if root.is_dir() else [];status('execution_masks',len(masks),'actual layer masks with dense first/last layers')
    if not dry_run:
        for file in masks:
            data=np.load(file);keys=list(data);fig,axes=plt.subplots(len(keys),1,figsize=(9,2*len(keys)),squeeze=False)
            for ax,key in zip(axes[:,0],keys):
                a=data[key];l,c,n=a.shape;im=ax.imshow(a.reshape(l,c*8,n//8).mean(-1),aspect='auto',vmin=0,vmax=1,cmap='viridis');ax.set(xlabel='Packed native time',ylabel='Layer',title=key);fig.colorbar(im,ax=ax)
            save(fig,out,'execution_'+file.parent.name)
        table=['# 实测结果','','按各配置已完成完整测试的 EMA 峰值选择；不同 backbone 分开计算前沿。测试选模范围和未完成课程须披露。延迟不参与路线判决。','','| 配置/峰值轮次 | Backbone | mAP (%) | GFLOPs | mean ms |','|---|---|---:|---:|---:|']
        for r in best:table.append(f"| {r['id']} | {r.get('backbone','')} | {100*r['metrics']['average_mAP']:.4f} | {r.get('gflops','未测')} | {r.get('latency_ms','未测')} |")
        if not best:table.append('\n尚无本批完整测试结果；没有推断或补填性能。')
        (out/'REPORT.zh.md').write_text('\n'.join(table)+'\n',encoding='utf-8')
        for name,obj in [('source_manifest',dict(records=records)),('coverage',coverage),('training_distribution',train),('best_models',best)]:
            (out/f'{name}.json').write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    return coverage

def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--output',required=True);p.add_argument('--figspec');p.add_argument('--dry-run',action='store_true');a=p.parse_args();print(json.dumps(analyze(a.manifest,a.output,a.figspec,a.dry_run),indent=2))
if __name__=='__main__':main()
