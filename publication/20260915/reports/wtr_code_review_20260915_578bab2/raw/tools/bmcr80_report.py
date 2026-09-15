"""Summarize the BMCR80 trajectory, completed full tests, and training distributions."""
import argparse
import csv
import json
import math
import statistics
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
EXP=ROOT/'bmcr80_20260913'
EPOCHS=tuple(range(25,81,5))
RECIPE='bmcr_corrected_warm20_joint60_v1'


def write_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)


def select_best(runs,backbone):
    candidates=[]
    for epoch in EPOCHS:
        folder=runs/f'{backbone}_bmcr_test_epoch_{epoch:02}';path=folder/'metrics.json'
        if not path.exists():continue
        data=json.loads(path.read_text());init=data['initialization'];metrics=data['metrics']
        if data['test_videos']!=211 or data['test_windows']!=792 or init.get('recipe')!=RECIPE or init['total_epochs']!=epoch:
            raise ValueError(f'Incomplete or wrong BMCR80 candidate: {path}')
        score=metrics['average_mAP']
        if not math.isfinite(score) or abs(score-sum(metrics[f'mAP@0.{i}'] for i in range(3,8))/5)>1e-10:
            raise ValueError(f'Invalid mAP arithmetic: {path}')
        candidates.append(dict(total_epochs=epoch,average_mAP=score,metrics=metrics,
                               checkpoint=init['checkpoint'],state_key=init['state_key'],folder=str(folder)))
    best=max(candidates,key=lambda x:x['average_mAP']) if candidates else None
    through60=[x for x in candidates if x['total_epochs']<=60]
    return dict(backbone=backbone,recipe=RECIPE,test_based_selection=True,
                criterion='maximum full211-test average_mAP,earliest tie',expected_total_epochs=list(EPOCHS),
                all_candidates_evaluated=len(candidates)==len(EPOCHS),evaluated=len(candidates),best=best,
                best_through60=max(through60,key=lambda x:x['average_mAP']) if through60 else None,
                at60=next((x for x in candidates if x['total_epochs']==60),None),
                at80=next((x for x in candidates if x['total_epochs']==80),None),candidates=candidates)


def save_selection(runs,backbone):
    result=select_best(runs,backbone);write_json(runs/f'{backbone}_best_test_checkpoint.json',result);return result


def quantile(values,q):
    values=sorted(values);pos=(len(values)-1)*q;low=int(pos);high=math.ceil(pos)
    return values[low]+(values[high]-values[low])*(pos-low)


def training_summary(runs,backbone):
    folder=runs/f'{backbone}_bmcr';path=folder/'progress.json'
    progress=json.loads(path.read_text()) if path.exists() else None
    records={};raw_count=0
    log=folder/'train.jsonl'
    if log.exists():
        for line in log.read_text().splitlines():
            try:item=json.loads(line)
            except json.JSONDecodeError:continue  # the live final line may still be being written
            raw_count+=1;records[item['successful_updates']]=item
    grouped={}
    for item in records.values():grouped.setdefault(item['epoch']+21,[]).append(item)
    epochs=[]
    for epoch,items in sorted(grouped.items()):
        losses={k:statistics.mean(x['losses'][k] for x in items if k in x['losses'])
                for k in sorted({k for x in items for k in x['losses']})}
        costs=[x['losses']['cost'] for x in items];grads=[x['grad_norm'] for x in items]
        epochs.append(dict(total_epoch=epoch,logged_updates=len(items),complete_epoch=len(items)==100,
                           losses_mean=losses,cost_median=statistics.median(costs),cost_p10=quantile(costs,.1),
                           cost_p90=quantile(costs,.9),grad_norm_mean=statistics.mean(grads),
                           grad_norm_p90=quantile(grads,.9),learning_rates_last=items[-1].get('learning_rates'),
                           logged_step_seconds=sum(x['seconds'] for x in items)))
    return dict(progress=progress,raw_logged_updates=raw_count,retained_update_ids=len(records),epochs=epochs,
                note='Latest log per update retained after epoch-resume; partial epoch is not a saved checkpoint.')


def build_report(exp=EXP,runs=None,state=None):
    runs=runs or exp/'runs';exp.mkdir(parents=True,exist_ok=True)
    result=dict(recipe=RECIPE,course='existing corrected warm20 EMA + fresh joint60',
                train_videos=200,test_videos=211,test_windows=792,seed=3407,
                training={b:training_summary(runs,b) for b in ('s','b')},
                selection={b:save_selection(runs,b) for b in ('s','b')},
                deployment=state or {},comparison_limit='The80-course checkpoint at60 has a different LR history from the earlier60-course terminal.')
    write_json(exp/'progress_summary.json',result)
    lines=['# Corrected BMCR: 20+60 course','',
           'All200 train /211 test /792 windows; seed3407. Existing warm20 is reused once, not retrained.',
           'EMA selected on complete test at total25..80 every5; total60 and80 are retained separately.',
           'The80-course checkpoint at60 differs in LR history from the old60-course terminal. No DS3/clip-selection experiment is resumed.','',
           '| Backbone | Last saved total epoch | Saved joint updates | Complete full tests | Current observed peak |',
           '|---|---:|---:|---:|---:|']
    csv_rows=[]
    for b in ('s','b'):
        p=result['training'][b]['progress'] or {};s=result['selection'][b];best=s['best']
        score=f'{best["average_mAP"]*100:.4f}% @ {best["total_epochs"]}' if best else 'Not available'
        lines.append(f'|{b.upper()}|{p.get("total_epochs","Not started")}|{p.get("successful_updates",0)}|{s["evaluated"]}/12|{score}|')
        for x in s['candidates']:csv_rows.append(dict(backbone=b,total_epoch=x['total_epochs'],**x['metrics']))
    lines+=['','| Backbone | Total epoch | Average mAP (%) | @0.3 | @0.4 | @0.5 | @0.6 | @0.7 |',
            '|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in csv_rows:
        lines.append('|'+row['backbone'].upper()+'|'+str(row['total_epoch'])+'|'+
                     '|'.join(f'{row[k]*100:.4f}' for k in ['average_mAP']+[f'mAP@0.{i}' for i in range(3,8)])+'|')
    if state:
        counts={}
        for stage in state.get('stages',{}).values():counts[stage.get('status','WAITING')]=counts.get(stage.get('status','WAITING'),0)+1
        lines+=['','Stage status: '+json.dumps(counts), 'Updated: '+str(state.get('updated_at',''))]
    lines+=['','Training component means, loss p10/median/p90, gradient norms and LR histories are in progress_summary.json.',
            'Per-batch logs and raw predictions remain in runs/. A single seed does not estimate training variance.']
    (exp/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    with (exp/'test_curve.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=['backbone','total_epoch','average_mAP']+[f'mAP@0.{i}' for i in range(3,8)])
        writer.writeheader();writer.writerows(csv_rows)
    return result


def plot_summary(summary,folder):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    for b,color in [('s','#247c8a'),('b','#b96a36')]:
        points=summary['selection'][b]['candidates']
        if points:axes[0].plot([x['total_epochs'] for x in points],[100*x['average_mAP'] for x in points],'o-',label=b.upper(),color=color)
        rows=[x for x in summary['training'][b]['epochs'] if x['complete_epoch']]
        if rows:
            x=[v['total_epoch'] for v in rows]
            axes[1].plot(x,[v['losses_mean']['cost'] for v in rows],label=b.upper(),color=color)
            axes[1].fill_between(x,[v['cost_p10'] for v in rows],[v['cost_p90'] for v in rows],alpha=.15,color=color)
    axes[0].set(title='Full211-test EMA trajectory',xlabel='Total epoch (warm20 included)',ylabel='Average mAP (%)')
    axes[1].set(title='Training cost: mean and batch p10–p90',xlabel='Total epoch',ylabel='Training loss')
    for ax in axes:
        ax.grid(alpha=.2)
        if ax.lines:ax.legend()
    folder.mkdir(exist_ok=True);fig.savefig(folder/'training_curve.png',dpi=180);fig.savefig(folder/'training_curve.svg');plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--exp',type=Path,default=EXP);p.add_argument('--plot',action='store_true');args=p.parse_args()
    state_path=args.exp/'deployment.json';state=json.loads(state_path.read_text()) if state_path.exists() else None
    summary=build_report(args.exp,state=state)
    if args.plot:plot_summary(summary,args.exp/'figures')
    print(json.dumps({b:{'progress':summary['training'][b]['progress'],'full_tests':summary['selection'][b]['evaluated']} for b in ('s','b')}))
