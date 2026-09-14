"""Fixed-epoch comparisons from complete P0 receipts; no interpolation of results."""
import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def family(name):
    if name.startswith('FULL_V2'):return 'Full-V2'
    if name.startswith('P01'):return 'PBD-style'
    if name.startswith('P00'):return 'Static'
    return {'full':'Full-V1','uniform':'Uniform'}.get(name)

def main(args):
    snapshot=json.loads(Path(args.snapshot).read_text());out=Path(args.output);out.mkdir(parents=True,exist_ok=True);rows=[]
    for items in snapshot['evidence'].values():
        for item in items:
            d=item['record'];cfg=d.get('config',{});label=family(cfg.get('comparison',''))
            if not label or d.get('epoch')!=args.epoch or cfg.get('seed')!=42:continue
            assert d['test_videos']==211 and d['test_windows']==792
            plans=d['plan_distribution'];n=sum(plans.values())
            rows.append(dict(model=label,backbone=cfg['backbone'],epoch=args.epoch,map=100*d['metrics']['average_mAP'],
                mean_gflops=d['dataset_mean_gflops'],representative_gflops=d['representative_gflops'],
                full_dynamic_depth_capacity_percent=100*sum(v for k,v in plans.items() if '_D100_' in k)/n,
                plan_distribution=plans,source=item['source']))
    order={'Full-V1':0,'Full-V2':1,'Uniform':2,'PBD-style':3,'Static':4};rows.sort(key=lambda r:(r['backbone'],order[r['model']]))
    colors={'Full-V1':'#3779ad','Full-V2':'#b54c3a','Uniform':'#718b5a','PBD-style':'#8865a0','Static':'#b68a30'}
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(11,4.8))
    for ax,bb in zip(axes,('s','b')):
        selected=[r for r in rows if r['backbone']==bb]
        for row in selected:
            ax.scatter(row['mean_gflops'],row['map'],s=90,color=colors[row['model']],label=row['model'],zorder=3)
            ax.annotate(f"{row['map']:.3f}",(row['mean_gflops'],row['map']),xytext=(0,12),textcoords='offset points',ha='center',color=colors[row['model']])
        if selected:
            x=[r['mean_gflops'] for r in selected];y=[r['map'] for r in selected]
            padx=max(15,(max(x)-min(x))*.2);pady=max(.10,(max(y)-min(y))*.3)
            ax.set_xlim(min(x)-padx,max(x)+padx);ax.set_ylim(min(y)-pady,max(y)+pady)
            frontier=[r for r in selected if not any(q['mean_gflops']<=r['mean_gflops'] and q['map']>=r['map'] and (q['mean_gflops']<r['mean_gflops'] or q['map']>r['map']) for q in selected)]
            frontier.sort(key=lambda r:r['mean_gflops']);ax.plot([r['mean_gflops'] for r in frontier],[r['map'] for r in frontier],':',color='#a0a5ae',lw=1)
        ax.set(title=f'VideoMAE-{bb.upper()}',xlabel='Full-test mean complete-model GFLOPs / window',ylabel='mAP @ tIoU 0.3–0.7 (%)');ax.grid(alpha=.18)
        ax.legend(frameon=False,loc='lower right',fontsize=9)
    fig.suptitle(f'Epoch {args.epoch}: complete THUMOS tests, single seed 42',fontsize=15,y=.99)
    fig.text(.5,.025,'80-epoch courses are ongoing. Policies choose different costs; this is not a matched-budget causal ablation. Unmeasured controls are omitted.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.05,1,.94))
    for ext in ('png','svg'):fig.savefig(out/f'epoch_{args.epoch:03}_comparison.{ext}',dpi=200)
    plt.close(fig)
    changes=[]
    for bb in ('s','b'):
        group={r['model']:r for r in rows if r['backbone']==bb}
        for control in ('Full-V1','Uniform'):
            if 'Full-V2' in group and control in group:
                a,b=group['Full-V2'],group[control]
                changes.append(dict(backbone=bb,comparison='Full-V2 minus '+control,map_pp=a['map']-b['map'],mean_gflops_change=a['mean_gflops']-b['mean_gflops'],compute_change_percent=100*(a['mean_gflops']/b['mean_gflops']-1)))
    result=dict(snapshot_time=snapshot['time'],epoch=args.epoch,seed=42,rows=rows,changes=changes)
    (out/'comparison.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--snapshot',required=True);parser.add_argument('--epoch',type=int,required=True);parser.add_argument('--output',required=True)
    main(parser.parse_args())
