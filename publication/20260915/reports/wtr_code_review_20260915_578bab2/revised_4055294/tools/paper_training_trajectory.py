"""Plot complete observed checkpoints; never infer unmeasured milestone scores."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tools.paper_epoch_comparison import family
from h65.paper.runtime import json_write


def main(args):
    source=json.loads(Path(args.snapshot).read_text(encoding='utf-8'))
    rows=[]
    for values in source['evidence'].values():
        for item in values:
            r=item['record'];cfg=r.get('config',{});name=family(cfg.get('comparison',''))
            if cfg.get('seed')!=42 or not name or r.get('checkpoint_state')!='ema' or not isinstance(r.get('epoch'),int):continue
            if r['test_videos']!=211 or r['test_windows']!=792:raise ValueError('Incomplete THUMOS result')
            rows.append(dict(model=name,backbone=cfg['backbone'],epoch=r['epoch'],map=100*r['metrics']['average_mAP'],
                             gflops=r['dataset_mean_gflops'],source=item['source']))
    colors={'Full-V1':'#3779ad','Full-V2':'#b54c3a','Uniform':'#718b5a','PBD-style':'#8865a0','Static':'#b68a30'}
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
    for ax,bb in zip(axes,('s','b')):
        group=[r for r in rows if r['backbone']==bb]
        for name,color in colors.items():
            measured=sorted([r for r in group if r['model']==name],key=lambda r:r['epoch'])
            if not measured:continue
            peak=max(measured,key=lambda r:r['map'])
            ax.plot([r['epoch'] for r in measured],[r['map'] for r in measured],'-o',color=color,lw=1.3,ms=5,
                    label=f'{name}: best {peak["map"]:.3f} @ {peak["epoch"]}')
            ax.scatter(peak['epoch'],peak['map'],marker='*',s=120,color=color,zorder=4)
        ax.set(title=f'VideoMAE-{bb.upper()}',xlabel='Completed evaluation milestone (epoch)',ylabel='Full-test mAP (%)')
        ax.set_xticks(sorted({r['epoch'] for r in group}));ax.grid(alpha=.15);ax.legend(frameon=False,fontsize=8,loc='best')
    fig.suptitle('Observed learning trajectories; complete 80-epoch courses remain active',fontsize=13)
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    for ext in ('png','svg'):fig.savefig(out/f'training_trajectory.{ext}',dpi=200)
    plt.close(fig)
    json_write(out/'training_trajectory.sources.json',dict(snapshot_time=source['time'],rows=rows,
        scope='Markers are complete 211-video/792-window tests; lines connect measured checkpoints only; stars denote test-selected observed peaks, not final epoch80'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--snapshot',required=True);p.add_argument('--output',required=True);main(p.parse_args())
