"""Plot completed historical BMCR80 courses and checkpoint-matched compute receipts."""
import argparse,json,math
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main(args):
    root=Path(args.receipts);out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    summary=json.loads((root/'progress_summary.json').read_text())
    historical=json.loads(Path(args.historical_json).read_text())
    rows=[];comparison=[]
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,2,figsize=(12,7.2),sharex='col')
    colors={'s':'#236b8e','b':'#765090'}
    for col,bb in enumerate(('s','b')):
        selection=summary['selection'][bb]
        assert selection['all_candidates_evaluated'] and selection['evaluated']==12
        best=selection['best'];terminal=selection['at80'];folder=root/'runs'/Path(best['folder']).name
        receipt=json.loads((folder/'completed.json').read_text());profile=json.loads((folder/'profile.json').read_text())
        assert receipt['test_videos']==211 and receipt['test_windows']==792 and not profile['unresolved_matrix_ops']
        candidates=selection['candidates'];x=[r['total_epochs'] for r in candidates];y=[100*r['average_mAP'] for r in candidates]
        ax=axes[0,col];ax.plot(x,y,'o-',color=colors[bb],lw=2,ms=5)
        ax.scatter(best['total_epochs'],100*best['average_mAP'],marker='*',s=170,color=colors[bb],zorder=3)
        ax.text(.04,.94,f"Peak {100*best['average_mAP']:.3f} @ {best['total_epochs']}",
                transform=ax.transAxes,va='top',color=colors[bb])
        if best['total_epochs']!=80:
            ax.annotate(f"Terminal {100*terminal['average_mAP']:.3f}",(80,100*terminal['average_mAP']),
                        xytext=(-108,-28),textcoords='offset points',arrowprops={'arrowstyle':'-','color':'#6b7280'})
        ax.set(title='BMCR-'+bb.upper()+' | complete test mAP',ylabel='mAP @ tIoU 0.3–0.7 (%)',ylim=(min(y)-.5,max(y)+.8))
        epochs=summary['training'][bb]['epochs'];ax=axes[1,col];t=[e['total_epoch'] for e in epochs]
        ax.fill_between(t,[e['cost_p10'] for e in epochs],[e['cost_p90'] for e in epochs],color=colors[bb],alpha=.15,label='P10–P90 across updates')
        ax.plot(t,[e['cost_median'] for e in epochs],color=colors[bb],label='Median training loss')
        ax.set(xlabel='Total epoch (20 warm + 60 joint)',ylabel='Training objective',title='Within-epoch loss distribution')
        ax.legend(frameon=False,fontsize=9)
        for ax in axes[:,col]:ax.set_xlim(20,84);ax.set_xticks(range(20,81,10));ax.grid(alpha=.18)
        dense=historical['models'][bb]['dense'];cross=historical['models'][bb]
        row=dict(backbone=bb,seed=summary['seed'],peak_epoch=best['total_epochs'],peak_map=100*best['average_mAP'],
                 terminal_map=100*terminal['average_mAP'],peak_gflops=profile['matrix_conv_flops']/1e9,
                 peak_latency_ms=1000*profile['latency_mean_seconds'],peak_memory_gib=profile['peak_gib'],
                 latency_scope=profile['scope'],compute_scope=profile['flops_convention'],
                 peak_receipt=str(folder/'completed.json'),peak_profile=str(folder/'profile.json'))
        row.update(cross_gain_pp=cross['map']-row['peak_map'],cross_extra_gflops=cross['gflops']-row['peak_gflops'],
                   saved_vs_dense_percent=100*(1-row['peak_gflops']/dense['gflops']))
        rows.append(row)
        comparison.extend([dict(label='BMCR-'+bb.upper(),family='BMCR',backbone=bb,map=row['peak_map'],gflops=row['peak_gflops'],dense=dense),
                           dict(label='Cross-'+bb.upper(),family='Cross',backbone=bb,map=cross['map'],gflops=cross['gflops'],dense=dense),
                           dict(label='Dense-'+bb.upper(),family='Dense',backbone=bb,map=dense['average_mAP'],gflops=dense['gflops'],dense=dense)])
    fig.suptitle('BMCR80 completed: best checkpoints and terminal behavior',fontsize=16,y=.98)
    fig.text(.5,.015,'Historical seed 3407; 211 test videos / 792 windows at all 12 EMA checkpoints. Loss bands describe updates, not multi-seed uncertainty.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.045,1,.95))
    for ext in ('png','svg'):fig.savefig(out/f'bmcr80_trajectory.{ext}',dpi=200)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(13,4.9));palette={'BMCR':'#25806d','Cross':'#2465a7','Dense':'#555b66'}
    offsets={'BMCR-s':(18,-25),'Cross-s':(18,14),'Dense-s':(-14,14),'BMCR-b':(18,-27),'Cross-b':(18,13),'Dense-b':(-86,12)}
    for r in comparison:
        marker='s' if r['backbone']=='s' else '^';color=palette[r['family']]
        axes[0].scatter(r['gflops'],r['map'],s=80,marker=marker,color=color,zorder=3)
        axes[0].annotate(r['label'],(r['gflops'],r['map']),xytext=offsets[r['family']+'-'+r['backbone']],textcoords='offset points',fontsize=9,color=color,
                         arrowprops={'arrowstyle':'-','lw':.6,'color':color})
        axes[1].scatter(100*r['gflops']/r['dense']['gflops'],100*r['map']/r['dense']['average_mAP'],s=80,marker=marker,color=color,label=r['label'])
    frontier=[];highest=-math.inf
    for r in sorted(comparison,key=lambda r:r['gflops']):
        if r['map']>highest:frontier.append(r);highest=r['map']
    axes[0].plot([r['gflops'] for r in frontier],[r['map'] for r in frontier],'--',color='#a3aab4',lw=1)
    axes[0].set(xlabel='Complete-model matrix/conv GFLOPs per full window',ylabel='Full-test mAP (%)',title='Absolute accuracy and compute',xlim=(800,8700),ylim=(62,72.2))
    axes[1].set(xlabel='Compute relative to same-backbone dense (%)',ylabel='mAP retention relative to dense (%)',title='Same-backbone efficiency',xlim=(45,107),ylim=(90,102))
    axes[1].legend(frameon=False,ncol=2,loc='lower right',fontsize=9)
    for ax in axes:ax.grid(alpha=.18)
    fig.suptitle('Historical efficiency references updated with completed BMCR80',fontsize=15,y=.99)
    fig.text(.5,.025,'Different historical training recipes; descriptive comparison. Dashed frontier uses only these profiled points. H65-S (65.386 mAP) has no matched cost.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.055,1,.94))
    for ext in ('png','svg'):fig.savefig(out/f'bmcr80_efficiency.{ext}',dpi=200)
    plt.close(fig)
    payload=dict(protocol='Historical seed3407; warm20 + joint60; test-based selection among12 EMA checkpoints',models=rows,
                 comparison=comparison,current_seed42_results_included=False,all_test_candidates=summary['selection'])
    (out/'results.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(rows))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--receipts',required=True);p.add_argument('--historical-json',required=True);p.add_argument('--output',required=True)
    main(p.parse_args())
