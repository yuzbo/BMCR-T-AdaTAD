#!/usr/bin/env python3
"""Publication exports from measured reverse-mini and historical B0 summaries."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--reverse-report',required=True)
    parser.add_argument('--rise-report',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();r=json.loads(Path(args.reverse_report).read_text());b=json.loads(Path(args.rise_report).read_text())
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
        'axes.spines.right':False,'savefig.bbox':'tight','pdf.fonttype':42,'svg.fonttype':'none'})
    fig,axes=plt.subplots(2,2,figsize=(13.2,8.5))
    names=['p0','p1','p1_bi','p2'];labels=['P0','P1','P1-bi','P2'];x=np.arange(4)
    colors=['#64748b','#94a3b8','#4776a7','#147d78']
    a=axes[0,0]
    means=[r['aggregate']['held8'][k]['mean']['regret']*1000 for k in names]
    a.bar(x,means,color=colors,width=.65)
    for i,key in enumerate(names):
        values=[v['mean']['regret']*1000 for v in r['seed_evaluations']['held8'][key]]
        a.scatter(i+np.array([-.12,0,.12]),values,s=22,color='#17212b',zorder=3)
        a.text(i,max(means[i],max(values))+.075,f'{means[i]:.3f}',ha='center',fontsize=9)
    stop=r['controls']['held8']['stop']['mean']['regret']*1000
    a.axhline(stop,color='#a74239',ls='--',lw=1.3,label=f'STOP {stop:.3f}')
    a.set(xticks=x,xticklabels=labels,ylabel='Raw loss regret x 1,000 (lower is better)',
        title='A  Unseen exchanges: no learnability recovery',ylim=(0,2.65))
    a.legend(frameon=False,loc='upper left')
    a=axes[0,1]
    keys=['p1_bi_minus_p1','p2_minus_p1_bi','p2_minus_p0']
    for i,key in enumerate(keys):
        m=r['comparisons'][key]['regret'];mean=m['mean']*1000;lo,hi=np.array(m['ci95'])*1000
        a.errorbar(mean,i,xerr=[[mean-lo],[hi-mean]],fmt='o',color=colors[i+1],capsize=4,lw=1.7)
    a.axvline(0,color='#64748b',lw=1,ls='--')
    a.set(yticks=range(3),yticklabels=['P1-bi - P1','P2 - P1-bi','P2 - P0'],ylim=(2.5,-.5),
        xlabel='Paired regret difference x 1,000; video 95% CI',
        title='B  Primary structural comparison crosses zero')
    a=axes[1,0]
    for group,label,color,marker in [('fit8','Seen fit exchanges','#64748b','o'),('held8','Unseen exchanges','#a74239','s')]:
        values=[r['aggregate'][group][k]['mean']['spearman'] for k in names]
        a.plot(x,values,marker=marker,color=color,label=label)
    a.axhline(0,color='#94a3b8',lw=.8)
    a.set(xticks=x,xticklabels=labels,ylim=(-.17,1.08),ylabel='Video-averaged Spearman',
        title='C  Strong fit ranking does not transfer')
    a.legend(frameon=False,loc='center left')
    a=axes[1,1];groups=['fit','calibration','holdout'];positions=np.arange(3)
    total=np.array([b['summary'][k]['seed_state_rows'] for k in groups])
    rank=np.array([b['summary'][k]['rank_order_changes'] for k in groups])
    choice=np.array([b['summary'][k]['choice_changes'] for k in groups])
    a.bar(positions-.18,rank/total*100,width=.34,color='#94a3b8',label='Candidate order changed')
    a.bar(positions+.18,choice/total*100,width=.34,color='#147d78',label='Final choice changed')
    for i,(n,d) in enumerate(zip(choice,total)):
        a.text(i+.18,100*n/d+3,f'{n}/{d}',ha='center',fontsize=9)
    a.set(xticks=positions,xticklabels=['Fit','Calibration','Inner development'],
        ylabel='Fraction of seed-state rows (%)',ylim=(0,115),
        title='D  Historical RISE: held decisions unchanged')
    a.legend(frameon=True,facecolor='white',edgecolor='none',framealpha=.95,loc='center left')
    fig.suptitle('RFV-T: completed reverse-consistency mini and RISE decision diagnosis',fontsize=15,y=.995)
    fig.text(.01,.005,'Reverse: 25 videos, fixed 8/8 split, 3 head seeds; packing-position OOD retained. Dots are seeds; CIs cluster by video.\n'
        'RISE: fixed beta 1.1, saved historical 20/40/60 trajectory; row counts are not independent videos. No task training unlocked.',
        fontsize=9,color='#475569')
    fig.tight_layout(rect=[0,.07,1,.96],w_pad=3,h_pad=2.4)
    for extension in ('png','pdf','svg'):fig.savefig(out/('rfv_reverse_rise_evidence.'+extension),dpi=180)
    plt.close(fig)
    print(json.dumps(dict(output=str(out),reverse_source=r['config']['source_revision'],rise_source=b['binding']['source_revision'])))

if __name__=='__main__':main()
