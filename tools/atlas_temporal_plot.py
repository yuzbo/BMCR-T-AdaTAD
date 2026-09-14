#!/usr/bin/env python3
"""Render the complete temporal slice only, from measured JSON, as PNG/SVG."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.atlas_plot import COLORS,NAMES,panel,plt,style

COUNTS=(4,6,8,10,12,16)
POLICIES=('random','static','attention','uniform','marginal_cf')
SCOPE='Full-data T allocation only; D/S, population and recovery are not covered by these figures.'


def check_complete(data):
    for backbone in ('s','b'):
        part=data[backbone];p=part['provenance']
        if (p['videos'],p['windows'],p['axis'],p['bootstrap'])!=(211,792,'T',10000):
            raise RuntimeError('Temporal figures require complete S/B publication statistics')
        scores=part['scores']
        if set(scores)!={f'{policy}:{n}' for policy in POLICIES for n in COUNTS}:
            raise RuntimeError('All five policies and six budgets must be present')
        for key,record in scores.items():
            c=record['analysis_contract']
            if (record['windows']!=792 or len(set(record['video_ids']))!=211
                or not record['official_AP_reproduced'] or c['backbone']!=backbone
                or c['bootstrap']!=10000 or c['kind']!='allocation'
                or c['variant']!=key or len(record['bootstrap_average'])!=10000):
                raise RuntimeError(f'Incomplete or mismatched AP cache: {backbone}/{key}')


def selection_ledger(raw,backbone):
    # Stream full raw records to recover the measured ranking-forward costs.
    # No predictions are re-evaluated and no model is instantiated.
    files=sorted((raw/f'allocation_{backbone}_T/windows').glob('*.json'))
    if len(files)!=792:raise RuntimeError('Selection ledger requires all 792 windows')
    videos=set();dense=[];base=[];query=[];candidate_counts=[]
    for path in files:
        row=json.loads(path.read_text());videos.add(row['meta']['video_id'])
        b=row['base_gflops'];actions=list(row['marginal_actions'].values())
        dense.append(row['dense_gflops']);base.append(b)
        query.append(b+sum(b+a['delta_gflops'] for a in actions))
        candidate_counts.append(len(actions))
    if len(videos)!=211:raise RuntimeError('Selection ledger lacks complete video coverage')
    return dict(videos=211,windows=792,unit='GFLOPs per window',
        dense_attention_forward=float(np.mean(dense)),
        cf_base_forward=float(np.mean(base)),
        cf_base_and_all_candidate_forwards=float(np.mean(query)),
        candidate_forward_counts=sorted(set(candidate_counts)),
        scope='Ranking forwards shared by the six budgets; additional to the selected execution curve. '
              'Host-side sorting and score arithmetic are not FLOP-profiled. '
              'Random, Uniform and frozen Static require no model forward for online ranking.')


def save(fig,out,name):
    for extension in ('png','svg'):
        fig.savefig(out/f'{name}.{extension}',bbox_inches='tight',pad_inches=.07)
    plt.close(fig)


def budget_figure(data,out):
    fig,axes=plt.subplots(2,2,figsize=(7.16,4.95),sharex='col')
    for col,backbone in enumerate(('s','b')):
        scores=data[backbone]['scores']
        for policy in POLICIES:
            records=[scores[f'{policy}:{n}'] for n in COUNTS]
            x=np.asarray([r['mean_gflops'] for r in records])
            for row,metric in enumerate(('average_mAP','AP07')):
                ax=axes[row,col]
                ax.plot(x,[r[metric] for r in records],color=COLORS[policy],
                    marker='*' if policy=='marginal_cf' else 'o',
                    ls='--' if policy in ('random','static') else '-',
                    lw=1.9 if policy=='marginal_cf' else 1.1,label=NAMES[policy])
                if policy in ('uniform','marginal_cf'):
                    ci=np.asarray([r['average_ci' if row==0 else 'AP07_ci'] for r in records])
                    ax.fill_between(x,ci[:,0],ci[:,1],color=COLORS[policy],alpha=.10,lw=0)
        for row in range(2):
            ax=axes[row,col]
            panel(ax,chr(97+row*2+col),f'AdaTAD-{backbone.upper()} / T allocation')
            ax.set_ylabel('Avg-mAP (%)' if row==0 else 'mAP at tIoU 0.7 (%)')
            if row==1:ax.set_xlabel('Executed GFLOPs / window')
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,ncol=3,loc='lower center',bbox_to_anchor=(.5,.043),frameon=False)
    fig.text(.5,.018,'T-only: 211 videos / 792 windows per model; 95% video-cluster intervals.',
        ha='center',fontsize=7)
    fig.tight_layout(rect=(0,.14,1,1),h_pad=1.7,w_pad=2)
    save(fig,out,'temporal_budget_curves')


def paired_figure(data,out):
    fig,axes=plt.subplots(1,2,figsize=(7.16,2.72))
    for col,backbone in enumerate(('s','b')):
        part=data[backbone];ax=axes[col]
        x=np.asarray([part['scores'][f'uniform:{n}']['mean_gflops'] for n in COUNTS])
        d=[part['paired_headroom'][str(n)] for n in COUNTS]
        y=np.asarray([r['delta_map'] for r in d]);ci=np.asarray([r['ci'] for r in d])
        ax.axhline(0,color='#888888',lw=.8,ls=':')
        ax.vlines(x,ci[:,0],ci[:,1],color='#222222',lw=1.0)
        cap=(x.max()-x.min())*.012
        ax.hlines(ci[:,0],x-cap,x+cap,color='#222222',lw=1.0)
        ax.hlines(ci[:,1],x-cap,x+cap,color='#222222',lw=1.0)
        ax.plot(x,y,color=COLORS['marginal_cf'],marker='o',lw=1.2)
        ax.set(xlabel='Matched execution GFLOPs / window',ylabel='CF reference - Uniform (pp)')
        panel(ax,chr(97+col),f'AdaTAD-{backbone.upper()} / paired Avg-mAP')
    fig.text(.5,.027,'T-only; 10,000 paired video bootstrap samples. Whiskers: pointwise 95% intervals.',
        ha='center',fontsize=7)
    fig.tight_layout(rect=(0,.10,1,1),w_pad=2)
    save(fig,out,'temporal_paired_difference')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--analysis',type=Path,default=ROOT/'analysis/temporal_only.json')
    parser.add_argument('--raw',type=Path,default=ROOT/'results')
    parser.add_argument('--output',type=Path,default=ROOT/'output/temporal')
    args=parser.parse_args();data=json.loads(args.analysis.read_text());check_complete(data)
    ledger={b:selection_ledger(args.raw,b) for b in ('s','b')}
    out=args.output;out.mkdir(parents=True,exist_ok=True);style()
    budget_figure(data,out);paired_figure(data,out)
    captions=dict(temporal_budget_curves=
        'Frozen AdaTAD-S/B on all 211 THUMOS test videos and 792 windows per model. '
        'Six budgets use 4/6/8/10/12/16 predeclared groups of actual frames. '
        'Every point uses full-dataset official AP. Shading shows pointwise 95% '
        'video-cluster intervals for Uniform and the marginal CF reference. '
        'CF is a GT-assisted reference over finite groups, not an oracle or a trained router. '
        'Attention uses dense diagnostic scores. The x-axis is selected execution cost '
        '(encoder, recovery and detection head); additional ranking forwards are in cost_ledger.json.',
        temporal_paired_difference=
        'Marginal CF minus Uniform at the six matched T execution budgets. '
        'Intervals use the same 10,000 video resamples for both strategies. '
        'These are pointwise intervals, not a simultaneous confidence band across budgets. '
        'No D/S, observation-redundancy or new-WTR performance claim follows from this slice.')
    revision=ROOT/'PLOT_REVISION'
    manifest=dict(scope=SCOPE,source_analysis=str(args.analysis.resolve()),
        renderer_revision=revision.read_text().strip() if revision.exists() else 'working-tree',
        source_provenance={b:data[b]['provenance'] for b in ('s','b')},captions=captions,
        files=[f'{name}.{ext}' for name in captions for ext in ('png','svg')],
        visual_review='Pending visual inspection of both PNGs; no PDF generated by this early-slice renderer.')
    (out/'cost_ledger.json').write_text(json.dumps(ledger,indent=2)+'\n')
    (out/'figures.json').write_text(json.dumps(manifest,indent=2)+'\n')
    lines=['# Temporal allocation: complete S/B data','',SCOPE,'',
        '| Model | Groups | Exec. GFLOPs | Uniform mAP | CF mAP | Delta (pp) | Paired 95% CI |',
        '|---|---:|---:|---:|---:|---:|---|']
    for b in ('s','b'):
        for n in COUNTS:
            scores=data[b]['scores'];u=scores[f'uniform:{n}'];cf=scores[f'marginal_cf:{n}']
            d=data[b]['paired_headroom'][str(n)];lo,hi=d['ci']
            lines.append(f'| {b.upper()} | {n} | {u["mean_gflops"]:.1f} | {u["average_mAP"]:.3f} | '
                f'{cf["average_mAP"]:.3f} | {d["delta_map"]:+.3f} | [{lo:+.3f}, {hi:+.3f}] |')
    lines+=['','## Additional ranking model forwards','',
        'Costs below are shared by the six budget points and excluded from their execution x-axes. '
        'They describe the measurement procedure, not the overhead of a trained deployable selector.','',
        '| Model | Dense attention GFLOPs/window | CF base + candidates GFLOPs/window | Candidate forwards/window |',
        '|---|---:|---:|---:|']
    for b,c in ledger.items():
        lines.append(f'| {b.upper()} | {c["dense_attention_forward"]:.1f} | '
            f'{c["cf_base_and_all_candidate_forwards"]:.1f} | {c["candidate_forward_counts"]} |')
    for name,caption in captions.items():lines+=['',f'## {name}','',caption]
    (out/'report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(manifest),flush=True)


if __name__=='__main__':main()
