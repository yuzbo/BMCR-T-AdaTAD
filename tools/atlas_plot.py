#!/usr/bin/env python3
"""Publication figures from frozen JSON only. This script never executes a model."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.offsetbox import OffsetImage,AnnotationBbox
from PIL import Image

COLORS={'O_interpolate':'#7A5195','D':'#0072B2','S':'#009E73','T':'#D55E00',
        'uniform':'#0072B2','random':'#56B4E9','static':'#777777','attention':'#E69F00','marginal_cf':'#111111'}
NAMES={'O_interpolate':'Observation replacement','D':'Depth upgrade','S':'FFN upgrade','T':'Frame exchange',
       'uniform':'Uniform','random':'Random','static':'Static (development)',
       'attention':'Attention (diagnostic)','marginal_cf':'Marginal CF reference'}
METHODS=('packed_naive','physical_interpolation','cross_without_multidepth','cross_multidepth')
METHOD_NAMES=('Packed linear','Physical interpolation','Cross, MD disabled','Cross + multidepth')


def style():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.labelsize':8,
        'axes.titlesize':9,'xtick.labelsize':7,'ytick.labelsize':7,'legend.fontsize':7,
        'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,
        'axes.spines.right':False,'axes.linewidth':.6,'lines.linewidth':1.4,
        'lines.markersize':3.7,'savefig.dpi':400,'figure.dpi':150,
        'axes.grid':True,'grid.alpha':.16,'grid.linewidth':.5,'axes.axisbelow':True})


def panel(ax,label,title):
    ax.set_title(f'({label}) {title}',loc='left',pad=7,fontweight='semibold')


def export(fig,name,caption,out,book,manifest):
    fig.savefig(out/f'{name}.pdf',bbox_inches='tight',pad_inches=.06)
    fig.savefig(out/f'{name}.svg',bbox_inches='tight',pad_inches=.06)
    fig.savefig(out/f'{name}.png',bbox_inches='tight',pad_inches=.06)
    book.savefig(fig,bbox_inches='tight',pad_inches=.06)
    manifest.append(dict(name=name,pdf=f'{name}.pdf',svg=f'{name}.svg',png=f'{name}.png',caption=caption))
    plt.close(fig)


def figure2(data,out,book,manifest):
    fig,axes=plt.subplots(2,3,figsize=(7.16,4.45))
    keys=('O_interpolate','D','S','T')
    for row,b in enumerate(('s','b')):
        for key in keys:
            d=data[b]['distributions'][key];v=d['video_balanced'];color=COLORS[key]
            x=np.asarray(d['cdf_x']);keep=x>0
            axes[row,0].plot(x[keep],100*np.asarray(v['cdf'])[keep],color=color,
                label=NAMES[key]+(' (eligible only)' if key=='T' else ''))
            lo,hi=np.asarray(v['cdf_ci'])*100
            axes[row,0].fill_between(x[keep],lo[keep],hi[keep],color=color,alpha=.09,lw=0)
            c=v['concentration']
            if c is not None:
                axes[row,1].plot(d['lorenz_x'],c['lorenz'],color=color)
                if v['lorenz_ci'] is not None:
                    axes[row,1].fill_between(d['lorenz_x'],*v['lorenz_ci'],color=color,alpha=.08,lw=0)
                p=np.asarray([5,10,20,40,60])
                axes[row,2].plot(p,100*np.asarray(c['top_p']),marker='o',color=color)
                if v['top_p_ci'] is not None:
                    lo,hi=np.asarray(v['top_p_ci'])*100
                    axes[row,2].fill_between(p,lo,hi,color=color,alpha=.08,lw=0)
        axes[row,0].set_xscale('log');axes[row,0].set_ylim(0,102)
        axes[row,0].set_xlabel('Relative task-effect tolerance');axes[row,0].set_ylabel('Actions below tolerance (%)')
        axes[row,0].axvline(.01,color='#999999',ls=':',lw=.8)
        axes[row,1].plot([0,1],[0,1],color='#999999',ls=':',lw=.8)
        axes[row,1].set(xlabel='Cumulative fraction of actions',ylabel='Cumulative positive effect',xlim=(0,1),ylim=(0,1))
        axes[row,2].set(xlabel='Highest-effect actions (%)',ylabel='Share of positive effect (%)',ylim=(0,102))
        for col,title in enumerate(('Low marginal task effect','Positive-effect concentration','Top-p concentration')):
            panel(axes[row,col],chr(97+row*3+col),f'AdaTAD-{b.upper()}: {title}')
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=2,bbox_to_anchor=(.5,-.025),frameon=False)
    fig.tight_layout(rect=(0,.06,1,1),w_pad=1.5,h_pad=2)
    export(fig,'fig2_task_necessity',
        'All 211 videos and 792 windows per model. Video-balanced estimates and 95% video-cluster intervals. '
        'Observation replacement, light-to-heavy upgrades and fixed-K frame exchanges are distinct conditional quantities. '
        'Concentration is computed on the positive part; signed raw effects and action-weighted estimates are retained in the data. '
        'Frame-exchange curves use the eligible action sets; per-axis video/action counts are included in the source data.',out,book,manifest)


def coalition_figure(data,out,book,manifest):
    fig,axes=plt.subplots(2,3,figsize=(7.16,4.1))
    for r,b in enumerate(('s','b')):
        for c,axis in enumerate(('O','D','S')):
            ax=axes[r,c]
            for policy,color,marker in [('stratified','#0072B2','o'),('low_single_effect','#D55E00','s')]:
                sizes=[];joint=[];summed=[];lower=[];upper=[]
                for size in (1,2,4,8,16):
                    record=data[b]['coalitions'].get(f'{axis}:{policy}:{size}')
                    if record and record['mean'] is not None:
                        sizes.append(size);joint.append(record['mean'][0]);summed.append(record['mean'][1])
                        lower.append(record['ci'][0][0]);upper.append(record['ci'][1][0])
                ax.plot(sizes,joint,color=color,marker=marker,label=f'{policy.replace("_"," ")}: joint')
                ax.fill_between(sizes,lower,upper,color=color,alpha=.10,lw=0)
                ax.plot(sizes,summed,color=color,ls='--',alpha=.65,label='sum of single effects' if policy=='stratified' else None)
            ax.axhline(0,color='#999999',lw=.6);ax.set_xscale('log',base=2)
            ax.set_xticks([1,2,4,8,16],[1,2,4,8,16]);ax.set_xlabel('Number of jointly changed actions')
            ax.set_ylabel('Relative task effect')
            panel(ax,chr(97+r*3+c),f'AdaTAD-{b.upper()} / {axis}')
    axes[0,0].legend(frameon=False,fontsize=6.5)
    fig.tight_layout(w_pad=1.5,h_pad=2)
    export(fig,'fig2_joint_interventions',
        'Joint intervention effects versus the sum of individually measured effects. Every coalition is actually re-executed. '
        'Solid and dashed lines distinguish joint measurements from additive approximations. Neither sign of interaction establishes global submodularity.',out,book,manifest)


def figure3(data,out,book,manifest):
    fig,axes=plt.subplots(2,3,figsize=(7.16,4.65))
    model_colors={'s':'#0072B2','b':'#D55E00'}
    centers=np.arange(10)*.05+.025
    for component in (0,1):
        ax=axes[0,component]
        for b in ('s','b'):
            values=[data[b]['boundary'].get(f'D:{i}') for i in range(10)]
            y=np.array([v['mean'][component] if v else np.nan for v in values])
            lo=np.array([v['ci'][0][component] if v else np.nan for v in values])
            hi=np.array([v['ci'][1][component] if v else np.nan for v in values])
            ax.plot(centers,y,color=model_colors[b],marker='o',label=f'AdaTAD-{b.upper()}')
            ax.fill_between(centers,lo,hi,color=model_colors[b],alpha=.13,lw=0)
        ax.axhline(0,color='#777777',lw=.6);ax.set_xlabel('Boundary distance / action duration')
        ax.set_ylabel('Relative marginal task effect')
        panel(ax,chr(97+component),('Classification','Localization')[component])
    names=('start','interior','end','background')
    for b,offset in [('s',-.12),('b',.12)]:
        values=[data[b]['regions'].get(f'D:{name}') for name in names]
        y=[sum(v['mean']) if v else np.nan for v in values]
        axes[0,2].bar(np.arange(4)+offset,y,.23,color=model_colors[b],label=b.upper())
        lo=np.asarray([v['total_ci'][0] if v else np.nan for v in values]);hi=np.asarray([v['total_ci'][1] if v else np.nan for v in values])
        axes[0,2].errorbar(np.arange(4)+offset,y,yerr=np.maximum(0,np.stack((np.asarray(y)-lo,hi-np.asarray(y)))),fmt='none',ecolor='#333333',capsize=2,lw=.6)
        d=[data[b]['duration'].get(f'D:{i}') for i in range(3)]
        y=np.asarray([v['total_mean'] if v else np.nan for v in d])
        lo=np.asarray([v['total_ci'][0] if v else np.nan for v in d]);hi=np.asarray([v['total_ci'][1] if v else np.nan for v in d])
        axes[1,0].errorbar(np.arange(3),y,yerr=np.maximum(0,np.stack((y-lo,hi-y))),marker='o',color=model_colors[b],capsize=2,label=b.upper())
    axes[0,2].set_xticks(range(4),['Start','Interior','End','Background'],rotation=20)
    axes[0,2].set_ylabel('Relative total task effect');axes[0,2].axhline(0,color='#777777',lw=.6)
    panel(axes[0,2],'c','GT-conditional locations')
    axes[1,0].set_xticks(range(3),['Short','Medium','Long']);axes[1,0].set_ylabel('Relative total task effect')
    panel(axes[1,0],'d','Action duration')
    proxies=('attention_score','actionness','entropy','feature_norm')
    labels=('Attention','Actionness','Entropy','Feature norm')
    for col,metric in [(1,'spearman'),(2,'ndcg')]:
        ax=axes[1,col]
        for b,offset in [('s',-.12),('b',.12)]:
            records=[data[b]['proxies'].get(f'D:{p}:{metric}') for p in proxies]
            y=np.asarray([v['mean'] if v else np.nan for v in records])
            lo=np.asarray([v['ci'][0] if v else np.nan for v in records]);hi=np.asarray([v['ci'][1] if v else np.nan for v in records])
            ax.errorbar(np.arange(4)+offset,y,yerr=np.maximum(0,np.stack((y-lo,hi-y))),fmt='o',capsize=2,color=model_colors[b])
        ax.set_xticks(range(4),labels,rotation=22,ha='right')
        ax.set_ylabel('Spearman correlation' if metric=='spearman' else 'NDCG at top 20%')
        panel(ax,chr(101+col-1),'Can simple scores rank update value?')
    axes[0,0].legend(frameon=False)
    fig.tight_layout(w_pad=1.5,h_pad=2)
    export(fig,'fig3_localization_structure',
        'TAD-conditional analysis is separate from population sampling. Classification and localization effects, duration strata, '
        'and proxy ranking quality are reported without assuming boundary dominance. Intervals resample videos; duration cutoffs come from training annotations.',out,book,manifest)


def figure4(data,out,book,manifest):
    for b in ('s','b'):
        fig,axes=plt.subplots(2,3,figsize=(7.16,4.5))
        for col,axis in enumerate(('T','D','S')):
            scores=data[b][axis]['scores']
            for policy in ('random','static','attention','uniform','marginal_cf'):
                records=[scores[f'{policy}:{n}'] for n in (4,6,8,10,12,16)]
                x=np.asarray([r['mean_gflops'] for r in records])
                for row,metric in enumerate(('average_mAP','AP07')):
                    y=np.asarray([r[metric] for r in records])
                    axes[row,col].plot(x,y,color=COLORS[policy],marker='*' if policy=='marginal_cf' else 'o',
                        ls='--' if policy in ('random','static') else '-',lw=1.9 if policy=='marginal_cf' else 1.1,label=NAMES[policy])
                    ci=np.asarray([r['average_ci' if row==0 else 'AP07_ci'] for r in records])
                    if policy in ('uniform','marginal_cf'):
                        axes[row,col].fill_between(x,ci[:,0],ci[:,1],color=COLORS[policy],alpha=.10,lw=0)
                    axes[row,col].set_xlabel('Executed GFLOPs / window')
                    axes[row,col].set_ylabel('Avg-mAP (%)' if row==0 else 'mAP at tIoU 0.7 (%)')
                    panel(axes[row,col],chr(97+row*3+col),f'{axis} allocation / AdaTAD-{b.upper()}')
        h,l=axes[0,0].get_legend_handles_labels()
        fig.legend(h,l,loc='lower center',ncol=3,bbox_to_anchor=(.5,-.025),frameon=False)
        fig.tight_layout(rect=(0,.065,1,1),w_pad=1.4,h_pad=1.8)
        export(fig,f'fig4_allocation_{b}',
            f'Frozen AdaTAD-{b.upper()}, all 211 videos/792 windows. Every plotted budget is actually executed and evaluated with full-dataset AP. '
            'The reference searches a finite, predeclared set of legal action groups and is not an oracle. Shading is a 95% video-cluster interval. '
            'Execution cost includes recovery/head; privileged CF queries and dense-attention scoring are reported separately.',out,book,manifest)


def figure5(data,out,book,manifest):
    fig,axes=plt.subplots(2,3,figsize=(7.16,4.55))
    palette=['#999999','#E69F00','#56B4E9','#0072B2']
    for r,b in enumerate(('s','b')):
        records=[data[b]['scores'][m] for m in METHODS]
        x=np.arange(4)
        axes[r,0].bar(x,[v['average_mAP'] for v in records],color=palette,width=.66)
        axes[r,0].set_xticks(x,['Packed','Physical','Cross\nMD off','Cross\nMD on'])
        axes[r,0].set_ylabel('Avg-mAP (%)')
        panel(axes[r,0],chr(97+r*3),f'V2-{b.upper()}: same-support recovery')
        gap=np.asarray(data[b]['gap_edges'])
        centers=np.r_[(gap[:-1]+gap[1:])/2,gap[-1]*1.5]
        for m,color in zip(METHODS,palette):
            values=[data[b]['gap'].get(f'{m}:{i}') for i in range(len(gap))]
            y=[v['mean'] if v else np.nan for v in values]
            axes[r,1].plot(centers,y,marker='o',color=color,label=METHOD_NAMES[METHODS.index(m)])
        axes[r,1].set_xlabel('Distance to observed support (s)');axes[r,1].set_ylabel('Feature NMSE')
        panel(axes[r,1],chr(98+r*3),'Fidelity across support gaps')
        error=[];missing=[]
        for method in METHODS:
            record=data[b]['endpoint'][method]
            error.append(record['error']['mean'] if record['error']['mean'] is not None else np.nan)
            missing.append(record['missing_percent']['mean'] if record['missing_percent']['mean'] is not None else np.nan)
        axes[r,2].bar(x,error,color=palette,width=.66)
        axes[r,2].set_xticks(x,['Packed','Physical','Cross\nMD off','Cross\nMD on'])
        axes[r,2].set_ylabel('Matched endpoint error (s)')
        panel(axes[r,2],chr(99+r*3),'Boundary error and missing matches')
        for i,(y,miss) in enumerate(zip(error,missing)):
            if np.isfinite(y):axes[r,2].annotate(f'{miss:.1f}% miss',(i,y),xytext=(0,4),textcoords='offset points',ha='center',fontsize=6)
    axes[0,1].legend(frameon=False,fontsize=6.5)
    fig.tight_layout(w_pad=1.5,h_pad=2)
    export(fig,'fig5_original_axis_recovery',
        'All recovery variants consume identical selected RGB support, heavy anchors and head weights. '
        'Disabling multidepth is a component intervention on an existing checkpoint, not a separately trained architecture. '
        'NMSE uses the same frozen encoder with full observation/full updates as its reference. Missing prediction matches are reported alongside conditional endpoint error.',out,book,manifest)


def figure1(data,raw,out,book,manifest):
    cases=data['s']['cases']
    if len(cases)!=3:raise RuntimeError('Three predeclared empirical quantile cases are required')
    fig=plt.figure(figsize=(7.16,6.6));grid=fig.add_gridspec(4,1,height_ratios=[.65,1.5,1.5,1.5],hspace=.6)
    top=fig.add_subplot(grid[0]);top.axis('off')
    boxes=[(.08,'Fixed RGB\n& physical time'),(.36,'Frozen dense\nTAD model'),(.66,'Original prediction\n+ cls / loc loss')]
    for x,label in boxes:
        top.text(x,.64,label,ha='left',va='center',fontsize=9,bbox=dict(boxstyle='round,pad=.35',fc='#F2F5F7',ec='#78909C',lw=.7),transform=top.transAxes)
    for a,b in [(.28,.35),(.59,.65)]:top.annotate('',xy=(b,.64),xytext=(a,.64),xycoords='axes fraction',arrowprops=dict(arrowstyle='->',lw=1,color='#455A64'))
    top.text(.5,-.03,'Change one observation OR one computation action; retain the same model and task coordinates.',ha='center',fontsize=8,transform=top.transAxes)
    for r,case in enumerate(cases):
        row=json.loads((raw/'population_s/windows'/f'{case["window_index"]:05d}.json').read_text())
        sub=grid[r+1].subgridspec(2,1,height_ratios=[.68,1],hspace=.1)
        upper=fig.add_subplot(sub[0]);lower=fig.add_subplot(sub[1],sharex=upper)
        meta=row['meta'];times=np.asarray(meta['frame_indices'])/meta['fps'];valid=meta['valid_candidates']
        left,right=times[0],times[valid-1]
        upper.set_xlim(left,right);upper.set_ylim(-.35,1.25);upper.axis('off')
        upper.text(0,1.2,f'{int(case["quantile"]*100)}th-percentile concentration: {case["video_id"]}',transform=upper.transAxes,fontweight='semibold',fontsize=8)
        strip_path=raw/'population_s'/row['thumbnail_path']
        strip=Image.open(strip_path)
        indices=row['thumbnail_candidate_indices']
        for i in (1,5,9,13):
            frame=strip.crop((i*160,0,(i+1)*160,160))
            x=times[indices[i]]
            upper.add_artist(AnnotationBbox(OffsetImage(np.asarray(frame),zoom=.18),(x,.68),frameon=False))
        for target in meta['gt']:
            start,end=target['segment'];a=max(left,start);b=min(right,end)
            if b>a:upper.broken_barh([(a,b-a)],(-.22,.11),facecolors='#0072B2')
        predicted=row['predictions'].get(meta['video_id'],[])
        for target in sorted(predicted,key=lambda p:-p['score'])[:5]:
            start,end=target['segment'];a=max(left,start);b=min(right,end)
            if b>a:upper.broken_barh([(a,b-a)],(-.07,.08),facecolors='#E69F00',alpha=.75)
        for axis,color in [('O','#7A5195'),('D','#0072B2')]:
            values={}
            for item in row['records']:
                if item['sampling']!='population' or item['action']['axis']!=axis:continue
                if axis=='O' and item['action']['operation']!='interpolate':continue
                t=item['position']['time_seconds'];values.setdefault(t,[]).append(item['value']/max(sum(row['dense_loss_cls_reg']),1e-8))
            points=sorted(values);lower.plot(points,[np.mean(values[t]) for t in points],color=color,marker='o',label='Observation' if axis=='O' else 'Depth update')
        lower.axhline(0,color='#999999',lw=.6);lower.set_ylabel('Relative effect');lower.set_xlabel('Physical time (s)')
        if r==0:lower.legend(frameon=False,ncol=2,loc='upper right',fontsize=6.5)
    fig.subplots_adjust(left=.1,right=.985,bottom=.06,top=.96)
    export(fig,'fig1_controlled_interventions',
        'Controlled measurement and three videos selected at the predeclared 10th, 50th and 90th percentiles of positive depth-effect concentration. '
        'Blue/orange temporal bars show GT/dense predictions. Observations and internal computation are distinct interventions. '
        'Illustrative cases do not substitute for the full-dataset statistics; depth points average sampled layer actions at the same time.',out,book,manifest)


def write_cost_ledger(data,out):
    ledger=dict(allocation={b:{axis:data['allocation'][b][axis]['cost_ledger']
        for axis in ('T','D','S')} for b in ('s','b')},
        recovery={b:data['recovery'][b]['cost_ledger'] for b in ('s','b')})
    lines=['# Model-forward compute ledger','',
        'All costs are mean GFLOPs per window over the complete 211-video / 792-window cohort.','',
        'The allocation figure uses selected execution cost. Marginal-CF ranking forwards '
        '(one base plus all candidates) and the dense-attention diagnostic forward are additional. '
        'A ranking is shared by the six budget points. Host-side score and sort arithmetic is not profiled.','',
        '| Model | Axis | Cheap base | CF ranking | Dense attention | Candidates |',
        '|---|---|---:|---:|---:|---|']
    for b in ('s','b'):
        for axis,c in ledger['allocation'][b].items():
            lines.append(f'| {b.upper()} | {axis} | {c["base_execution"]:.3f} | '
                f'{c["marginal_cf_ranking_forwards"]:.3f} | {c["attention_diagnostic_forward"]:.3f} | '
                f'{c["candidate_forward_counts"]} |')
    lines+=['','## Same-support recovery','',
        'Every variant includes shared support plus recovery and detector execution. '
        'The full-observation feature target is a separate extra encoder/interpolation pass '
        'that reuses the shared preview.','',
        '| Model | Shared support | Extra full-observation target |',
        '|---|---:|---:|']
    for b,c in ledger['recovery'].items():
        lines.append(f'| {b.upper()} | {c["shared_support"]:.3f} | {c["full_observation_target_extra"]:.3f} |')
    lines+=['','| Model | Variant | Recovery + detector only |','|---|---|---:|']
    for b,c in ledger['recovery'].items():
        for method in METHODS:
            lines.append(f'| {b.upper()} | {METHOD_NAMES[METHODS.index(method)]} | '
                f'{c["recovery_and_head"][method]:.3f} |')
    (out/'compute_cost_ledger.json').write_text(json.dumps(ledger,indent=2)+'\n')
    (out/'compute_cost_ledger.md').write_text('\n'.join(lines)+'\n')
    return dict(json='compute_cost_ledger.json',markdown='compute_cost_ledger.md')


def main():
    p=argparse.ArgumentParser();p.add_argument('--analysis',required=True);p.add_argument('--raw',required=True);p.add_argument('--output',required=True)
    args=p.parse_args();analysis=Path(args.analysis);raw=Path(args.raw);out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    data={name:json.loads((analysis/f'{name}.json').read_text()) for name in ('population','allocation','recovery')}
    if data['population']['protocol']['videos']!=211 or data['population']['protocol']['windows']!=792:
        raise RuntimeError('Formal figures require complete publication coverage')
    for b in ('s','b'):
        collections=[data['allocation'][b][a]['scores'] for a in ('T','D','S')]+[data['recovery'][b]['scores']]
        for scores in collections:
            for name,record in scores.items():
                if record.get('windows')!=792 or len(set(record.get('video_ids',[])))!=211 or not record.get('official_AP_reproduced'):
                    raise RuntimeError(f'Formal performance figure has incomplete or unverified AP: {b}/{name}')
                if record.get('analysis_contract',{}).get('backbone')!=b:
                    raise RuntimeError('Performance cache backbone provenance mismatch')
    style();manifest=[]
    with PdfPages(out/'publication_atlas.pdf') as book:
        figure1(data['population'],raw,out,book,manifest)
        figure2(data['population'],out,book,manifest)
        coalition_figure(data['population'],out,book,manifest)
        figure3(data['population'],out,book,manifest)
        figure4(data['allocation'],out,book,manifest)
        figure5(data['recovery'],out,book,manifest)
    revision_file=Path(__file__).resolve().parents[1]/'PLOT_REVISION'
    record=dict(figures=manifest,bundle='publication_atlas.pdf',source_analysis=str(analysis.resolve()),
                renderer_revision=revision_file.read_text().strip() if revision_file.exists() else 'working-tree',
                sources={b:dict(population=data['population'][b]['provenance'],
                    allocation={a:data['allocation'][b][a]['provenance'] for a in ('T','D','S')},
                    recovery=data['recovery'][b]['provenance']) for b in ('s','b')},
                full_videos=211,full_windows=792,models=['s','b'],cost_ledger=write_cost_ledger(data,out),
                visual_review='Rendered PNGs require visual inspection')
    (out/'figures.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record),flush=True)


if __name__=='__main__':main()
