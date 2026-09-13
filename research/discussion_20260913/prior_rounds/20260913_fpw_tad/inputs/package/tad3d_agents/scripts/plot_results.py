#!/usr/bin/env python3
"""CVPR-sized plots from complete, sourced records. No invented result points.

Every plot is separate; combine panels in LaTeX. Default matplotlib colors are
used with distinct markers. No project weights or videos are required here.
"""
from pathlib import Path
import argparse, csv, json, math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

COLUMN_WIDTH=3.28125
DOUBLE_WIDTH=6.875
NUMERIC=['mean_map_ratio','gflops_2mac','model_mean_ms','model_median_ms']+[f'map_{i}_ratio' for i in ['03','04','05','06','07']]

def load_records(path):
    with open(path,encoding='utf-8-sig',newline='') as f: rows=list(csv.DictReader(f))
    if not rows:raise ValueError('No records: do not plot missing results as zero')
    for row in rows:
        if row.get('status')!='complete':raise ValueError('Partial/planned results cannot be used here')
        if row.get('evidence_kind') not in ('repository_record','new_full_eval'):raise ValueError('No toy/estimated accuracy allowed')
        if not row.get('source_url'):raise ValueError('Every row requires provenance')
        if int(row['test_videos'])!=211 or int(row['test_windows'])!=792:
            raise ValueError('This plotting recipe is only for the pinned 211/792 protocol')
        for key in NUMERIC:
            row[key]=float(row[key])
            if not math.isfinite(row[key]):raise ValueError('Nonfinite/missing metric')
        if not all(0 <= row[k] <= 1 for k in NUMERIC if 'map' in k):raise ValueError('Use AP ratios [0,1]')
        average=sum(row[f'map_{i}_ratio'] for i in ['03','04','05','06','07'])/5
        if abs(average-row['mean_map_ratio'])>1e-8:raise ValueError('Mean mAP differs from five-threshold mean')
    return rows

def finish(fig,ax,out,name,rows):
    ax.tick_params(labelsize=8)
    ax.xaxis.label.set_size(8);ax.yaxis.label.set_size(8)
    ax.legend(fontsize=7,frameon=False,loc='best')
    fig.tight_layout(pad=.8)
    for ext in ('pdf','svg','png'):
        fig.savefig(out/f'{name}.{ext}',dpi=300)
    (out/f'{name}.source.json').write_text(json.dumps(dict(
        evidence_kind='repository_record' if all(r['evidence_kind']=='repository_record' for r in rows) else 'mixed_verified_records',
        width_in=COLUMN_WIDTH,plot_name=name,rows=rows,
        warning='Historical records: distinct official/new training; nodes may differ; not a new paired GPU benchmark.'),indent=2))
    plt.close(fig)

def plot(input_file,out):
    rows=load_records(input_file);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    plt.rcParams['pdf.fonttype']=42;plt.rcParams['ps.fonttype']=42
    plt.rcParams['svg.fonttype']='none'
    markers=['o','s','^','D','v','P']
    for backbone in sorted({r['backbone'] for r in rows}):
        part=[r for r in rows if r['backbone']==backbone]
        if len({r['timing_cohort'] for r in part})!=1:
            raise ValueError('Do not mix timing cohorts in one comparison')
        title=f'VideoMAE-{backbone}: archived records'
        for field,label,suffix in [('gflops_2mac','Matrix/conv GFLOPs (2 MAC)','flops'),
                                   ('model_median_ms','Fixed-window model median (ms)','latency')]:
            fig,ax=plt.subplots(figsize=(COLUMN_WIDTH,2.85))
            for idx,row in enumerate(part):
                ax.plot(row[field],100*row['mean_map_ratio'],marker=markers[idx%len(markers)],
                        linestyle='none',markersize=5,label=row['method'])
            ax.set_title(title,fontsize=9)
            ax.set_xlabel(label);ax.set_ylabel('Average mAP (%)')
            ax.margins(x=.18,y=.25)
            finish(fig,ax,out,f'historical_{backbone}_{suffix}',part)
        fig,ax=plt.subplots(figsize=(COLUMN_WIDTH,2.85))
        for idx,row in enumerate(part):
            ax.plot([.3,.4,.5,.6,.7],[100*row[f'map_{i}_ratio'] for i in ['03','04','05','06','07']],
                    marker=markers[idx%len(markers)],linewidth=1.2,markersize=3.5,label=row['method'])
        ax.set_title(title,fontsize=9);ax.set_xlabel('Temporal IoU threshold');ax.set_ylabel('mAP (%)')
        ax.set_xticks([.3,.4,.5,.6,.7]);ax.margins(y=.15)
        finish(fig,ax,out,f'historical_{backbone}_tiou',part)
    return len(rows)

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input',required=True);ap.add_argument('--out',required=True)
    args=ap.parse_args();print('Plotted verified rows:',plot(args.input,args.out))
