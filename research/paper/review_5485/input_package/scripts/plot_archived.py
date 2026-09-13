"""Plot only supplied archived numerical records; never creates new model data."""
from __future__ import annotations
from pathlib import Path
import csv,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def render(root: Path) -> list[str]:
    out=root/'figures';out.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':9,'axes.titlesize':10,'axes.labelsize':9,
                         'legend.fontsize':8,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none'})
    with (root/'data/factorial_records.csv').open(encoding='utf-8-sig') as f: rows=list(csv.DictReader(f))
    files=[]
    def save(fig,name,source,selection,meaning):
        for ext in ('pdf','svg','png'):
            path=out/f'{name}.{ext}'
            fig.savefig(path,dpi=220)
            files.append(str(path.relative_to(root)))
        (out/f'{name}.sources.json').write_text(json.dumps({'source_csv':source,'filter':selection,'meaning':meaning,'data_scope':'archived full tests, not new seed42 results','units':'mAP percentage, not mAP fraction'},ensure_ascii=False,indent=2),encoding='utf-8')
        plt.close(fig)
    for bb in ('S','B'):
        m={r['T']+r['D']+r['S']:float(r['average_mAP_pct']) for r in rows if r['backbone']==bb}
        fig,ax=plt.subplots(figsize=(6.875,3.6))
        for s,mark,sty in [('0','o','-'),('1','s','--')]:
            keys=[t+d+s for t,d in [('0','0'),('0','1'),('1','0'),('1','1')]]
            values=[m[k] for k in keys]
            ax.plot(range(4),values,marker=mark,linestyle=sty,label='FFN capacity = '+('1.00' if s=='0' else '0.48'))
            for x,y in enumerate(values):ax.annotate(f'{y:.2f}',(x,y),xytext=(0,7 if s=='0' else -13),textcoords='offset points',ha='center',fontsize=8)
        ax.set_xticks(range(4),['K768 / D1','K768 / D0.5','K384 / D1','K384 / D0.5'])
        ax.set_ylabel('Average mAP (%)');ax.set_ylim(57.5,72.0)
        ax.set_title(f'Archived J01-{bb}: same checkpoint, epoch 5 EMA, seed3407')
        ax.legend(loc='lower left');ax.grid(axis='y',alpha=.25)
        fig.subplots_adjust(left=.10,right=.97,top=.86,bottom=.21)
        fig.text(.10,.035,'D is capacity in routed layers, not fraction of network layers retained.\nThese are interventions, not eight independently trained seed42 models.',fontsize=8)
        save(fig,f'archived_factorial_{bb}','data/factorial_records.csv',{'backbone':bb},'Existing conditional damage and interactions; no claim of recovered accuracy')
    with (root/'data/historical_frontier.csv').open(encoding='utf-8-sig') as f: front=list(csv.DictReader(f))
    fig,ax=plt.subplots(figsize=(6.875,3.6))
    for row in front:
        x,y=float(row['gflops']),float(row['mAP_pct'])
        is_internal=row['role']=='internal_precursor'
        ax.scatter([x],[y],marker='s' if is_internal else 'o',s=42)
        ax.annotate(row['method'],(x,y),xytext=(7,-13 if is_internal else 7),textcoords='offset points',fontsize=8)
    ax.set_xlabel('Complete-model GFLOPs / fixed full 768-candidate window (2MAC)')
    ax.set_ylabel('Best reported average mAP (%)');ax.set_xlim(0,10300);ax.set_ylim(62.5,73)
    ax.set_title('Archived evidence: cross-backbone absolute comparison')
    ax.grid(alpha=.25)
    fig.subplots_adjust(left=.10,right=.97,top=.86,bottom=.23)
    fig.text(.10,.035,'Internal Cross results are not published competitors or current seed42 outcomes.\nOfficial S dominates archived Cross-B in both accuracy and absolute computation.',fontsize=8)
    save(fig,'archived_absolute_comparison','data/historical_frontier.csv',{},'Correctly exposes absolute cross-backbone dominance; first-full-window cost only')
    return files

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);a=p.parse_args()
    print('\n'.join(render(a.root)))
