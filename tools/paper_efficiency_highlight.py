"""Lead with best audited efficiency results; retain absolute comparisons in appendices."""
import json
from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
import paper_focus_plot as focus
from h65.paper.runtime import json_write


def data():
    folder=ROOT/'research/paper/unit_focus_20260913';snapshot=json.loads((folder/'current_snapshot.json').read_text())
    rows,fixed,_=focus.load_data(folder/'plot_data.json');result={}
    for b in ('s','b'):
        candidates=[r for r in snapshot['frame'] if r['record']['config']['id']==f'R03_cross_{b}' and r['record'].get('id')==f'R03_cross_{b}' and r['record']['checkpoint_state']=='ema']
        best=max(candidates,key=lambda r:r['record']['metrics']['average_mAP']);record=best['record']
        dense=next(r for r in fixed if r['backbone']==b and r['official'])
        anchor=next(r for r in fixed if r['backbone']==b and not r['official'])
        bmc=max((r for r in snapshot['bmcr80'] if Path(r['source']).parent.name.startswith(b+'_')),key=lambda r:r['record']['metrics']['average_mAP'])
        result[b]=dict(backbone=b.upper(),map=100*record['metrics']['average_mAP'],gflops=record['gflops'],epoch=record['epoch'],
            source=best['source'],dense=dense,anchor=anchor,bmcr80_map=100*bmc['record']['metrics']['average_mAP'],bmcr80_source=bmc['source'],
            bmcr80_epoch=int(Path(bmc['source']).parent.name.split('_')[-1]))
        r=result[b];r.update(cost_used_pct=100*r['gflops']/dense['gflops'],cost_saved_pct=100*(1-r['gflops']/dense['gflops']),
            map_retention_pct=100*r['map']/dense['average_mAP'],dense_gap_pp=r['map']-dense['average_mAP'],
            anchor_gain_pp=r['map']-anchor['average_mAP'],bmcr80_gain_pp=r['map']-r['bmcr80_map'])
    return rows,fixed,result,snapshot['snapshot_time']


def hero(result):
    fig=plt.figure(figsize=(12.2,6.5));fig.suptitle('Best audited compressed models: accuracy retained, compute reduced',fontsize=15,y=.975)
    for index,b in enumerate(('b','s')):
        r=result[b];left=.10+index*.49;center=left+.175
        fig.text(center,.88,f'VideoMAE-{b.upper()} backbone',ha='center',fontsize=12)
        fig.text(center,.795,f'{r["cost_saved_pct"]:.1f}% fewer FLOPs',ha='center',fontsize=22,fontweight='bold',color='#187B57')
        fig.text(center,.735,f'{r["map_retention_pct"]:.1f}% of dense mAP retained',ha='center',fontsize=13,color='#273E49')
        ax=fig.add_axes([left,.30,.35,.34]);y=np.array([1.,0.]);width=np.array([r['map_retention_pct'],r['cost_used_pct']])
        ax.barh(y,[100,100],height=.46,color='#E2E5E6',zorder=1)
        ax.barh(y,width,height=.28,color=['#263D49','#2E906B'],zorder=2)
        for yi,xi in zip(y,width):ax.text(xi+1.8,yi,f'{xi:.1f}%',va='center',fontsize=11,fontweight='bold')
        ax.set(xlim=(0,111),ylim=(-.65,1.65),xticks=[0,25,50,75,100],yticks=y,
               yticklabels=['mAP retained','FLOPs used'],xlabel='Percent of same-backbone dense model')
        ax.grid(axis='x',color='#DADADA',lw=.5);ax.set_axisbelow(True)
        for edge in ('top','right','left'):ax.spines[edge].set_visible(False)
        ax.spines['bottom'].set_color('#AAA');ax.tick_params(axis='y',length=0)
        fig.text(center,.215,f'Dense: {r["dense"]["average_mAP"]:.2f}% mAP  |  {r["dense"]["gflops"]:,.0f} GFLOPs',ha='center',fontsize=10.2,color='#63676B')
        fig.text(center,.164,f'Cross: {r["map"]:.2f}% mAP  |  {r["gflops"]:,.0f} GFLOPs',ha='center',fontsize=11.5,fontweight='bold')
        fig.text(center,.115,f'Absolute mAP difference: {r["dense_gap_pp"]:+.2f} pp; selected EMA epoch {r["epoch"]}',ha='center',fontsize=9.6,color='#6B5650')
    focus.footer(fig,'Zero-based bars; grey bars are the dense reference (100%). All absolute mAP values and gaps remain visible.\n'
        'THUMOS14 full tests; complete-model GFLOPs per full 768-candidate window (2 MAC). This is a within-backbone comparison.')
    return fig


def gains(result):
    fig,axes=plt.subplots(1,2,figsize=(12.2,6.1));fig.subplots_adjust(left=.10,right=.975,bottom=.24,top=.78,wspace=.38)
    for ax,b in zip(axes,('b','s')):
        r=result[b];values=[r['anchor']['average_mAP'],r['bmcr80_map'],r['map']]
        labels=[r['anchor']['label'],f'BMCR80 (e{r["bmcr80_epoch"]})',f'Cross (e{r["epoch"]} EMA)']
        ax.scatter(values,[0,1,2],s=[90,100,155],marker='o',color=['#A9B4B8','#6F8F99','#187B57'],zorder=4)
        for y,value in enumerate(values):ax.annotate(f'{value:.3f}%',(value,y),xytext=(7,7),textcoords='offset points',fontsize=10)
        ax.annotate('',(values[2],1.55),(values[1],1.55),arrowprops=dict(arrowstyle='->',color='#187B57',lw=1.8))
        ax.text((values[1]+values[2])/2,1.70,f'{r["bmcr80_gain_pp"]:+.2f} pp',ha='center',color='#187B57',fontsize=12,fontweight='bold')
        ax.set(yticks=[0,1,2],yticklabels=labels,xlabel='TAD mAP (%)',ylim=(-.55,2.8),title=f'VideoMAE-{b.upper()} / K=384 observations')
        if b=='b':ax.set_xlim(67.1,69.3)
        else:ax.set_xlim(63.1,65.85);ax.axvline(65.3857244379457,ls=':',lw=1.1,color='#A19583')
        ax.grid(axis='x',alpha=.25,lw=.6)
        for edge in ('top','right','left'):ax.spines[edge].set_visible(False)
        ax.tick_params(axis='y',length=0,labelsize=8.8);ax.spines['bottom'].set_color('#AAA')
        if b=='s':ax.text(65.3857244379457,2.62,'Historical H65-S\n65.386%, cost unverified',ha='right',fontsize=8.5,color='#8B8070')
    fig.suptitle('Best measured checkpoints at the same K384 observation budget',fontsize=14.5,y=.97)
    fig.text(.5,.88,f'Cross versus the latest corrected BMCR peaks: B {result["b"]["bmcr80_gain_pp"]:+.2f} pp; S {result["s"]["bmcr80_gain_pp"]:+.2f} pp',ha='center',fontsize=11,color='#187B57')
    focus.footer(fig,'Same observation budget, not identical total FLOPs or training recipes. Corrected BMCR costs are not imputed.\n'
        'Cross includes decoder cost; its audited FLOPs appear on the main page. Historical S accuracy is retained without inventing a cost point.')
    return fig


def main():
    folder=ROOT/'research/paper/unit_focus_20260913';rows,fixed,result,stamp=data()
    best={b:max((r for r in rows if r['id']==f'R03_cross_{b}' and focus.normal(r)),key=lambda r:r['average_mAP']) for b in ('s','b')}
    figs=[hero(result),gains(result),focus.overview(rows,fixed,best),focus.budget_curves(rows,fixed)[0],focus.checkpoint_spread(rows,fixed)]
    names=['00_best_efficiency_summary','00b_same_frame_budget_gains','01_compute_performance_distribution','02_same_checkpoint_budget_curves','03_checkpoint_performance_spread']
    with PdfPages(folder/'compute_performance_curves.pdf',metadata={'Title':'H65/BMCR best-model efficiency results','Author':'H65/BMCR research'}) as pdf:
        for fig,name in zip(figs,names):
            pdf.savefig(fig);fig.savefig(folder/(name+'.png'),dpi=220);fig.savefig(folder/(name+'.svg'));plt.close(fig)
    json_write(folder/'best_efficiency_data.json',dict(snapshot_time=stamp,models=result,main_claim='within-backbone inference efficiency; no claim of exceeding dense mAP',
        historical_s=dict(map=65.3857244379457,gflops=None,source='diagnostics/h65_65_gap_20260911/historical_run/terminal_evaluation.json'),
        appendix='Absolute cross-backbone distribution, all tested T/D/S budgets and checkpoint trajectories are retained.'))
    manifest=json.loads((folder/'manifest.json').read_text());manifest.update(pdf_pages=5,main_panels=names[:2],appendix_panels=names[2:],
        best_model_snapshot=stamp,interpretation='Emphasize audited efficiency and current K384 gains; keep absolute dense gaps and all original distributions.')
    json_write(folder/'manifest.json',manifest)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
