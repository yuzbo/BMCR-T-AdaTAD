"""Uni-AdaFocus-inspired figures using measured H65/BMCR TAD records only."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import re
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from h65.paper.runtime import json_write

COLORS={'Cross':'#202020','Joint T/D/S':'#397BA8','Depth':'#C5663D','Spatial':'#378D6B',
        'TCN':'#83629E','Feature loss':'#C27C96','Interpolation / controls':'#77A6AF','Original-head control':'#A6A6A6'}
POLICIES=[(100,100,'Full D/S','#181818','*','-'),(100,48,'Spatial 48%','#3C80AA','D',':'),
          (50,100,'A-MoD 50%','#429580','^',':'),(50,48,'A-MoD 50% + spatial 48%','#C16B46','s',':')]
FONT=10.5
plt.rcParams.update({'font.family':'DejaVu Serif','font.size':FONT,'axes.titlesize':12,'axes.labelsize':11,
    'legend.fontsize':8.5,'axes.linewidth':.8,'xtick.labelsize':9,'ytick.labelsize':9,
    'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','axes.axisbelow':True})


def family(row):
    name=row['id']
    if 'anchor_head' in name:return 'Original-head control'
    for prefix,label in [('R03','Cross'),('J01','Joint T/D/S'),('D02','Depth'),('S02','Spatial'),('R02','TCN'),('R04','Feature loss')]:
        if name.startswith(prefix):return label
    return 'Interpolation / controls'


def normal(row):
    return row['state']=='ema' and not row.get('runtime_override') and row['selector']==row['training_selector']


def load_data(path):
    saved=json.loads(Path(path).read_text());rows=saved['records'] if isinstance(saved,dict) else saved
    for row in rows:
        if row['dataset']!='thumos' or row['head']!='point':raise ValueError('This figure bundle uses the THUMOS point-head cohort')
        if not np.isfinite(row['gflops']) or not 0<=row['average_mAP']<=100:raise ValueError('Invalid measured point')
        if not np.isclose(row['average_mAP'],row['metrics']['average_mAP']*100,atol=1e-9):raise ValueError('mAP units differ')
    fixed=[];comparison=json.loads((ROOT/'fidelity_20260911/FINAL_COMPARISON.json').read_text())
    for row in comparison['legacy_reference']:
        if row['method']=='official' or (row['method']=='bmcr' and row['backbone']=='B'):
            fixed.append(dict(label=('Official '+row['backbone']) if row['method']=='official' else 'Legacy BMCR-B',
                              backbone=row['backbone'].lower(),gflops=row['matrix_conv_gflops'],
                              average_mAP=100*row['metrics']['average_mAP'],official=row['method']=='official',
                              source='fidelity_20260911/FINAL_COMPARISON.json'))
    fixed.append(next(dict(label='Corrected H65-S',backbone='s',gflops=r['matrix_conv_gflops'],
                          average_mAP=100*r['metrics']['average_mAP'],official=False,
                          source='fidelity_20260911/FINAL_COMPARISON.json')
                      for r in comparison['results'] if r['backbone']=='S' and r['selection']=='test_peak'))
    best={b:max((r for r in rows if r['id']==f'R03_cross_{b}' and normal(r)),key=lambda r:r['average_mAP']) for b in ('s','b')}
    return rows,fixed,best


def joint_grid(rows,b):
    selected=[r for r in rows if r['id']==f'J01_joint_{b}' and r['epoch']==5 and r['state']=='ema' and r['selector']=='anchor']
    cfg=json.loads((ROOT/'configs/frame'/f'J01_joint_{b}.json').read_text());grid={}
    for row in selected:
        match=re.search(r'_K(\d+)_D(\d+)_S(\d+)_Q(\d+)',row['runtime_variant'])
        if match:
            k,d,s,q=map(int,match.groups())
            if q!=100:continue
        else:k,d,s=cfg['budget'],round(cfg['engine']['depth_ratio']*100),round(cfg['engine']['spatial_ratio']*100)
        key=(k,d,s)
        if key in grid:raise ValueError('Duplicate same-checkpoint operating point')
        grid[key]=dict(row,frames=k,depth=d,space=s)
    expected={(k,d,s) for k in (384,768) for d in (50,100) for s in (48,100)}
    if set(grid)!=expected:raise ValueError('Incomplete eight-point same-checkpoint grid')
    return grid


def style(ax):
    ax.grid(True,color='#D8D8D8',linewidth=.55,alpha=.75)
    ax.tick_params(direction='out',length=3)
    for spine in ax.spines.values():spine.set_color('#606060')


def footer(fig,text):fig.text(.5,.028,text,ha='center',va='bottom',fontsize=8.4,linespacing=1.45)


def measured_envelope(rows):
    ordered=sorted(rows,key=lambda r:(r['gflops'],-r['average_mAP']));out=[];highest=-np.inf
    for row in ordered:
        if row['average_mAP']>highest+1e-10:out.append(row);highest=row['average_mAP']
    return out


def overview(rows,fixed,best):
    fig,axes=plt.subplots(1,2,figsize=(12.2,6.2));fig.subplots_adjust(left=.07,right=.985,top=.87,bottom=.32,wspace=.24)
    envelope=measured_envelope(rows+fixed)
    for ax in axes:
        for label,color in COLORS.items():
            group=[r for r in rows if family(r)==label]
            ax.scatter([r['gflops'] for r in group],[r['average_mAP'] for r in group],s=26,color=color,alpha=.63,
                       edgecolors='white',linewidths=.3,zorder=3)
        ax.plot([r['gflops'] for r in envelope],[r['average_mAP'] for r in envelope],color='#333333',lw=1.4,ls='--',zorder=2)
        for row in fixed:
            ax.scatter(row['gflops'],row['average_mAP'],s=90 if row['official'] else 62,marker='P' if row['official'] else 's',
                       color='#333333' if row['official'] else '#416782',edgecolor='white',linewidth=.7,zorder=6)
        for row in best.values():ax.scatter(row['gflops'],row['average_mAP'],s=150,marker='*',color='black',edgecolor='white',linewidth=.6,zorder=7)
        style(ax);ax.set_xlabel('Inference cost (GFLOPs / full window)');ax.set_ylabel('TAD mAP (%)')
    axes[0].set(xlim=(500,8750),ylim=(49,73),xticks=[1000,3000,5000,7000,8500],title='(a) All measured operating points')
    axes[1].set(xlim=(840,2500),ylim=(58.5,70.3),xticks=[1000,1400,1800,2200,2500],title='(b) Lower-compute region')
    for row in fixed:
        ax=axes[0]
        offset=(-8,7) if row['label']=='Official B' else (8,5) if row['official'] else (8,-12)
        ax.annotate(row['label'],(row['gflops'],row['average_mAP']),xytext=offset,ha='right' if row['label']=='Official B' else 'left',textcoords='offset points',fontsize=8.2)
        if row['backbone']=='s':axes[1].annotate(row['label'],(row['gflops'],row['average_mAP']),xytext=(-8,6) if row['official'] else (8,-12),
            ha='right' if row['official'] else 'left',textcoords='offset points',fontsize=8.2)
    axes[0].annotate('Cross-B best',(best['b']['gflops'],best['b']['average_mAP']),xytext=(8,6),textcoords='offset points',fontsize=8.5)
    axes[1].annotate('Cross-S best',(best['s']['gflops'],best['s']['average_mAP']),xytext=(10,4),textcoords='offset points',fontsize=8.5)
    box=Rectangle((840,58.5),1660,11.8,fill=False,edgecolor='#7F7F7F',lw=.8)
    axes[0].add_patch(box)
    handles=[Line2D([],[],ls='',marker='o',ms=5,color=color,label=label) for label,color in COLORS.items()]
    handles += [Line2D([],[],ls='--',color='#333333',label='Observed envelope*'),
                Line2D([],[],ls='',marker='*',ms=8,color='black',label='Best measured Cross EMA')]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.105),ncol=4,frameon=False,columnspacing=1.2,handletextpad=.45)
    fig.suptitle('Compute-performance distribution of H65/BMCR routes',fontsize=15,y=.985)
    footer(fig,f'{len(rows)} complete evaluations + {len(fixed)} fixed references; THUMOS14, 211 videos / 792 windows, mAP@0.3:0.1:0.7.\n'
           'FLOPs: full-window operator profiles (2 MAC), not video totals. *Envelope mixes models/checkpoints; no fitted points.')
    return fig


def budget_curves(rows,fixed):
    fig,axes=plt.subplots(1,2,figsize=(12.2,6.2));fig.subplots_adjust(left=.07,right=.985,top=.86,bottom=.26,wspace=.25)
    grids={b:joint_grid(rows,b) for b in ('s','b')}
    for ax,b in zip(axes,('s','b')):
        grid=grids[b]
        for d,s,label,color,marker,line in POLICIES:
            points=[grid[(k,d,s)] for k in (384,768)]
            ax.plot([p['gflops'] for p in points],[p['average_mAP'] for p in points],ls=line,marker=marker,
                    color=color,lw=1.7 if d==s==100 else 1.4,ms=9 if marker=='*' else 5.5,label=label,zorder=3)
        for ref in (r for r in fixed if r['backbone']==b):
            ax.scatter(ref['gflops'],ref['average_mAP'],marker='P' if ref['official'] else 's',s=78,
                       color='#505050' if ref['official'] else '#93AFB5',edgecolors='white',linewidth=.6,zorder=5)
            ax.annotate(ref['label'],(ref['gflops'],ref['average_mAP']),xytext=(-7,6) if ref['official'] else (8,-12),
                        ha='right' if ref['official'] else 'left',textcoords='offset points',fontsize=8.2,
                        bbox=dict(facecolor='white',edgecolor='none',alpha=.92,pad=.15))
        style(ax);ax.set_ylabel('TAD mAP (%)');ax.set_xlabel('Inference cost (GFLOPs / full window)')
        ax.set_title(f'({"a" if b=="s" else "b"}) VideoMAE-{b.upper()} - J01, epoch 5 EMA')
        low=grid[(384,100,100)];high=grid[(768,50,48 if b=='s' else 100)]
        y=62.55 if b=='s' else 66.35;ratio=high['gflops']/low['gflops']
        ax.annotate('',(low['gflops'],y),(high['gflops'],y),arrowprops=dict(arrowstyle='<->',color='#1A9B62',lw=1.4))
        ax.text((low['gflops']+high['gflops'])/2,y-.47,f'{ratio:.2f}x FLOP ratio',ha='center',color='#118551',fontsize=9.3,
                bbox=dict(facecolor='white',edgecolor='none',alpha=.92,pad=.15))
        for p in (low,high):ax.plot([p['gflops'],p['gflops']],[y,p['average_mAP']],color='#91B99F',ls=':',lw=.8,zorder=1)
        ax.annotate('T=384',(low['gflops'],low['average_mAP']),xytext=(-8,8),ha='right',textcoords='offset points',fontsize=8)
        full=grid[(768,100,100)];ax.annotate('T=768',(full['gflops'],full['average_mAP']),xytext=(-4,-13),ha='right',textcoords='offset points',fontsize=8)
        if b=='s':ax.set(xlim=(830,2530),ylim=(58.4,70.3),xticks=[1000,1400,1800,2200,2500])
        else:ax.set(xlim=(2850,8650),ylim=(59.2,72.4),xticks=[3000,4500,6000,7500,8500])
    handles=[Line2D([],[],color=c,marker=m,ls=l,lw=1.5,label=label) for d,s,label,c,m,l in POLICIES]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.15),ncol=4,frameon=False,columnspacing=1.6)
    fig.suptitle('Measured budget curves from the same checkpoint',fontsize=15,y=.985)
    footer(fig,'Each backbone uses one checkpoint and all 8 measured T/D/S settings; each curve changes T only (384 -> 768).\n'
        'Segments connect observations; no fitted or extrapolated points. Green brackets compare measured FLOPs, not runtime speed.')
    return fig,grids


def checkpoint_spread(rows,fixed):
    fig,axes=plt.subplots(1,2,figsize=(12.2,5.7));fig.subplots_adjust(left=.07,right=.985,top=.85,bottom=.24,wspace=.23)
    for ax,b in zip(axes,('s','b')):
        for ident,label,color,marker in [(f'R03_cross_{b}','Cross', '#202020','*'),(f'R04_feature_only_{b}','Feature loss only','#C27C96','o'),(f'R02_tcn_{b}','TCN','#83629E','D')]:
            group=sorted((r for r in rows if r['id']==ident and normal(r)),key=lambda r:r['epoch'])
            if not group:continue
            if np.ptp([r['gflops'] for r in group])>.01:raise ValueError('Checkpoint spread requires a fixed compute setting')
            ax.plot([r['epoch'] for r in group],[r['average_mAP'] for r in group],color=color,marker=marker,ms=8 if marker=='*' else 5,
                    lw=1.5,ls='-' if len(group)>1 else '',label=f'{label}: {group[0]["gflops"]:.0f} G')
            if ident.startswith('R03'):
                peak=max(group,key=lambda r:r['average_mAP']);ax.annotate(f'{peak["average_mAP"]:.3f}%',(peak['epoch'],peak['average_mAP']),
                    xytext=(0,10),ha='center',textcoords='offset points',fontsize=9)
                for row in rows:
                    if row['id']==ident and row['state']=='learned' and not row.get('runtime_override'):
                        ax.scatter(row['epoch'],row['average_mAP'],marker='x',s=52,color='#D06541',zorder=4,label='Cross online endpoint')
        baseline=next(r for r in fixed if r['backbone']==b and not r['official'])
        ax.axhline(baseline['average_mAP'],ls=':',color='#708F9C',lw=1.1,label=f'{baseline["label"]}: {baseline["gflops"]:.0f} G')
        style(ax);ax.set(xlim=(3.5,21.5),xticks=[5,10,15,20],xlabel='Decoder-course epochs (fixed anchor)',ylabel='TAD mAP (%)',
                        title=f'({"a" if b=="s" else "b"}) VideoMAE-{b.upper()}')
        ax.set_ylim((63.1,64.88) if b=='s' else (67.1,68.85))
        ax.legend(loc='lower right',frameon=True,facecolor='white',edgecolor='#D6D6D6',framealpha=.98,fontsize=8,borderpad=.6)
    fig.suptitle('Performance spread across checkpoints at fixed computation',fontsize=15,y=.975)
    footer(fig,'Fixed-compute checkpoint trajectories, not budget curves or confidence intervals; GFLOPs are shown in the legends.\n'
               'Feature-loss-only retains GT boundary weighting. Epochs denote additional decoder training on a fixed anchor.')
    return fig


def main(args):
    rows,fixed,best=load_data(args.records);out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    first=overview(rows,fixed,best);second,grids=budget_curves(rows,fixed);third=checkpoint_spread(rows,fixed)
    stems=['01_compute_performance_distribution','02_same_checkpoint_budget_curves','03_checkpoint_performance_spread']
    pdf=out/'compute_performance_curves.pdf'
    with PdfPages(pdf,metadata={'Title':'H65/BMCR compute-performance curves','Author':'H65/BMCR research','Subject':'Measured TAD results; visual reference: Uni-AdaFocus Figs 9-10'}) as book:
        for fig,stem in zip((first,second,third),stems):
            fig.savefig(out/(stem+'.png'),dpi=220);fig.savefig(out/(stem+'.svg'));book.savefig(fig);plt.close(fig)
    json_write(out/'plot_data.json',dict(records=rows,fixed_references=fixed,
        same_checkpoint_grids={b:[r for key,r in sorted(g.items())] for b,g in grids.items()}))
    ratios={}
    for b in ('s','b'):
        low=grids[b][(384,100,100)];high=grids[b][(768,50,48 if b=='s' else 100)]
        ratios[b]=dict(low_cost_point=[low['gflops'],low['average_mAP']],high_cost_point=[high['gflops'],high['average_mAP']],
                       high_over_low_flops=high['gflops']/low['gflops'],low_minus_high_map_pp=low['average_mAP']-high['average_mAP'])
    manifest=dict(records=len(rows),fixed_references=len(fixed),same_checkpoint_points={b:len(g) for b,g in grids.items()},
        dataset='THUMOS14',test_videos=211,test_windows=792,metric='mean mAP@0.3:0.1:0.7 in percent',
        cost_unit='actual complete-model GFLOPs per fixed first full 768-candidate window; matrix/conv 2 MAC',
        source_records=str(Path(args.records).resolve()),source_snapshot='2026-09-13 20:35:36 +0800',
        style_reference=dict(title='Uni-AdaFocus: Spatial-temporal Dynamic Computation for Video Recognition',
            url='https://arxiv.org/html/2412.11228v1',figures=[9,10],pdf_page=11,
            reused='axes, overview-plus-zoom layout, markers, dotted measured-setting connectors; no author result values copied'),
        curve_semantics='J01 epoch5 EMA, seed3407, same weights; D/S fixed within each two-point T curve',
        envelope_semantics='upper envelope of available evaluated models/checkpoints, not a single-model curve',
        green_brackets=ratios,pdf=pdf.name,panels=stems,
        exclusions=['unmeasured paper_v2 models','ANet and InternVideo results not yet measured','cancelled original 16-clip selection routes'],
        limitations=['no whole-video GFLOPs available','no multi-seed confidence bands fabricated',
                     'historical H65-S 65.3857% is retained in project records; no paired operator profile added here'])
    json_write(out/'manifest.json',manifest)
    readme=f'''**计算量与性能分布图：参考 Uni-AdaFocus 图9/10**

采用当前已保存的{len(rows)}次完整THUMOS测试与{len(fixed)}个固定参照。横轴为完整模型的实际GFLOPs/768候选帧完整窗口（2 MAC），纵轴为211视频/792窗口上的mAP@0.3:0.1:0.7。原论文是视频识别，数值不与我们的TAD指标混合；这里只参考呈现方式。

01：全部已测点与低计算区放大。虚线是不同模型/检查点的已观察上包络，不是单个模型的连续预算曲线。

02：严格同checkpoint预算曲线。每个骨干都固定J01 epoch5 EMA、seed3407；每条线固定D/S，仅改变T384→768，共8点。绿色括号均由真实端点计算，不表示延迟加速：S为{ratios['s']['high_over_low_flops']:.3f}倍计算量差，低计算点mAP高{ratios['s']['low_minus_high_map_pp']:.3f}个百分点；B为{ratios['b']['high_over_low_flops']:.3f}倍，低计算点mAP高{ratios['b']['low_minus_high_map_pp']:.3f}个百分点。单checkpoint结果不宣称统计显著，也不作为新80轮模型的最终结论。

03：相同计算设置下的checkpoint性能分布。连线沿实际训练epoch，EMA与online区分，没有虚构多seed置信区间。

未测的新论文模型、ANet、InternVideo不补点。旧16候选clip路线不重新纳入。历史H65-S 65.3857%的高分保留在项目协议中，本图不为它填造配对FLOP值。

参考：[Uni-AdaFocus论文图9/10](https://arxiv.org/html/2412.11228v1)。实际查看了PDF第11页。

数据与每点路径：plot_data.json；口径与数值核验：manifest.json。PNG用于展示，SVG和三页PDF用于论文排版。

使用本次固定数据重绘：`python tools/paper_focus_plot.py --records research/paper/unit_focus_20260913/plot_data.json --output research/paper/unit_focus_20260913`
'''
    (out/'README.zh.md').write_text(readme,encoding='utf-8')
    print(json.dumps(dict(pdf=str(pdf),records=len(rows),same_checkpoint_points=16,ratios=ratios),ensure_ascii=False,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--records',default=str(ROOT/'research/paper/figures/full_test_records.json'))
    p.add_argument('--output',default=str(ROOT/'research/paper/unit_focus_20260913'));main(p.parse_args())
