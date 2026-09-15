"""Reproducible paper figures from executed traces and measured action labels."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle,FancyArrowPatch

COLORS={'frame':'#3665a8','temporal':'#2c9380','depth':'#d68435','spatial':'#9956a0','joint':'#686c73'}
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})


def save(fig,out,name):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    for suffix in ('png','pdf','svg'):fig.savefig(out/f'{name}.{suffix}',dpi=180,bbox_inches='tight')
    plt.close(fig)


def plot_calibration(records,report,out):
    fig,axes=plt.subplots(2,2,figsize=(11,8),layout='constrained')
    for component,title in enumerate(('Classification','Localization')):
        ax=axes[0,component]
        for kind,color in COLORS.items():
            subset=[r for r in records if r['action_type']==kind]
            ax.scatter([r['repair_delta'][component] for r in subset],[r['actual_delta'][component] for r in subset],s=11,alpha=.4,color=color,label=kind)
        ax.axhline(0,color='.65',lw=.6);ax.axvline(0,color='.65',lw=.6)
        ax.set(xlabel='Teacher feature-repair proxy: loss reduction',ylabel='Actual re-execution: loss reduction',title=title+' action value')
    axes[0,0].legend(frameon=False,ncol=2)
    rows=report['oof'];actual=np.asarray([r['actual_delta'] for r in rows]);mu=np.asarray([r['mean'] for r in rows]);sigma=np.asarray([r['sigma'] for r in rows])
    for c,title in enumerate(('Classification','Localization')):
        ax=axes[1,c];ax.scatter(sigma[:,c],np.abs(actual[:,c]-mu[:,c]),s=10,alpha=.35,color='#3665a8')
        ax.set(xlabel='Predicted standard deviation (OOF)',ylabel='Absolute error (OOF)',title=title+' uncertainty')
    fig.suptitle('Fixed final student; router folds split by training video. Test labels are excluded.',fontsize=11)
    save(fig,out,'action_value_and_uncertainty')
    fig,ax=plt.subplots(figsize=(7,4.2),layout='constrained')
    coverage=np.linspace(.1,.95,18)
    from scipy.stats import norm
    for kind in ('budget','frame'):
        subset=[r for r in rows if (r['action_type']=='frame')==(kind=='frame')]
        if not subset:continue
        y=np.asarray([r['actual_delta'] for r in subset]);m=np.asarray([r['mean'] for r in subset]);s=np.asarray([r['sigma'] for r in subset])
        empirical=[float((np.abs(y-m)<=norm.ppf((1+p)/2)*s).mean()) for p in coverage]
        ax.plot(coverage,empirical,'o-',ms=3,label=kind+' before residual calibration')
    ax.plot([0,1],[0,1],'--',color='.6');ax.set(xlabel='Nominal central interval coverage',ylabel='Observed OOF coverage',xlim=(0,1),ylim=(0,1))
    ax.legend(frameon=False,fontsize=9);save(fig,out,'oof_coverage')


def plot_case(case,out,name):
    f=case;fig=plt.figure(figsize=(13,10),layout='constrained');grid=fig.add_gridspec(4,3,height_ratios=[1.1,1.2,1.2,1.2])
    for i in range(3):
        ax=fig.add_subplot(grid[0,i]);ax.imshow(f['rgb'][i]);mask=f['overlays'][i]
        ax.imshow(np.ma.masked_where(mask<=0,mask),cmap='Greens',vmin=0,vmax=1,alpha=.35,extent=(-.5,f['rgb'].shape[2]-.5,f['rgb'].shape[1]-.5,-.5),interpolation='nearest')
        ax.set_title(f'Executed heavy FFN mask; block {f["overlay_block"]}, native {f["overlay_indices"][i]}',fontsize=9);ax.axis('off')
    ax=fig.add_subplot(grid[1,:]);times=f['candidate_times'];valid=f['candidate_valid']
    ax.plot(times[valid],f['actionness'][valid],color='#7085ab',lw=.8,label='Scout actionness')
    chosen=f['selected_indices'][f['selected_valid']];ax.scatter(times[chosen],np.full(len(chosen),-.08),s=4,color='#2c9380',label='Selected real observations')
    for a,b in f['gt_segments_seconds']:ax.axvspan(a,b,color='#db9e41',alpha=.16)
    ax.set(xlim=(times[valid].min(),times[valid].max()),xlabel='Physical video time (s)',ylabel='Actionness / observations',title='Full candidate timeline, actual observation placement, and GT intervals')
    ax.legend(frameon=False,loc='upper right',fontsize=8,ncol=3)
    for slot,key,title in ((grid[2,0:2],'depth_fraction','Admitted attention fraction'),(grid[3,0:2],'heavy_fraction','Heavy FFN fraction')):
        ax=fig.add_subplot(slot);im=ax.imshow(f[key],aspect='auto',vmin=0,vmax=1,cmap='viridis',origin='lower')
        ax.set(xlabel='Selected native state index',ylabel='Block (0 based)',title=title);fig.colorbar(im,ax=ax,shrink=.8)
    ax=fig.add_subplot(grid[2,2]);ax.plot(f['query_times'],f['feature_error'],lw=.8)
    ax.set(xlabel='Physical time (s)',ylabel='1 - cosine similarity',title='Recovery versus full reference')
    ax=fig.add_subplot(grid[3,2]);ax.plot(f['layer_ids'],f['layer_drift'],'o-',ms=3)
    ax.set(xlabel='Block',ylabel='Mean adjacent-layer feature drift',title='State change through depth')
    fig.suptitle(f'{name} | {f["plan"]} | inference uses RGB and masks; GT is drawn after execution',fontsize=11)
    save(fig,out,name)


def architecture(out):
    fig,ax=plt.subplots(figsize=(13,5.8));ax.set(xlim=(0,13),ylim=(0,5.7));ax.axis('off')
    def box(x,y,w,h,text,color):
        ax.add_patch(Rectangle((x,y),w,h,facecolor=color,edgecolor='#536171',lw=.8));ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=10)
    def arrow(a,b):ax.add_patch(FancyArrowPatch(a,b,arrowstyle='->',mutation_scale=14,color='#536171'))
    box(.1,3.5,2,1.2,'Full RGB timeline\n768 candidates','#e6edf6')
    box(2.7,3.5,2,1.2,'H65 / BMCR scout\nframe selection\njoint budget policy','#e2f0e9');arrow((2.1,4.1),(2.7,4.1))
    box(5.3,3.5,2,1.2,'Selected real RGB\nT capacity','#e2f0e9');arrow((4.7,4.1),(5.3,4.1))
    box(7.9,3.5,2.1,1.2,'Persistent states\nA-MoD depth route\nheavy / light FFN','#faeadb');arrow((7.3,4.1),(7.9,4.1))
    box(10.6,3.5,2.2,1.2,'Multi-depth memory\nfull-axis decoder','#eee3f3');arrow((10.,4.1),(10.6,4.1))
    ax.plot([3.7,3.7,11.7],[4.7,5.18,5.18],color='#536171',lw=.9);arrow((11.7,5.18),(11.7,4.7))
    ax.text(7.7,5.28,'Lightweight context over the full time axis',ha='center',fontsize=9)
    box(10.6,1.55,2.2,1.1,'Trainable TAD readout\npoint head / TadTR','#e6edf6');arrow((11.7,3.5),(11.7,2.65))
    box(4.0,1.55,5.6,1.1,'Dense first/last; alternating routed blocks\nPrevious dense attention ranks token capacity\nSkipped states persist; global TIA remains active','#fff5e9')
    arrow((8.9,3.5),(8.9,2.65))
    box(.1,1.55,3.2,1.1,'Training only\nGT + external / shared full\nReal-action supervision','#f4f4f4')
    ax.text(.2,.55,'Primary decision: measured complete-model FLOPs versus full-test mAP.\nLatency, memory, decoding/NMS and training overhead are reported separately.',fontsize=11)
    ax.set_title('Budgeted full-axis state recovery for temporal action detection',fontsize=15,pad=15)
    save(fig,out,'model_and_three_axes')
