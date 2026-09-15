"""Render the measured RFV mini results; never reads checkpoints or test labels."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
DATA=ROOT/'evidence'
read=lambda name:json.loads((DATA/name).read_text(encoding='utf8'))
g0,g1,held,b0=map(read,['T_LOCAL_CF.json','GRAPH_G1_T_MINI.json','ACTION_HOLDOUT_DIAGNOSTIC.json','RISE_B_T.json'])
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
    'axes.spines.right':False,'axes.titleweight':'bold','svg.fonttype':'none','pdf.fonttype':42})
fig,axs=plt.subplots(2,2,figsize=(12,8.2),layout='constrained')
fig.suptitle('RFV mini: measurable headroom, unresolved Value generalization',fontsize=15,fontweight='bold')
ax=axs[0,0]
mean=g0['ap_delta_pp'];lo,hi=g0['ap_delta_ci95_pp']
ax.errorbar([mean],[0],xerr=[[mean-lo],[hi-mean]],fmt='o',color='#167B76',capsize=6,linewidth=2)
ax.axvline(0,color='#CBD5E1',linewidth=1)
ax.set(xlim=(-.05,.85),ylim=(-.6,.6),yticks=[],xlabel='Local-CF minus Uniform (mAP percentage points)',
    title='A  Current action-space headroom')
ax.text(.02,.86,'Training-side calibration: 20 videos / 46 windows',transform=ax.transAxes)
ax.text(mean,.12,f'+{mean:.3f} pp  [ +{lo:.3f}, +{hi:.3f} ]',ha='center',fontsize=10)
ax.text(.02,.04,'4 rounds x at most 16 swaps; paired video 95% CI\nGT-assisted reference, not a learned-router test result',transform=ax.transAxes,color='#475569',fontsize=9)

ax=axs[0,1]
keys=['plain_m','plain_l','static_graph','dynamic_graph']
labels=['Plain-M','Plain-L','Static','Dynamic']
values=[g1['holdout'][k]['mean']['regret']*1000 for k in keys]
bars=ax.bar(labels,values,color=['#94A3B8','#64748B','#318A84','#5BAAA4'],width=.65)
for label in ax.bar_label(bars,fmt='%.3f',padding=3,fontsize=9):
    label.set_bbox(dict(facecolor='white',edgecolor='none',pad=.5))
ax.axhline(g1['controls']['stop']['mean']['regret']*1000,color='#9D4544',linestyle='--',label='STOP')
ax.axhline(g1['controls']['uniform_swap']['mean']['regret']*1000,color='#455E88',linestyle=':',label='Random legal swap')
ax.set(ylabel='Regret x1000 (lower is better)',ylim=(0,4.8),title='B  Plain / Graph mini')
ax.legend(frameon=False,fontsize=9,loc='upper right')
ax.text(.02,.04,'10 inner videos; 3-head-seed means\nStatic minus Plain-L 95% CI crosses zero',transform=ax.transAxes,
    fontsize=9,color='#475569',bbox=dict(facecolor='white',edgecolor='none',alpha=.94,pad=4))

ax=axs[1,0]
keys=['fit','held_actions','calibration']
values=[held['aggregate'][key]['mean']['spearman'] for key in keys]
bars=ax.bar(['Fit-8\n25 videos','Held-8\n25 videos','Calibration\n8 videos'],values,
    color=['#318A84','#94A3B8','#B06564'],width=.58)
ax.bar_label(bars,fmt='%.3f',padding=4)
ax.axhline(0,color='#CBD5E1',linewidth=1)
ax.set(ylim=(-.4,1.25),ylabel='Spearman (3-head-seed mean)',title='C  Same-video unseen actions')
ax.text(.02,.96,'Fixed physical 8/8 split; no additional CF labels',transform=ax.transAxes,va='top',fontsize=9,color='#475569')

ax=axs[1,1]
labels=['Current = Post','True EMA']
for index,key in enumerate(['current_post','true_ema']):
    metric=b0['future_minus_control'][key]['regret']
    mean=metric['mean']*1000;lo,hi=np.asarray(metric['ci95'])*1000
    ax.errorbar(mean,index,xerr=[[mean-lo],[hi-mean]],fmt='o',color='#9D4544',capsize=6,linewidth=2)
ax.axvline(0,color='#CBD5E1',linewidth=1)
ax.set(yticks=[0,1],yticklabels=labels,ylim=(-.65,1.8),xlim=(-.015,.055),
    xlabel='Future minus control regret x1000 (negative is better)',title='D  Historical same-state forecast')
ax.text(.02,.92,f'Calibration-selected beta = {b0["beta"]:.2f}; 10 inner videos',transform=ax.transAxes,fontsize=9)
ax.text(.02,.03,'Paired video 95% CI; Future does not beat Post / EMA\nOffline historical diagnosis; FVD remains locked',transform=ax.transAxes,fontsize=9,color='#475569')
for extension in ['png','pdf','svg']:
    fig.savefig(ROOT/f'RFV_MINI_EVIDENCE.{extension}',dpi=180)
plt.close(fig)
print('Rendered RFV_MINI_EVIDENCE.png/pdf/svg from saved experimental reports')
