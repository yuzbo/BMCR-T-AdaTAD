"""Plot measured checkpoints of the completed historical BMCR80 courses."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
data=json.loads((ROOT/'live_snapshot.json').read_text(encoding='utf-8'))
rows=[]
for item in data['evidence']['bmcr80']:
    r=item['record']
    rows.append(dict(backbone=r['backbone'],epoch=r['initialization']['total_epochs'],map=100*r['metrics']['average_mAP'],source=item['source']))
fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
for ax,bb,color in zip(axes,('s','b'),('#2563a6','#b35a35')):
    line=sorted((r for r in rows if r['backbone']==bb),key=lambda r:r['epoch'])
    best=max(line,key=lambda r:r['map'])
    ax.plot([r['epoch'] for r in line],[r['map'] for r in line],'-o',color=color,lw=1.8,ms=4)
    ax.scatter([best['epoch']],[best['map']],marker='*',s=120,color=color,zorder=3)
    ax.annotate(f"Best {best['map']:.4f}% at {best['epoch']}",(best['epoch'],best['map']),xytext=(0,13),textcoords='offset points',ha='center',fontsize=9)
    ax.axvline(80,color='.6',ls=':',lw=1)
    ax.set(title=f'Internal BMCR-{bb.upper()}: historical seed 3407',xlabel='Total epoch (warm20 + joint)',ylabel='THUMOS14 average mAP (%)',xlim=(22,83),ylim=(50,71))
    ax.set_xticks(range(25,81,5));ax.grid(axis='y',alpha=.2);ax.spines[['top','right']].set_visible(False)
    if bb=='s':ax.annotate('New epoch75: 63.7866%',(75,line[-1]['map']),xytext=(-8,-25),textcoords='offset points',ha='right',fontsize=9)
fig.suptitle('Completed BMCR80 training; only fully evaluated checkpoints are plotted',fontsize=12)
fig.supxlabel('Full 211 videos / 792 windows; EMA peak selection disclosed. Epoch80 test is still missing.',fontsize=9)
fig.savefig(ROOT/'bmcr80_measured_course.png',dpi=180)
fig.savefig(ROOT/'bmcr80_measured_course.svg')
plt.close(fig)
(ROOT/'plot_data.json').write_text(json.dumps(rows,indent=2)+'\n')
