"""Render the saved full-evaluation trajectory; never evaluates a model."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    p=argparse.ArgumentParser();p.add_argument('--snapshot',required=True);p.add_argument('--output',required=True)
    args=p.parse_args();r=json.loads(Path(args.snapshot).read_text());out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
        'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})
    fig,axes=plt.subplots(1,2,figsize=(12,4.6),sharey=True)
    for axis,letter in zip(axes,('d','s')):
        for policy,label,color in [('u','Uniform','#64748b'),('v','Value','#147d78')]:
            source=r['courses'][f'wtr_{letter}_{policy}_s42']['metrics']
            epochs=sorted(map(int,source));values=[source[str(e)]['metrics']['average_mAP']*100 for e in epochs]
            axis.plot(epochs,values,'o-',color=color,label=label,ms=5,lw=1.7)
        axis.axvline(80,color='#a74239',ls='--',lw=1)
        axis.text(78,63.05,'Primary endpoint\nnot yet available',ha='right',color='#a74239',fontsize=9)
        axis.set(xlim=(6,84),ylim=(63,65.8),xticks=[10,20,40,60,80],xlabel='Adaptation epoch',title=letter.upper()+' axis: complete-dataset milestones')
        axis.legend(frameon=False,loc='lower left')
        axis.grid(axis='y',color='#e2e8f0',lw=.6)
    axes[0].set_ylabel('Average mAP (%)')
    fig.suptitle('Existing D/S component courses: Value has not established a gain',fontsize=13)
    fig.text(.02,.015,'211 videos / 792 windows per point; EMA; one training seed. All available milestones shown; no test-peak selection.\n'
        'Science4055294; snapshot '+r['recorded_at']+'. These courses use fixed T and group quotas.',fontsize=9,color='#475569')
    fig.tight_layout(rect=[0,.12,1,.93])
    for ext in ('png','pdf','svg'):fig.savefig(out/('ds_course_trajectory.'+ext),dpi=180,bbox_inches='tight')
    plt.close(fig)

if __name__=='__main__':main()
