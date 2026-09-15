"""Five decision figures and a faithful network diagram from completed evidence only."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle,FancyArrowPatch
from h65.paper.figures import save
from h65.paper.runtime import json_write

def family(row):
    name=row['comparison']
    if row.get('config',{}).get('recipe')=='graph_tad_v1':return name
    if row.get('config',{}).get('recipe')=='native_adatad_uniform_v1':
        if row['config']['frames']==768:return 'Native dense AdaTAD'
        return 'Native uniform / direct' if row['role']=='external_retested' else 'Native uniform / adapted'
    if row['role']=='external_retested':return 'Official AdaTAD'
    if name.startswith('FULL_V2'):return 'Full-V2'
    if name=='full':return 'Full-V1'
    if name=='uniform':return 'Uniform Full'
    if name.startswith('P01'):return 'PBD-style'
    if name.startswith('P00'):return 'Static'
    return name

def best_rows(rows,limit=None):
    best={}
    for row in rows:
        if not row['standard'] or row['state'] not in ('ema','official_ema'):continue
        if limit is not None and row['epoch'] is not None and row['epoch']>limit:continue
        if row['seed']!=42 and row['role']!='external_retested':continue
        key=row['run_id']
        if key not in best or row['map']>best[key]['map']:best[key]=row
    return list(best.values())

def emit(fig,out,name,rows,scope):
    save(fig,out,name);json_write(Path(out)/(name+'.sources.json'),dict(rows=rows,scope=scope,unmeasured_points_added=False))

def architecture(out):
    fig,ax=plt.subplots(figsize=(13,6));ax.set(xlim=(0,13),ylim=(0,6));ax.axis('off')
    def box(x,y,w,h,text,color='#eef2f6'):
        ax.add_patch(Rectangle((x,y),w,h,facecolor=color,edgecolor='#374151',lw=.8));ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=9)
    def arrow(a,b,dashed=False):ax.add_patch(FancyArrowPatch(a,b,arrowstyle='->',mutation_scale=13,color='#374151',linestyle='--' if dashed else '-'))
    box(.1,3.2,1.65,1.3,'RGB timeline\n768 candidates')
    box(2.1,3.2,1.8,1.3,'H65 / BMCR\nScout + frame route\n15-plan budget')
    box(4.25,3.2,1.8,1.3,'Selected RGB S_K\nPatch embed + pack\nK/2 x P tokens')
    box(6.4,3.2,2.45,1.3,'First 4 blocks dense\nLate alternating A-MoD\nselected Q / full KV\ndepth-light + FFN light','#e5eee9')
    box(9.2,3.2,1.7,1.3,'Original-axis Cross\nK/2 anchors\n384 time queries','#eee8f3')
    box(11.25,3.2,1.65,1.3,'Trainable TAD\nreadout\n768 positions\nclass / boundaries')
    for left,right in ((1.75,2.1),(3.9,4.25),(6.05,6.4),(8.85,9.2),(10.9,11.25)):arrow((left,3.85),(right,3.85))
    box(4.25,1.1,4.6,1.2,'TRAINING ONLY: frozen initial encoder\nExactly the same S_K / packing / padding\nFull D/S state targets; stop gradient','#f4f4f4')
    arrow((5.15,3.2),(5.15,2.3),True);arrow((7.7,2.3),(7.7,3.2),True)
    box(9.2,1.1,1.7,1.2,'TRAINING ONLY\nOfficial teacher\nFull RGB / frozen\nFeature target','#f4f4f4');arrow((10.05,2.3),(10.05,3.2),True)
    box(11.25,1.1,1.65,1.2,'TRAINING ONLY\nShared-full\nstudent path\nGT + self KD','#f4f4f4');arrow((12.07,2.3),(12.07,3.2),True)
    ax.plot([3,3,10.05],[4.5,5.05,5.05],color='#64748b',lw=.8);arrow((10.05,5.05),(10.05,4.5))
    ax.text(6.5,5.2,'Full-axis lightweight context and physical-time provenance',ha='center',fontsize=9)
    ax.text(.1,.35,'Full-V2 implemented candidate. Full KV is within each packed attention domain; global TIA communicates over selected K/2 support.\nThe complete original spatial-time grid is not reconstructed at every block. Performance remains an empirical question.',fontsize=9)
    ax.set_title('Support-consistent full model for temporal action detection',fontsize=14)
    emit(fig,out,'model_full_v2',[],scope='architecture from current code; not a claim of validated performance')

def dot_comparison(rows,out,name,labels,title):
    selected=[r for r in best_rows(rows,40) if r['dataset']=='thumos' and r['backbone']=='s' and r['head']=='point' and r['comparison'] in labels]
    if not selected:return False
    selected.sort(key=lambda r:list(labels).index(r['comparison']));fig,axes=plt.subplots(1,2,figsize=(10,max(3,len(selected)*.45)),layout='constrained');y=np.arange(len(selected))
    axes[0].scatter([r['map'] for r in selected],y,color='#245b91');axes[0].set_yticks(y,labels=[labels[r['comparison']] for r in selected]);axes[0].set_xlabel('Best mAP through epoch40 (%)')
    x=[r['dataset_mean_gflops'] for r in selected];valid=[i for i,v in enumerate(x) if v is not None]
    axes[1].scatter([x[i] for i in valid],y[valid],color='#397957');axes[1].set_yticks(y,labels=['']*len(y));axes[1].set_xlabel('Full-test mean GFLOPs / window')
    for ax in axes:ax.grid(axis='x',alpha=.2)
    fig.suptitle(title);emit(fig,out,name,selected,'Single seed42; same 10/20/40 selection candidates; missing results omitted; GFLOPs include the complete model');return True

def plot_all(rows,out,manifest=None):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);manifest=manifest or {};architecture(out);made=['model_full_v2'];missing=[]
    core=[r for r in best_rows(rows) if r['dataset']=='thumos' and r['head']=='point' and family(r) in ('Official AdaTAD','Full-V1','Full-V2','Uniform Full','PBD-style','Static','Native dense AdaTAD','Native uniform / direct','Native uniform / adapted','G-repair','G-context','G-full')]
    if core:
        fixed_references=manifest.get('fixed_window_references',[])
        colors={'Official AdaTAD':'#555555','Full-V1':'#3779ad','Full-V2':'#b54c3a','Uniform Full':'#718b5a','PBD-style':'#8865a0','Static':'#b68a30'}
        colors.update({'Native dense AdaTAD':'#242424','Native uniform / direct':'#59a6a6','Native uniform / adapted':'#155e63'})
        colors.update({'G-repair':'#ba63a3','G-context':'#bd7a30','G-full':'#c82537'})
        fig,axes=plt.subplots(1,2,figsize=(11,4.5),sharey=True,layout='constrained')
        for ax,key,title in zip(axes,('dataset_mean_gflops','representative_gflops'),('Complete-test mean window cost','Representative full-window cost')):
            valid=[r for r in core if r[key] is not None]
            for r in valid:
                label=family(r)+' / '+r['backbone'].upper();ax.scatter(r[key],r['map'],s=65,marker='o' if r['backbone']=='s' else '^',color=colors[family(r)],label=label)
                terminal=[q for q in rows if q['run_id']==r['run_id'] and q['standard'] and q['state']=='ema' and q['epoch']==q['epochs'] and q[key] is not None]
                if terminal:
                    t=terminal[-1];ax.plot([r[key],t[key]],[r['map'],t['map']],color=colors[family(r)],lw=.6,alpha=.5)
                    ax.scatter(t[key],t['map'],s=100,marker='o' if r['backbone']=='s' else '^',facecolors='none',edgecolors=colors[family(r)])
            reference_rows=[]
            if key=='representative_gflops':
                for reference in fixed_references:
                    ax.scatter(reference[key],reference['map'],s=90,marker='D',facecolors='none',edgecolors='#555555',label=reference['label'],zorder=4)
                    reference_rows.append(reference)
            candidates=[*valid,*reference_rows]
            frontier=[r for r in candidates if not any(q[key]<=r[key] and q['map']>=r['map'] and (q[key]<r[key] or q['map']>r['map']) for q in candidates)]
            frontier.sort(key=lambda r:r[key]);ax.plot([r[key] for r in frontier],[r['map'] for r in frontier],color='.35',ls=':',lw=.9)
            ax.set(xlabel='Complete model GFLOPs / window',title=title);ax.grid(alpha=.15)
        axes[0].set_ylabel('Best full-test average mAP (%)');axes[1].legend(frameon=False,fontsize=7.5,loc='lower right')
        emit(fig,out,'fig1_complete_method_pareto',[*core,*fixed_references],'Filled best and hollow terminal EMA, each with its own measured cost; hollow diamonds are explicitly dated earlier official fixed-window references, shown only on the right because dataset-mean cost is not available; dotted lines are the displayed measured envelopes, not an exhaustive published-method frontier');made.append('fig1_complete_method_pareto')
    else:missing.append('fig1_complete_method_pareto')
    groups=[('fig2_axes_frontier',{'axes_T1D0S0':'T only','axes_T1D1S0':'T + D','axes_T1D0S1':'T + S','full':'Full TDS / V1','FULL_V2_S':'Full TDS / V2'},'Independent complete recipes: T / TD / TS / TDS'),
            ('fig3_depth_failure_diagnosis',{'N00_S':'Selected KV / hold','N01_S':'Full KV / hold','N02_S':'Full KV / light','N04_S':'Full KV / light / support','N06_S':'Late route / support','N07_S':'Uniform depth / late'},'Context, bypass state and layer placement'),
            ('fig4_decoder_initialization',{'interpolate':'Interpolation','tcn':'TCN','R00_S':'Fresh Cross','mae_random':'MAE random','mae_pretrained':'MAE pretrained','R01_S':'MAE + input alignment','full':'Hot-R03 Cross (extra history)'},'Decoder architecture and initialization; training history is explicit')]
    for name,labels,title in groups:
        (made if dot_comparison(rows,out,name,labels,title) else missing).append(name)
    graph_labels={'G-fixed-local':'Fixed local graph','G-no-referral':'No referral','G-full-kv':'Full KV + time graph','G-full':'Dynamic graph / full'}
    (made if dot_comparison(rows,out,'fig6_graph_mechanisms',graph_labels,'Graph mechanism controls; same retrieval budget, actual cost reported') else missing).append('fig6_graph_mechanisms')
    diagnostic=[]
    for root in manifest.get('run_roots',[]):
        for path in Path(root).glob('**/review_diagnostics/case_measurements.json'):
            for row in json.loads(path.read_text(encoding='utf-8')):diagnostic.append(dict(run=path.parent.parent.name,source=str(path),**row))
    if diagnostic:
        fig,axes=plt.subplots(1,3,figsize=(12,3.7),layout='constrained')
        for run in sorted({r['run'] for r in diagnostic}):
            for ax,point in zip(axes,('attention','pre_tia','post_tia')):
                by_layer={}
                for row in diagnostic:
                    if row['run']!=run:continue
                    for x in row['same_checkpoint_support_errors']:
                        if x['point']==point:by_layer.setdefault(x['original_block_id'],[]).append(x['normalized_mse'])
                ids=sorted(by_layer);ax.plot(ids,[np.mean(by_layer[i]) for i in ids],'-o',ms=3,lw=1,label=run.replace('review5485_','').replace('_seed42',''))
                ax.set(xlabel='Original block ID',ylabel='Normalized MSE',title=point)
        axes[-1].legend(frameon=False,fontsize=7);emit(fig,out,'fig5_layer_state_error',diagnostic,'Fixed diagnostic videos; current-weight full computation on the same selected support; policies and support can differ across trained recipes');made.append('fig5_layer_state_error')
    else:missing.append('fig5_layer_state_error')
    json_write(out/'figure_manifest.json',dict(generated=made,awaiting_real_results=missing,all_results_source_linked=True))
    return made,missing

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--figures');p.add_argument('--output',required=True);a=p.parse_args()
    from tools.paper_review_analyze import collect
    m=json.loads(Path(a.manifest).read_text(encoding='utf-8'));plot_all(collect(m),a.output,m)
