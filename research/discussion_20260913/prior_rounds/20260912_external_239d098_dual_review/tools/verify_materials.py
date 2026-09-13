"""Verify archived inputs and reported arithmetic; summarize existing receipts only."""
import json
import math
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[1] / 'ds3_20260912'
EVIDENCE = ROOT / 'evidence'


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def numeric_differences(a, b, prefix=''):
    if isinstance(a, dict):
        assert a.keys() == b.keys(), prefix
        return [x for k in a for x in numeric_differences(a[k], b[k], prefix+'/'+k)]
    if isinstance(a, list):
        assert len(a) == len(b), prefix
        return [x for i,(x1,x2) in enumerate(zip(a,b)) for x in numeric_differences(x1,x2,prefix+'/'+str(i))]
    if isinstance(a, (float,int)) and not isinstance(a,bool):
        assert math.isclose(a,b,rel_tol=1e-9,abs_tol=1e-10), (prefix,a,b)
        return [(prefix,a,b)] if a != b else []
    assert a == b, (prefix,a,b)
    return []


def main():
    manifest = read(ROOT/'ARCHIVE_MANIFEST.json')
    archive_checks=[]
    for item in manifest['inputs']:
        for kind in ('text','zip'):
            a=Path(item['original_'+kind]);b=Path(item['archived_'+kind])
            assert a.read_bytes()==b.read_bytes(), str(a)
        with zipfile.ZipFile(item['archived_zip']) as z:
            entries=[]
            for member in z.infolist():
                if member.is_dir():continue
                output=Path(item['extracted'])/member.filename
                assert z.read(member)==output.read_bytes(),member.filename
                entries.append({'path':member.filename,'bytes':member.file_size,'byte_equal':True})
            archive_checks.append({'review':item['id'],'text_copy_equal':True,'zip_copy_equal':True,'entries':entries})
    cases=[('review_1','bmcr_review','toy_audit.py','toy_results.json','--out'),
           ('review_2','BMCR_239d098_review','mechanism_checks.py','mechanism_checks.json','--output')]
    toy_checks=[]
    for review,folder,script,result,option in cases:
        source=ROOT/'inputs'/review/'package'/folder
        dest=EVIDENCE/(review+'_'+result)
        run=subprocess.run([sys.executable,str(source/script),option,str(dest)],capture_output=True,text=True,encoding='utf-8')
        (EVIDENCE/(review+'_cpu_run.log')).write_text(run.stdout+run.stderr,encoding='utf-8')
        assert run.returncode==0,(review,run.stderr)
        difference=numeric_differences(read(source/result),read(dest))
        toy_checks.append({'review':review,'exit_code':run.returncode,'saved_result_matches':True,'roundoff_differences':difference})

    snapshot=read(EVIDENCE/'live_snapshot.json')
    rows=[]
    for path,value in sorted(snapshot['metrics'].items()):
        metrics=value['metrics']
        average=sum(metrics[f'mAP@0.{i}'] for i in range(3,8))/5
        assert math.isclose(average,metrics['average_mAP'],abs_tol=1e-12)
        assert value['test_videos']==211 and value['test_windows']==792 and value['seed']==3407,path
        receipt_path=path.replace('metrics.json','completed.json')
        assert receipt_path in snapshot['receipts'],path
        init=value.get('initialization',{})
        rows.append({'path':path,'policy':value['policy'],'epoch':init.get('completed_epochs'),
                     'updates':init.get('successful_updates'),'aux_state':init.get('aux_state'),
                     'metrics_percent':{k:100*v for k,v in metrics.items()},
                     'duration_recall':value.get('duration_recall'),
                     'e2e_seconds':value['e2e_seconds'],'complete_211_792_receipt':True})
    legacy=read(PROJECT/'phase2_20260910/comparison.json')
    corrected=read(PROJECT/'fidelity_20260911/FINAL_COMPARISON.json')
    legacy_rows=[{'backbone':x['backbone'],'method':x['method'],'mAP_percent':100*x['metrics']['average_mAP'],
                  'five_threshold_percent':{k:100*v for k,v in x['metrics'].items()}} for x in legacy['results']]
    corrected_rows=[{'backbone':x['backbone'],'selection':x.get('selection'),
                     'metrics':x.get('metrics'),'mAP':x.get('average_mAP'),
                     'total_epoch':x.get('total_epoch')} for x in corrected['results']]
    summary={'metrics_receipt_cutoff':snapshot['deployment']['updated_at'],'rows':rows,
             'legacy':legacy_rows,'corrected_h65_records':corrected['results'],
             'EMA_initial_coefficients':{str(n):.999**n for n in (500,1500,3000,3700,8000)},
             'interpretation':'Existing metrics/receipts checked; no weights, predictions or GPU inference rerun.'}
    (EVIDENCE/'metrics_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    checks={'archive':archive_checks,'CPU_toys':toy_checks,'metric_means_and_211_792_receipts':len(rows),
            'python':sys.version,'scope':'Only archive byte comparisons, provided CPU mechanisms, and recorded metrics arithmetic.'}
    (EVIDENCE/'VERIFICATION.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    plot(rows)
    print(json.dumps({'archive_files_verified':sum(len(x['entries']) for x in archive_checks),
                      'toy_scripts_passed':len(toy_checks),'metric_receipts_verified':len(rows),
                      'T24A':[x for x in rows if x['policy']=='T24A'],
                      'legacy':legacy_rows,'corrected_result_keys':list(corrected['results'][0]),
                      'EMA':summary['EMA_initial_coefficients']},ensure_ascii=False,indent=2))


def plot(rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    points=sorted((x for x in rows if x['policy']=='T24A'),key=lambda x:x['epoch'])
    refs={x['policy']:x for x in rows if x['epoch'] is None}
    fig,axes=plt.subplots(1,2,figsize=(11.5,4.6),layout='constrained')
    for ax,metric,title in zip(axes,['average_mAP','mAP@0.7'],['Average mAP (tIoU 0.3–0.7)','mAP at tIoU 0.7']):
        for policy,color,style in [('D768G','#6e7886',':'),('D768L','#6e7886','--'),('Z24','#187d70','-')]:
            ax.axhline(refs[policy]['metrics_percent'][metric],color=color,ls=style,lw=1.4,label=policy)
        xs=[x['epoch'] for x in points];ys=[x['metrics_percent'][metric] for x in points]
        ax.plot(xs,ys,'o-',color='#b24730',lw=2,label='D1 T24A / EMA')
        for x,y in zip(xs,ys):ax.annotate(f'{y:.2f}',(x,y),xytext=(0,8),textcoords='offset points',ha='center',fontsize=8)
        ax.set(xlabel='Completed test checkpoint (epoch)',ylabel='mAP (%)',title=title,xticks=xs,xlim=(3,32))
        ax.grid(axis='y',alpha=.2)
    axes[0].legend(loc='center right',frameon=False,fontsize=8)
    fig.suptitle('VideoMAE-S: completed T24A checkpoints remain below Z24',fontsize=13,fontweight='bold')
    fig.supxlabel('211 videos / 792 windows per point. Snapshot: 2026-09-13 00:03 CST. The 80-epoch run is unfinished.',fontsize=9)
    out=ROOT/'figures';out.mkdir(exist_ok=True)
    fig.savefig(out/'d1_checkpoint_comparison.png',dpi=180)
    fig.savefig(out/'d1_checkpoint_comparison.svg')
    plt.close(fig)


if __name__=='__main__':main()
