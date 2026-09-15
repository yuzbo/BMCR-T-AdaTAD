"""Read-only compact experiment snapshot for a progress report."""
import argparse,json,math,statistics,subprocess,time
from pathlib import Path

def read(path):
    path=Path(path)
    return json.loads(path.read_text()) if path.is_file() else None

def tail(path,limit=16000):
    path=Path(path)
    if not path.is_file():return ''
    with path.open('rb') as stream:
        stream.seek(max(0,path.stat().st_size-limit));return stream.read().decode(errors='replace')

def last_record(path):
    for line in reversed(tail(path,100000).splitlines()):
        try:return json.loads(line)
        except json.JSONDecodeError:pass
    return None

def command(*args):
    p=subprocess.run(args,text=True,capture_output=True)
    return dict(code=p.returncode,stdout=p.stdout.strip(),stderr=p.stderr.strip())

def metrics(row):
    if row is None:return None
    out={}
    for key,value in row.items():
        if 'map' in key.lower() or any(x in key.lower() for x in ('gflop','latency','videos','windows','checkpoint_state','source_revision','epoch')):
            out[key]=value
        elif key in ('metrics','results','evaluation') and isinstance(value,dict):out[key]=metrics(value)
    return out

def course(root,name,fit=None):
    folder=root/'research/paper/runs'/name
    row=last_record(folder/'train.jsonl') or {}
    result=dict(path=str(folder),last={k:row.get(k) for k in ('epoch','successful_updates','sample_cursor','losses','grad_norm','peak_gib','query_counts','plan','exposure')},
        progress=read(folder/'progress.json'),completed=metrics(read(folder/'completed.json')),
        checkpoints=[p.name for p in sorted(folder.glob('*.pth'))],evaluations={})
    for out in sorted(folder.glob('eval_*')):
        done=read(out/'completed.json')
        if done:result['evaluations'][out.name]=metrics(done)
        elif (out/'progress.json').exists():result['evaluations'][out.name]=dict(in_progress=read(out/'progress.json'))
    p=folder/'operator_interventions.jsonl'
    if p.exists():
        rows=[json.loads(x) for x in p.read_text().splitlines() if x.strip()]
        gains=[sum(r['gain_cls_loc']) for r in rows]
        result['online_value']=dict(count=len(rows),labels_outside_fit=sum(r['video_id'] not in fit for r in rows) if fit else None,
            positive=sum(g>0 for g in gains),zero=sum(g==0 for g in gains),gain_min=min(gains),gain_max=max(gains),gain_mean=statistics.mean(gains),
            source_revisions=sorted({r.get('source_revision','unknown') for r in rows}))
    return result

def state_summary(path):
    row=read(path)
    if not row:return None
    stages=row.get('stages',{})
    if isinstance(stages,list):stages={str(i):s for i,s in enumerate(stages)}
    return dict(updated=row.get('updated_at',row.get('updated')),owner_pid=row.get('controller_pid',row.get('owner_pid')),
        stages={k:{p:v.get(p) for p in ('status','job_id','pid','gpu','failure','exit_code','waiting_files','submission_error')} for k,v in stages.items()
                if k.startswith('wtr_train') or v.get('status') not in ('COMPLETED','CANCELLED')})

def raw_stats(root):
    folder=root/'results/mini_bank';groups=[]
    for p in sorted((folder/'groups').glob('*.json')):groups.append(read(p))
    result=dict(gate=read(root/'results/r2_six/passed.json'),completed=read(folder/'completed_0.json'),
        diagnostics=read(root/'results/mini_diagnostics/bank_diagnostics.json'),domains={})
    if result['completed']:result['completed'].pop('group_files',None)
    for domain in ('O','R+'):
        rows=[r for r in groups if r['domain']==domain]
        actions=[(r,a) for r in rows for a in r['actions']]
        gains=[sum(a['gain_cls_loc']) for _,a in actions]
        off=[a['insert'] not in set(r['episode']['official_frame_ids']) for r,a in actions]
        result['domains'][domain]=dict(states=len(rows),actions=len(actions),positive=sum(g>0 for g in gains),
            gain_mean=statistics.mean(gains) if gains else None,gain_min=min(gains) if gains else None,gain_max=max(gains) if gains else None,
            offgrid_insertions=sum(off),changed_pairs=sorted({a['n_changed_pairs'] for _,a in actions}),
            changed_packs=sorted({a['n_changed_packs'] for _,a in actions}))
    index={(r['episode']['video_id'],r['episode']['window_index'],r['state_id'],r['domain']):r for r in groups}
    comparisons=[]
    for key,row in index.items():
        if key[-1]!='O':continue
        raw=index.get((*key[:-1],'R+'))
        if not raw:continue
        old={(a['remove'],a['insert']) for a in row['actions']};new={(a['remove'],a['insert']) for a in raw['actions']}
        best=lambda r:max([0.]+[sum(a['gain_cls_loc']) for a in r['actions']])
        comparisons.append(dict(video=key[0],state=key[2],split=row['split'],same_actions=old==new,
            shared_actions=len(old&new),best_o=best(row),best_raw=best(raw),delta=best(raw)-best(row),
            same_support=row['selection']==raw['selection']))
    result['paired_states']=comparisons
    result['value_reports']=[str(p) for p in root.glob('results/**/training_report.json')]
    return result

p=argparse.ArgumentParser();p.add_argument('--site',required=True);a=p.parse_args()
out=dict(time=time.strftime('%Y-%m-%dT%H:%M:%S%z'),host=command('hostname'))
if a.site=='4090':
    core=Path('/data/run01/sczc063/yuzibo/wtr_fasttrack_20260915')
    owner=Path('/data/run01/sczc063/yuzibo/h65_clean_adatad_20260910/paper_20260913')
    res=read(core/'research/paper/resources.local.json');fit=set(res['wtr_router_splits']['fit'])
    out['queue']=command('squeue','-u','sczc063','-h','-o','%i|%j|%T|%R')
    out['owner']=state_summary(owner/'research/paper/deployment.json')
    out['core']=[course(core,n,fit) for n in ('wtr_d_v_s42','wtr_d_u_s42')]
    out['raw']=raw_stats(Path('/data/run01/sczc063/yuzibo/wtr_raw_v1_20260915/revision_27d557e'))
    old=owner.parent/'support_review_20260914'
    out['references']=[course(old,'review5485_full_v2_b_seed42'),course(owner,'thumos_s_point_uniform_seed42'),course(owner,'thumos_b_point_uniform_seed42')]
elif a.site=='a100':
    root=Path('/HOME/pxyai/pxyai_0057/HDD_POOL/yzb/wtr_fasttrack_20260915')
    res=read(root/'research/paper/resources.local.json');fit=set(res['wtr_router_splits']['fit'])
    out['queue']=command('squeue','-u','pxyai_0057','-h','-o','%i|%j|%T|%R')
    out['owner']=state_summary(root/'research/paper/deployment.json')
    out['core']=[course(root,n,fit) for n in ('wtr_s_v_s42','wtr_s_u_s42')]
    out['owner_log']=tail(root/'research/paper/owner.log',3000)
elif a.site=='atlas':
    root=Path('/root/autodl-tmp/wtr_characterization_20260915')
    out['owner']=read(root/'queue/status.json')
    out['failure']=read(root/'queue/failed.json')
    out['ready']=read(root/'queue/measurements_and_figures_ready.json')
    out['progress']={p.parent.name:read(p) for p in root.glob('results/*/progress_0.json')}
    out['completed']={p.parent.name:metrics(read(p)) for p in root.glob('results/*/completed.json')}
    out['analysis_files']=[str(p.relative_to(root)) for p in root.glob('analysis/**/*') if p.is_file()]
print(json.dumps(out,ensure_ascii=False))
