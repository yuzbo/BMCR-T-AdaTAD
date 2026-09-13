"""Add isolated review runs to the existing owned dispatcher, prioritizing complete answers."""
import argparse,copy,json,os,signal,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from h65.paper.runtime import json_write

def absolute_stage(stage,revision):
    stage=copy.deepcopy(stage);stage['done']=str(ROOT/'research/paper'/stage['done']);stage['source_revision']=revision
    stage['requires']=[str(ROOT/x) for x in stage.get('requires',[])]
    if 'resume_checkpoint' in stage:stage['resume_checkpoint']=str(ROOT/stage['resume_checkpoint'])
    if 'args' in stage:stage['args']=[str(ROOT/x) if x.startswith(('tools/','configs/','research/')) else x for x in stage['args']]
    return stage

def main(args):
    base=Path(args.owner_root);exp=base/'research/paper';out=ROOT/'research/paper/review_5485'
    review=json.loads((out/'plan.json').read_text());revision=(ROOT/'source_revision.txt').read_text().strip()
    added={n:absolute_stage(s,revision) for n,s in review['stages'].items()}
    resources=json.loads((ROOT/'research/paper/resources.local.json').read_text())
    for bb in ('s','b'):
        current=ROOT/f'research/paper/runs/review5485_full_v2_{bb}_seed42'
        controls={'v1':base/f'research/paper/runs/thumos_{bb}_point_full_seed42',
                  'uniform':base/f'research/paper/runs/thumos_{bb}_point_uniform_seed42',
                  'pbd':ROOT/f'research/paper/runs/review5485_p01_{bb}_seed42',
                  'static':ROOT/f'research/paper/runs/review5485_p00_{bb}_seed42'}
        for name,reference in controls.items():
            for epoch in ([40,80] if name=='v1' else [40]):
                a=current/f'eval_{epoch:03}_ema';b=reference/f'eval_{epoch:03}_ema';dest=ROOT/f'research/paper/review_5485/paired/{bb}_{name}_{epoch}'
                added[f'paired_review_{bb}_{name}_{epoch}']=dict(kind='analysis',priority=15,dependencies=[],assets=[],requires=[str(a/'completed.json'),str(b/'completed.json')],done=str(dest/'completed.json'),source_revision=revision,
                    args=[str(ROOT/'tools/frame_errors.py'),'--predictions',str(a/'result_detection.json'),'--reference-predictions',str(b/'result_detection.json'),'--ground-truth',resources['datasets']['thumos']['annotations'],'--replicates','1000','--seed','42','--output',str(dest)])
    if not args.register:
        print(json.dumps(dict(stages=len(added),configurations=len(review['configs']),register=False)));return
    if not (out/'assets_complete.json').exists():raise RuntimeError('Real asset construction has not completed')
    state=json.loads((exp/'deployment.json').read_text());old=state['controller_pid'];proc=Path(f'/proc/{old}/cmdline')
    if proc.exists():
        if b'paper_dispatch.py' not in proc.read_bytes():raise RuntimeError('Unexpected controller PID')
        os.kill(old,signal.SIGTERM)
        for _ in range(25):
            if not proc.exists():break
            time.sleep(.2)
        else:raise RuntimeError('Controller has not stopped')
    state=json.loads((exp/'deployment.json').read_text());json_write(out/'owner_before_registration.json',state)
    current_plan=json.loads((exp/'plan.json').read_text());queue=subprocess.check_output(['squeue','-u',os.environ['USER'],'-h','-o','%i|%j|%T'],text=True)
    live={r[0]:(r[1],r[2]) for line in queue.splitlines() if len(r:=line.split('|'))==3}
    core={'thumos_s_point_full_seed42':0,'thumos_b_point_full_seed42':1,'thumos_s_point_uniform_seed42':4,'thumos_b_point_uniform_seed42':5}
    p1={'tcn','interpolate','mae_pretrained','mae_random','axes_T1D0S0','axes_T1D1S0','axes_T1D0S1','dense'}
    by_config={x['id']:x for x in current_plan['configs']};cancel=[]
    for name,stage in state['stages'].items():
        ident=stage.get('config_id');cfg=by_config.get(ident,{})
        if ident and ident.startswith('review5485_'):continue
        if stage['kind']=='train':
            if ident in core:stage['priority']=core[ident]
            elif cfg.get('dataset')!='thumos' or cfg.get('head')=='tadtr' or cfg.get('backbone')=='internvideo_mq':stage['priority']=35
            elif cfg.get('backbone')=='s' and cfg.get('comparison') in p1:stage['priority']=20
            else:stage['priority']=60
        if stage['kind']=='preflight':stage['priority']=12
        jid=stage.get('job_id');entry=live.get(str(jid))
        if entry and entry[1]=='PENDING' and ident not in core and stage['kind'] in ('train','preflight') and not name.startswith(('train_review5485','preflight_review5485')):
            cancel.append(dict(stage=name,job_id=jid,reason='user-priority reorder; same recipe remains queued'))
            stage.setdefault('priority_deferrals',[]).append(dict(job_id=jid,time=time.strftime('%Y-%m-%dT%H:%M:%S%z')))
            stage.pop('job_id',None);stage['status']='WAITING'
    if cancel:subprocess.run(['scancel',*[str(x['job_id']) for x in cancel]],check=True)
    # Uniform courses get their own inline technical check, not a wait for V1's score.
    for bb in ('s','b'):
        ident=f'thumos_{bb}_point_uniform_seed42';name='train_'+ident;check='preflight_'+ident
        if state['stages'][name].get('status') in ('COMPLETED','RUNNING'):continue
        target=dict(kind='inline_preflight',config_id=ident,runs_with=name,done=f'runs/{check}/completed.json',dependencies=[],assets=[])
        state['stages'].setdefault(check,target);current_plan['stages'].setdefault(check,target)
        state['stages'][name].update(dependencies=[],contains_preflight=check,args=['tools/paper_course.py','--config',f'configs/paper/{ident}.json','--preflight-output',f'research/paper/runs/{check}','--slice-hours','10'])
    for name,stage in added.items():
        if name not in state['stages']:state['stages'][name]=stage
        current_plan['stages'][name]=stage
    for name,stage in state['stages'].items():
        if name in current_plan['stages']:
            for key in ('priority','dependencies','contains_preflight','args'):
                if key in stage:current_plan['stages'][name][key]=stage[key]
    known={x['id'] for x in current_plan['configs']}
    current_plan['configs'].extend(x for x in review['configs'] if x['id'] not in known)
    current_plan.update(total_scientific_courses=len(current_plan['configs']),review_runtime=str(ROOT),performance_gates=False)
    state.update(review_runtime=str(ROOT),priority_policy='10 P0 complete courses first, then P1/P3/P2; no mAP gates')
    json_write(exp/'plan.json',current_plan);json_write(exp/'deployment.json',state)
    json_write(out/'analysis_manifest.json',dict(run_roots=[str(exp/'runs'),str(ROOT/'research/paper/runs'),str(base.parent/'fpw_3d_20260913/research/frame/runs'),str(base.parent/'bmcr80_20260913/bmcr80_20260913/runs')],snapshot_files=[]))
    with (exp/'slurm/controller_review_priority.log').open('a') as log:
        child=subprocess.Popen([sys.executable,'-u',str(base/'tools/paper_dispatch.py'),'--submit','--max-live','10','--legacy-receipt',str(base.parent/'fpw_3d_20260913/research/frame/deployment.json'),'--inherit-legacy'],cwd=base,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    receipt=dict(time=time.strftime('%Y-%m-%dT%H:%M:%S%z'),old_controller=old,new_controller=child.pid,new_runtime=str(ROOT),new_stages=len(added),total_courses=len(current_plan['configs']),deferred_pending_jobs=cancel,existing_running_jobs_preserved=True,source_revision=revision)
    json_write(out/'registration.json',receipt);print(json.dumps(receipt))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--owner-root',default='/data/run01/sczc063/yuzibo/h65_clean_adatad_20260910/paper_20260913');p.add_argument('--register',action='store_true');main(p.parse_args())
