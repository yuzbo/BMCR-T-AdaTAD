#!/usr/bin/env python3
"""Single owner for this two-GPU measurement suite; no Slurm or training jobs."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
PYTHON=sys.executable


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)


def stages():
    out=[]
    def add(name,command,gpu,needs,receipt,**extra):
        out.append(dict(id=name,command=[PYTHON,*command],gpu=gpu,needs=needs,receipt=receipt,**extra))
    def run(name,backbone,mode,gpu,needs,**kw):
        command=['tools/atlas_run.py','--resources','resources.json','--backbone',backbone,
                 '--mode',mode,'--output',f'results/{name}']
        for key,value in kw.items():
            command.append('--'+key.replace('_','-'))
            if value is not True:command.append(str(value))
        receipt=f'results/{name}/completed.json' if mode=='baseline' else f'results/{name}/shard_0_done.json'
        add(name,command,gpu,needs,receipt,baseline=backbone if mode=='baseline' else None)
    run('baseline_s','s','baseline',0,[],split='publication')
    run('preflight_s_full_video','s','preflight',1,[],split='development',limit_videos=1,workers=0)
    add('validation_s',['tools/atlas_validate.py','--resources','resources.json','--backbone','s',
        '--input','results/preflight_s_full_video','--output','results/validation_s'],None,
        ['preflight_s_full_video'],'results/validation_s/passed.json')
    run('recovery_preflight_s','s','recovery',1,['validation_s'],split='development',limit_videos=1,one_window=True,workers=0)
    for backbone,gpu in [('s',0),('b',1)]:
        if backbone=='b':
            run('baseline_b','b','baseline',1,['baseline_s','validation_s','recovery_preflight_s'],split='publication')
            run('preflight_b_full_video','b','preflight',1,['baseline_b'],split='development',limit_videos=1,workers=0)
            add('validation_b',['tools/atlas_validate.py','--resources','resources.json','--backbone','b',
                '--input','results/preflight_b_full_video','--output','results/validation_b'],None,
                ['preflight_b_full_video'],'results/validation_b/passed.json')
            run('recovery_preflight_b','b','recovery',1,['validation_b'],split='development',limit_videos=1,one_window=True,workers=0)
        previous=[f'baseline_{backbone}',f'validation_{backbone}',f'recovery_preflight_{backbone}']
        calibration=[]
        for axis in ('T','D','S'):
            name=f'calibration_{backbone}_{axis}'
            run(name,backbone,'calibration',gpu,previous,split='development',limit_videos=32,one_window=True,axis=axis)
            previous=[name];calibration.append(name)
        add(f'static_{backbone}',['tools/atlas_analyze.py','--mode','calibrate','--input','results',
            '--resources','resources.json','--backbone',backbone,'--output','static_orders.json'],None,
            calibration,'static_orders.json',static_backbone=backbone)
        previous=[f'static_{backbone}']
        # T headroom and population data lead; D/S curves and recovery follow.
        for mode,axis in [('allocation','T'),('population',None),('allocation','D'),('allocation','S'),('recovery',None)]:
            name=f'{mode}_{backbone}'+(f'_{axis}' if axis else '')
            run(name,backbone,mode,gpu,previous,split='publication',**({'axis':axis} if axis else {}))
            previous=[name]
    for mode in ('population','allocation','recovery'):
        needs=([f'allocation_{b}_{a}' for b in ('s','b') for a in ('T','D','S')] if mode=='allocation'
               else [f'{mode}_{b}' for b in ('s','b')])
        add(f'analyze_{mode}',['tools/atlas_analyze.py','--mode',mode,'--input','results',
            '--resources','resources.json','--output','analysis','--bootstrap','10000'],None,needs,f'analysis/{mode}.json')
    add('figures',['tools/atlas_plot.py','--analysis','analysis','--raw','results','--output','output/pdf'],None,
        ['analyze_population','analyze_allocation','analyze_recovery'],'output/pdf/figures.json')
    return out


def alive(pid):
    try:os.kill(int(pid),0);return True
    except (ProcessLookupError,TypeError,ValueError):return False


def completed(stage):
    path=ROOT/stage['receipt']
    if not path.exists():return False
    receipt=json.loads(path.read_text())
    if stage.get('static_backbone'):
        return stage['static_backbone'] in receipt
    if stage.get('baseline'):
        expected={'s':69.0126,'b':71.1204}[stage['baseline']]
        measured=100*receipt['metrics']['average_mAP']
        if abs(measured-expected)>.10:
            raise RuntimeError(f'{stage["id"]}: baseline {measured:.4f} differs from {expected:.4f} beyond locked tolerance')
    return True


def coordinate(revision):
    directory=ROOT/'queue';directory.mkdir(exist_ok=True)
    lock=(directory/'owner.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (directory/'failed.json').exists():
        history=directory/'failures';history.mkdir(exist_ok=True)
        (directory/'failed.json').replace(history/f'{time.time_ns()}.json')
    plan=stages();write(directory/'plan.json',plan)
    state_path=directory/'status.json'
    old=json.loads(state_path.read_text()) if state_path.exists() else {}
    state={s['id']:dict(status='WAITING',attempts=old.get('stages',{}).get(s['id'],{}).get('attempts',[])) for s in plan}
    processes={};logs={}
    adopted={name:value['pid'] for name,value in old.get('stages',{}).items()
             if value.get('status')=='RUNNING' and alive(value.get('pid'))}
    for name,pid in adopted.items():state[name].update(status='RUNNING',pid=pid)
    (ROOT/'logs').mkdir(exist_ok=True)
    while True:
        for stage in plan:
            name=stage['id']
            if completed(stage):state[name]['status']='COMPLETED'
        for name,process in list(processes.items()):
            code=process.poll()
            if code is None:continue
            state[name]['exit_code']=code
            if state[name]['status']!='COMPLETED':state[name]['status']='FAILED'
            logs.pop(name).close();processes.pop(name)
        for name,pid in list(adopted.items()):
            if not alive(pid):
                if state[name]['status']!='COMPLETED':state[name]['status']='FAILED'
                adopted.pop(name)
        busy={next(s['gpu'] for s in plan if s['id']==name) for name in [*processes,*adopted]}
        for stage in plan:
            name=stage['id']
            if state[name]['status']!='WAITING' or stage['gpu'] in busy:continue
            if not all(state[n]['status']=='COMPLETED' for n in stage['needs']):continue
            env=dict(os.environ,ATLAS_SOURCE_REVISION=revision,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
            env['CUDA_VISIBLE_DEVICES']='' if stage['gpu'] is None else str(stage['gpu'])
            log=(ROOT/'logs'/f'{name}.log').open('a')
            process=subprocess.Popen(stage['command'],cwd=ROOT,env=env,stdin=subprocess.DEVNULL,
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            processes[name]=process;logs[name]=log;busy.add(stage['gpu'])
            state[name].update(status='RUNNING',pid=process.pid,gpu=stage['gpu'])
            state[name]['attempts'].append(dict(pid=process.pid,started=time.strftime('%Y-%m-%dT%H:%M:%S%z'),source_revision=revision))
            print(f'Started {name}: pid={process.pid}, gpu={stage["gpu"]}',flush=True)
        status=dict(owner_pid=os.getpid(),updated=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
                    source_revision=revision,stages=state,training_updates=0)
        write(state_path,status)
        if all(v['status']=='COMPLETED' for v in state.values()):
            write(directory/'measurements_and_figures_ready.json',dict(**status,visual_review='pending human-visible PNG inspection'))
            return
        waiting=[s for s in plan if state[s['id']]['status']=='WAITING']
        ready=any(all(state[d]['status']=='COMPLETED' for d in s['needs']) for s in waiting)
        if not processes and not adopted and not ready:
            write(directory/'failed.json',status)
            raise RuntimeError('Remaining stages depend on failed technical or measurement jobs; inspect stage logs')
        time.sleep(15)


def main():
    p=argparse.ArgumentParser();p.add_argument('--launch',action='store_true');p.add_argument('--revision',required=True)
    args=p.parse_args()
    if args.launch:
        directory=ROOT/'queue';directory.mkdir(exist_ok=True)
        receipt=directory/'launch.json'
        if receipt.exists():
            previous=json.loads(receipt.read_text())
            if alive(previous.get('pid')):
                print(json.dumps(dict(already_running=True,**previous)));return
        stream=(directory/'owner.log').open('a')
        process=subprocess.Popen([PYTHON,str(Path(__file__).resolve()),'--revision',args.revision],
            cwd=ROOT,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        value=dict(pid=process.pid,source_revision=args.revision,started=time.strftime('%Y-%m-%dT%H:%M:%S%z'))
        write(receipt,value);print(json.dumps(value));return
    coordinate(args.revision)


if __name__=='__main__':main()
