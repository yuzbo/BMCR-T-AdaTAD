"""Move queued CPU preprocessing into a verified owned training allocation."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];EXP=ROOT/'research/paper'

def main(args):
    path=EXP/'data_preparation_job.json';previous=json.loads(path.read_text())
    if previous.get('phase')=='cpu_step_in_owned_training_allocation':
        print(json.dumps(previous));return
    state=json.loads((EXP/'deployment.json').read_text())
    hosts=[s for s in state['stages'].values() if s.get('job_id')==args.allocation_id and s['kind']=='train']
    assert len(hosts)==1,'The selected allocation must belong to this paper training program'
    host=hosts[0]
    queue=dict(line.split('|') for line in subprocess.check_output(['squeue','-u',os.environ['USER'],'-h','-o','%i|%T'],text=True).splitlines())
    assert queue.get(str(args.allocation_id))=='RUNNING'
    assert queue.get(str(previous['job_id']))=='PENDING','Keep any already-running preparation allocation'
    out=EXP/'data_cpu_steps';out.mkdir(exist_ok=True)
    started=out/f'{args.allocation_id}_started.json';log=EXP/'slurm'/f'anet_cpu_{args.allocation_id}.log'
    command=['srun','--jobid='+str(args.allocation_id),'--overlap','--exact','--nodes=1','--ntasks=1','--cpus-per-task=1',
             '--gres=none','--cpu-bind=cores','--immediate=15','--output='+str(log),'--error='+str(log),
             'env','CUDA_VISIBLE_DEVICES=','OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','nice','-n','15',
             sys.executable,'-u',str(ROOT/'tools/paper_data/prepare_anet_archives.py'),
             '--workers','1','--ready-output',str(EXP/'assets/anet_ready.json'),'--started-output',str(started)]
    launcher_pid=None
    if not started.exists():
        with (out/f'{args.allocation_id}_launcher.log').open('a') as stream:
            process=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        launcher_pid=process.pid
        for _ in range(150):
            if started.exists():break
            if process.poll() is not None:raise RuntimeError('CPU preparation step did not start; queued allocation retained')
            time.sleep(.2)
        else:raise RuntimeError('CPU start receipt not yet available; queued allocation retained')
    proof=json.loads(started.read_text());assert proof['preparation_lock_acquired'] and proof['workers']==1
    record=dict(job_id=str(args.allocation_id)+'.'+proof['slurm_step_id'],allocation_job_id=args.allocation_id,
                phase='cpu_step_in_owned_training_allocation',host_course=host['config_id'],log_path=str(log),
                previous_pending_job=previous['job_id'],source_revision=args.data_revision,additional_gpus=0,cpus_per_task=1,
                started=proof,command=command,launcher_pid=launcher_pid,
                interruption_policy='Host completion/time-slice may end this step; resume from the existing successful-video journal after diagnosis')
    (out/f'{args.allocation_id}_previous_job.json').write_text(json.dumps(previous,indent=2)+'\n')
    subprocess.run(['scancel','--state=PENDING',str(previous['job_id'])],check=True)
    path.write_text(json.dumps(record,indent=2)+'\n')
    (out/f'{args.allocation_id}_receipt.json').write_text(json.dumps(record,indent=2)+'\n')
    run=EXP/'runs'/host['config_id'];run.mkdir(exist_ok=True)
    (run/'cpu_colocation.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--allocation-id',type=int,required=True);parser.add_argument('--data-revision',required=True)
    main(parser.parse_args())
