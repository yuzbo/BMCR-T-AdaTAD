"""Read-only experiment/queue/evidence snapshot for the complete paper program."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import subprocess
import time


def read(path):return json.loads(path.read_text()) if path.exists() else None


def main(args):
    base=Path(args.site_root);root=base/'paper_20260913';exp=root/'research/paper'
    query=subprocess.run(['squeue','-u',os.environ['USER'],'-h','-o','%i|%j|%T|%R|%M|%l'],text=True,capture_output=True,check=True)
    queue={row[0]:dict(job_id=row[0],name=row[1],state=row[2],reason_or_node=row[3],elapsed=row[4],limit=row[5])
           for line in query.stdout.splitlines() if len(row:=line.split('|'))==6}
    value=dict(snapshot_time=time.strftime('%Y-%m-%dT%H:%M:%S%z'),account_jobs=len(queue),programs={},queue=queue,
               scientific_revision=(root/'source_revision.txt').read_text().strip())
    locations={'paper':exp,'fpw':base/'fpw_3d_20260913/research/frame','bmcr80':base/'bmcr80_20260913/bmcr80_20260913'}
    owned=set()
    for name,folder in locations.items():
        state=read(folder/'deployment.json');stages=state['stages'];counts=Counter();kinds={}
        for ident,stage in stages.items():
            jid=str(stage.get('job_id'));live=queue.get(jid)
            if stage.get('job_id'):owned.add(jid)
            status=live['state'] if live else stage.get('status','WAITING')
            counts[status]+=1;kinds.setdefault(stage['kind'],Counter())[status]+=1
        pid=state.get('controller_pid');proc=Path(f'/proc/{pid}/cmdline')
        value['programs'][name]=dict(source=str(folder/'deployment.json'),controller_pid=pid,controller_present=proc.exists(),
            controller_command=proc.read_bytes().replace(b'\0',b' ').decode(errors='replace') if proc.exists() else None,
            inherited_by=state.get('paper_queue_owner'),status_counts=dict(counts),kind_counts={k:dict(v) for k,v in kinds.items()},stages=stages)
    plan=read(exp/'plan.json');value['paper_configs']=plan['configs'];value['paper_courses']=[]
    review_root=Path(value['programs']['paper']['source']).parents[2]
    owner_state=read(exp/'deployment.json')
    if owner_state.get('review_runtime'):review_root=Path(owner_state['review_runtime'])
    for cfg in plan['configs']:
        course_exp=review_root/'research/paper' if cfg['id'].startswith('review5485_') else exp
        folder=course_exp/'runs'/cfg['id'];progress=read(folder/'progress.json');done=read(folder/'completed.json');last=None
        log=folder/'train.jsonl'
        if log.exists():
            with log.open('rb') as stream:
                stream.seek(max(0,log.stat().st_size-10000));lines=stream.read().decode(errors='replace').splitlines()
            for line in reversed(lines):
                try:last=json.loads(line);break
                except json.JSONDecodeError:continue
        value['paper_courses'].append(dict(id=cfg['id'],metadata_exists=(folder/'metadata.json').exists(),progress=progress,completed=done,
            last_update=None if last is None else {k:last.get(k) for k in ('epoch','successful_updates','sample_cursor','losses','peak_gib')}))
    resource=read(exp/'resources.local.json');value['resources']=resource
    value['asset_sizes']={k:dict(path=v['checkpoint'],bytes=Path(v['checkpoint']).stat().st_size if Path(v['checkpoint']).exists() else 0)
        for k,v in {**resource.get('verified_downloads',{}),**resource.get('pending_downloads',{})}.items()}
    value['anet_preparation']=read(Path(resource['datasets']['anet']['prepared_report']))
    value['anet_ready']=read(Path(resource['datasets']['anet']['ready_file']))
    prep=read(exp/'data_preparation_job.json');value['data_preparation_job']=prep
    if prep:
        jid=prep.get('job_id',prep.get('slurm_job_id',1288466));owned.add(str(jid))
    else:owned.add('1288466')
    value['owned_live_jobs']=[r for jid,r in queue.items() if jid in owned]
    value['evidence']={}
    for name,folder in locations.items():
        records=[]
        scan=[folder/'runs']
        if name=='paper' and review_root!=root:scan.append(review_root/'research/paper/runs')
        for directory in scan:
            for file in directory.glob('**/completed.json'):
                record=read(file)
                if 'metrics' in record and record.get('test_videos') in (211,4728):records.append(dict(source=str(file),record=record))
        if name=='bmcr80':
            records=[]
            for file in (folder/'runs').glob('*_bmcr_test_epoch_*/metrics.json'):
                record=read(file)
                if record.get('test_videos')==211:records.append(dict(source=str(file),record=record))
        value['evidence'][name]=records
    print(json.dumps(value,ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--site-root',default='/data/run01/sczc063/yuzibo/h65_clean_adatad_20260910');main(p.parse_args())
