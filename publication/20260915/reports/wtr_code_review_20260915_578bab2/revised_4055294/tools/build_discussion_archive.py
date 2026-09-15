"""Collect existing research inputs and receipts for the GitHub discussion handoff."""
import json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT.parent
OUT=ROOT/'research/discussion_20260913';OUT.mkdir(exist_ok=True)
records=[]
def copy_tree(source,dest,exclude=()):
    for p in sorted(source.rglob('*')):
        if not p.is_file():continue
        rel=p.relative_to(source)
        if any(x in rel.parts for x in ('__pycache__','.git','official_decoder_weights',*exclude)):continue
        if p.suffix in ('.pth','.pt','.pyc','.mp4','.npy','.npz') or p.name.endswith('.tar.gz'):continue
        q=dest/rel;q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
        records.append(dict(source=str(p),archived=q.relative_to(ROOT).as_posix(),bytes=p.stat().st_size))
copy_tree(BASE/'reviews',OUT/'prior_rounds')
for name,sub in [('fpw_3d_20260913','research/frame'),('bmcr80_20260913','bmcr80_20260913')]:
    source=BASE/name/sub
    copy_tree(source,OUT/'legacy_runtime'/name,exclude=('runs','raw_snapshots','resources','status_20260913_0920'))
shutil.copy2(BASE/'STATE.md',OUT/'STATE_at_handoff.md')
(OUT/'archive_manifest.json').write_text(json.dumps(records,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
commands=[r for r in records if any(x in Path(r['archived']).name.lower() for x in ('handoff','commands','tasks','experiments','coordinator','next_model_instructions','pro_completeness')) or '/agents/' in r['archived']]
(OUT/'command_sources.json').write_text(json.dumps(commands,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
snapshot=json.loads((OUT/'live_snapshot.json').read_text(encoding='utf-8'))
plan=json.loads((ROOT/'research/paper/plan.json').read_text(encoding='utf-8'))
assert len(plan['configs'])==52 and all(c['seed']==42 for c in plan['configs'])
rows=[]
for c in plan['configs']:
    stage=snapshot['programs']['paper']['stages']['train_'+c['id']]
    rows.append(dict(id=c['id'],comparison=c['comparison'],dataset=c['dataset'],backbone=c['backbone'],head=c['head'],seed=c['seed'],epochs=c['epochs'],status=stage.get('status','WAITING'),job_id=stage.get('job_id'),config_path=f'configs/paper/{c["id"]}.json'))
(OUT/'current_experiments.json').write_text(json.dumps(rows,indent=2)+'\n')
table=['**当前52个完整课程：每配置seed42一次**','','状态取自'+snapshot['snapshot_time']+'。WAITING表示持久计划中等待依赖或资源，不等于已提交Slurm。','', '|配置|数据/骨干/检测头|轮次|状态|Slurm|','|---|---|---:|---|---:|']
for r in rows:table.append(f'|[{r["id"]}](../../{r["config_path"]})|{r["dataset"]}/{r["backbone"]}/{r["head"]}|{r["epochs"]}|{r["status"]}|{r["job_id"] or "—"}|')
(OUT/'EXPERIMENTS.zh.md').write_text('\n'.join(table)+'\n',encoding='utf-8')
print(json.dumps(dict(archived_files=len(records),bytes=sum(r['bytes'] for r in records),command_sources=len(commands),configurations=len(rows),snapshot_time=snapshot['snapshot_time']),ensure_ascii=False))
