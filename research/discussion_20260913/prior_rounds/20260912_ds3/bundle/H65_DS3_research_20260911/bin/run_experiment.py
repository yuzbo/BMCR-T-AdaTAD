#!/usr/bin/env python3
"""Plan by default; run ONLY a sealed, explicitly authorized H65 train/eval manifest.
This script supplies no DS3 model implementation. Missing proposed configs fail closed.
"""
from __future__ import annotations
import argparse, hashlib, json, os, shlex, subprocess, sys
from pathlib import Path
BASE='04c35a3b76897e6c1569eeede41ed3aecaf7f854'

def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()

def git(repo: Path,*args: str) -> str:
    return subprocess.check_output(['git','-C',str(repo),*args],text=True,stderr=subprocess.PIPE).strip()

def resolve_file(root: Path,value: str) -> Path:
    p=Path(value).expanduser()
    return (root/p).resolve() if not p.is_absolute() else p.resolve()

def build(manifest: dict) -> tuple[list[str],list[str],Path,Path,dict]:
    issues=[]
    repo=Path(manifest.get('repo','__UNSET_REPO__')).expanduser().resolve()
    out=Path(manifest.get('output_dir','__UNSET_OUTPUT__')).expanduser().resolve()
    if repo==out or repo in out.parents: issues.append('output_dir must be outside source worktree')
    cfg=resolve_file(repo,manifest.get('config','__UNSET_CONFIG__'))
    if not cfg.is_file(): issues.append(f'Config does not exist; agents must implement it: {cfg}')
    elif sha256(cfg)!=manifest.get('config_sha256'): issues.append('Config SHA256 mismatch/missing')
    head=None
    try:
        head=git(repo,'rev-parse','HEAD')
        if head!=manifest.get('expected_commit'): issues.append('HEAD does not match sealed expected_commit')
        if git(repo,'status','--porcelain'): issues.append('Source worktree must be clean for a sealed run')
        git(repo,'merge-base','--is-ancestor',BASE,'HEAD')
    except (OSError,subprocess.CalledProcessError): issues.append('Missing repository/anchor ancestry')
    if manifest.get('schema')!='h65_ds3_launch_v1': issues.append('Wrong launch manifest schema')
    if manifest.get('review_status')!='APPROVED_FOR_THIS_RUN': issues.append('Manifest is not reviewed/approved')
    stage=manifest.get('stage')
    if stage not in ('train','eval'): issues.append('stage must be train or eval')
    if manifest.get('regime') not in ('dense','Z0','D1','C2','J3'): issues.append('Explicit regime required')
    cmd=[sys.executable,'tools/train.py' if stage=='train' else 'tools/test.py',str(cfg),
         '--seed',str(manifest.get('seed',3407))]
    ckpt=manifest.get('checkpoint')
    if stage=='eval' and not ckpt: issues.append('eval requires checkpoint with provenance')
    if ckpt:
        cp=resolve_file(repo,ckpt.get('path','__UNSET_CHECKPOINT__'))
        if not cp.is_file(): issues.append(f'Checkpoint missing: {cp}')
        elif sha256(cp)!=ckpt.get('sha256'): issues.append('Checkpoint SHA256 mismatch/missing')
        if ckpt.get('provenance_verified') is not True: issues.append('Checkpoint provenance not verified')
        if stage=='eval':
            cmd+=['--checkpoint',str(cp),'--checkpoint-state-key',str(ckpt.get('state_key','state_dict_ema')),
                  '--expected-checkpoint-epoch',str(ckpt.get('epoch',-1)),
                  '--metrics-json',str(out/'metrics.json')]
    # Training initialization is configured by the NEW config; never invent --resume options.
    for dep in manifest.get('dependencies',[]):
        f=resolve_file(repo,dep.get('path','__UNSET_DEP__'))
        if not f.is_file() or sha256(f)!=dep.get('sha256'): issues.append(f'Dependency missing/hash mismatch: {f}')
    gates=manifest.get('gates',[])
    if not gates: issues.append('No admission gate evidence supplied')
    for gate in gates:
        p=resolve_file(repo,gate.get('path','__UNSET_GATE__'))
        if not p.is_file(): issues.append(f'Gate evidence missing: {p}'); continue
        if sha256(p)!=gate.get('sha256'): issues.append(f'Gate evidence hash mismatch: {p}'); continue
        try:
            g=json.loads(p.read_text(encoding='utf-8'))
            if g.get('status')!='PASS' or g.get('code_commit')!=head or g.get('gate_id')!=gate.get('gate_id'):
                issues.append(f'Gate not PASS for this code/gate id: {p}')
            if not g.get('evidence'): issues.append(f'Gate lacks traceable evidence: {p}')
        except (ValueError,OSError): issues.append(f'Invalid gate JSON: {p}')
    if 'configuration_effective_snapshot' not in manifest: issues.append('Resolved configuration snapshot required')
    else:
        snap=manifest['configuration_effective_snapshot']
        p=resolve_file(repo,snap.get('path','__UNSET_SNAPSHOT__'))
        if not p.is_file() or sha256(p)!=snap.get('sha256'): issues.append('Resolved config snapshot missing/hash mismatch')
    for required in ('dataset_manifest_sha256','checkpoint_selection_rule','successful_update_budget','budget_description'):
        if not manifest.get(required): issues.append(f'Missing {required}')
    # Only output location can be overridden here; scientific options must live in the sealed config.
    cmd+=['--cfg-options','work_dir='+str(out/'work')]
    env=dict(os.environ)
    env.update(LOCAL_RANK='0',RANK='0',WORLD_SIZE='1',MASTER_ADDR='127.0.0.1',
               MASTER_PORT=str(manifest.get('master_port',29531)),DUCA_EXPECTED_COMMIT=str(head or ''))
    if not (1024<=int(env['MASTER_PORT'])<=65535): issues.append('Invalid master_port')
    # Allow only named provenance variables, not arbitrary shell commands or secret env dumps.
    allowed={'DUCA_STAGE1_CHECKPOINT','DUCA_STAGE1_CHECKPOINT_SHA256','DUCA_STAGE1_CHECKPOINT_EPOCH',
             'DS3_TEACHER_CHECKPOINT','DS3_TEACHER_SHA256','DS3_TEACHER_STATE_KEY','DS3_TEACHER_EPOCH'}
    for k,v in manifest.get('environment',{}).items():
        if k not in allowed: issues.append(f'Unsupported manifest environment variable: {k}')
        else: env[k]=str(v)
    return cmd,issues,repo,out,env

def main() -> None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('manifest',type=Path)
    p.add_argument('--execute',action='store_true')
    args=p.parse_args()
    try:
        raw=args.manifest.read_bytes(); m=json.loads(raw)
        cmd,issues,repo,out,env=build(m)
    except (ValueError,OSError,KeyError,TypeError) as exc:
        raise SystemExit(f'Invalid launch manifest: {exc}')
    plan=dict(command=shlex.join(cmd),cwd=str(repo),output=str(out),issues=issues,
              action='EXECUTE_REQUESTED' if args.execute else 'PLAN_ONLY',
              note='No DS3 model/config is implemented by this launcher; readiness must come from agents/tests.')
    print(json.dumps(plan,ensure_ascii=False,indent=2))
    if not args.execute: return
    if issues: raise SystemExit('Refusing execution: resolve every admission issue first.')
    if os.environ.get('H65_DS3_ALLOW_GPU_RUN')!='1': raise SystemExit('Explicit H65_DS3_ALLOW_GPU_RUN=1 required.')
    if not os.environ.get('SLURM_JOB_ID') and os.environ.get('H65_DS3_COMPUTE_ALLOCATION_CONFIRMED')!='1':
        raise SystemExit('No Slurm job or explicitly confirmed compute allocation. Never train on login nodes.')
    if out.exists() and any(out.iterdir()): raise SystemExit('Refusing nonempty output directory.')
    check=subprocess.run([sys.executable,'-c','import torch; assert torch.cuda.is_available(), "CUDA unavailable"'],env=env)
    if check.returncode: raise SystemExit('CUDA preflight failed.')
    out.mkdir(parents=True,exist_ok=True)
    (out/'launch_manifest.json').write_bytes(raw)
    (out/'launch_plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    with (out/'run.log').open('w',encoding='utf-8') as f:
        result=subprocess.run(cmd,cwd=repo,env=env,stdout=f,stderr=subprocess.STDOUT)
    (out/'exit_status.json').write_text(json.dumps({'returncode':result.returncode,
        'manifest_sha256':hashlib.sha256(raw).hexdigest()},indent=2)+'\n',encoding='utf-8')
    raise SystemExit(result.returncode)

if __name__=='__main__': main()
