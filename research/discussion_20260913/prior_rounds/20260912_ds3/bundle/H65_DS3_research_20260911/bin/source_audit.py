#!/usr/bin/env python3
"""Read-only anchor/working-tree provenance audit. Never imports torch or prints file contents."""
from __future__ import annotations
import argparse, hashlib, json, subprocess
from pathlib import Path

BUNDLE=Path(__file__).resolve().parents[1]
SPEC=json.loads((BUNDLE/'code_audit.json').read_text(encoding='utf-8'))

def git(repo: Path, *args: str) -> str:
    return subprocess.check_output(['git','-C',str(repo),*args],text=True,stderr=subprocess.PIPE).strip()

def audit(repo: Path) -> dict:
    repo=Path(git(repo,'rev-parse','--show-toplevel')).resolve()
    base=SPEC['base_commit']
    head=git(repo,'rev-parse','HEAD')
    git(repo,'cat-file','-e',base+'^{commit}')
    files=[]
    for item in SPEC['files']:
        actual=git(repo,'rev-parse',base+':'+item['path'])
        p=repo/item['path']
        files.append(dict(path=item['path'],expected_anchor_blob=item['blob_sha'],anchor_blob=actual,
                          anchor_match=actual==item['blob_sha'],working_file_exists=p.is_file(),
                          working_sha256=hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None))
    return dict(schema='h65_ds3_readonly_source_audit_v1',repo=str(repo),base_commit=base,head_commit=head,
                exact_anchor=head==base,status_porcelain=git(repo,'status','--porcelain'),
                files=files,anchor_files_match=all(f['anchor_match'] for f in files),
                checkpoint_provenance='NOT_CHECKED',h65_runtime_tests='NOT_RUN',gpu_available='NOT_CHECKED',
                note='Source identity is not model correctness or experiment evidence.')

def main() -> None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    try:
        result=audit(args.repo)
        out=args.out.resolve()
        root=Path(result['repo'])
        if root==out or root in out.parents:
            raise ValueError('Audit output must be outside the source worktree.')
        out.parent.mkdir(parents=True,exist_ok=True)
        out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        print(json.dumps({'audit':str(out),'anchor_files_match':result['anchor_files_match'],
                          'exact_anchor':result['exact_anchor'],'runtime':'NOT_RUN'},ensure_ascii=False))
        if not result['anchor_files_match']:
            raise SystemExit(2)
    except (subprocess.CalledProcessError,ValueError,OSError) as exc:
        raise SystemExit(f'Audit failed: {type(exc).__name__}: {exc}')

if __name__=='__main__':
    main()
