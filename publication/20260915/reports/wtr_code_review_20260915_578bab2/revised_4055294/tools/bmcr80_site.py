"""Bind the new owned experiment to read-only completed warm/data resources."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
EXP=ROOT/'bmcr80_20260913'


def link(path,target):
    if not target.exists():raise FileNotFoundError(target)
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists() or path.is_symlink():
        if path.resolve()!=target.resolve():raise ValueError(f'Existing resource differs: {path}')
    else:path.symlink_to(target,target_is_directory=target.is_dir())


def main():
    reference=ROOT.parent/'fidelity_20260911'
    link(ROOT/'resources',reference/'resources')
    sources={}
    for b in ('s','b'):
        parent=reference/'fidelity_20260911/runs'/f'{b}_warm'
        info=json.loads((parent/'completed.json').read_text())
        if (info['backbone']!=b or info['phase']!='warm' or info['completed_epochs']!=20 or
            info['successful_updates']!=2000 or info['fidelity_revision']!='lr_identity_crop_validity_v1'):
            raise ValueError('Parent is not the completed corrected warm20')
        link(EXP/'runs'/f'{b}_warm/terminal.pth',parent/'terminal.pth')
        sources[b]=dict(checkpoint=str((parent/'terminal.pth').resolve()),completed=info)
    EXP.mkdir(exist_ok=True);(EXP/'slurm').mkdir(exist_ok=True)
    (EXP/'warm_sources.json').write_text(json.dumps(sources,indent=2)+'\n')
    print(json.dumps({b:s['checkpoint'] for b,s in sources.items()},indent=2))


if __name__=='__main__':main()
