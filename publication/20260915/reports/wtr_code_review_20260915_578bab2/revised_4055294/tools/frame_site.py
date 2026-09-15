"""Bind only this owned FPW export to already verified read-only assets."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def main():
    base=ROOT.parent;target=base/'fidelity_20260911/resources';link=ROOT/'resources'
    if not target.exists():raise FileNotFoundError(target)
    if link.exists():
        if link.resolve()!=target.resolve():raise RuntimeError('Wrong resources binding')
    else:link.symlink_to(target,target_is_directory=True)
    resources=dict(anchors={
        's':dict(checkpoint=str(base/'fidelity_20260911/fidelity_20260911/runs/s_h65/epoch_40.pth'),variant='h65',recipe='corrected_h65_60',mean_map=.6340937787007751),
        'b':dict(checkpoint=str(base/'phase2_20260910/runs/b_bmcr/terminal.pth'),variant='bmcr',recipe='legacy_bmcr_60',mean_map=.6734044636181724)},
        official={b:str(link/f'checkpoints/adatad_{b}_ema.pth') for b in ('s','b')},
        resource_root=str(link),protocol='200/211/792,seed3407',decoder_pretrain={})
    for value in [x['checkpoint'] for x in resources['anchors'].values()]+list(resources['official'].values()):
        if not Path(value).exists():raise FileNotFoundError(value)
    out=ROOT/'research/frame';out.mkdir(parents=True,exist_ok=True);(out/'runs').mkdir(exist_ok=True);(out/'slurm').mkdir(exist_ok=True)
    (out/'resources.local.json').write_text(json.dumps(resources,indent=2)+'\n');print(json.dumps(resources))


if __name__=='__main__':main()
