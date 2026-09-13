"""CPU asset/model construction; it cannot substitute for the Slurm GPU preflight."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'upstream')]


def main(args):
    import torch
    from h65.paper.runtime import read_config,read_resources,build_config,json_write
    from h65.paper.model import PaperModel
    from h65.paper.training import optimizer_groups
    from opentad.datasets.builder import build_dataset
    torch.set_num_threads(2);resources=read_resources(args.resources);records=[]
    plan=json.loads((ROOT/'research/paper/plan.json').read_text())
    for stage in plan['stages'].values():
        if stage['kind']!='preflight':continue
        from tools.paper_dispatch import blocked_assets
        blocked=blocked_assets(stage,resources)
        if blocked:records.append(dict(config_id=stage['config_id'],status='asset_pending',assets=blocked));continue
        cfg=read_config(ROOT/'configs/paper'/f'{stage["config_id"]}.json');mc=build_config(cfg,resources)
        model=PaperModel(mc,cfg,resources)
        groups=optimizer_groups(model,cfg);parameters=[p for group in groups for p in group['params']]
        if len({id(p) for p in parameters})!=len(parameters):raise ValueError('Optimizer parameter aliases')
        if model.teacher is not None and any(p.requires_grad for p in model.teacher.parameters()):raise ValueError('Teacher trainable')
        expected={n for n,p in model.named_parameters() if p.requires_grad}
        if not expected<=set(model.learned_state()):raise ValueError('Trainable parameters omitted from checkpoints')
        if cfg['dataset']=='thumos':
            train=build_dataset(mc.dataset.train);test=build_dataset(mc.dataset.test)
            if len(train)!=200 or len(test)!=792:raise ValueError('THUMOS dataset protocol changed')
        record=dict(config_id=cfg['id'],status='constructed',head=cfg['head'],depth=model.encoder.depth,
                    trainable_parameters=sum(p.numel() for p in parameters),checkpoint_keys=len(model.learned_state()))
        records.append(record);json_write(args.output,dict(cpu_only=True,models=records,complete=False))
        print(json.dumps(record),flush=True);del model
    json_write(args.output,dict(cpu_only=True,models=records));print('CPU model construction complete',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--resources',default=str(ROOT/'research/paper/resources.local.json'))
    p.add_argument('--output',default=str(ROOT/'research/paper/validation/cpu_construction.json'));main(p.parse_args())
