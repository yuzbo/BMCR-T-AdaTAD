"""Evaluate an exact trained state on the complete dataset and measured budgets."""
import argparse
import copy
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'upstream')]


def load_model(args):
    import torch
    from h65.paper.runtime import read_config,read_resources,build_config
    from h65.paper.model import PaperModel
    from h65.full.runtime import initialize_gpu,seed_all
    cfg=read_config(args.config);resources=read_resources(args.resources)
    hardware=initialize_gpu();seed_all(cfg['seed']);model_cfg=build_config(cfg,resources)
    payload=torch.load(args.checkpoint,map_location='cpu')
    if payload['metadata']['config']!=cfg:raise ValueError('Checkpoint/config mismatch')
    model=PaperModel(model_cfg,cfg,resources).cuda().eval();model.load_learned(payload[args.state])
    if getattr(args,'budget_fraction',None) is not None:model.config['budget_fraction']=args.budget_fraction
    if getattr(args,'selector',None) is not None:model.config['selector']=args.selector
    if getattr(args,'disable_frame',False):model.config['frame_utility']=False
    metadata=dict(**hardware,config=copy.deepcopy(model.config),checkpoint=str(args.checkpoint),checkpoint_state=args.state,
                  epoch=payload.get('epoch_index'),successful_updates=payload.get('successful_updates'),recipe=cfg['recipe'],
                  source_revision=payload['metadata']['source_revision'],training_config=cfg)
    return model,model_cfg,resources,metadata


def main(args):
    if args.dry_run:print(json.dumps(vars(args),indent=2));return
    from h65.paper.evaluation import evaluate
    from h65.paper.profile import calibrate_costs
    from h65.paper.runtime import json_write
    model,model_cfg,resources,metadata=load_model(args)
    if model.config['dataset']=='anet':
        ready=Path(resources['datasets']['anet']['ready_file'])
        if not ready.exists() or json.loads(ready.read_text())['status']!='READY':raise RuntimeError('Full ANet evaluation requires READY data')
    if args.selector or args.disable_frame:
        from torch.utils.data import DataLoader
        from opentad.datasets.builder import build_dataset,collate
        from h65.full.runtime import to_gpu
        from h65.paper.geometry import candidate_mask
        for data in DataLoader(build_dataset(model_cfg.dataset.test),batch_size=1,num_workers=2,collate_fn=collate):
            data=to_gpu(data)
            if int(candidate_mask(data).sum())==768:break
        else:raise RuntimeError('No full window for changed routing cost')
        json_write(Path(args.output)/'cost_table.json',calibrate_costs(model,data))
    evaluate(model,model_cfg,resources,args.output,metadata,args.force_plan)


def parser():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--checkpoint',required=True)
    p.add_argument('--resources',default=str(ROOT/'research/paper/resources.local.json'))
    p.add_argument('--output',required=True);p.add_argument('--state',choices=['ema','learned'],default='ema')
    p.add_argument('--force-plan',type=int);p.add_argument('--budget-fraction',type=float)
    p.add_argument('--selector',choices=['anchor','uniform','random']);p.add_argument('--disable-frame',action='store_true')
    p.add_argument('--dry-run',action='store_true');return p


if __name__=='__main__':main(parser().parse_args())
