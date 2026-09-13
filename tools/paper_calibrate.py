"""Collect final-student T/D/S/frame re-executions, fit OOF routers, save deployment."""
import argparse
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'upstream')]
from tools.paper_eval import load_model


def main(args):
    import numpy as np
    import torch
    from torch.utils.data import DataLoader,Subset
    from opentad.datasets.builder import build_dataset,collate
    from h65.full.runtime import to_gpu
    from h65.paper.training import EpochDataset,cpu_state
    from h65.paper.interventions import collect_action
    from h65.paper.calibration import calibrate
    from h65.paper.runtime import json_write
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True);start=time.perf_counter()
    model,mc,resources,metadata=load_model(args)
    dataset=build_dataset(mc.dataset.train);wrapped=EpochDataset(dataset,model.config['seed']+500)
    train_ids={x[0] for x in dataset.data_list};test_ids=set(resources['datasets'][model.config['dataset']]['test_ids'])
    if train_ids&test_ids:raise RuntimeError('Calibration and test video IDs overlap')
    ids=np.random.default_rng(model.config['seed']).permutation(len(wrapped))[:args.videos].tolist()
    labels=out/'actual_actions.jsonl';records=[json.loads(x) for x in labels.read_text().splitlines()] if labels.exists() else []
    provenance=out/'source.json';identity=dict(checkpoint=str(args.checkpoint),state=args.state,config=model.config,indices=ids)
    if provenance.exists() and json.loads(provenance.read_text())!=identity:raise ValueError('Calibration source changed')
    json_write(provenance,identity);done={(r['video_name'],r['action_type']) for r in records}
    loader=DataLoader(Subset(wrapped,ids),batch_size=1,shuffle=False,num_workers=2,collate_fn=collate,pin_memory=True)
    for index,cpu in enumerate(loader):
        data=to_gpu(cpu);name=data['metas'][0]['video_name']
        for offset,kind in enumerate(('frame','temporal','depth','spatial')):
            if (name,kind) in done:continue
            record=collect_action(model,data,kind,index*4+offset,measure=True)
            if record is None:continue
            record['dataset_index']=ids[index];records.append(record)
            with labels.open('a') as stream:stream.write(json.dumps(record)+'\n')
        json_write(out/'progress.json',dict(videos=index+1,total_videos=len(ids),actions=len(records)))
        if index%20==0:print(f'Actual intervention labels {index+1}/{len(ids)}; {len(records)} actions',flush=True)
    fitted,report=calibrate(records,model.config['seed'],steps=args.fit_steps)
    for kind,router in fitted.items():
        destination=model.frame_router if kind=='frame' else model.budget_router
        destination.network.load_state_dict(router.network.state_dict(),strict=True)
        destination.scales.copy_(router.scales);destination.sigma_calibration.copy_(router.sigma_calibration)
    payload=torch.load(args.checkpoint,map_location='cpu');payload['ema']=cpu_state(model.learned_state())
    payload['calibration']=dict(source=str(args.checkpoint),report=str(out/'oof_report.json'),training_video_labels_only=True)
    torch.save(payload,out/'calibrated.tmp');(out/'calibrated.tmp').replace(out/'calibrated.pth')
    json_write(out/'oof_report.json',report)
    from h65.paper.figures import plot_calibration
    plot_calibration(records,report,out)
    json_write(out/'completed.json',dict(**metadata,actual_interventions=len(records),video_disjoint_router_oof=True,
        calibration_videos=report['videos'],test_labels_used=False,seconds=time.perf_counter()-start,
        paired_student_forward_gflops=sum((r['base_forward_flops']+r['action_forward_flops'])/1e9 for r in records),
        cost_scope='paired actual student/head scoring forwards; repair-target generation and fit overhead separately timed',
        representation_scope=report['representation_scope'],calibrated_checkpoint=str(out/'calibrated.pth')))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--checkpoint',required=True)
    p.add_argument('--resources',default=str(ROOT/'research/paper/resources.local.json'));p.add_argument('--output',required=True)
    p.add_argument('--state',default='ema',choices=['ema','learned']);p.add_argument('--videos',type=int,default=200)
    p.add_argument('--fit-steps',type=int,default=600);main(p.parse_args())
