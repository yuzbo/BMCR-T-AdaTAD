"""Transfer ONLY a fitted frame refiner to a later frozen detector; no future labels.

Reports a separate cross-checkpoint policy-transfer test. Never overwrite original weights.
"""
import argparse
from pathlib import Path
import torch
from .core import file_digest,atomic_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--router-checkpoint',required=True);p.add_argument('--target-checkpoint',required=True)
    p.add_argument('--target-state',choices=['ema','learned'],default='learned');p.add_argument('--output',required=True)
    args=p.parse_args()
    if Path(args.output).exists():raise FileExistsError(args.output)
    source=torch.load(args.router_checkpoint,map_location='cpu',weights_only=False)
    target=torch.load(args.target_checkpoint,map_location='cpu',weights_only=False)
    if not source.get('evaluation_only'):raise ValueError('Source must be an offline router export.')
    if source['metadata']['config']!=target['metadata']['config']:raise ValueError('Configuration mismatch.')
    if source['successful_updates']>=target['successful_updates']:raise ValueError('Target must be a later detector.')
    result={k:v.clone() for k,v in target[args.target_state].items()}
    moved=[]
    for k,v in source['learned'].items():
        if not k.startswith('frame_router.'):continue
        if k not in result or result[k].shape!=v.shape:raise ValueError('State mismatch '+k)
        result[k]=v.clone();moved.append(k)
    if not moved:raise ValueError('No frame router state.')
    metadata=dict(target['metadata']);metadata['wtr_transfer']=dict(source=file_digest(args.router_checkpoint),
        target=file_digest(args.target_checkpoint),changed_keys=moved,future_labels_used_for_fit=False,
        caveat='Later detector feature/state drift can change calibration; report this test separately.')
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    data=dict(learned=result,ema={k:v.clone() for k,v in result.items()},metadata=metadata,
        successful_updates=target['successful_updates'],epoch_index=target.get('epoch_index'),
        evaluation_only=True,optimizer_resume_forbidden=True)
    tmp=out.with_suffix('.tmp');torch.save(data,tmp);tmp.replace(out)
    atomic_json(out.with_suffix('.receipt.json'),metadata['wtr_transfer'])


if __name__=='__main__':main()
