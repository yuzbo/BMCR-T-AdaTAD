"""Read original checkpoints and stream a compact, provenance-bearing CPU asset."""
import argparse
import json
import sys
from pathlib import Path
import torch

p=argparse.ArgumentParser()
p.add_argument('--backbone',choices=['s','b'],required=True)
args=p.parse_args()
root=Path('/data/run01/sczc063/yuzibo/h65_clean_adatad_20260910')
b=args.backbone
cfg=f'review5485_full_v2_{b}_seed42'
source=root/'support_review_20260914/research/paper/runs'/cfg/'epoch_040.pth'
payload=torch.load(source,map_location='cpu')
resources=json.loads((root/'paper_20260913/research/paper/resources.local.json').read_text())
original=resources['encoders'][f'thumos:{b}']['checkpoint']
base=torch.load(original,map_location='cpu')['state_dict_ema']
base={k.removeprefix('module.'):v for k,v in base.items()}
teacher=torch.load(resources['teachers'][f'thumos:{b}'],map_location='cpu')['state_dict_ema']
teacher={k.removeprefix('module.'):v for k,v in teacher.items()}
different={}
for key,value in base.items():
    if key.startswith('backbone.') and (key not in teacher or not torch.equal(value,teacher[key])):
        different[key]=value
scout={k:v for k,v in base.items() if k.startswith('scout.')}
out=dict(ema=payload['ema'], metadata=payload['metadata'], epoch_index=payload['epoch_index'],
    successful_updates=payload['successful_updates'], original_checkpoint=str(source),
    original_encoder_checkpoint=original, backbone_overrides=different, scout_initial=scout,
    reconstruction='Official checkpoint plus exact differing initial backbone tensors, then complete V2 EMA learned state',
    extraction='Read-only CPU extraction; no optimization; no checkpoint hash added')
torch.save(out,sys.stdout.buffer)
