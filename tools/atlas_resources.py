#!/usr/bin/env python3
"""Map transferred, existing assets to the isolated AutoDL experiment."""
import argparse
import json
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--root', default='/root/autodl-tmp/wtr_characterization_20260915')
p.add_argument('--data', default='/root/autodl-tmp/thumos14')
args = p.parse_args()
root = Path(args.root)
data = Path(args.data)
ann = data/'annotations/thumos_14_anno.json'
db = json.loads(ann.read_text())['database']
train = sorted(k for k,v in db.items() if v['subset'] == 'training' and (data/'train'/f'{k}.mp4').exists())
test = sorted(k for k,v in db.items() if v['subset'] == 'validation' and (data/'test'/f'{k}.mp4').exists())
if (len(train),len(test)) != (200,211):
    raise RuntimeError(f'THUMOS population mismatch: {len(train)}/{len(test)}')
res = dict(schema=2, datasets=dict(thumos=dict(annotations=str(ann),
    class_map=str(data/'annotations/category_idx.txt'), train_videos=str(data/'train'),
    test_videos=str(data/'test'), train_ids=train, test_ids=test)),
    teachers={f'thumos:{b}':str(root/'assets'/f'adatad_{b}_ema.pth') for b in ('s','b')},
    atlas_light={b:str(root/'assets'/f'v2_{b}_epoch040_light.pth') for b in ('s','b')})
(root/'resources.json').write_text(json.dumps(res, ensure_ascii=False, indent=2)+'\n')
print(json.dumps(dict(train=len(train), test=len(test), resources=str(root/'resources.json'))))
