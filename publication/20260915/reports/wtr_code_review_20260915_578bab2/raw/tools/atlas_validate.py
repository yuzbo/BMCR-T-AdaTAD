"""Validate full-video allocation reconstruction/AP on development measurements."""
import argparse
import copy
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'upstream')]
from h65.atlas.data import CharacterizationData
from h65.atlas.statistics import ap_cache
from h65.paper.native_adatad import build_native_config
from h65.paper.evaluation import merge_windows
from h65.paper.runtime import json_write

p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--resources',required=True)
p.add_argument('--backbone',choices=['s','b'],required=True);p.add_argument('--output',required=True)
args=p.parse_args()
rows=[json.loads(x.read_text()) for x in sorted((Path(args.input)/'windows').glob('*.json'))]
ids=sorted({r['meta']['video_id'] for r in rows})
if not rows or any(r['meta']['split']!='development' or not r.get('passed') for r in rows):
    raise RuntimeError('Validation requires passed development preflight records')
resources=json.loads(Path(args.resources).read_text())
cfg=build_native_config(dict(backbone=args.backbone,frames=768),resources)
data=CharacterizationData(cfg,resources,'development',ids)
if len(rows)!=len(data):raise RuntimeError('AP smoke test must reconstruct the complete chosen development videos')
out=Path(args.output);out.mkdir(parents=True,exist_ok=True);gt=out/'development_ground_truth.json'
data.ground_truth(gt)
post=copy.deepcopy(cfg.post_processing);post.sliding_window=True
checks=[]
for axis in ('T','D','S'):
    for policy in ('uniform','random','static','attention','marginal_cf'):
        pred={v:[] for v in ids}
        for row in rows:
            item=next(r for r in row['allocation'][axis]['variants'] if r['policy']==policy)
            if not item['matched']:raise RuntimeError('Allocation failed predeclared matched-cost tolerance')
            for video,values in item['predictions'].items():pred[video].extend(values)
        pred=merge_windows(pred,post)
        caches,official=ap_cache(pred,gt,'training',ids)
        checks.append(dict(axis=axis,policy=policy,official=official,cache_matches_official=True))
receipt=dict(passed=True,videos=len(ids),windows=len(rows),checks=checks,training_updates=0,
    scope='Technical validation only; development results are not publication evidence')
json_write(out/'passed.json',receipt);print(json.dumps(receipt),flush=True)
