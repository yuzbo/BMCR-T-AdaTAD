#!/usr/bin/env python3
"""Register training-only cohorts and explicit resources; never inspect test outcomes."""
import argparse
import json
from pathlib import Path
import random
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from h65.paper.runtime import json_write


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--train',required=True)
    p.add_argument('--test',required=True)
    p.add_argument('--annotations',required=True)
    p.add_argument('--class-map',required=True)
    p.add_argument('--assets',required=True)
    p.add_argument('--output',required=True)
    args = p.parse_args()
    db = json.loads(Path(args.annotations).read_text())['database']
    train = sorted(k for k,v in db.items() if v['subset'] == 'training')
    test = sorted(k for k,v in db.items() if v['subset'] == 'validation')
    if (len(train),len(test)) != (200,211):
        raise ValueError('THUMOS split contract is 200/211')
    missing = [str(Path(folder)/f'{name}.mp4') for folder,ids in [(args.train,train),(args.test,test)]
               for name in ids if not (Path(folder)/f'{name}.mp4').is_file()]
    if missing:
        raise FileNotFoundError(f'{len(missing)} missing required videos: {missing[:6]}')
    asset = Path(args.assets)
    for name in ('adatad_s_ema.pth','v2_s_epoch040_light.pth','v2_s_initial_source.pth'):
        if not (asset/name).is_file():
            raise FileNotFoundError(asset/name)
    resources = dict(datasets=dict(thumos=dict(annotations=args.annotations,class_map=args.class_map,
        train_videos=args.train,test_videos=args.test,train_ids=train,test_ids=test)),
        teachers={'thumos:s':str(asset/'adatad_s_ema.pth')},atlas_light={'s':str(asset/'v2_s_epoch040_light.pth')})
    shuffled = train.copy()
    random.Random(42).shuffle(shuffled)
    splits = dict(fit=sorted(shuffled[:160]),calibration=sorted(shuffled[160:180]),holdout=sorted(shuffled[180:]))
    mini = dict(fit=splits['fit'][:16],calibration=splits['calibration'][:4],holdout=splits['holdout'][:4])
    protocol = dict(schema='raw_v1_pilot_1',seed=42,phase='P1 Evidence Acquisition Prototype',
        parallel_scientific_line='Batch-1A Atlas D/S; independent existing owner',
        input=dict(preview_count=192,preview_resolution=64,offsets=[-.375,-.125,.125,.375],
            heavy_slots=384,depth_capacity=1.,space_capacity=1.,cross_queries=384,detector_positions=768,
            acquisition_unit='individual_frame',domain='O or O union bounded off-grid',
            tail_valid_capacity='min(384, distinct valid official frames), shared between domains'),
        search=dict(rounds=4,pairs_per_round=16,stop='nonpositive predicted or measured utility'),
        splits=splits,mini=mini,gate_videos=sorted(sum([v[:2] for v in splits.values()],[])),
        mini_bank=dict(videos=24,states=2,domains=2,pairs_per_state=4,maximum_swaps=384),
        full_bank=dict(videos=200,states=2,domains=2,pairs_per_state=8,maximum_swaps=6400),
        episode_policy='official deterministic sliding windows; bank uses middle window; no detector augmentation training',
        training_scope='Value-only pilot, not a replacement detector training sampler',
        explicit_domain_indicator=False,graph=False,fvd=False,db=False,
        publication_policy='no test-driven tuning; Value full test requires holdout gate; publication CF only after freezing',
        full_bank_status='requires mini-bank evidence review',publication_status='not released')
    output = Path(args.output)
    for filename,value in [('resources.json',resources),('protocol.json',protocol)]:
        path=output/filename
        if path.exists() and json.loads(path.read_text()) != value:
            raise ValueError(f'Existing registration differs: {path}')
        json_write(path,value)
    print(json.dumps(dict(resources=str(output/'resources.json'),protocol=str(output/'protocol.json'),train=200,test=211,mini_swaps=384)))


if __name__ == '__main__':
    main()
