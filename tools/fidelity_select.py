"""Track user-requested joint-stage test peaks without changing training duration."""
import argparse
import json
import math
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = Path(os.environ.get('H65_RUNS_DIR', ROOT/'fidelity_20260911/runs')).resolve()
EPOCHS = tuple(range(25,61,5))


def select_best(runs, backbone):
    candidates = []
    for epoch in EPOCHS:
        folder = runs/f'{backbone}_h65_test_epoch_{epoch:02}'
        path = folder/'metrics.json'
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        if data['test_videos'] != 211 or data['test_windows'] != 792:
            raise ValueError(f'{path}: candidate is not a complete test')
        init = data['initialization']
        if init['total_epochs'] != epoch or init.get('fidelity_revision') != 'lr_identity_crop_validity_v1':
            raise ValueError(f'{path}: wrong candidate provenance')
        mean = data['metrics']['average_mAP']
        if not math.isfinite(mean):
            raise ValueError(f'{path}: nonfinite mAP')
        candidates.append(dict(total_epochs=epoch, average_mAP=mean, metrics=data['metrics'],
                               checkpoint=init['checkpoint'], state_key=init['state_key'], folder=str(folder)))
    # Candidates are epoch-ascending; a tie retains the earliest checkpoint.
    best = max(candidates, key=lambda x:x['average_mAP']) if candidates else None
    return dict(backbone=backbone, criterion='maximum full211-test average_mAP at joint milestones; earliest tie',
                test_based_selection=True, expected_total_epochs=list(EPOCHS),
                evaluated=len(candidates), all_candidates_evaluated=len(candidates)==len(EPOCHS),
                best=best, terminal=next((c for c in candidates if c['total_epochs']==60),None), candidates=candidates)


def save_selection(runs, backbone):
    result = select_best(runs, backbone)
    path = runs/f'{backbone}_best_test_checkpoint.json'
    path.parent.mkdir(parents=True,exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(result,indent=2)+'\n')
    temp.replace(path)
    return result


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--backbone',choices=['s','b'],required=True)
    args=parser.parse_args()
    print(json.dumps(save_selection(RUNS,args.backbone),indent=2))
