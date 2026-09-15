"""Fixed T24A full-test peak selection; no per-policy checkpoint shopping."""
import json
import os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
EXP=ROOT/'ds3_20260912'
RUNS=Path(os.environ.get('DS3_RUNS_DIR',EXP/'runs'))
EPOCHS=tuple(range(5,81,5))
ALIASES={'F100':'T24A','F010':'S75','F101':'DAD'}


def selection(backbone,runs=RUNS):
    rows=[]
    for epoch in EPOCHS:
        path=runs/f'{backbone}_T24A_epoch_{epoch:02}/metrics.json'
        if not path.exists():continue
        value=json.loads(path.read_text())
        if value['test_videos']!=211 or value['test_windows']!=792:raise ValueError('partial test cannot select a checkpoint')
        if value['initialization']['successful_updates']!=epoch*100:raise ValueError('wrong checkpoint update count')
        rows.append(dict(epoch=epoch,average_mAP=value['metrics']['average_mAP']))
    best=max(rows,key=lambda row:(row['average_mAP'],-row['epoch'])) if rows else None
    return dict(backbone=backbone,candidates=rows,best=best,all_candidates_evaluated=len(rows)==len(EPOCHS),
                rule='highest complete-test T24A average_mAP; earliest tie; same selected aux EMA for all final policies',
                warning='test-based checkpoint selection requested by user; not an untouched-test estimate')


def save_selection(backbone,runs=RUNS):
    value=selection(backbone,runs);path=runs/f'{backbone}_selection.json';path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)
    return value


if __name__=='__main__':
    for b in ('s','b'):print(json.dumps(save_selection(b),indent=2))
