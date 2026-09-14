#!/usr/bin/env python3
"""Precompute the completed T slices of the existing allocation analysis.

No model execution, no GPU, no raw-data changes, and no second queue owner.
The original analyze_allocation stage reuses these exact validated AP caches.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'upstream')]


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)


def run():
    os.environ['CUDA_VISIBLE_DEVICES']=''
    import fcntl
    import numpy as np
    import torch
    from tools.atlas_analyze import load_windows,evaluate_variants
    torch.set_num_threads(4)
    analysis=ROOT/'analysis';analysis.mkdir(exist_ok=True)
    lock=(analysis/'temporal_precompute.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    resources=json.loads((ROOT/'resources.json').read_text())
    output={}
    for backbone in ('s','b'):
        name=f'allocation_{backbone}_T'
        if not (ROOT/'results'/name/'shard_0_done.json').exists():
            raise RuntimeError('Temporal precompute requires both completed full-data measurement slices')
        print(f'Loading full {name}',flush=True)
        rows=load_windows(ROOT/'results'/name,792)
        if len({row['meta']['video_id'] for row in rows})!=211:
            raise RuntimeError('Full AP requires exactly 211 measured videos')
        if any(not variant['matched'] for row in rows for variant in row['variants']):
            raise RuntimeError('A temporal execution violated the predeclared matched-cost contract')
        folder=analysis/'ap'/name;folder.mkdir(parents=True,exist_ok=True)
        scores=evaluate_variants(rows,resources,folder,'allocation',10000,backbone)
        paired={}
        for count in (4,6,8,10,12,16):
            cf=scores[f'marginal_cf:{count}'];uniform=scores[f'uniform:{count}']
            draw=np.asarray(cf['bootstrap_average'])-np.asarray(uniform['bootstrap_average'])
            paired[str(count)]=dict(delta_map=cf['average_mAP']-uniform['average_mAP'],
                ci=np.quantile(draw,[.025,.975]).tolist(),cost_difference=cf['mean_gflops']-uniform['mean_gflops'])
        output[backbone]=dict(scores=scores,paired_headroom=paired,
            provenance=dict(source_revisions=sorted({row['source_revision'] for row in rows}),
                heavy_checkpoint=resources['teachers'][f'thumos:{backbone}'],
                videos=211,windows=792,axis='T',bootstrap=10000))
        write(analysis/f'temporal_{backbone}_ready.json',output[backbone])
        del rows
    output['scope']='Full-data temporal slices only; D/S and population conclusions remain pending.'
    output['completed_at']=time.strftime('%Y-%m-%dT%H:%M:%S%z')
    write(analysis/'temporal_only.json',output)
    print('Full S/B temporal AP and paired video-cluster intervals ready.',flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--launch',action='store_true');args=p.parse_args()
    analysis=ROOT/'analysis';analysis.mkdir(exist_ok=True)
    receipt=analysis/'temporal_precompute_process.json'
    if args.launch:
        if (analysis/'temporal_only.json').exists():
            print(json.dumps(dict(already_complete=True)));return
        if receipt.exists():
            previous=json.loads(receipt.read_text())
            try:
                os.kill(previous['pid'],0)
                print(json.dumps(dict(already_running=True,**previous)));return
            except ProcessLookupError:pass
        stream=(ROOT/'logs/analysis_temporal_precompute.log').open('a')
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
        process=subprocess.Popen([sys.executable,str(Path(__file__).resolve())],cwd=ROOT,env=env,
            stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        value=dict(pid=process.pid,started=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
            scope='CPU precompute of existing analyze_allocation T caches; original 30-stage queue unchanged',gpu=False)
        write(receipt,value);print(json.dumps(value));return
    run()


if __name__=='__main__':main()
