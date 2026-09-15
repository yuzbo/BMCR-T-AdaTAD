"""Read the fixed repository's measured results; do not query floating live status."""
import argparse
import json
from pathlib import Path
from .integration import setup_repository
from .core import atomic_json


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--repo',required=True);p.add_argument('--output',required=True)
    args=p.parse_args();root=setup_repository(args.repo)
    rows=json.loads((root/'research/project_status_20260914/current_results.json').read_text())
    wanted=[]
    ids={f'review5485_full_v2_{b}_seed42' for b in ['s','b']}|{f'thumos_{b}_point_uniform_seed42' for b in ['s','b']}
    for r in rows:
        if r['run_id'] not in ids:continue
        dist=r.get('plan_distribution') or {};n=sum(dist.values())
        active_d=sum(v for k,v in dist.items() if '_D100_' not in k)
        active_s=sum(v for k,v in dist.items() if not k.endswith('_S100'))
        joint=sum(v for k,v in dist.items() if '_D100_' not in k and not k.endswith('_S100'))
        wanted.append(dict(run_id=r['run_id'],epoch=r['epoch'],map=r['map'],
            dataset_mean_gflops=r['dataset_mean_gflops'],latency_ms=r['latency_ms'],
            windows=n,depth_compressed_fraction=active_d/n if n else None,
            spatial_compressed_fraction=active_s/n if n else None,
            simultaneous_ds_fraction=joint/n if n else None,plan_distribution=dist))
    comparisons=[]
    for b in ['s','b']:
        a={r['epoch']:r for r in wanted if r['run_id']==f'review5485_full_v2_{b}_seed42'}
        u={r['epoch']:r for r in wanted if r['run_id']==f'thumos_{b}_point_uniform_seed42'}
        for epoch in sorted(a.keys()&u.keys()):
            comparisons.append(dict(backbone=b,epoch=epoch,map_delta_pp=a[epoch]['map']-u[epoch]['map'],
                matrix_flops_reduction_pct=100*(1-a[epoch]['dataset_mean_gflops']/u[epoch]['dataset_mean_gflops'])))
    atomic_json(args.output,dict(measured_rows=wanted,matched_epoch_comparisons=comparisons,
        caveat='Uniform uses a different recipe/init path; this is matched epoch, not full causal matching.'))
    print(json.dumps(comparisons,indent=2))


if __name__=='__main__':main()
