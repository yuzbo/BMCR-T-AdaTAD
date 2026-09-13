"""Independently recompute review arithmetic from local experiment records."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
phase = ROOT / 'phase2_20260910'
rows = json.loads((phase/'comparison.json').read_text())['results']
lookup = {(r['backbone'], r['method']): r for r in rows}
calculations = []
for backbone, width in [('S',384), ('B',768)]:
    official, h65, bmcr = [lookup[(backbone,m)] for m in ('official','h65','bmcr')]
    head_extra = sum(official['component_macs'][k]-bmcr['component_macs'][k] for k in ('projection','head')) / 1e9
    fusion = 768 * (96*96 + width*96 + 96*width + 200*32 + 32*1) / 1e9
    estimate = bmcr['gmacs'] + head_extra + fusion
    calculations.append(dict(backbone=backbone, avg_map_drop_pp=100*(official['metrics']['average_mAP']-bmcr['metrics']['average_mAP']),
        bmcr_over_h65_pp={k:100*(bmcr['metrics'][k]-h65['metrics'][k]) for k in bmcr['metrics']},
        median_latency_ratio=bmcr['latency_median_ms']/official['latency_median_ms'],
        allocated_reduction_percent=100*(1-bmcr['peak_gib']/official['peak_gib']),
        proposed_head_projection_extra_gmac=head_extra, proposed_fusion_gmac=fusion,
        proposed_total_gmac=estimate, proposed_mac_reduction_percent=100*(1-estimate/official['gmacs'])))

training=[]
for backbone in ('s','b'):
    warm=json.loads((phase/f'runs/{backbone}_warm/completed.json').read_text())
    joint=json.loads((phase/f'runs/{backbone}_bmcr/completed.json').read_text())
    logs=[json.loads(line) for line in (phase/f'runs/{backbone}_bmcr/train.jsonl').read_text().splitlines() if line]
    planned=actual_events=forwards=0
    for r in logs:
        update=r['successful_updates']-1
        if r['weights']['contribution']>0 and update%8==0:
            planned+=1
        observations=r['diagnostics'].get('counterfactuals',[])
        if observations:
            actual_events+=1
            baselines={o['row'] for o in observations}
            swaps={(o['row'],o['remove'],o['insert']) for o in observations}
            forwards+=len(baselines)+len(swaps)
    training.append(dict(backbone=backbone,warm_plus_joint_hours=(warm['elapsed_seconds']+joint['elapsed_seconds'])/3600,
        full_course_max_recorded_allocated_gib=max(warm['peak_gib'],joint['peak_gib']),
        scheduled_teacher_events=planned,nonempty_teacher_events_from_logs=actual_events,
        teacher_route_forwards_reconstructed_from_logged_unique_exchanges=forwards,
        teacher_route_forward_upper_bound=planned*3,student_joint_forward_rows=8000,
        reconstruction_limit='Call count reconstructed from records and source control flow, not independently traced GPU time/FLOPs.'))

lr_records=[]
for name in ('s_warm','b_warm','s_h65','b_h65','s_bmcr','b_bmcr'):
    config=json.loads((phase/'runs'/name/'completed.json').read_text())
    # Scout groups are appended after detector groups and one adapter group.
    groups=config['optimizer_groups'][-6:] if not name.endswith('warm') else config['optimizer_groups'][-4:]
    lr_records.append(dict(stage=name,scout_groups=groups))

def iou(a,b):
    overlap=max(0,min(a[1],b[1])-max(a[0],b[0]))
    return overlap/((a[1]-a[0])+(b[1]-b[0])-overlap)
output=dict(results=calculations,training=training,actual_optimizer_group_records=lr_records,
    monotone_mapping_example=dict(rank_ious=[iou((0,3),(0,2)),iou((0,3),(1,3))],
        true_time_ious=[iou((0,12),(0,2)),iou((0,12),(1,12))]),
    proposed_cost_scope='Exact arithmetic for the stated 200->32->1 gate and projection dimensions; prospective costs, no new model measured.')
(OUT/'records_and_math.json').write_text(json.dumps(output,indent=2)+'\n',encoding='utf-8')
print(json.dumps(output,indent=2))
