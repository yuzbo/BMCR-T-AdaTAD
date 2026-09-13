"""Independent arithmetic/predicate checks; not repository execution or model testing.

Inputs are transcribed from the audited fixed-commit source/results.
No network access, training, checkpoint loading, or dataset evaluation occurs.
"""
from __future__ import annotations
import json
import re
from pathlib import Path

def interval_iou(a: tuple[float,float], b: tuple[float,float]) -> float:
    intersection = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = (a[1] - a[0]) + (b[1] - b[0]) - intersection
    if union <= 0:
        raise ValueError('Intervals must have positive union.')
    return intersection / union

# Source-declared ASFormer: two 96->2 classifiers; four internal 48->96 projections.
param_shapes = {}
for parent in ('temporal.encoder', 'temporal.decoders.0'):
    param_shapes[parent+'.conv_out.weight'] = 96*2
    param_shapes[parent+'.conv_out.bias'] = 2
    for layer in range(2):
        key = f'{parent}.layers.{layer}.att_layer.conv_out'
        param_shapes[key+'.weight'] = 48*96
        param_shapes[key+'.bias'] = 96
pattern = re.compile(r'^temporal\.(?:encoder|decoders\.\d+)\.conv_out\.(?:weight|bias)$')
wrongly_assigned = {n:v for n,v in param_shapes.items()
                   if '.conv_out.' in n and pattern.fullmatch(n) is None}
assert sum(wrongly_assigned.values()) == 18816
assert sum(v for n,v in param_shapes.items() if pattern.fullmatch(n)) == 388
assert sum(v for n,v in param_shapes.items() if n.endswith('weight')) == 18816
assert sum(v for n,v in param_shapes.items() if n.endswith('bias')) == 388

metrics = {}
rows = {
    'S': dict(official_avg=69.01255355, bmcr_avg=63.15629268,
              h65_avg=62.85931571, official_t07=48.24017508, bmcr_t07=41.00304018,
              h65_t07=41.30506780, official_gmac=1173.947019264,
              bmcr_gmac=605.172676608, official_median_ms=51.174364984,
              bmcr_median_ms=121.590739582, official_mem_gib=.935388565,
              bmcr_mem_gib=.861325741, detector_increment_gmac=7.580399616,
              embed_dim=384),
    'B': dict(official_avg=71.12799253, bmcr_avg=67.34044636,
              h65_avg=66.61280115, official_t07=49.62205511, bmcr_t07=45.35031671,
              h65_t07=44.31160228, official_gmac=4041.077354496,
              bmcr_gmac=2038.737844224, official_median_ms=118.172101676,
              bmcr_median_ms=153.743255883, official_mem_gib=1.507989407,
              bmcr_mem_gib=1.123399258, detector_increment_gmac=7.806892032,
              embed_dim=768),
}
for name, d in rows.items():
    ce = d['embed_dim']; t = 768; width = 96
    # One C projection, one fine projection, scalar gate (200->32->1), output projection.
    fusion_gmac = t*(width*width+ce*width+200*32+32+width*ce)/1e9
    hypothetical_gmac = d['bmcr_gmac']+d['detector_increment_gmac']+fusion_gmac
    metrics[name] = dict(
        avg_gap_pp=d['bmcr_avg']-d['official_avg'],
        strict_gap_pp=d['bmcr_t07']-d['official_t07'],
        bmcr_vs_h65_avg_pp=d['bmcr_avg']-d['h65_avg'],
        bmcr_vs_h65_t07_pp=d['bmcr_t07']-d['h65_t07'],
        mac_reduction_pct=100*(1-d['bmcr_gmac']/d['official_gmac']),
        median_latency_ratio=d['bmcr_median_ms']/d['official_median_ms'],
        allocated_memory_reduction_pct=100*(1-d['bmcr_mem_gib']/d['official_mem_gib']),
        proposed_fusion_only_gmac=fusion_gmac,
        proposed_full_model_gmac_estimate=hypothetical_gmac,
        proposed_mac_reduction_pct_estimate=100*(1-hypothetical_gmac/d['official_gmac']))

out = {
    'scope':'Independent source-derived predicate and arithmetic checks ONLY; no repository tests or model execution.',
    'code_commit':'15280e5dc29e3df18d085aaba21067e04107ce84',
    'optimizer_group_check': {
        'wrongly_assigned_internal_projection_parameters':sum(wrongly_assigned.values()),
        'intended_action_classifier_parameters':388,
        'actual_action_lr_group_parameters_from_predicate':sum(param_shapes.values()),
        'actual_weight_subgroup':18816,
        'actual_bias_subgroup':388,
        'wrong_names':wrongly_assigned},
    'monotone_time_map_counterexample':{
        'rank_grid':[0,1,2,3], 'real_grid':[0,1,2,12],
        'gt_rank':[0,3], 'gt_real':[0,12],
        'prediction_A_rank':[0,2], 'prediction_A_real':[0,2],
        'prediction_B_rank':[1,3], 'prediction_B_real':[1,12],
        'rank_iou_A':interval_iou((0,3),(0,2)),
        'rank_iou_B':interval_iou((0,3),(1,3)),
        'real_iou_A':interval_iou((0,12),(0,2)),
        'real_iou_B':interval_iou((0,12),(1,12)),
        'note':'Mathematical counterexample, not an estimate of observed dataset error.'},
    'metrics_derived':metrics,
    'teacher_upper_bound': {
        'events':len([u for u in range(4000) if u>667 and u%8==0]),
        'max_single_row_forwards':3*len([u for u in range(4000) if u>667 and u%8==0]),
        'joint_student_row_forwards':4000*2,
        'note':'Forward-row ratio is not the training FLOP ratio or wall-time ratio.'},
}
path=Path(__file__).with_name('independent_checks.json')
path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(out, ensure_ascii=False, indent=2))
