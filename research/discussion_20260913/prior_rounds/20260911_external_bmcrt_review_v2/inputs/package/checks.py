"""CPU-only arithmetic and geometry checks for the fixed-commit review.
No repository training, inference, dataset loading or mAP re-evaluation is run.
Input metrics are transcribed from phase2_20260910/comparison.json at
15280e5dc29e3df18d085aaba21067e04107ce84; these checks do not authenticate weights.
"""
from pathlib import Path
import json
import numpy as np

OUT = Path(__file__).resolve().parent
rows = {
    'S_official': dict(ap=[.8383380539306469,.7908809643471865,.7232891216449084,.6157177867327084,.48240175077146424], gmac=1173.947019264, median_ms=51.17436498, mean_ms=51.20790554, peak_gib=.935388565),
    'S_H65C': dict(ap=[.7838136215419993,.7307661121238492,.6633324165974027,.5520029573608478,.4130506780025591], gmac=605.155081728, median_ms=86.67039499, mean_ms=92.83603, peak_gib=.861283302),
    'S_BMCRT': dict(ap=[.7877310917656425,.7389128709858096,.6591456733429285,.5619945960365763,.4100304017711937], gmac=605.172676608, median_ms=121.59073958, mean_ms=121.91008334, peak_gib=.861325741),
    'B_official': dict(ap=[.8598400814011409,.8184182052281697,.7493058903658388,.6326148983979525,.4962205511035524], gmac=4041.077354496, median_ms=118.17210168, mean_ms=118.31026, peak_gib=1.507989407),
    'B_H65C': dict(ap=[.8236880920909322,.775707521174154,.6992535172486127,.5888749043865396,.4431160228210951], gmac=2038.720249344, median_ms=123.01701680, mean_ms=123.084753, peak_gib=1.123346329),
    'B_BMCRT': dict(ap=[.8301970669741479,.7836110680175928,.7067274850351016,.5929835309762771,.45350316708774274], gmac=2038.737844224, median_ms=153.74325588, mean_ms=153.748753, peak_gib=1.123399258),
}
for row in rows.values():
    row['mean_ap_percent'] = float(np.mean(row['ap']) * 100)
derived = {}
for scale, dim in [('S',384), ('B',768)]:
    off, new, old = (rows[f'{scale}_{kind}'] for kind in ('official','BMCRT','H65C'))
    head_extra = 7.580399616 if scale == 'S' else 7.806892032
    # A proposed 96->D coarse projection and a (96+6)->32->1 scalar gate.
    coarse_gmac = 768 * 96 * dim / 1e9
    gate_gmac = 768 * (102*32+32) / 1e9
    proposed_gmac = new['gmac'] + head_extra + coarse_gmac + gate_gmac
    derived[scale] = dict(
        avg_drop_pp=off['mean_ap_percent']-new['mean_ap_percent'],
        ap07_drop_pp=100*(off['ap'][-1]-new['ap'][-1]),
        gains_vs_H65C_pp=[100*(n-o) for n,o in zip(new['ap'],old['ap'])],
        avg_gain_vs_H65C_pp=new['mean_ap_percent']-old['mean_ap_percent'],
        mac_reduction_percent=100*(1-new['gmac']/off['gmac']),
        median_latency_ratio=new['median_ms']/off['median_ms'],
        scoped_peak_allocated_reduction_percent=100*(1-new['peak_gib']/off['peak_gib']),
        proposed_dense_fusion_gmac_analytical_not_measured=proposed_gmac,
        proposed_mac_reduction_percent_analytical_not_measured=100*(1-proposed_gmac/off['gmac']),
    )

# Non-affine monotone coordinate changes need not preserve interval IoU.
positions = np.array([0,1,2,3,8,9,10,11,12.],dtype=float)
ranks = np.arange(9,dtype=float)
gt = np.array([3.,9.]); pred = np.array([2.,10.])
def iou(a,b):
    inter = max(0., min(a[1], b[1])-max(a[0], b[0]))
    return float(inter / ((a[1]-a[0])+(b[1]-b[0])-inter))
rank_gt = np.interp(gt,positions,ranks); rank_pred = np.interp(pred,positions,ranks)
assert np.allclose(np.interp(rank_gt,ranks,positions),gt)
assert np.allclose(np.interp(rank_pred,ranks,positions),pred)
assert iou(gt,pred) == .75 and iou(rank_gt,rank_pred) == .5

# A simple two-frame temporal convolution demonstrates that zero padding is
# not interchangeable with edge replication, even when output tokens are masked.
x = 1.; w = np.array([.4,.6])
edge_pair = float(w @ np.array([x,x])); zero_pair = float(w @ np.array([x,0.]))
assert edge_pair != zero_pair
checks = dict(
    scope='Only arithmetic and toy geometry/padding; no model training or inference.',
    source_commit='15280e5dc29e3df18d085aaba21067e04107ce84',
    rows=rows, derived=derived,
    toy_geometry=dict(positions=positions.tolist(),rank_gt=rank_gt.tolist(),rank_prediction=rank_pred.tolist(),true_iou=iou(gt,pred),rank_iou=iou(rank_gt,rank_pred),round_trip_correct=True),
    toy_padding=dict(edge_pair=edge_pair,zero_pair=zero_pair,interpretation='Demonstrates a possible feature change, not an observed mAP effect.'),
    teacher_budget=dict(max_extra_forward_equivalents_per_student_example_on_active_steps=3/(8*2),active_events_for_updates_672_through_3992=416,upper_bound_joint_average_extra_F_equivalents=416*3/(4000*2)),
)
(OUT/'arithmetic_checks.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(derived,ensure_ascii=False,indent=2))
print('Toy geometry:', checks['toy_geometry'])
