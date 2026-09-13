"""CPU mathematical checks for the fixed 239d098 review.

These are independent, deliberately small reimplementations of mathematical
mechanisms. They do NOT load the repository, checkpoints, videos, or a GPU,
and their results are NOT TAD accuracy or latency measurements.
Run: python mechanism_checks.py --output mechanism_checks.json
Requires: Python >=3.9 and NumPy.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import numpy as np


def arithmetic(channels: int, detector_macs: int) -> dict:
    clips, layers, t, side = 48, 12, 8, 10
    n = t * side**2
    d = channels // 4
    patch = clips * n * (3 * 2 * 16**2) * channels
    attention_linear = clips * layers * 4 * n * channels**2
    attention_qk_av = clips * layers * 2 * n**2 * channels
    mlp = clips * layers * 8 * n * channels**2
    tia = clips * layers * n * (2 * channels * d + d*d + 3*d)
    heavy = attention_linear + attention_qk_av + mlp
    total = patch + heavy + tia + detector_macs
    fixed = patch + tia + detector_macs
    at128 = ((patch + attention_linear + mlp + tia) * .64
             + attention_qk_av * .4096 + detector_macs)
    return {
        'channels': channels, 'matrix_convolution_MAC_only': True,
        'patch_GMAC': patch/1e9,
        'attention_linear_GMAC': attention_linear/1e9,
        'attention_QK_AV_GMAC': attention_qk_av/1e9,
        'attention_total_GMAC': (attention_linear+attention_qk_av)/1e9,
        'FFN_GMAC': mlp/1e9, 'TIA_GMAC': tia/1e9,
        'backbone_GMAC': (patch+heavy+tia)/1e9,
        'detector_GMAC': detector_macs/1e9,
        'total_GMAC': total/1e9, 'total_GFLOP_2MAC': total/1e9*2,
        'TIA_total_share': tia/total,
        'late4_MLP75_max_total_saving_before_modules': mlp/3*.25/total,
        'late4_MLP50_max_total_saving_before_modules': mlp/3*.5/total,
        'D8_total_ratio_before_exit': (patch+(heavy+tia)*8/12+detector_macs)/total,
        'all_time_128_ratio_before_modules': at128/total,
        'global_all_patch_TIA_half_clip_expensive_ops_ratio': (fixed+heavy/2)/total,
        'global_all_patch_TIA_first2_full_then_half_ratio': (fixed+heavy*7/12)/total,
        'full_spatial_state_B1_bf16_MiB': clips*n*channels*2/(1024**2),
        'native384_state_B1_bf16_MiB': 384*channels*2/(1024**2),
        'all12_spatial_states_B1_bf16_MiB': clips*n*channels*2*12/(1024**2),
        'excluded': ['new router/reconstruction/exit operators', 'bias', 'softmax',
                     'normalization', 'GELU', 'gather/scatter', 'interpolation',
                     'memory traffic', 'decode', 'NMS'],
    }


def main() -> dict:
    s = arithmetic(384, 9_648_820_224 + 4_807_434_240)
    b = arithmetic(768, 10_101_805_056 + 4_807_434_240)
    assert abs(s['total_GMAC'] - 1173.947019264) < 1e-8
    assert abs(b['total_GMAC'] - 4041.077354496) < 1e-8
    anchors = np.rint(np.linspace(0, 47, 6)).astype(int)
    candidates = [i for i in range(48) if i not in set(anchors)]
    zero_route = [int(i) for i in sorted(list(anchors) + candidates[:18])]
    uniform = np.rint(np.linspace(0,47,24)).astype(int).tolist()
    assert zero_route != uniform and len(set(zero_route)) == 24
    # Scalar-gradient proxy can vanish while replacement incurs nonzero loss.
    loss = lambda f: (f*f-1.0)**2
    assert loss(0.0)-loss(1.0) == 1.0
    # Non-affine monotone coordinate change changes interval IoU.
    transform = lambda x: np.interp(x, [0., 1., 2., 3.], [0., 1., 2., 20.])
    iou = lambda a,b: max(0.,min(a[1],b[1])-max(a[0],b[0])) / (max(a[1],b[1])-min(a[0],b[0]))
    pred, gt = np.array([1.,3.]), np.array([0.,2.])
    rank_iou = iou(pred,gt)
    real_iou = iou(transform(pred),transform(gt))
    assert abs(rank_iou-real_iou) > .1
    # Local independent tokenwise operation: dense-mask == packed-scatter.
    rng = np.random.default_rng(3407)
    x = rng.normal(size=(8,3)); w = rng.normal(size=(3,5))
    selected = np.array([0,2,5]); mask = np.zeros(8); mask[selected] = 1
    dense_mask = np.maximum(x@w,0)*mask[:,None]
    compact = np.zeros((8,5)); compact[selected] = np.maximum(x[selected]@w,0)
    assert np.allclose(dense_mask, compact)
    # Attention differs when omitted tokens are also removed from K/V.
    def attention(z):
        scores = z @ z.T / math.sqrt(z.shape[1])
        p = np.exp(scores-scores.max(1,keepdims=True)); p /= p.sum(1,keepdims=True)
        return p@z
    attention_error = np.max(np.abs(attention(x)[selected]-attention(x[selected])))
    assert attention_error > 1e-3
    # A cross-clip temporal convolution is not independently clip-cacheable.
    z = np.zeros(8); z[3] = 1.
    global_conv = np.convolve(z,[1.,1.,1.],mode='same')
    local_conv = np.concatenate([np.convolve(q,[1.,1.,1.],mode='same') for q in np.split(z,2)])
    assert global_conv[4] == 1 and local_conv[4] == 0
    # A zero-last-layer residual preserves an available interpolation baseline.
    base = rng.normal(size=(384,4)); h = rng.normal(size=(384,6)); last = np.zeros((6,4))
    assert np.array_equal(base+h@last, base)
    # Calibrated sigmoid capacity constraint: independent finite-difference check.
    logits = np.array([-.9,-.2,.3,.8,1.5]); temperature = .7; budget = 2.
    upstream = np.array([.2,-.6,.7,1.1,-.3])
    def rates(scores):
        lo,hi=-50.,50.
        for _ in range(100):
            lam=(lo+hi)/2
            r=1/(1+np.exp(-(scores-lam)/temperature))
            if r.sum()>budget: lo=lam
            else: hi=lam
        return 1/(1+np.exp(-(scores-(lo+hi)/2)/temperature))
    r=rates(logits); slope=r*(1-r)/temperature
    analytic=slope*(upstream-(slope*upstream).sum()/slope.sum())
    eps=1e-5; numerical=[]
    for i in range(len(logits)):
        e=np.zeros_like(logits);e[i]=eps
        numerical.append(np.dot(upstream,rates(logits+e)-rates(logits-e))/(2*eps))
    error=float(np.max(np.abs(analytic-np.array(numerical))))
    assert error < 1e-8
    result = {
        'evidence_level': 'CPU arithmetic and toy mechanisms only; no real TAD/GPU experiment',
        'main_commit': '239d098cd899936c35989fae6243c70259a85adb',
        'S_cost': s, 'B_cost': b,
        'EMA': {'decay':.999,'initial_coefficient_after_500':.999**500,
                'initial_coefficient_after_1500':.999**1500,
                'initial_coefficient_after_8000':.999**8000,
                'half_life_updates':math.log(.5)/math.log(.999),
                'warning':'parameter coefficient, NOT output or mAP proportion'},
        'zero_logit_route': {'anchors':anchors.tolist(),'stable_adaptive':zero_route,
                            'uniform24':uniform,
                            'adaptive_max_clip_id_gap':int(np.diff(zero_route).max()),
                            'uniform_max_clip_id_gap':int(np.diff(uniform).max())},
        'non_additivity': {'complementarity':{'U_a':0,'U_b':0,'U_ab':1},
                          'redundancy':{'U_a':1,'U_b':1,'U_ab':1}},
        'zero_teacher_gradient_counterexample': {'teacher_f':1,'student_f':0,'gradient':0,'true_loss_increase':1},
        'rank_geometry_counterexample':{'rank_IoU':rank_iou,'physical_IoU':real_iou},
        'dense_mask_vs_compact': {'pointwise_max_abs':float(np.max(np.abs(dense_mask-compact))),
                                'attention_selected_queries_removed_KV_max_abs':float(attention_error),
                                'note':'conditional equivalence is operator-specific'},
        'global_vs_local_convolution': {'global':global_conv.tolist(),'local':local_conv.tolist()},
        'calibrated_sigmoid':{'sum_rates':float(r.sum()),'max_gradient_finite_difference_error':error},
        'zero_residual_baseline_equal':True,
        'recorded_contrast_pp': {'global_to_local_loss':100*(.690125535485383-.6753443989239092),
                                'local_to_Z24_loss':100*(.6753443989239092-.5610650345213524),
                                'uniform24_minus_random24':100*(.5610650345213524-.5300220714609446),
                                'T24A_epoch5_minus_Z24':100*(.5064583163849076-.5610650345213524)},
        'timing_ratios_of_recorded_means_not_new_measurements': {
            'current_GPU_model_dense_over_Z24':67.87/42.60,
            'current_end_to_end_dense_over_Z24':1236.41/1229.65},
    }
    return result


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('mechanism_checks.json'))
    args=parser.parse_args()
    result=main()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))
