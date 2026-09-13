"""CPU algebra checks, NOT a run of BMCR/DS3, checkpoint loading, or GPU validation.
Run: python toy_audit.py --out toy_results.json
Uses only Python's standard library. Counts matrix/convolution MACs; excludes
bias, activations, normalization, interpolation, masks, selection, and memory I/O.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1 / (1 + z)
    z = math.exp(x)
    return z / (1 + z)


def calibrated(logits: list[float], budget: float, tau: float) -> list[float]:
    if not 0 < budget < len(logits) or tau <= 0:
        raise ValueError('Require 0 < budget < n and positive temperature')
    lo, hi = min(logits) - 100 * tau, max(logits) + 100 * tau
    for _ in range(100):
        mid = (lo + hi) / 2
        if sum(sigmoid((x - mid) / tau) for x in logits) > budget:
            lo = mid
        else:
            hi = mid
    return [sigmoid((x - (lo + hi) / 2) / tau) for x in logits]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=Path('toy_results.json'))
    args = parser.parse_args()
    total = 1173947019264  # Reported dense S matrix/conv MACs, fixed repository record.
    clips, layers, channels, tokens, hidden = 48, 12, 384, 800, 96
    qkv_out = clips * layers * 4 * tokens * channels**2
    qkav = clips * layers * 2 * tokens**2 * channels
    mlp = clips * layers * 8 * tokens * channels**2
    tia = clips * layers * tokens * (2 * channels * hidden + hidden**2 + 3 * hidden)
    patch = clips * tokens * channels * (3 * 2 * 16**2)
    backbone = qkv_out + qkav + mlp + tia + patch
    other = total - backbone
    assert mlp == 543581798400
    assert qkv_out + qkav == 554906419200
    assert qkav == 283115520000
    assert tia == 38353305600
    assert backbone == 1159490764800
    r = 64 / 100
    res128 = ((backbone - qkav) * r + qkav * r * r + other) / total
    depth8_base = (patch + (backbone-patch) * 8 / 12 + other) / total
    # ExitHead in the audited code: Conv C->192, DWConv3, Conv192->C,
    # applied to 384 native times. Its elementwise work is excluded.
    exit_head = 384 * (2 * channels * 192 + 3 * 192)
    depth8_with_exit = depth8_base + exit_head / total
    surrogate = 4 * clips * tokens * (2 * channels * 32 + channels)
    cost = dict(dense_reported_macs=total, derived_patch_macs=patch,
                derived_qkv_and_output_macs=qkv_out, derived_qkav_macs=qkav,
                derived_mlp_macs=mlp, derived_tia_macs=tia,
                derived_backbone_macs=backbone, residual_detector_macs=other,
                tia_fraction=tia/total,
                late4_mlp_keep75_ideal_saving=mlp/3*.25/total,
                late4_mlp_keep50_ideal_saving=mlp/3*.5/total,
                late4_surrogate32_plus_score_fraction=surrogate/total,
                late4_keep75_net_saving_before_other_ops=(mlp/3*.25-surrogate)/total,
                late4_keep50_net_saving_before_other_ops=(mlp/3*.5-surrogate)/total,
                all_time_128_fraction=res128,
                depth8_fraction_without_exit=depth8_base,
                depth8_fraction_with_exit=depth8_with_exit,
                bf16_full_grid_S_MiB=384*100*384*2/2**20,
                bf16_native_S_MiB=384*384*2/2**20,
                bf16_native8_and12_S_MiB=2*384*384*2/2**20)
    logits = [-1.2, -.5, .1, .7, 1.6]
    grad = [1.1, -.8, .4, 1.7, -.2]
    tau, k, eps = .7, 2.3, 1e-5
    rates = calibrated(logits, k, tau)
    slopes = [x*(1-x)/tau for x in rates]
    center = sum(g*s for g,s in zip(grad,slopes)) / sum(slopes)
    analytic = [s*(g-center) for g,s in zip(grad,slopes)]
    numerical=[]
    for i in range(len(logits)):
        plus, minus = logits[:], logits[:]
        plus[i] += eps; minus[i] -= eps
        numerical.append(sum(g*(a-b)/(2*eps) for g,a,b in
                             zip(grad,calibrated(plus,k,tau),calibrated(minus,k,tau))))
    error=max(abs(a-b) for a,b in zip(analytic,numerical))
    assert error < 1e-8
    n,budget,anchor_count=48,24,6
    uniform=lambda count: sorted(set(round(i*(n-1)/(count-1)) for i in range(count)))
    anchors=uniform(anchor_count)
    zero_logits=sorted(anchors+[i for i in range(n) if i not in anchors][:budget-len(anchors)])
    uniform24=uniform(budget)
    gap=lambda ids:max(b-a-1 for a,b in zip(ids,ids[1:]))
    assert zero_logits != uniform24
    # Exact inverse mapping does not preserve interval IoU under a nonlinear warp.
    # Physical selected positions [0,5,6,7,8,9,10] map to ranks [0,1,2,3,4,5,6].
    rank_iou=dict(gt_physical=[0,10], pred_physical=[0,5], physical_iou=.5,
                  selected_positions=[0,5,6,7,8,9,10], gt_rank=[0,6],
                  pred_rank=[0,1], rank_iou=1/6,
                  inference='Counterexample to metric invariance, NOT attribution of observed TAD mAP.')
    # Unobservable event and complementarity examples are logical constructions.
    result=dict(scope=__doc__, source_commit='239d098cd899936c35989fae6243c70259a85adb',
                ema=dict(initial_parameter_mass_500=.999**500,
                         initial_parameter_mass_1500=.999**1500,
                         initial_parameter_mass_8000=.999**8000,
                         parameter_half_life_updates=math.log(.5)/math.log(.999),
                         warning='These are parameter coefficients, not output or mAP ratios.'),
                cost=cost, calibrated_continuous_rate_gradient=dict(analytic=analytic,
                finite_difference=numerical,max_abs_error=error,
                warning='Does NOT validate hard sampling or its RGB transport surrogate.'),
                zero_logits_topk=dict(anchors=anchors,selected=zero_logits,uniform=uniform24,
                                     selected_max_empty_clip_gap=gap(zero_logits),
                                     uniform_max_empty_clip_gap=gap(uniform24)),
                gradient_counterexamples=dict(stationary_teacher_gradient=0,
                       loss_increase_after_deletion=1,
                       signed_dot_cancellation=0,absolute_product_sum=2),
                nonlinear_rank_iou=rank_iou,
                complementarity=dict(utility_A_alone=0,utility_B_alone=0,utility_A_and_B=1),
                guarantee_boundary='Exact equality only for the same computable state and operator contract. '
                'Two videos with identical acquired evidence but different hidden events cannot both '
                'be recovered by a deterministic function of that evidence alone.')
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__ == '__main__':
    main()
