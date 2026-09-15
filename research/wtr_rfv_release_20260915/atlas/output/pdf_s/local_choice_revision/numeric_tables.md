# Exact values for the revised paper figures

User-requested post-hoc description of the existing frozen publication measurements. All intervals use 10,000 video-cluster resamples; the original analysis is retained.

## AdaTAD-S: benign reference

epsilon_benign,95 = 0.0015930387; 95% CI [0.0011344262, 0.0028151936].
This is an empirical RGB-perturbation scale, not pure numerical noise or a significance threshold. The threshold is re-estimated within each bootstrap draw. Percentages below use strict greater-than.

| Measured change | Above reference (%) [95% CI] | Videos | Samples |
|---|---:|---:|---:|
| Observation replacement | 26.1673 [19.1759, 31.8541] | 211 | 12672 |
| Layer-local refinement | 0.2062 [0.0666, 0.4717] | 211 | 12672 |
| Spatial FFN choice | 0.2568 [0.0385, 0.5172] | 211 | 12672 |
| Frame exchange | 30.5284 [21.8541, 37.5161] | 187 | 12284 |
| Benign control | 4.8846 [3.8342, 4.9950] | 211 | 792 |
| No-op control | 0.0000 [0.0000, 0.0000] | 211 | 792 |

## AdaTAD-S: positive effect captured

| Measured change | Top 5% [95% CI] | Top 10% [95% CI] | Top 20% [95% CI] |
|---|---:|---:|---:|
| Observation replacement | 75.269% [71.972, 78.136] | 87.741% [85.822, 89.368] | 96.581% [95.918, 97.150] |
| Layer-local refinement | 83.599% [68.162, 91.434] | 91.447% [83.130, 95.626] | 97.396% [94.735, 98.696] |
| Spatial FFN choice | 80.787% [67.824, 88.955] | 89.944% [82.828, 94.365] | 96.877% [94.523, 98.279] |
| Frame exchange | 66.359% [61.809, 70.723] | 81.669% [78.728, 84.417] | 94.457% [93.300, 95.446] |
| Benign control | 63.360% [52.730, 71.053] | 79.008% [71.293, 84.810] | 93.412% [90.132, 95.710] |
| No-op control | N/A: no positive mass | N/A: no positive mass | N/A: no positive mass |

## AdaTAD-S: within-axis interaction

Mean joint-minus-sum effects are normalized by the same window dense loss. Failure to reject zero interaction is not proof of additivity.

| Axis / order / size | Mean [95% video CI] | Videos |
|---|---:|---:|
| D:stratified:1 | 0 [0, 0] | 211 |
| D:stratified:2 | -2.289557e-07 [-1.260405e-06, 7.141582e-07] | 211 |
| D:stratified:4 | -1.636667e-06 [-4.834332e-06, 2.7321e-07] | 211 |
| D:stratified:8 | -2.386984e-06 [-6.274985e-06, 5.995368e-08] | 211 |
| D:stratified:16 | -2.560009e-06 [-6.490993e-06, 8.404576e-08] | 211 |
| D:low_single_effect:1 | 0 [0, 0] | 211 |
| D:low_single_effect:2 | -5.553037e-08 [-1.333444e-07, 1.205878e-08] | 211 |
| D:low_single_effect:4 | -6.512073e-09 [-1.321472e-07, 1.265119e-07] | 211 |
| D:low_single_effect:8 | -3.533917e-07 [-7.469696e-07, -3.088412e-08] | 211 |
| D:low_single_effect:16 | -2.560009e-06 [-6.490993e-06, 8.404576e-08] | 211 |
| S:stratified:1 | 0 [0, 0] | 211 |
| S:stratified:2 | 1.372443e-07 [-4.486306e-07, 1.008249e-06] | 211 |
| S:stratified:4 | -8.209472e-08 [-6.946214e-07, 6.625207e-07] | 211 |
| S:stratified:8 | -6.446632e-07 [-2.327819e-06, 6.790253e-07] | 211 |
| S:stratified:16 | -4.740829e-08 [-1.953988e-06, 1.61746e-06] | 211 |
| S:low_single_effect:1 | 0 [0, 0] | 211 |
| S:low_single_effect:2 | 2.112027e-08 [-1.798072e-08, 6.045653e-08] | 211 |
| S:low_single_effect:4 | 4.070496e-09 [-8.176945e-08, 9.354067e-08] | 211 |
| S:low_single_effect:8 | 3.280156e-08 [-1.686097e-07, 2.403904e-07] | 211 |
| S:low_single_effect:16 | -4.740829e-08 [-1.953988e-06, 1.61746e-06] | 211 |
| O:stratified:1 | 0 [0, 0] | 211 |
| O:stratified:2 | -1.552978e-05 [-4.935388e-05, 1.196479e-05] | 211 |
| O:stratified:4 | 2.411938e-05 [-8.098756e-05, 0.0001277422] | 211 |
| O:stratified:8 | 0.0001953218 [-0.0001091706, 0.0005796218] | 211 |
| O:stratified:16 | -5.781122e-05 [-0.0006254712, 0.000505028] | 211 |
| O:low_single_effect:1 | 0 [0, 0] | 211 |
| O:low_single_effect:2 | 1.296595e-06 [-1.161228e-07, 3.906266e-06] | 211 |
| O:low_single_effect:4 | -1.287487e-06 [-6.785729e-06, 2.873309e-06] | 211 |
| O:low_single_effect:8 | -2.178203e-05 [-6.765087e-05, 5.052463e-06] | 211 |
| O:low_single_effect:16 | -5.781122e-05 [-0.0006254712, 0.000505028] | 211 |
| T:stratified:1 | 0 [0, 0] | 187 |
| T:stratified:2 | -2.36593e-06 [-1.09914e-05, 6.237247e-06] | 187 |
| T:stratified:4 | 9.65339e-06 [-2.88012e-05, 6.406695e-05] | 187 |
| T:stratified:8 | 3.150681e-05 [-1.901538e-05, 9.123001e-05] | 187 |
| T:stratified:16 | 4.197653e-05 [-3.908491e-05, 0.0001287421] | 186 |
| T:high_single_value:1 | 0 [0, 0] | 187 |
| T:high_single_value:2 | -1.225101e-05 [-2.833214e-05, 2.054521e-07] | 187 |
| T:high_single_value:4 | -6.579686e-05 [-0.0001442787, -1.287752e-05] | 187 |
| T:high_single_value:8 | -8.983634e-05 [-0.0001990356, -1.811178e-05] | 187 |
| T:high_single_value:16 | 4.197653e-05 [-3.908491e-05, 0.0001287421] | 186 |

## AdaTAD-S: layer-local choice ranking

First-order top-k marginal-value ranking diagnostic. Combined proxy selections are not re-executed. No STOP; not RFV single-swap-plus-STOP regret or detection AP regret.

Sum of the k largest signed measured single-point values minus the sum selected by a proxy; task-loss units
Raw task-loss regret is the primary regret display. Relative regret divides each window result by its dense total loss.

| Metric | Ranking | Mean [95% video CI] | Videos |
|---|---|---:|---:|
| spearman | random | 0 [0, 0] | 211 |
| spearman | actual | 1 [1, 1] | 211 |
| spearman | attention_score | -0.00233091 [-0.027027, 0.0227466] | 211 |
| spearman | actionness | 0.00549892 [-0.0205886, 0.0324019] | 211 |
| spearman | entropy | 0.00227452 [-0.0233587, 0.028627] | 211 |
| spearman | feature_norm | 0.0133352 [-0.0122727, 0.0387395] | 211 |
| ndcg | random | 0.221688 [0.217673, 0.22584] | 211 |
| ndcg | actual | 1 [1, 1] | 211 |
| ndcg | attention_score | 0.291435 [0.265741, 0.318346] | 211 |
| ndcg | actionness | 0.375043 [0.344574, 0.405799] | 211 |
| ndcg | entropy | 0.375961 [0.346349, 0.406782] | 211 |
| ndcg | feature_norm | 0.213877 [0.187318, 0.241574] | 211 |
| regret | attention_score | 0.000216261 [0.000144259, 0.000339692] | 211 |
| regret | actionness | 0.000215764 [0.000147667, 0.000332989] | 211 |
| regret | entropy | 0.000225716 [0.000151488, 0.000348346] | 211 |
| regret | feature_norm | 0.000226692 [0.0001567, 0.000346263] | 211 |
| regret | random | 0.00023408 [0.000166447, 0.000332941] | 211 |
| regret | actual | 0 [0, 0] | 211 |
| relative_regret | attention_score | 0.000668372 [0.00029605, 0.00136651] | 211 |
| relative_regret | actionness | 0.000647784 [0.000290962, 0.00132766] | 211 |
| relative_regret | entropy | 0.000665372 [0.000296178, 0.00135958] | 211 |
| relative_regret | feature_norm | 0.00067076 [0.000311799, 0.00135169] | 211 |
| relative_regret | random | 0.000627373 [0.000330638, 0.00117005] | 211 |
| relative_regret | actual | 0 [0, 0] | 211 |

## AdaTAD-S: paired proxy minus Random

Differences pair the same eligible windows and videos. Positive NDCG differences favor the proxy; negative regret differences favor the proxy. Marginal interval overlap is not a paired comparison.

| Metric | Proxy | Paired mean difference [95% CI] | Videos |
|---|---|---:|---:|
| ndcg | attention_score | 0.0697472 [0.0441846, 0.0967296] | 211 |
| ndcg | actionness | 0.153355 [0.122178, 0.184964] | 211 |
| ndcg | entropy | 0.154274 [0.124474, 0.185744] | 211 |
| ndcg | feature_norm | -0.00781052 [-0.0344755, 0.0200854] | 211 |
| regret | attention_score | -1.78191e-05 [-5.73876e-05, 2.1989e-05] | 211 |
| regret | actionness | -1.83157e-05 [-5.88366e-05, 2.14776e-05] | 211 |
| regret | entropy | -8.36331e-06 [-4.771e-05, 3.04436e-05] | 211 |
| regret | feature_norm | -7.38722e-06 [-4.59113e-05, 3.08074e-05] | 211 |
| relative_regret | attention_score | 4.09994e-05 [-7.11287e-05, 0.000225546] | 211 |
| relative_regret | actionness | 2.04115e-05 [-9.54599e-05, 0.00020782] | 211 |
| relative_regret | entropy | 3.79997e-05 [-7.61223e-05, 0.000223325] | 211 |
| relative_regret | feature_norm | 4.33878e-05 [-7.03947e-05, 0.000227162] | 211 |

Coverage: {"spearman:reference": 792, "ndcg:reference": 792, "regret:attention_score": 792, "regret:actionness": 792, "regret:entropy": 792, "regret:feature_norm": 792, "regret:random": 792, "regret:actual": 792}.

## AdaTAD-S: one local choice change

The existing benign-reference epsilon is unchanged. These are point estimates: loss increases beyond the band, within the reference band, or loss decreases beyond it. Gray is a descriptive reference band, not an equivalence test. O replacement is a separate input probe and remains in the raw-sign appendix.

| Choice / control | Loss increases (%) | Within reference (%) | Loss decreases (%) | Videos |
|---|---:|---:|---:|---:|
| Frame exchange (fixed K) | 16.148904 | 69.471571 | 14.379525 | 187 |
| One token: FFN choice | 0.096391 | 99.743158 | 0.160450 | 211 |
| One token: layer-local refinement | 0.048124 | 99.793805 | 0.158070 | 211 |
| Benign control | 3.173493 | 95.115439 | 1.711068 | 211 |
| No-op control | 0.000000 | 100.000000 | 0.000000 | 211 |
