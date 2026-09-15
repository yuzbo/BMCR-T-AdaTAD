# Temporal allocation: complete S/B data

Full-data T allocation only; D/S, population and recovery are not covered by these figures.

| Model | Groups | Exec. GFLOPs | Uniform mAP | CF mAP | Delta (pp) | Paired 95% CI |
|---|---:|---:|---:|---:|---:|---|
| S | 4 | 664.0 | 45.206 | 45.206 | +0.000 | [+0.000, +0.000] |
| S | 6 | 945.0 | 56.982 | 61.267 | +4.285 | [+3.051, +5.855] |
| S | 8 | 1225.3 | 61.828 | 65.335 | +3.507 | [+2.337, +4.692] |
| S | 10 | 1506.8 | 63.915 | 66.866 | +2.951 | [+1.668, +4.119] |
| S | 12 | 1787.5 | 66.641 | 68.670 | +2.028 | [+1.035, +2.855] |
| S | 16 | 2347.9 | 68.977 | 68.977 | +0.000 | [+0.000, +0.000] |
| B | 4 | 2235.0 | 49.823 | 49.823 | +0.000 | [+0.000, +0.000] |
| B | 6 | 3210.6 | 60.819 | 62.963 | +2.144 | [+1.044, +3.511] |
| B | 8 | 4184.1 | 65.801 | 67.193 | +1.392 | [-0.058, +2.594] |
| B | 10 | 5161.7 | 67.333 | 68.638 | +1.305 | [+0.055, +2.404] |
| B | 12 | 6136.4 | 69.399 | 69.812 | +0.413 | [-0.655, +1.331] |
| B | 16 | 8082.2 | 71.142 | 71.142 | +0.000 | [+0.000, +0.000] |

## Additional ranking model forwards

Costs below are shared by the six budget points and excluded from their execution x-axes. They describe the measurement procedure, not the overhead of a trained deployable selector.

| Model | Dense attention GFLOPs/window | CF base + candidates GFLOPs/window | Candidate forwards/window |
|---|---:|---:|---:|
| S | 2347.9 | 10320.5 | [12] |
| B | 8082.2 | 34919.1 | [12] |

## temporal_budget_curves

Frozen AdaTAD-S/B on all 211 THUMOS test videos and 792 windows per model. Six budgets use 4/6/8/10/12/16 predeclared groups of actual frames. Every point uses full-dataset official AP. Shading shows pointwise 95% video-cluster intervals for Uniform and the marginal CF reference. CF is a GT-assisted reference over finite groups, not an oracle or a trained router. Attention uses dense diagnostic scores. The x-axis is selected execution cost (encoder, recovery and detection head); additional ranking forwards are in cost_ledger.json.

## temporal_paired_difference

Marginal CF minus Uniform at the six matched T execution budgets. Intervals use the same 10,000 video resamples for both strategies. These are pointwise intervals, not a simultaneous confidence band across budgets. No D/S, observation-redundancy or new-WTR performance claim follows from this slice.
