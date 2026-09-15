# Model-forward compute ledger

All costs are mean GFLOPs per window over the complete 211-video / 792-window cohort.

The allocation figure uses selected execution cost. Marginal-CF ranking forwards (one base plus all candidates) and the dense-attention diagnostic forward are additional. A ranking is shared by the six budget points. Host-side score and sort arithmetic is not profiled.

| Model | Axis | Cheap base | CF ranking | Dense attention | Candidates |
|---|---|---:|---:|---:|---|
| S | T | 663.985 | 10320.533 | 2347.894 | [12] |
| S | D | 1652.422 | 28786.651 | 2347.894 | [16] |
| S | S | 2003.788 | 34408.495 | 2347.894 | [16] |

## Same-support recovery

Every variant includes shared support plus recovery and detector execution. The full-observation feature target is a separate extra encoder/interpolation pass that reuses the shared preview.

| Model | Shared support | Extra full-observation target |
|---|---:|---:|
| S | 1196.558 | 2318.982 |

| Model | Variant | Recovery + detector only |
|---|---|---:|
| S | Packed linear | 28.913 |
| S | Physical interpolation | 28.913 |
| S | Cross, MD disabled | 29.806 |
| S | Cross + multidepth | 29.919 |
