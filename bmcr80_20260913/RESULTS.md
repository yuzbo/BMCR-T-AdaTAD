# Corrected BMCR: 20+60 course

All200 train /211 test /792 windows; seed3407. Existing warm20 is reused once, not retrained.
EMA selected on complete test at total25..80 every5; total60 and80 are retained separately.
The80-course checkpoint at60 differs in LR history from the old60-course terminal. No DS3/clip-selection experiment is resumed.

| Backbone | Last saved total epoch | Saved joint updates | Complete full tests | Current observed peak |
|---|---:|---:|---:|---:|
|S|21|100|0/12|Not available|
|B|21|100|0/12|Not available|

| Backbone | Total epoch | Average mAP (%) | @0.3 | @0.4 | @0.5 | @0.6 | @0.7 |
|---|---:|---:|---:|---:|---:|---:|---:|

Stage status: {"COMPLETED": 4, "RUNNING": 2, "WAITING": 30}
Updated: 2026-09-13T01:30:37+0800

Training component means, loss p10/median/p90, gradient norms and LR histories are in progress_summary.json.
Per-batch logs and raw predictions remain in runs/. A single seed does not estimate training variance.
