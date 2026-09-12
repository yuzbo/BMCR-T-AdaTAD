# DS3 deployment receipt

2026-09-12 18:46 CST: S/B realGPU preflight job1287657 completed successfully. Each backbone ran two batch2 auxiliary optimizer updates, preserved every teacher/detector parameter and buffer, and passed strict EMA reload, local full/compact equivalence and finite sparse-deployment checks. These four preflight updates are separate from formal training; no DS3 full-test accuracy result is claimed yet. Current independent controller PID3000068 is proceeding to the density-conversion references and Z0 before the80epoch trajectory.

Authoritative live state is `ds3_20260912/deployment.json` under the remote source root `/data/run01/sczc063/yuzibo/h65_clean_adatad_20260910/ds3_20260912`. Training outputs are in its nested `ds3_20260912/runs/`; the controller log and Slurm logs are in that same experiment folder. Shared source data/checkpoints/environment are linked read-only from this project's `resources/`.

Seven remoteLinuxCPU tests passed before submission; actual decoding/data construction additionally confirmed200 disjoint training IDs,211 test IDs,792 windows,100 batch2 updates/epoch and the source-frame metadata interface. RealGPU receipts are in `validation/gpu_preflight_s.json` and `validation/gpu_preflight_b.json`.

Two diagnosed launcher failures are preserved:1287651 enabled `set -u` before the site's profile had defined XDG_DATA_DIRS;1287655 inferred the project directory from Slurm's spool copy of the wrapper. Neither entered model execution. The launcher now loads the site environment before strict variable checks and uses SLURM_SUBMIT_DIR. Only the owned controller was stopped/restarted for these corrections; attempts and explanations remain in the authoritative receipt.

A failed stage retains its jobID and log and is not automatically retried. The pilot100updates is part of the80epoch trajectory and resumes at epoch2. Every5epoch checkpoint makes a full T24A test eligible while training continues;60/80 get independent profiles, and the selected peak gets a profile and fixed-policy ablations. A thread follow-up checks every30minutes and remains quiet when no actionable state has changed.

The earlier completed H65 controller and its checkpoints are untouched. Do not start another DS3 controller or reuse an unrelated GPU job allocation.
