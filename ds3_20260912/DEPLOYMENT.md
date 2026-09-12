# DS3 deployment receipt

2026-09-12 18:36 CST: new independent controller PID2870398 is running from the owned DS3 source directory. Slurm job1287651 (`ds3-preflight`) was submitted and is PENDING(Priority) in the verified4090 pool. No DS3 optimizer step or full-test result is claimed yet.

Authoritative live state is `ds3_20260912/deployment.json` under the remote source root `/data/run01/sczc063/yuzibo/h65_clean_adatad_20260910/ds3_20260912`. Training outputs are in its nested `ds3_20260912/runs/`; the controller log and Slurm logs are in that same experiment folder. Shared source data/checkpoints/environment are linked read-only from this project's `resources/`.

Seven remoteLinuxCPU tests passed before submission. The GPU gate must complete for bothS/B before density-conversion references, Z0 and D1 are launched. A failed stage retains its jobID and log and is not automatically retried. The pilot100updates is part of the80epoch trajectory and resumes at epoch2. Every5epoch checkpoint makes a full T24A test eligible while training continues;60/80 get independent profiles, and the selected peak gets a profile and fixed-policy ablations.

The earlier completed H65 controller and its checkpoints are untouched. Do not start another DS3 controller or reuse an unrelated GPU job allocation.
