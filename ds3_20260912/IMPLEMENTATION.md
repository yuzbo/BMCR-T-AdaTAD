# DS3-L implementation and experiment entry points

This branch implements D1 dense-forward auxiliary distillation and Z0/conditional deployment on pristine OpenTAD. It is a new local-TIA8 variant, not a claim that the provided global-TIA384 official checkpoint is functionally unchanged. The parameter-identical local/global dense comparison quantifies this architectural conversion separately.

- `h65/ds3/model.py`: frozen teacher and original detector interface; compact 0/8/12 clip execution; late-block sparse heavy MLP with full-raster TIA.
- `h65/ds3/auxiliary.py`: low-resolution ImageNet MobileNetV3Small preview, tubelet-preserving exit8, low-rank MLP surrogates and spatial scores.
- `h65/ds3/losses.py`: complete dense teacher targets, legal-boundary feature losses and gradient-weighted residual routing proxy. The teacher detector is frozen while gradients with respect to its input features remain available.
- `tools/ds3_preflight.py`: real batch2 S/B checks, two auxiliary optimizer updates, direct frozen tensor equality, strict EMA reload, compact path identity and finite deployment outputs.
- `tools/ds3_train.py`: one seed3407 / all200 training videos / batch2 /80epochs. The100-step pilot is epoch1 of the same8000-update schedule. Only auxiliary modules are optimized.
- `tools/ds3_eval.py`: all211 test videos /792 windows; original coordinates, detector and NMS; matrix/conv FLOPs from actual executed operators including fused attention QK/AV. Full/partial/short fixed-window profiles and complete dataset timing are separate scopes.
- `tools/ds3_dispatch.py`: independent Slurm controller; at most2 own train/preflight plus1 test; no cancellation of other projects or automatic retries of failed jobs.

Policies: D768G original global dense; D768L converted local dense; Z16/24/36 complete-clip uniform inference with physical-center interpolation; ZR24 seeded random counterpart; PONLY preview; T24U/T24A fixed24 uniform/adaptive clips; D8 allclips exit8; DAD24 clips through8layers/12 through12layers; S75/S50 late heavy MLP fractions. F(T,S,D) evaluates fixed-budget factor combinations. F100=T24A, F010=S75 and F101=DAD reuse identical policy evaluations; no redundant training is introduced. The inference-only uniform controls are not separate uniform training experiments.

T24A is fully tested at epochs5,10,...,80. Select the highest mean AP (earliest tie), retain60 and80 separately, and use this same selected auxiliary EMA for all final policies. This intentionally follows the user's test-peak selection request; it is not an untouched-test estimate. C2 sparsity-aware calibration and J3 joint sparse training are not part of this implemented D1 regime.

Remote Linux CPU validation:7 focused tests pass, including actual upstream tiny-ViT clip equivalence, measured sparse FFN execution, physical-coordinate interpolation, legal boundary weighting, frozen-detector input gradients and nonzero LR at60/zero at80. Real4090 S/B preflight job1287657 also passed:2 auxiliary optimizer updates each, strict EMA reload, unchanged teacher parameters/buffers, compact equivalence and finite sparse outputs. These are implementation checks, not a full-test performance result.

Deployment from this repository uses `tools/ds3_site.py` to create own-directory symlinks to read-only resources, then `tools/ds3_dispatch.py`. Site-specific shell/environment paths are in `ds3_20260912/site/run_job.sh`. Full checkpoints and videos stay in the owned remote experiment directory and are excluded from Git.
