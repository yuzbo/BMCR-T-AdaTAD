# DS3-L experiment results in progress

Snapshot2026-09-12 20:02 CST. BothS/B real4090 preflights passed. S global/local dense references and zero-additional-training Z24 are complete; learned D1 policies have no complete result yet. S pilot1287766 and Z16 test1287768 are PENDING(Priority). The predeclared D1 budget remains80epochs8000updates per backbone, all200 training videos, seed3407, with full211-video/792-window tests every5epochs and separate60/80 reporting. The100-update pilot is epoch1 of this same trajectory.

| Backbone / policy | Mean mAP (%) | Matrix/conv TFLOPs | FLOPs / corresponding official | Model mean / median / p95 (ms) | Backbone mean (ms) | Full-test elapsed (min) |
|---|---:|---:|---:|---:|---:|---:|
| S / D768G official globalTIA384 |69.012554|2.347894|100.00%|67.867 /64.772 /76.322|48.560|20.607|

The other completed references are:

| Policy | Mean mAP (%) | Matrix/conv TFLOPs | FLOPs / D768G | Model mean / median (ms) | Full-test elapsed (min) |
|---|---:|---:|---:|---:|---:|
| S / D768L localTIA8 dense conversion |67.534440|2.347894|100.0000%|64.109 /64.014|20.504|
| S / Z24 complete-clip uniform selection + interpolation, no aux training |56.106503|1.188403|50.6157%|42.599 /42.561|20.494|

The fixed-weight global-to-local conversion loses1.478114 percentage points before any sparse sampling, with no FLOPs reduction. Z24 loses another11.427936pp relative to local dense, or12.906050pp versus global official. Z24 genuinely executes24clips/layer on a full window, but its lower fixed-window GPU latency comes with a large accuracy drop; complete-data elapsed is close to the dense reference and has uncontrolled cache/runtime variation. This is not an80-epoch trained D1 result or evidence that auxiliary training will necessarily recover the gap.

D768G uses the fixed provided official checkpoint, with no new training or checkpoint selection. Its five-threshold scores exactly match the previously completed officialS evaluation:83.833805 /79.088096 /72.328912 /61.571779 /48.240175 percent at tIoU0.3/0.4/0.5/0.6/0.7. This confirms that the new evaluation entry point preserves official detection behavior.

The complete211 video-ID set and422000 nonempty post-NMS predictions were verified. All three fixed profile cases contain12 actual attention calls,48 physical clips/layer and38400 heavy-MLP tokens/layer; fused QK/AV products are counted and no matrix operation is unresolved. MACs are1173.947019264G; the reported FLOPs use2MAC for matrix/convolution arithmetic, not a count of every elementwise operation.

The primary profile uses the first full768-candidate test window. GPU model timing starts with raw input already onGPU and includes this DS3 evaluation runtime's preprocessing, feature assembly, route metadata and detector, excluding decoding/NMS. The backbone measurement starts with normalized clips and the route prepared, excluding preview/routing, native reconstruction and detector. All20 timing samples are retained. Full-test elapsed includes data loading/decoding, CPU transforms, transfer, inference and window/video NMS, but excludes AP calculation and prediction-file writing; filesystem cache state is uncontrolled. These scopes are distinct. New runtime/node timings must not be directly substituted for the older official51.21ms measurement when computing a speedup.

The official raw metrics, profiles, completed receipt and verification are in[evaluations/s_D768G](evaluations/s_D768G). Full predictions remain in the corresponding owned remote `ds3_20260912/runs/` folders. The[complete retrospective](retrospective_20260912/REPORT.zh.md),[current snapshot](retrospective_20260912/remote_snapshot.json) and[local/Z24 verification](retrospective_20260912/ds3_reference_verification.json) preserve the added references. No claim of both official accuracy retention and useful end-to-end acceleration is supported.
