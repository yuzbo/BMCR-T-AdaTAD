# DS3-L experiment results in progress

Snapshot2026-09-12 19:27 CST. BothS/B real4090 preflights passed. The first complete reference test is available; the local-TIA conversion and learned DS3 policies have no complete result in this snapshot. The predeclared D1 budget remains80epochs8000updates per backbone, all200 training videos, seed3407, with full211-video/792-window tests every5epochs and separate60/80 reporting.

| Backbone / policy | Mean mAP (%) | Matrix/conv TFLOPs | FLOPs / corresponding official | Model mean / median / p95 (ms) | Backbone mean (ms) | Full-test elapsed (min) |
|---|---:|---:|---:|---:|---:|---:|
| S / D768G official globalTIA384 |69.012554|2.347894|100.00%|67.867 /64.772 /76.322|48.560|20.607|

D768G uses the fixed provided official checkpoint, with no new training or checkpoint selection. Its five-threshold scores exactly match the previously completed officialS evaluation:83.833805 /79.088096 /72.328912 /61.571779 /48.240175 percent at tIoU0.3/0.4/0.5/0.6/0.7. This confirms that the new evaluation entry point preserves official detection behavior.

The complete211 video-ID set and422000 nonempty post-NMS predictions were verified. All three fixed profile cases contain12 actual attention calls,48 physical clips/layer and38400 heavy-MLP tokens/layer; fused QK/AV products are counted and no matrix operation is unresolved. MACs are1173.947019264G; the reported FLOPs use2MAC for matrix/convolution arithmetic, not a count of every elementwise operation.

The primary profile uses the first full768-candidate test window. GPU model timing starts with raw input already onGPU and includes this DS3 evaluation runtime's preprocessing, feature assembly, route metadata and detector, excluding decoding/NMS. The backbone measurement starts with normalized clips and the route prepared, excluding preview/routing, native reconstruction and detector. All20 timing samples are retained. Full-test elapsed includes data loading/decoding, CPU transforms, transfer, inference and window/video NMS, but excludes AP calculation and prediction-file writing; filesystem cache state is uncontrolled. These scopes are distinct. New runtime/node timings must not be directly substituted for the older official51.21ms measurement when computing a speedup.

The raw metrics, profiles, completed receipt and verification are in[evaluations/s_D768G](evaluations/s_D768G). Full predictions remain in the owned remote `ds3_20260912/runs/s_D768G/result_detection.json`. The current local-TIA8 test measures the conversion effect separately; no accuracy-retention or acceleration conclusion about DS3 is supported yet.
