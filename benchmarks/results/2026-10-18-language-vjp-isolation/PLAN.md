# Fixed-Operand Language VJP Isolation

Base: 6e51d6db2245b06dbd4927d79373648875a426a1. Before execution:

- Keep the previous width16/seed11, ordinary/learned/frozen Poincare saved
  models, and training batch indices 0,127,511,1023. No optimizer updates.
- Extract projection inputs and upstream gradients from the independent
  module-associated Torch float32 model at the same saved parameter bytes.
- Replay every linear projection and each Attention VJP on exactly those
  float32 operands. This is local-kernel isolation, not a native full-model
  trace or a new corpus/learning-quality result.
- Compare the three existing matrix accumulation policies, without changing
  the default or choosing a passing arm. Retain all outcomes.
- Compute independent Torch float32 and float64 local references from identical
  float32 operands. Separately round the float64 Key gradient to float32 before
  summing, to distinguish output quantization from upstream arithmetic error.
- Repair only reproduced defects. Keep finite/inherited guards, all parameter
  coordinates and the frozen long-run gate. Do not enforce Key-bias zero.
- Preserve private input tensors and full gradients locally; publish derived
  errors, hashes, controls and reproduction instructions.

Request SHA-256: b0a612eec9aeb2f890f1d37c7f4edc50acedd5d898c456cadc3c1c717849d29c.
Saved model report SHA-256: 858a5576ba95c643fc223044fe6eaab4f502fa9afd0fc15b744eaa7bdb0b3dfe.

## Full-Learning Follow-Up (Declared Before Execution)

After reproducing and repairing the Attention range witnesses, repeat all
1,024 updates in the same three width16/seed11 cases from the same initial
request. Keep the frozen Torch float32 reference and its strict tolerances.
This follow-up includes the preceding exact dense-bias reduction repair, so
it is not a causal ablation of Attention alone. Do not select a seed, change
matmul policy, suppress Key gradients, or reinterpret a failed gate as passed.
Keep parameters, corpus, complete reports and optimizer checkpoints private;
publish all case metrics, strict outcomes, source/build hashes and limitations.
Record wall time as execution cost, not an isolated kernel-speed benchmark.

The first full follow-up completed before the independent-review denominator
repair. Retain that failed strict comparison. Repeat the identical request
once on the reviewed final source, without using either outcome to select a
seed or change a gate; publish both outcomes.
