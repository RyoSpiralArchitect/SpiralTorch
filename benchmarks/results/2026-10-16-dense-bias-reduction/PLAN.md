# Dense Bias Reduction Isolation

Base: ec19510e0a0d1764bbed2a37f38bdd9f4a2a52ea. Before executing the new probes:

- Reproduce plain float32 bias-sum cancellation without Attention or AdamW.
- Reuse the exact width16/seed11 request and completed checkpoint from the
  Attention-conservation follow-up. Keep ordinary, learned and frozen Poincare.
- Probe training-batch indices 0, 127, 511 and 1023 at the same saved state in
  every case. No update, retry-based selection, new data or generation.
- Compare the unchanged native model before/after any reduction repair to
  independent CPU Torch float32 and float64 gradients on identical float32
  parameter bytes. Keep the original residual association as a separate
  float32 control; use the module association for the matched-program control.
- Float64 is an accuracy diagnostic, not a replacement for the frozen language
  oracle or a new gate for the previous failed studies.
- Retain all parameters and cases, including any worsening. Native checkpoint
  values, optimizer state and revision must remain unchanged by each probe.
- Exercise the shared reduction in native and browser clients, including
  existing nonfinite/overflow guards. Measure cost separately from accuracy.

Pinned SHA-256:

| Input | Hash |
| --- | --- |
| request | b0a612eec9aeb2f890f1d37c7f4edc50acedd5d898c456cadc3c1c717849d29c |
| saved study checkpoint | 9e8ae028c81705e05935e70ba8386e9db602ebfa4603d78ce874670f7b6c491c |
| saved native report | 858a5576ba95c643fc223044fe6eaab4f502fa9afd0fc15b744eaa7bdb0b3dfe |

Raw model values, gradients and documents remain private. Publish derived
numerical results, source/build identities and reproduction instructions.
No full-language retraining or serving promotion is included in this isolation.
