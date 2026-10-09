# Resident attention projection training: correctness only

This extends the [attention VJP](../2026-10-09-resident-attention-vjp/README.md)
through fused QKV and output projections and a single parameter owner. It is not
a speed benchmark, complete LLM training run, geometry-quality ablation, public
Python/JS training binding, or evidence about the published 0.4.29 wheel.

## Frozen experiment

- Independent oracle: unmodified CPU-f32 PyTorch 2.12.1, one thread, separate
  Q/K/V/output projections and autograd. Optional global Python startup patches
  were disabled (`-I -S` with only the installed framework site-packages appended).
- Three input/head/output shapes: `[2,3,4] / H2 D2 / O5`,
  `[1,4,3] / H3 D3 / O4`, and `[1,1,3] / H1 D2 / O2`.
- Both causal and noncausal masks, each with no bias, Z, pair, both, and both-zero
  biases. Inputs, weights and upstream are matched across the five bias modes.
- All 30 conditions compare prediction, logical input gradient, all four fused
  parameter gradients, and exactly the optional geometry gradients requested.
- A separate 16-step MSE/plain-SGD run uses both fixed geometry biases, rate 0.03,
  seed 1737 and target seed 901. All updates execute before any loss, parameter,
  activation, gradient or update-receipt readback from that loop.
- Fixed admission: `abs(error) <= 3e-6 + 5e-5 * abs(reference)`, finite values,
  exact shapes/counts and accepted update revisions. No tolerance was widened.

The source fixture is
`crates/st-nn/tests/fixtures/resident_attention_training_torch.json`, SHA-256
`87d490a6f45664d86d4f1ebd081a8d77e3f57c59ddbf6027c07ef9e84bbcc58c`.
A fresh isolated regeneration compared byte-for-byte equal; the generator refuses
to overwrite an existing fixture.

## Observed results

| Observation | Native Metal / Apple M4 | Browser WebGPU |
| --- | ---: | ---: |
| Matched forward/VJP conditions | 30 / 30 | 30 / 30 |
| Accepted contiguous SGD updates | 16 / 16 | 16 / 16 |
| Maximum prediction absolute error | 2.9802322387695312e-8 | 2.9802322387695312e-8 |
| Maximum parameter-gradient absolute error | 2.9802322387695312e-8 | 2.9802322387695312e-8 |
| Maximum input-gradient absolute error | 1.4901161193847656e-8 | 1.4901161193847656e-8 |
| Maximum final-parameter absolute error | 1.862645149230957e-9 | 1.862645149230957e-9 |
| Maximum final-prediction absolute error | 7.450580596923828e-9 | 7.450580596923828e-9 |
| Rejected update preserves all parameters | yes | yes |
| Rejected update invalidates old gradients | yes | yes |

The browser reports `BrowserWebGpu` without a hardware model, so it is not labeled
as a separately identified M4 adapter. Native uses a release build; the standalone
WASM probe uses a dev build. No elapsed time is interpreted as performance.
The trajectory's first/last **pre-update** MSE is approximately
`0.16081056 -> 0.12581828`; these are synthetic projection-learning observations,
not language-loss or heldout-quality measurements.

`native.json` and `browser.json` retain every condition and update, while
`validation.json` records artifact/source hashes, build commands and checks.
Raw logs and generated binaries are kept locally, not committed; the browser
JSON is the exact page download. The native JSON is extracted from test stdout.

## Regression boundary and replay

Native unit tests separately check finite gradients whose candidate overflows
only QKV or only output: the other projection would change, but all four tensors
remain bitwise unchanged. Distinct cotangents preserve earlier gradients, which
can still be successfully applied. Tests also cover stale/foreign tokens,
zero-rate validation and host-plan/snapshot immutability. Concatenation covers
strides, broadcasts, offsets, aliases, last-axis multi-workgroup dispatch and
failures hidden in empty/cropped views, including empty outer dimensions.

Independent read-only review found two false-pass risks in the probe (not in the
runtime implementation): truncated reference arrays could shorten `zip`
comparisons, and unrelated errors could count as expected rejection. The final
probe requires exact reference/result counts and exact `Rejected` /
`ParameterVersion` error variants. Fourteen missing/extra-array mutations fail
the fixture preflight. Both native and browser results were rerun after these
checks were strengthened; fixture bytes and numerical tolerances did not change.

The focused strict backend Clippy and workspace format checks pass. A separately
attempted strict `st-nn` Clippy check with 1.99.0 remains red on 23 diagnostics in
unchanged files; the non-denying diagnostic pass reports none in the new files.
These existing warnings were not suppressed or mixed into this feature. The
checkout's older default Clippy 1.97.0 additionally does not recognize an existing
backend allowance. Runtime validation used Rust 1.97.0, not the routing directory's
toolchain. The manifest keeps these validation scopes distinct.

See [the API and exact replay commands](../../../docs/resident_attention_projection_training.md).
Native runtime tests require the real-GPU environment opt-in; default test
success alone does not prove runtime execution. CI runs native regressions and
checks the WASM build. The actual browser replay recorded here remains distinct
from CI compilation and from the existing public attention-primitive wrappers.
