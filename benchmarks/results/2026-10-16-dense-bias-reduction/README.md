# Exact Dense Bias Reduction And Fixed-State Language Probes

This is a numerical repair, **not a new language-quality result**. It follows
the [failed long-run Attention comparison](../2026-10-16-attention-shift-invariance/README.md)
without changing its gate, retraining, selecting a new seed or promoting a model.
Base: `ec19510e0a0d1764bbed2a37f38bdd9f4a2a52ea`.

## Reproduced Defects And Repair

The shared dense/QKV bias VJP previously summed rows naively in float32.
`[1, 2^-25, -1]` returned zero instead of `2^-25`; `[2^24, 1, -2^24]`
returned zero instead of one. The initial four-component compensated prototype
fixed these, but independent review identified a finite overflow boundary:
`[f32::MAX, 2^102, 2^102, -2^77]` incorrectly failed during collapse.
Its correctly rounded sum is still `f32::MAX`.

`st_kernel_contracts::f32_sum` now owns a fixed 320-bit signed integer sum in
units of `2^-149`, with one final round-to-nearest-even conversion. Every finite
float32 term is represented exactly; at most `u32::MAX` terms fit. Rust provides
the checked reference and the shared WGSL source. Dense training and graph
autograd consume it without host readbacks or a separate Python implementation.
The old sequential-prefix overflow rejection is retained separately, including
`[MAX, MAX, -MAX]`. Nonfinite inputs, an overflowing final result and inherited
failures remain rejected. Exact zero is canonical positive zero.

An independent Python-integer fixture then exposed a second issue on Metal:
the non-GELU VJP's unnecessary multiplication by one flushed subnormal seeds
before reduction. It is now a checked copy; GELU's multiplication is unchanged.
This does not make every GPU operation subnormal-preserving.

CPU checks cover 92 signed exact-oracle cases. Both
[native Metal](native.json) and [Chrome WebGPU](browser.json) execute all 84
finite cases, eight explicit cancellation/boundary cases, and five typed
rejection/recovery cases. Browser physical hardware execution is unverified.
All pass; the original failures and rejected prototype remain retained privately.

## Same Saved Language Models

The [pre-execution plan](PLAN.md) pins the existing width16/seed11 request and
1024-update checkpoint. At each ordinary, learned-Poincare and frozen-Poincare
saved state, we execute one backward on training-batch indices 0,127,511,1023.
No parameters, moments or revision change. This is **12 fixed-state probes**,
not 12 independent training runs. No new corpus or generation is included.

The independent Torch 2.12.1 CPU controls use float32 parameter bytes and either
float32 or float64 arithmetic. We retain the original float32 residual association
as a separate control and use module-matched association for the primary
float32 control. Float64 holds the declared float32 scalar controls fixed; it is
an accuracy diagnostic, not a replacement for the frozen language oracle.

Maximum absolute Key-bias gradient across the twelve probes:

| Control | Largest residual |
| --- | ---: |
| Native before | 1.058652e-9 |
| Native exact sum | 6.293703e-10 |
| Torch float32, legacy association | 4.190952e-9 |
| Torch float32, module association | 3.259629e-9 |
| Torch float64, module association | 8.328028e-18 |

All twelve native Key residual maxima decrease. Against float64, however, of
256 parameter/probe L2 errors **51 improve, 188 are identical, and 17 worsen**.
The exact sum and compensated prototype produce the same language probe output;
the exact method additionally fixes the adversarial boundary. Every parameter
and all five controls remain in [comparison.json](comparison.json).
This is not uniform full-model agreement or evidence that geometry improves
language quality. The previous failed long-run studies remain failed.

One preliminary exact build accidentally included default/selfsup features.
It is retained, but excluded from the matched comparison. The final baseline
and exact executables both use only `wgpu` and the same debug profile; their
output bytes are checked against the archived reports. Intermediate variants,
hashes and equality checks are listed in [summary.json](summary.json).

The [bounded cost diagnostic](timing.json) includes one warmup and four measured
processes per variant in alternating AB/BA order. Each restores three models,
executes twelve backwards, and includes startup, shader compilation, JSON and
readbacks. Median process time is 2.180s before versus 2.259s exact (1.036x).
This is not a kernel-throughput measurement, a PyTorch speed comparison or a
general performance bound. Register/spill costs remain unmeasured.

## Reproduce And Limits

```sh
python3 -I -S tools/generate_f32_sum_fixture.py > /tmp/f32_sum.txt
cmp /tmp/f32_sum.txt crates/st-kernel-contracts/tests/fixtures/f32_sum.txt
cargo +1.99.0 test --locked -p st-kernel-contracts
SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS=1 cargo +1.99.0 test --locked \
  -p st-backend-wgpu --test resident_bias_reduction -- --nocapture
SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS=1 cargo +1.99.0 test --locked \
  -p st-backend-wgpu --lib resident_training -- --test-threads=1
SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS=1 cargo +1.99.0 test --locked \
  -p st-nn --no-default-features --features wgpu --lib resident::byte_ \
  -- --test-threads=1
cargo +1.99.0 build --locked -p st-nn --no-default-features --features wgpu \
  --example resident_byte_gradient_probe --message-format json
target/debug/examples/resident_byte_gradient_probe "$REQUEST" "$CHECKPOINT"
python3 -I -B benchmarks/results/2026-10-16-dense-bias-reduction/test_probe.py
```

`probe.py reference` takes `--repo`, `--request`, `--saved` (the prior native
report) and a new `--output`. `probe.py compare` takes the same request/saved
inputs, `--reference`, `--before`, `--after` and a new `--output`. Inputs must
match the plan's pins. Comparison validates case/batch/parameter coverage,
finite gradients/losses, exact parameter bits and checkpoint immutability;
hashes bind the same raw snapshots that were parsed. Five tests include drift
rejection and the float64 reference's Key-shift invariant.

For browser checks build `resident_bias_reduction_browser` in `st-backend-wgpu`
for `wasm32-unknown-unknown`, then generate web assets with wasm-bindgen 0.2.129
in `target/resident-bias-reduction-web`. Serve the test page and those assets
over loopback, preserving relative paths; download its result. Complete failed
reports remain displayed and downloadable. A WASM build alone is not a test.

132 contract tests, 52 backend training tests, 120 byte-model/study tests,
the shared integration and five Python tests pass. These counts overlap in
behavior and are not independent samples. Native/backend and WASM-example
strict Clippy pass. Whole `st-nn` strict Clippy remains blocked by 24 diagnostics
in unchanged files; no lint exemptions were added. Rust formatting passes.

The [verification record](verification.json), [source hashes](source-hashes.json)
and [private file hashes](private-hashes.json) retain both successes and failed
attempts. Raw corpus, weights, moments, full gradients, binaries and logs remain
private. Hashes identify bytes, not independent execution attestation. Python
uses the shared backend after rebuilding; no wheel or full binding rerun is
claimed here. Checkpoints remain structurally readable, not bitwise continuous
across changed kernels. Parent PR #2270 still has an unstarted billing-lock CI
failure; this slice is not merged on local evidence alone.
