# Retained Wide Rounding

This is a shared backend numerical repair supporting geometric language training,
not a new geometry policy or evidence of better language quality. The same WGSL
converter serves native WGPU and browser WebGPU Attention, LayerNorm and geometric
reductions. Python uses that Rust backend rather than reproducing its arithmetic.

## Local Numerical Evidence

| Check | Before | After |
| --- | ---: | ---: |
| Synthetic retained-component conversion, Metal | 582 / 7,042 mismatches | 0 / 7,042 |
| Actual Attention midpoint cases, Metal | 4 / 12 failures | 0 / 12 |
| Synthetic conversion, browser WebGPU | not rerun | 0 / 7,042 |
| Actual Attention midpoint cases, browser WebGPU | not rerun | 0 / 12 |

The repair uses retained low/tail bits to distinguish a midpoint from either
side, preserves even-tie rounding, and handles the smaller gap below powers of
two, normal/subnormal transitions and the finite/overflow midpoint. It avoids
flushing the deciding tail through multiplication and guards saturated extended
exponents before bias addition. It does not make all preceding `Wide` arithmetic
exact; see [independent review](REVIEW.md) for a normalization-loss witness.

The 7,042 synthetic triples test conversion directly, including noncanonical
triples; they are not 7,042 full-model scenarios. The twelve Attention cases
separately exercise reachable production computations. The oracle sums integer
representations and rounds once, without an intermediate float64 rounding.

Additional local validation:

- Native Apple M4 / Metal: 144 resident-tensor tests passed; one explicit benchmark
  remained ignored. Attention VJP, Poincare bias and causal Z-Space wave integration
  suites passed (four top-level tests).
- Rebuilt browser Attention report: 36 reference checks, 16 SGD updates, 13 shift
  cases and 46 range cases passed. Physical browser hardware execution remains
  **unverified**; native adapter identity does not establish browser identity.
- Source-only tests: four integer-oracle tests, three evidence-publication tests,
  fourteen durable Python segment tests and three recovery-qualifier tests passed.

Complete synthetic before/after outputs and bounded language diagnostics are in
`results/`. Raw language requests, model parameters, optimizer states, binaries
and complete model outputs remain in local validation storage. `sources.json`
records the long-run build-time source identities; `receipts.json` identifies the
retained logs and rebuilt WASM modules. Hashes establish identity, not attestation.

## Long Learning Result

The fixed seed11 / width16 study completed 1,024 updates in each of three arms
in 287.96 seconds (debug profile, not a speed claim). The original strict gate
still **fails**. The complete final language report has the same SHA-256 as the
immediately preceding range-repair run:
`ddc47f8bcd053391daf265cb1a3f914bdc78bceeb8c4ea4371792772ef80f280`.

| Arm | Maximum parameter error vs Torch | Failed weight components |
| --- | ---: | ---: |
| Ordinary | 4.0636543e-6 | 3 |
| Learned Poincare | 4.8952114e-6 | 4 |
| Frozen Poincare | 4.5808555e-6 | 3 |

All ten failed weight components are in the Key slice of `block.0.qkv.bias`.
Maximum training and validation loss error is 9.5367432e-7 in every arm.
These figures are unchanged, not an improvement from final-rounding repair.
The synthetic defect is real but does not explain this long-run discrepancy.
No gradients were masked, and the original gate was not loosened or replaced.

## Reproduce

From the repository root, using Rust 1.99.0:

```sh
python3 -I -S -B tools/test_generate_wide_rounding_fixture.py
python3 -I -S -B benchmarks/results/2026-10-18-wide-rounding/test_publish.py
SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS=1 cargo +1.99.0 test --locked \
  -p st-backend-wgpu --test resident_wide_rounding -- --test-threads=1
cargo +1.99.0 run --locked -p st-backend-wgpu --example resident_wide_rounding
cargo +1.99.0 build --locked -p st-backend-wgpu \
  --target wasm32-unknown-unknown --release \
  --example resident_wide_rounding_browser --example resident_attention_vjp_browser
```

Use wasm-bindgen 0.2.129 (`--target web`) on those two example modules. The
`wide_rounding_browser.html` test imports the generated files from
`target/wide-rounding-web/`; the existing `attention_vjp_browser.html` imports
`target/resident-attention-vjp-web/`. Serve an isolated directory containing only
those pages and generated modules with `scripts/serve_wasm_demo.py`, not the
repository or private experiment directories. Both pages offer result downloads.

The longer learning follow-up deliberately reuses the pinned request/reference
and original strict comparator from the preceding study:

```sh
python3 -I -B benchmarks/results/2026-10-18-language-vjp-isolation/learning.py \
  --repo . --prior "$VALIDATION/2026-10-16-attention-shift-invariance" \
  --reference "$VALIDATION/2026-10-16-language-capacity-budget" \
  --output "$VALIDATION/2026-10-18-wide-rounding/learning"
python3 -I -S -B benchmarks/results/2026-10-18-wide-rounding/publish.py \
  --repo . --validation "$VALIDATION/2026-10-18-wide-rounding" \
  --previous "$VALIDATION/2026-10-18-language-vjp-isolation/learning-reviewed" \
  --reference "$VALIDATION/2026-10-16-language-capacity-budget/w16-seed11/torch.json" \
  --output benchmarks/results/2026-10-18-wide-rounding/results
```

`VALIDATION` is the local validation root. Output directories must be new. The
runner snapshots source files and checks for drift before and after execution;
do not edit guarded Rust/WGSL files during the run. Publication recomputes the
strict comparison and pins the immediately preceding result, rather than choosing
an older, more favorable comparison. This is one seed and debug-profile numerical
evidence, not a throughput benchmark or a model promotion decision.
