# Attention Shift-Conservation Repair

This follows the [shared AdamW repair](../2026-10-16-attention-adamw-numerics/README.md).
It fixes a reproduced normalization defect in the common WGPU Attention
pullback, not a Python-side correction or a special rule for Key-bias.
No coordinate is omitted and no comparison tolerance is relaxed.

## Isolated Failure And Correction

The statistics kernel previously accumulated exponential weights by online
rescaling whenever the maximum score increased. The pullback kernels recomputed
weights relative to the final maximum. Rounded float32 exponentials are not
multiplicatively consistent, so those two paths could disagree: probabilities
no longer summed to one and score cotangents no longer summed to zero.

A new GPU regression failed on the old shader at 16, 32, 64 and 257 keys.
It uses one query/head/dimension, ascending keys `i/32`, values `i/8`,
query/upstream one, and zero pair bias. Its bound allows one float32 output ULP
per component plus `2e-12` for summation; it does not assume exact floating-point
zero. All five tested sizes pass after the repair. The extracted
[before](conservation-before.txt) and [after](conservation-after.txt) records
retain every tested size and direction, including the original failure.

| 257-key direction | Expected sum | Before | After |
| --- | ---: | ---: | ---: |
| Value | 1 | 1.00000063817 | 1.00000000418 |
| Key / pair bias | 0 | -2.54263e-6 | 4.62751e-9 |

The statistics kernel now finds the final maximum in a score-only pass and
accumulates normalization and the weighted upstream/value product at that same
maximum. It preserves the score reduction tree, causal visibility, gradient
expressions and invalid-input guards. It neither subtracts a measured residual
from gradients nor zeros a parameter. No probability matrix is retained.

There is an additional score-only scan within the statistics dispatch, with no
upstream/value dot product in that scan. The three-dispatch layout is unchanged.
**Backward performance has not been measured.** The existing
forward-only timestamp test cannot establish this change's cost or speed.

## Native And Browser Coverage

The shared native/browser fixture adds 13 deterministic shift checks:

- Ascending/reversed key orders at 8, 16, 32, 64 and 257 keys.
- Batched, two-head, seven-dimensional queries with an additive pair bias.
- Causal query offsets 0 and 29 with actual Poincare pair biases and nonzero
  coordinate/raw-gain pullbacks, compared to the existing Rust reference.

Every case checks Q/K/V, per-key and pair-bias pullbacks against the existing
f64-accumulating reference. It also checks forward/pullback invariance under a
common per-head Key shift and Value/Key/bias directional sums. The existing
`3e-6 + 5e-5 * abs(reference)` comparison envelope is unchanged.
These added reference checks are Rust-based, not new independent Torch cases.

Both [native](native.json) and [browser](browser.json) clients pass, alongside
the existing 36 frozen PyTorch comparisons
(18 cases, contiguous and strided), 16 synthetic SGD updates, 12 inherited
guard cases, two range cases and three forward-reduction controls. Native
execution identifies Apple M4 / Metal. Browser execution identifies
`BrowserWebGpu`; selected-adapter physical hardware remains **unverified**.

## Language Follow-Up

The [predeclared plan](PLAN.md) fixes the same width16/seed11 request, all three
ordinary/learned/frozen Poincare cases and all 1,024 updates. The immediately
prior native debug run already contains the AdamW correction; the independent
Torch report is unchanged. No generation or promotion is part of this probe.

All three runs completed. The [strict comparison and all numerical
localization](language.json) remain public, including the failed gate:

| Case | Failed weight components before | After | Max weight error before | After |
| --- | ---: | ---: | ---: | ---: |
| Ordinary | 2 | 2 | 5.84920e-6 | 4.91948e-6 |
| Learned Poincare | 3 | 4 | 4.43087e-6 | 5.88897e-6 |
| Frozen Poincare | 3 | 3 | 4.52929e-6 | 4.71753e-6 |

All failed components remain in the Key slice of `block.0.qkv.bias`. The
diagnostics find no moment or other-weight envelope breaches in these three
cases. Training and development-loss errors are at most 9.536743e-7.
The failed component total **increased from eight to nine**. The isolated
conservation repair is not uniform full-model agreement, and this run does not
qualify any previous failed study. No additional seed was tried to find a pass.

One explicitly **post-hoc, read-only** [diagnostic](key-bias-diagnostic.json)
examines this same Key slice in all three retained reports. Its initial bias is
exactly zero. In exact arithmetic, adding a common Key vector shifts every
visible score in a query row equally and leaves softmax unchanged; hence the
data gradient in this direction is zero. Zero-initialized AdamW moments and
decay alone would keep that bias zero. The observed final maximum absolute
Key biases are:

| Case | Before | After | Independent float32 Torch |
| --- | ---: | ---: | ---: |
| Ordinary | 1.65922e-6 | 7.94616e-7 | 4.52305e-6 |
| Learned Poincare | 2.28207e-6 | 1.51789e-6 | 4.44892e-6 |
| Frozen Poincare | 1.37508e-6 | 1.40209e-6 | 4.23304e-6 |

Both implementations therefore accumulate numerical drift in an
exact-arithmetic invariant direction. The frozen case's drift grows slightly
after the repair. This diagnostic does not establish that either implementation
is generally more accurate, replace the independent oracle or change the strict
gate. The next isolation should compare Attention/projection reductions and
their AdamW amplification against an independent higher-precision control.

The runner builds the native example itself and consumes Cargo's matching
debug artifact rather than accepting an arbitrary executable path. Its receipt
records the compiler, command, profile, source identities and executable hash.
The pinned inputs, copied request, sources and executable are rechecked after
execution and again after comparison. This is a local reproducibility record,
not independent execution attestation.

The [started record](language-started.json) binds the exact archived runner,
request, comparison sources and build receipt. The [public build
receipt](language-build.json) replaces only the machine-specific target path
with `<repo>/target`; its original private receipt hash is retained.

```sh
python3 -I -B benchmarks/results/2026-10-16-attention-shift-invariance/language_probe.py \
  --repo "$PWD" --prior "$PRIOR_NUMERICS_ROOT" \
  --reference "$CAPACITY_STUDY_ROOT" --output "$NEW_PRIVATE_DIRECTORY"
```

The script refuses an existing output directory or changed pinned inputs.
Process success means measurement completion; the numerical decision remains
`after.strict_gate.passed`, which is false here.

## Reproduction And Scope

```sh
SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS=1 cargo +1.99.0 test --locked \
  -p st-backend-wgpu --lib resident_tensor::attention -- --test-threads=1
SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS=1 cargo +1.99.0 test --locked \
  -p st-backend-wgpu --test resident_attention_vjp -- --nocapture --test-threads=1
SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS=1 cargo +1.99.0 test --locked \
  -p st-nn --no-default-features --features wgpu --lib resident::byte_ \
  -- --test-threads=1
cargo +1.99.0 build --locked -p st-backend-wgpu \
  --example resident_attention_vjp_browser --target wasm32-unknown-unknown
```

Generate web assets with wasm-bindgen 0.2.129; serve only the generated assets
and `crates/st-backend-wgpu/tests/attention_vjp_browser.html` over loopback,
retaining the page's relative import layout. Download the browser result.
A WASM build alone is not an executed browser test.

Native tests pass: 26 Attention tests plus 120 byte-model/study tests. One
opt-in forward timestamp experiment is ignored, not counted as verified.
The separate shared integration fixture passes. Native/WASM strict Clippy and
Rust formatting checks pass. Four new runner tests and six reused-helper tests
pass; these are tests, not independent language samples.

The production change is consumed by the common Rust backend; no independent
Python/WASM mathematics was added. No wheel was rebuilt or published. Older
checkpoints remain structurally loadable; exact continuation across a changed
kernel/build is not promised. Earlier failed language studies remain failed.
Neither synthetic conservation nor numerical agreement establishes geometry
efficacy, free-running language quality, or readiness for serving promotion.

[Verification](verification.json), [source identities](source-hashes.json) and
[retained-file hashes](private-hashes.json) connect these records. Raw corpus
requests, trained weights, moments, checkpoints, executable and full logs remain
private; public reports contain scalar diagnostics and synthetic fixture results.
The parent PR's latest observed CI run did not start because of a billing lock.
No merge was performed on the strength of local tests alone.
