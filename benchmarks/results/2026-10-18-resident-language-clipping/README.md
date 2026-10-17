# Resident Language Gradient Clipping

This source-only Rust slice adds opt-in global active-gradient clipping to
`ResidentParameters` and `ResidentByteDecoder`, before plain SGD or AdamW.
It is a training mechanism, not a new model-quality or throughput result.
The parent is language-MLOps integration commit
`18a9ed5978b0fd1804556f81063266d6317072d9` (PR #2271).

## Contract And Scope

- Compute one L2 norm over nonzero-rate parameter tensors. Frozen ordinary or
  geometric tensors do not dilute the active budget, but their invalid values
  and inherited guards still reject the complete update.
- Clip before AdamW moment accumulation. Preserve immutable raw gradients,
  owner/revision checks and all-or-none weights/moments/clocks. An invalid
  attempt can be followed by a valid retry without hidden optimizer damage.
- Reduce scaled `(maximum, normalized_square_sum)` pairs on the GPU, then apply
  normal-f32 factors. Finite wide norms and underflowing aggregate scales do not
  become a zero update merely because an intermediate float32 square overflows.
- Reuse `st-kernel-contracts` WGSL norm/factor construction in GraphLearner and
  model-owned parameters. Existing unclipped paths do not allocate clip kernels.

Like update rates, this policy is an explicit step input, not model-checkpoint
state. Exact continuation below supplies it again. Durable study-request identity,
Python orchestration and the packaged WASM training API still need an explicit
versioned recipe. The browser test executes the shared Rust primitive via an
example export; it is **not** public-package API qualification.

## Validation

Native Metal on Apple M4 and actual browser WebGPU both passed all fifteen synthetic
boundary cases: global, zero, small-norm no-op, near-one no-op, wide norm, tiny
scale, frozen exclusion, all-frozen, multi-partial reduction, five floor-neighbor
cases and subnormal no-op bit preservation. Both reports also
exercise strided inputs, foreign/stale rejection, frozen invalid-gradient rejection
and AdamW recovery. Browser success does not attest physical adapter execution.
The subnormal case checks clipping and a zero-rate commit, not nonzero-rate SGD.

CPU Torch 2.12.1 independently computes float64 norms from the exact float32
operands and checks every clipped value (`atol=0`, `rtol=3e-6`), including bitwise
no-op/frozen outputs. Inputs, limits and case coverage are pinned. The oracle uses
SpiralTorch's existing small-norm/near-one policy, not Torch `clip_grad_norm_`'s
extra denominator epsilon. This is a same-policy numerical comparison, not a
standard-Torch API equivalence or speed claim.

A real two-sequence geometric byte-LM test exercises both SGD and AdamW with
Poincare/Euclidean/Fisher-Rao/complex-projective query mixtures. Clipping actually
activates, geometry learns, zero-rate geometry and its moments freeze, and a
JSON-restored model matches uninterrupted execution exactly over three further
mixed-freeze steps. CPU norm/scaling and optimizer candidates independently check
the updates. This tiny correctness fixture does not establish language quality.

Additional local checks passed: shared CPU contract (3), shared clipping suite
(7, including shader validation, the fifteen cases and GraphLearner regressions),
language integration (1), momentum regressions (4), and Torch oracle negative
controls (6). Native/WASM strict backend Clippy and pinned scoped
rustfmt passed. Existing dependency warnings were not suppressed.

The initial Rust test compile failed on a temporary-slice lifetime and was fixed.
The first browser kernel run passed, but independent input checking caught the
page's `JSON.parse`/`JSON.stringify` roundtrip changing signed zero. The page now
posts Rust's original JSON string. A fresh execution and independent comparison
then passed. The original attempt remains in private validation storage; failure
log/result identities are retained in `validation.json`, not erased.

A read-only independent reviewer found an inherited floor-comparison defect:
nine copies of float32 bits `0x332aaaab` have a norm just above epsilon, but the
rounded `floor / sqrt(sum)` test skipped clipping. It reproduced on Metal, leaving
about `3.973643e-8` instead of `3.3333333e-31`. The shared shader now compares an
exact 72-bit significand product for the **represented** norm pair against the
power-of-two floor, without additional division/sqrt/product rounding. Norm
accumulation itself remains approximate f32; this is not an exact-real norm claim.
Each normal significand is an integer `m` with value `m * 2^(exp - 24)`;
the squared maximum times the normalized sum therefore contributes `3 * 24 = 72`
significand bits to the comparator's exponent alignment.
Bit-based magnitude decomposition also avoids the subnormal-input uncertainty of
[WGSL frexp](https://www.w3.org/TR/WGSL/#frexp-builtin).

The added subnormal test exposed an existing, separate nonzero-rate SGD flush to
zero after clipping had preserved the original bits. The failed attempt remains
recorded; subnormal optimizer arithmetic is not qualified by this patch. The
final case explicitly tests no-op clipping and a zero-rate transaction rather
than relaxing an arithmetic tolerance or claiming that limitation was repaired.

Focused independent re-review found no actionable issue in the repair or that
explicit fixture boundary. The reviewer checked limb carries, exponent alignment,
cutoff equality, decomposition and CPU emulation against the Torch oracle, without
running Cargo or GPU tests. Native/browser execution was validated separately.

These checks do not repair or waive the earlier 1,024-update Key-bias numerical
gate. No gradient masking or tolerance relaxation is introduced. The geometry
quality question and the optimizer's geometric interpretation remain separate.

## Reproduce

Use Rust 1.99.0, wasm-bindgen 0.2.129 and an installed CPU Torch. Run GPU commands
serially. The `native.json` and `browser.json` files contain only deterministic
synthetic operands/results, not private corpus or model weights.

```sh
cargo +1.99.0 test --locked -p st-kernel-contracts --lib gradient_clip
SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS=1 cargo +1.99.0 test --locked \
  -p st-backend-wgpu --lib clip -- --test-threads=1
SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS=1 cargo +1.99.0 test --locked \
  -p st-backend-wgpu --lib momentum_gpu -- --test-threads=1
SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS=1 cargo +1.99.0 test --locked \
  -p st-nn --no-default-features --features wgpu --lib \
  clipped_language_sgd_and_adamw_preserve_geometry_freeze_and_resume -- --test-threads=1
cargo +1.99.0 run --locked -p st-backend-wgpu --example resident_parameter_clip > /tmp/clip-native.json
python3 -I -B tools/check_resident_parameter_clip.py \
  --input /tmp/clip-native.json --output /tmp/clip-native-torch.json
python3 -I -B tools/test_resident_parameter_clip.py
cargo +1.99.0 build --locked -p st-backend-wgpu --release \
  --target wasm32-unknown-unknown --example resident_parameter_clip
wasm-bindgen target/wasm32-unknown-unknown/release/examples/resident_parameter_clip.wasm \
  --target web --out-name spiraltorch_wasm --out-dir /tmp/clip-web
python3 -I -S -B benchmarks/results/2026-10-18-resident-language-clipping/stage.py \
  --repo "$PWD" --package /tmp/clip-web --output /tmp/clip-browser
python3 -I -S -B benchmarks/results/2026-10-18-packaged-language-mlops/qualify.py serve \
  --repo /tmp/clip-browser/site --package /tmp/clip-web --output /tmp/clip-browser --port 8828
```

Use new output paths. Open `http://127.0.0.1:8828/`, wait for `passed` and the
saved `main.json`, then run the same Torch checker over that file. The frozen
loopback server exposes only the staged synthetic page/assets and generated
package, not the private validation directory. CI runs the native clipping and
language tests through existing resident-suite filters, builds the example via
WASM all-target Clippy, and checks the committed reports/negative controls with
Torch. Local qualification does not substitute for green CI on the merge head.
